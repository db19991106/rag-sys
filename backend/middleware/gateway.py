"""
API网关中间件
实现功能：JWT鉴权、速率限制、请求校验、追踪上下文
"""

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
from jwt import PyJWTError as JWTError
from datetime import datetime
import hashlib
import json
import re
import uuid
from typing import Optional, Dict, Any
from config import settings
from utils.logger import logger


class TokenBucket:
    """Token Bucket 速率限制算法 (Redis实现)"""

    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.default_limit = 100  # 每分钟100请求
        self.default_burst = 20  # 突发20请求

    async def is_allowed(
        self, key: str, limit: int = None, burst: int = None
    ) -> tuple[bool, Dict]:
        """
        检查是否允许请求

        Returns:
            (is_allowed, rate_limit_info)
        """
        limit = limit or self.default_limit
        burst = burst or self.default_burst

        # 如果没有Redis，使用内存存储(仅适用于单实例)
        if self.redis is None:
            return True, {"limit": limit, "remaining": burst, "reset": 60}

        try:
            import redis

            pipe = self.redis.pipeline()
            now = datetime.utcnow().timestamp()
            window = 60  # 1分钟窗口

            # Token Bucket算法
            bucket_key = f"ratelimit:{key}"

            # 获取当前令牌数
            current = self.redis.get(bucket_key)
            if current is None:
                # 初始化令牌桶
                self.redis.setex(bucket_key, window, burst - 1)
                return True, {
                    "limit": limit,
                    "remaining": burst - 1,
                    "reset": int(now + window),
                }

            current = int(current)
            if current > 0:
                # 消耗一个令牌
                self.redis.decr(bucket_key)
                return True, {
                    "limit": limit,
                    "remaining": current - 1,
                    "reset": int(now + window),
                }
            else:
                # 令牌已用完
                ttl = self.redis.ttl(bucket_key)
                return False, {
                    "limit": limit,
                    "remaining": 0,
                    "reset": int(now + ttl) if ttl > 0 else int(now + window),
                }

        except Exception as e:
            logger.warning(f"Rate limit check failed: {e}")
            # 降级：允许请求
            return True, {"limit": limit, "remaining": 1, "reset": 60}


class APIGatewayMiddleware(BaseHTTPMiddleware):
    """
    API网关中间件
    - JWT/OAuth2验证
    - 速率限制 (Token Bucket)
    - 请求体校验
    - 敏感词过滤
    - 追踪上下文初始化
    """

    def __init__(self, app):
        super().__init__(app)
        self.token_bucket = TokenBucket()
        self.security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
        ]

        # 敏感词列表 (简化版，实际应使用更完整的词库)
        self.sensitive_words = [
            "script",
            "javascript",
            "vbscript",
            "onload",
            "onerror",
            "onclick",
            "eval",
            "expression",
            "url",
            "javascript:",
        ]

    async def dispatch(self, request: Request, call_next):
        # 生成追踪ID
        trace_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.trace_id = trace_id

        # 初始化追踪上下文
        span_id = f"{trace_id}-root"
        baggage = {
            "user_id": None,
            "client_version": request.headers.get("X-Client-Version", "unknown"),
            "trace_id": trace_id,
        }

        # 将追踪信息附加到请求状态
        request.state.span_id = span_id
        request.state.baggage = baggage

        try:
            # ===== 阶段1: JWT鉴权 =====
            auth_result = await self._authenticate(request)
            if not auth_result["success"]:
                return self._error_response(
                    401,
                    auth_result.get("error", "invalid_token"),
                    "AUTH_FAILED",
                    trace_id,
                )

            # 更新baggage中的user_id
            baggage["user_id"] = auth_result.get("user_id")
            request.state.user_id = auth_result.get("user_id")

            # ===== 阶段2: 速率限制 =====
            rate_key = auth_result.get("user_id", request.client.host)
            is_allowed, rate_info = await self.token_bucket.is_allowed(rate_key)

            if not is_allowed:
                # 记录限流事件
                logger.warning(f"Rate limit exceeded for user: {rate_key}")
                return self._error_response(
                    429,
                    "Too Many Requests",
                    "RATE_LIMIT_EXCEEDED",
                    trace_id,
                    headers={
                        "Retry-After": "60",
                        "X-RateLimit-Limit": str(rate_info["limit"]),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(rate_info["reset"]),
                    },
                )

            # ===== 阶段3: 请求体校验 =====
            if request.method in ["POST", "PUT", "PATCH"]:
                validation_result = await self._validate_request_body(request)
                if not validation_result["success"]:
                    return self._error_response(
                        400,
                        validation_result.get("error", "Invalid request body"),
                        "VALIDATION_FAILED",
                        trace_id,
                    )

            # ===== 阶段4: 处理请求 =====
            response = await call_next(request)

            # 添加追踪头
            response.headers["X-Request-ID"] = trace_id
            response.headers["X-Trace-ID"] = trace_id
            response.headers["X-B3-TraceId"] = trace_id
            response.headers["X-B3-SpanId"] = span_id
            response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])

            # 添加安全头
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"

            return response

        except Exception as e:
            logger.error(f"Gateway error: {e}", exc_info=True)
            return self._error_response(
                500, "Internal Server Error", "INTERNAL_ERROR", trace_id
            )

    async def _authenticate(self, request: Request) -> Dict[str, Any]:
        """JWT鉴权"""
        # 公开路径跳过鉴权
        public_paths = ["/docs", "/openapi.json", "/health", "/", "/api/auth"]
        if any(request.url.path.startswith(path) for path in public_paths):
            return {"success": True, "user_id": "anonymous"}

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return {
                "success": False,
                "error": "Missing or invalid Authorization header",
            }

        token = auth_header.split(" ")[1]

        try:
            payload = jwt.decode(
                token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
            )
            user_id = payload.get("sub")
            if user_id is None:
                return {"success": False, "error": "Invalid token payload"}

            # 检查token过期
            exp = payload.get("exp")
            if exp and datetime.utcnow().timestamp() > exp:
                return {"success": False, "error": "Token expired"}

            return {"success": True, "user_id": user_id}

        except JWTError:
            return {"success": False, "error": "Invalid token"}

    async def _validate_request_body(self, request: Request) -> Dict[str, Any]:
        """请求体验证"""
        try:
            body = await request.body()
            if not body:
                return {"success": True}

            # 检查Content-Type，只对JSON请求进行解析
            content_type = request.headers.get("Content-Type", "")
            if "application/json" in content_type:
                # JSON Schema验证
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    return {"success": False, "error": "Invalid JSON"}

                # 检查查询长度
                if "query" in data:
                    query = data["query"]
                    if len(query) > 2000:
                        return {
                            "success": False,
                            "error": "Query too long (max 2000 chars)",
                        }

                    # 敏感词检测
                    if self._contains_sensitive_words(query):
                        return {"success": False, "error": "Query contains sensitive words"}

            # 对于其他Content-Type（如multipart/form-data），跳过JSON解析
            return {"success": True}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _contains_sensitive_words(self, text: str) -> bool:
        """检测敏感词"""
        text_lower = text.lower()
        return any(word in text_lower for word in self.sensitive_words)

    def _error_response(
        self,
        status_code: int,
        message: str,
        code: str,
        trace_id: str,
        headers: Dict = None,
    ) -> JSONResponse:
        """构造错误响应"""
        response = JSONResponse(
            status_code=status_code,
            content={"error": message, "code": code, "trace_id": trace_id},
        )

        if headers:
            for key, value in headers.items():
                response.headers[key] = str(value)

        response.headers["X-Request-ID"] = trace_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """请求日志中间件 - 记录所有请求"""

    async def dispatch(self, request: Request, call_next):
        start_time = datetime.utcnow()
        trace_id = getattr(request.state, "trace_id", "unknown")

        # 记录请求
        logger.info(
            f"Request: {request.method} {request.url.path} [trace_id={trace_id}]"
        )

        try:
            response = await call_next(request)

            # 计算耗时
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000

            logger.info(
                f"Response: {response.status_code} "
                f"[duration={duration:.2f}ms, trace_id={trace_id}]"
            )

            return response

        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(
                f"Request failed: {e} [duration={duration:.2f}ms, trace_id={trace_id}]"
            )
            raise


def create_jwt_token(user_id: str, expires_delta: int = None) -> str:
    """
    创建JWT Token

    Args:
        user_id: 用户ID
        expires_delta: 过期时间(分钟)
    """
    from datetime import timedelta

    expire = datetime.utcnow() + timedelta(
        minutes=expires_delta or settings.jwt_access_token_expire_minutes
    )

    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access",
    }

    return jwt.encode(
        payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )


# 导出
__all__ = [
    "APIGatewayMiddleware",
    "RequestLoggingMiddleware",
    "TokenBucket",
    "create_jwt_token",
]
