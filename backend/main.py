from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import jwt
from jwt import PyJWTError as JWTError
from datetime import datetime, timedelta
from config import settings
from utils.logger import logger
from services.state_logger import state_logger
from api import (
    documents,
    chunking,
    embedding,
    vector_db,
    retrieval,
    rag,
    cleaning,
    sync,
    summary,
    conversations,
)
from api import settings as settings_api
from api import enhanced_rag


# CSRF保护中间件
class CSRFMiddleware:
    """CSRF保护中间件"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # 跳过GET、HEAD、OPTIONS请求（这些请求不修改状态）
            if scope["method"] in ["GET", "HEAD", "OPTIONS"]:
                await self.app(scope, receive, send)
                return

            # 对于POST、PUT、DELETE等修改状态的请求，检查CSRF token
            request = Request(scope, receive)

            # 检查CSRF token（从header中获取）
            csrf_token = request.headers.get("X-CSRF-Token")

            # 从cookie中获取CSRF token
            cookie_csrf_token = request.cookies.get("csrf_token")

            # 验证CSRF token
            if (
                not csrf_token
                or not cookie_csrf_token
                or csrf_token != cookie_csrf_token
            ):
                # 如果验证失败，返回403错误
                response = JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "CSRF token验证失败"},
                )
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info(f"启动 {settings.app_name} v{settings.app_version}")
    state_logger.log_system_state(
        "application",
        "starting",
        {"version": settings.app_version, "debug": settings.debug},
    )

    # 初始化向量数据库（使用默认配置）
    from models import VectorDBConfig, VectorDBType
    from services.vector_db import vector_db_manager

    state_logger.log_system_state(
        "vector_db",
        "initializing",
        {"type": settings.vector_db_type, "dimension": settings.faiss_dimension},
    )

    default_config = VectorDBConfig(
        db_type=VectorDBType.FAISS,
        dimension=settings.faiss_dimension,
        index_type=settings.faiss_index_type,
    )
    vector_db_manager.initialize(default_config)
    state_logger.log_system_state(
        "vector_db", "initialized", {"type": settings.vector_db_type, "status": "ready"}
    )

    # 自动加载默认嵌入模型（使用配置文件中的设置）
    from models import EmbeddingConfig, EmbeddingModelType
    from services.embedding import embedding_service

    state_logger.log_system_state(
        "embedding", "loading", {"model": settings.embedding_model_name, "device": settings.embedding_device}
    )

    try:
        # 根据模型名称自动选择模型类型
        if "Qwen3" in settings.embedding_model_name:
            model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
        elif "bge" in settings.embedding_model_name.lower():
            model_type = EmbeddingModelType.BGE
        else:
            model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
        
        embedding_config = EmbeddingConfig(
            model_type=model_type,
            model_name=settings.embedding_model_name,
            batch_size=settings.embedding_batch_size,
            device=settings.embedding_device,
        )
        logger.info(f"正在加载默认嵌入模型: {settings.embedding_model_name}")
        embedding_response = embedding_service.load_model(embedding_config)
        if embedding_response.status == "success":
            logger.info(
                f"默认嵌入模型加载成功: {embedding_response.model_name} (维度: {embedding_response.dimension})"
            )
            state_logger.log_system_state(
                "embedding",
                "loaded",
                {
                    "model": embedding_response.model_name,
                    "dimension": embedding_response.dimension,
                    "status": "ready",
                },
            )
        else:
            logger.warning(f"默认嵌入模型加载失败: {embedding_response.message}")
            state_logger.log_system_state(
                "embedding",
                "failed",
                {"model": "BAAI/bge-base-zh-v1.5", "error": embedding_response.message},
                "warning",
            )
    except Exception as e:
        logger.error(f"加载默认嵌入模型时出现异常: {str(e)}")
        state_logger.log_system_state(
            "embedding",
            "error",
            {"model": "BAAI/bge-base-zh-v1.5", "error": str(e)},
            "error",
        )

    state_logger.log_system_state(
        "application",
        "running",
        {"host": settings.host, "port": settings.port, "status": "ready"},
    )

    yield

    # 关闭时
    logger.info("关闭应用")
    state_logger.log_system_state(
        "application", "stopping", {"status": "shutting_down"}
    )

    # 保存向量数据库
    vector_db_manager.save()
    state_logger.log_system_state("vector_db", "saved", {"status": "persisted"})

    state_logger.log_system_state("application", "stopped", {"status": "closed"})

    # 添加分隔符
    logger.info("=========")


# 创建速率限制器
limiter = Limiter(key_func=get_remote_address)

# 创建 FastAPI 应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG 系统后端 API",
    lifespan=lifespan,
)

# 设置速率限制异常处理器
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# JWT 配置
security = HTTPBearer()
SECRET_KEY = (
    settings.jwt_secret_key
    if hasattr(settings, "jwt_secret_key") and settings.jwt_secret_key
    else "your-secret-key-change-in-production"
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """验证JWT令牌"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return username
    except JWTError:
        raise credentials_exception


# 配置 CORS - 更安全的配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins
    if hasattr(settings, "cors_origins")
    else ["http://localhost:5173", "http://localhost:3000"],  # 默认允许本地开发环境
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # 明确指定允许的方法
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-CSRF-Token",
    ],  # 明确指定允许的headers
    max_age=600,  # 预检请求缓存时间（秒）
)

# 添加受信任的主机中间件（防止Host header攻击）
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
    if settings.debug
    else ["localhost", "127.0.0.1"],  # 生产环境应该配置具体域名
)

# 添加CSRF保护（在开发环境可以禁用）
if not settings.debug:
    app.add_middleware(CSRFMiddleware)

# 添加API网关中间件（JWT鉴权、速率限制、请求校验）
from middleware.gateway import APIGatewayMiddleware, RequestLoggingMiddleware

app.add_middleware(APIGatewayMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# 添加全局异常处理器
from utils.error_handler import app_exception_handler, AppError
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

app.add_exception_handler(Exception, app_exception_handler)
app.add_exception_handler(AppError, app_exception_handler)
app.add_exception_handler(RequestValidationError, app_exception_handler)
app.add_exception_handler(StarletteHTTPException, app_exception_handler)


# 注册路由并应用速率限制
# 文档管理：上传和删除操作限制更严格
app.include_router(documents.router)

# 切分操作：限制中等
app.include_router(chunking.router)

# 嵌入操作：限制中等（计算密集）
app.include_router(embedding.router)

# 向量数据库操作：限制较宽松
app.include_router(vector_db.router)

# 检索操作：限制宽松
app.include_router(retrieval.router)

# RAG生成：限制中等（计算密集）
app.include_router(rag.router)

# 清理操作：限制宽松
app.include_router(cleaning.router)

# 同步操作：限制宽松
app.include_router(sync.router)

# 设置操作：限制严格
app.include_router(settings_api.router)

# 增强版RAG路由
app.include_router(enhanced_rag.router)

# 摘要生成路由
app.include_router(summary.router)

# 对话路由
app.include_router(conversations.router)

# 应用全局速率限制（通过中间件已经配置）
# 特定端点的速率限制已在各自的路由中配置


# 根路径
@app.get("/")
@limiter.limit("100/minute")  # 可选：给根路径也加限流
async def root(request: Request):  # 新增 request 参数
    """根路径"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
    }


# 健康检查
@app.get("/health")
@limiter.limit("100/minute")  # 可选：给健康检查加限流
async def health(request: Request):  # 新增 request 参数
    """健康检查"""
    return {"status": "healthy", "version": settings.app_version}


# 登录端点（核心修改：新增 request 参数）
@app.post("/auth/login")
@limiter.limit("5/minute")
async def login(
    request: Request, username: str, password: str
):  # 新增 request: Request 参数
    """登录获取JWT令牌"""
    # 简单的用户验证（生产环境应使用数据库）
    if username == "admin" and password == "123456":
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = jwt.encode(
            {"sub": username, "exp": datetime.utcnow() + access_token_expires},
            SECRET_KEY,
            algorithm=ALGORITHM,
        )
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误"
        )


# 全局异常处理（修改：补充 request 类型注解）
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):  # 补充类型注解
    """全局异常处理"""
    logger.error(f"未处理的异常: {str(exc)}")
    debug_mode = settings.debug if hasattr(settings, "debug") else False
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "服务器内部错误",
            "error": str(exc) if debug_mode else "Internal server error",
        },
    )


if __name__ == "__main__":
    import uvicorn

    # 增加默认值容错，避免 settings 缺少字段报错
    host = settings.host if hasattr(settings, "host") else "0.0.0.0"
    port = settings.port if hasattr(settings, "port") else 8000
    debug = settings.debug if hasattr(settings, "debug") else False
    log_level = settings.log_level.lower() if hasattr(settings, "log_level") else "info"

    uvicorn.run("main:app", host=host, port=port, reload=debug, log_level=log_level)
