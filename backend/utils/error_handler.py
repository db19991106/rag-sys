"""
统一错误处理工具
提供统一的异常处理机制，防止敏感信息泄露
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import Any, Dict, Optional
from utils.logger import logger
from config import settings


class AppError(Exception):
    """应用基础异常类"""
    
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or "INTERNAL_ERROR"
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(AppError):
    """验证错误"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="VALIDATION_ERROR",
            details=details
        )


class NotFoundError(AppError):
    """资源未找到错误"""
    def __init__(self, message: str, resource_type: Optional[str] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="NOT_FOUND",
            details={"resource_type": resource_type} if resource_type else None
        )


class UnauthorizedError(AppError):
    """未授权错误"""
    def __init__(self, message: str = "未授权访问"):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="UNAUTHORIZED"
        )


class ForbiddenError(AppError):
    """禁止访问错误"""
    def __init__(self, message: str = "无权访问此资源"):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="FORBIDDEN"
        )


class ConflictError(AppError):
    """冲突错误"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_409_CONFLICT,
            error_code="CONFLICT",
            details=details
        )


class RateLimitError(AppError):
    """速率限制错误"""
    def __init__(self, message: str = "请求过于频繁，请稍后再试"):
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED"
        )


def create_error_response(
    error: Exception,
    status_code: int,
    error_code: str,
    details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """
    创建统一的错误响应
    
    Args:
        error: 异常对象
        status_code: HTTP状态码
        error_code: 错误代码
        details: 错误详情
        
    Returns:
        JSON响应
    """
    response_data = {
        "error": error_code,
        "message": str(error),
        "status_code": status_code
    }
    
    # 在开发环境添加更多调试信息
    if settings.debug:
        response_data["details"] = details or {}
        if hasattr(error, "__traceback__"):
            import traceback
            response_data["traceback"] = traceback.format_exc()
    else:
        # 生产环境只返回用户友好的错误信息
        if details:
            # 只返回安全的详情信息
            safe_details = {k: v for k, v in details.items() 
                          if not any(sensitive in str(v).lower() 
                                   for sensitive in ['password', 'secret', 'token', 'key'])}
            if safe_details:
                response_data["details"] = safe_details
    
    return JSONResponse(
        status_code=status_code,
        content=response_data
    )


async def app_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    全局异常处理器
    
    Args:
        request: 请求对象
        exc: 异常对象
        
    Returns:
        JSON响应
    """
    # 记录完整的异常信息到日志（包含堆栈跟踪）
    logger.error(
        f"异常发生: {type(exc).__name__}: {str(exc)}",
        exc_info=True,
        extra={
            "path": request.url.path,
            "method": request.method,
            "client": request.client.host if request.client else "unknown"
        }
    )
    
    # 处理应用自定义异常
    if isinstance(exc, AppError):
        return create_error_response(
            error=exc,
            status_code=exc.status_code,
            error_code=exc.error_code,
            details=exc.details
        )
    
    # 处理FastAPI验证错误
    if isinstance(exc, RequestValidationError):
        return create_error_response(
            error=exc,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            details={"errors": exc.errors()}
        )
    
    # 处理Starlette HTTP异常
    if isinstance(exc, StarletteHTTPException):
        return create_error_response(
            error=exc,
            status_code=exc.status_code,
            error_code="HTTP_ERROR",
            details={"detail": exc.detail}
        )
    
    # 处理其他未捕获的异常
    return create_error_response(
        error=exc,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code="INTERNAL_ERROR",
        details={"request_id": getattr(request.state, "request_id", "unknown")}
    )


def log_sensitive_operation(
    operation: str,
    user_id: Optional[str] = None,
    resource_id: Optional[str] = None,
    success: bool = True,
    details: Optional[Dict[str, Any]] = None
):
    """
    记录敏感操作日志
    
    Args:
        operation: 操作类型
        user_id: 用户ID
        resource_id: 资源ID
        success: 是否成功
        details: 操作详情
    """
    from utils.security import secure_config
    
    # 脱敏处理
    safe_details = {}
    if details:
        for key, value in details.items():
            # 对敏感字段进行脱敏
            if any(sensitive in key.lower() 
                   for sensitive in ['password', 'secret', 'token', 'key', 'api_key']):
                safe_details[key] = secure_config.mask_sensitive(str(value))
            else:
                safe_details[key] = value
    
    log_data = {
        "operation": operation,
        "user_id": user_id,
        "resource_id": resource_id,
        "success": success,
        "details": safe_details
    }
    
    if success:
        logger.info(f"敏感操作成功: {operation}", extra=log_data)
    else:
        logger.warning(f"敏感操作失败: {operation}", extra=log_data)