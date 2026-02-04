from functools import wraps
from typing import List
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from services.auth_service import auth_service

security = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    获取当前用户

    Args:
        credentials: HTTP授权凭证

    Returns:
        用户信息
    """
    token = credentials.credentials
    payload = auth_service.verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


def require_permissions(required_permissions: List[str]):
    """
    权限验证装饰器

    Args:
        required_permissions: 所需权限列表

    Returns:
        装饰器函数
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, current_user=None, **kwargs):
            if not current_user:
                credentials = kwargs.get('credentials')
                if credentials:
                    current_user = get_current_user(credentials)
                else:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required",
                    )

            user_role = current_user.get('role', 'user')
            user_permissions = auth_service.get_user_permissions(user_role)

            # 检查是否有所需的权限
            for permission in required_permissions:
                if permission not in user_permissions:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission denied: {permission}",
                    )

            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator


def require_role(required_role: str):
    """
    角色验证装饰器

    Args:
        required_role: 所需角色

    Returns:
        装饰器函数
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, current_user=None, **kwargs):
            if not current_user:
                credentials = kwargs.get('credentials')
                if credentials:
                    current_user = get_current_user(credentials)
                else:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required",
                    )

            user_role = current_user.get('role', 'user')
            if user_role != required_role:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role denied: {required_role} required",
                )

            return await func(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator
