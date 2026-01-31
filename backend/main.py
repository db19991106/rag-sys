from fastapi import FastAPI, Depends, HTTPException, status, Request  # 新增 Request 导入
from fastapi.middleware.cors import CORSMiddleware
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
from api import documents, chunking, embedding, vector_db, retrieval, rag, cleaning, sync
from api import settings as settings_api


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info(f"启动 {settings.app_name} v{settings.app_version}")

    # 初始化向量数据库（使用默认配置）
    from models import VectorDBConfig, VectorDBType
    from services.vector_db import vector_db_manager

    default_config = VectorDBConfig(
        db_type=VectorDBType.FAISS,
        dimension=settings.faiss_dimension,
        index_type=settings.faiss_index_type
    )
    vector_db_manager.initialize(default_config)

    # 自动加载默认嵌入模型（BAAI/bge-base-zh-v1.5）
    from models import EmbeddingConfig, EmbeddingModelType
    from services.embedding import embedding_service

    try:
        embedding_config = EmbeddingConfig(
            model_type=EmbeddingModelType.BGE,
            model_name="BAAI/bge-base-zh-v1.5",
            batch_size=32,
            device="cpu"
        )
        logger.info("正在加载默认嵌入模型: BAAI/bge-base-zh-v1.5")
        embedding_response = embedding_service.load_model(embedding_config)
        if embedding_response.status == "success":
            logger.info(f"默认嵌入模型加载成功: {embedding_response.model_name} (维度: {embedding_response.dimension})")
        else:
            logger.warning(f"默认嵌入模型加载失败: {embedding_response.message}")
    except Exception as e:
        logger.error(f"加载默认嵌入模型时出现异常: {str(e)}")

    yield

    # 关闭时
    logger.info("关闭应用")
    # 保存向量数据库
    vector_db_manager.save()


# 创建速率限制器
limiter = Limiter(key_func=get_remote_address)

# 创建 FastAPI 应用
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG 系统后端 API",
    lifespan=lifespan
)

# 设置速率限制异常处理器
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# JWT 配置
security = HTTPBearer()
SECRET_KEY = settings.jwt_secret_key if hasattr(settings, 'jwt_secret_key') and settings.jwt_secret_key else "your-secret-key-change-in-production"
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
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return username
    except JWTError:
        raise credentials_exception

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins if hasattr(settings, 'cors_origins') else ["*"],  # 增加容错
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 注册路由
app.include_router(documents.router)
app.include_router(chunking.router)
app.include_router(embedding.router)
app.include_router(vector_db.router)
app.include_router(retrieval.router)
app.include_router(rag.router)
app.include_router(cleaning.router)
app.include_router(sync.router)
app.include_router(settings_api.router)


# 根路径
@app.get("/")
@limiter.limit("100/minute")  # 可选：给根路径也加限流
async def root(request: Request):  # 新增 request 参数
    """根路径"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running"
    }


# 健康检查
@app.get("/health")
@limiter.limit("100/minute")  # 可选：给健康检查加限流
async def health(request: Request):  # 新增 request 参数
    """健康检查"""
    return {
        "status": "healthy",
        "version": settings.app_version
    }


# 登录端点（核心修改：新增 request 参数）
@app.post("/auth/login")
@limiter.limit("5/minute")
async def login(request: Request, username: str, password: str):  # 新增 request: Request 参数
    """登录获取JWT令牌"""
    # 简单的用户验证（生产环境应使用数据库）
    if username == "admin" and password == "123456":
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = jwt.encode(
            {"sub": username, "exp": datetime.utcnow() + access_token_expires},
            SECRET_KEY,
            algorithm=ALGORITHM
        )
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误"
        )


# 全局异常处理（修改：补充 request 类型注解）
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):  # 补充类型注解
    """全局异常处理"""
    logger.error(f"未处理的异常: {str(exc)}")
    debug_mode = settings.debug if hasattr(settings, 'debug') else False
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "服务器内部错误",
            "error": str(exc) if debug_mode else "Internal server error"
        }
    )


if __name__ == "__main__":
    import uvicorn

    # 增加默认值容错，避免 settings 缺少字段报错
    host = settings.host if hasattr(settings, 'host') else "0.0.0.0"
    port = settings.port if hasattr(settings, 'port') else 8000
    debug = settings.debug if hasattr(settings, 'debug') else False
    log_level = settings.log_level.lower() if hasattr(settings, 'log_level') else "info"

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level=log_level
    )