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
    rag_stream,  # 流式RAG路由
    cleaning,
    sync,
    summary,
    conversations,
)
from api import settings as settings_api
from api import enhanced_rag


# 最顶层 OPTIONS 请求处理中间件 - 确保 CORS 预检请求能被正确处理
class OptionsMiddleware:
    """处理 OPTIONS 请求的中间件 - 必须在最顶层添加"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "OPTIONS":
            # 直接返回 200 OK，允许 CORS 预检请求通过
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    [b"access-control-allow-origin", b"*"],
                    [b"access-control-allow-methods", b"GET, POST, PUT, DELETE, OPTIONS"],
                    [b"access-control-allow-headers", b"Content-Type, Authorization, X-CSRF-Token, X-Request-ID"],
                    [b"access-control-max-age", b"600"],
                ],
            })
            await send({
                "type": "http.response.body",
                "body": b"",
            })
            return
        await self.app(scope, receive, send)


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

    # 根据 settings.vector_db_type 动态选择向量数据库类型
    db_type_map = {
        "faiss": VectorDBType.FAISS,
        "milvus": VectorDBType.MILVUS,
        "milvus_lite": VectorDBType.MILVUS_LITE,
        "qdrant": VectorDBType.QDRANT,
    }
    db_type = db_type_map.get(settings.vector_db_type, VectorDBType.MILVUS_LITE)

    default_config = VectorDBConfig(
        db_type=db_type,
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

    # 自动加载 Reranker（如果配置启用）
    if settings.reranker_enabled:
        from services.reranker import reranker_manager
        
        state_logger.log_system_state(
            "reranker",
            "loading",
            {"type": settings.reranker_type, "model": settings.reranker_model},
        )
        
        try:
            reranker_manager.initialize(
                reranker_type=settings.reranker_type,
                model_name=settings.reranker_model,
                device=settings.reranker_device,
                top_k=settings.reranker_top_k,
                threshold=settings.reranker_threshold,
            )
            logger.info(
                f"Reranker 加载成功: type={settings.reranker_type}, model={settings.reranker_model}"
            )
            state_logger.log_system_state(
                "reranker",
                "loaded",
                {
                    "type": settings.reranker_type,
                    "model": settings.reranker_model,
                    "status": "ready",
                },
            )
        except Exception as e:
            logger.warning(f"Reranker 加载失败: {str(e)}，将使用无重排序模式")
            state_logger.log_system_state(
                "reranker",
                "failed",
                {"error": str(e)},
                "warning",
            )
    else:
        logger.info("Reranker 未启用，跳过加载")

    # 初始化意图识别器（LLM 兜底）
    try:
        from api.rag import intent_recognizer
        from models import GenerationConfig
        
        intent_config = GenerationConfig(
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model,
            temperature=0.1,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        intent_recognizer.initialize_with_config(intent_config)
        logger.info(f"意图识别器 LLM 兜底已初始化: provider={settings.llm_provider}")
        state_logger.log_system_state(
            "intent_recognizer",
            "initialized",
            {"provider": settings.llm_provider, "model": settings.llm_model},
        )
    except Exception as e:
        logger.warning(f"意图识别器初始化失败: {str(e)}，将仅使用规则匹配")
        state_logger.log_system_state(
            "intent_recognizer",
            "failed",
            {"error": str(e)},
            "warning",
        )

    state_logger.log_system_state(
        "application",
        "running",
        {"host": settings.host, "port": settings.port, "status": "ready"},
    )

    # 后台任务：BM25 预热和文档同步（不阻塞启动）
    async def background_init_tasks():
        """后台初始化任务：BM25 预热和文档同步"""
        import asyncio
        
        # 1. BM25 预热（优先执行，快速，约 2-3 秒）
        try:
            from services.retriever import retriever
            logger.info("[后台任务] 开始预热 BM25 索引...")
            await asyncio.to_thread(retriever.warmup_bm25_index)
            state_logger.log_system_state(
                "bm25_index",
                "warmed_up",
                {"status": "ready"},
            )
        except Exception as e:
            logger.warning(f"[后台任务] BM25 索引预热失败: {str(e)}，首次查询可能较慢")
            state_logger.log_system_state(
                "bm25_index",
                "warmup_failed",
                {"error": str(e)},
                "warning",
            )

        # 2. 文档同步（稍后执行，较慢，约 3 分钟）
        try:
            from services.document_manager import document_manager
            
            # 检查 documents.json 是否为空
            if len(document_manager.documents) == 0:
                logger.info("[后台任务] documents.json 为空，开始从向量数据库同步元数据...")
                
                # 从 Milvus 获取所有元数据（在线程池中执行，避免阻塞）
                all_metadata = await asyncio.to_thread(vector_db_manager.get_all_metadata)
                
                if all_metadata:
                    logger.info(f"[后台任务] 从向量数据库获取到 {len(all_metadata)} 条元数据")
                    
                    # 提取唯一的文档信息
                    doc_set = {}  # doc_id -> {filename, chunk_count}
                    skipped_count = 0
                    for meta in all_metadata:
                        # 兼容多种字段名：document_id (新) / doc_id (旧) / source (备选)
                        doc_id = meta.get("document_id", meta.get("doc_id", ""))
                        # 兼容多种字段名：document_name (新) / filename (旧) / source (备选)
                        filename = meta.get("document_name", meta.get("filename", meta.get("source", "unknown")))
                        
                        if not doc_id:
                            skipped_count += 1
                            continue
                            
                        if doc_id not in doc_set:
                            doc_set[doc_id] = {
                                "filename": filename,
                                "chunk_count": 1,
                            }
                        else:
                            doc_set[doc_id]["chunk_count"] += 1
                    
                    if skipped_count > 0:
                        logger.warning(f"[后台任务] 跳过 {skipped_count} 条无 document_id 的元数据")
                    
                    logger.info(f"[后台任务] 识别到 {len(doc_set)} 个唯一文档")
                    
                    # 同步到 document_manager
                    for doc_id, info in doc_set.items():
                        if doc_id not in document_manager.documents:
                            from models import DocumentInfo, DocumentStatus
                            document_manager.documents[doc_id] = DocumentInfo(
                                id=doc_id,
                                name=info["filename"],
                                size=0,
                                status=DocumentStatus.COMPLETED,
                                chunk_count=info["chunk_count"],
                                upload_time=datetime.now(),
                            )
                    
                    # 保存到 documents.json
                    document_manager._save_documents()
                    logger.info(f"[后台任务] 从向量数据库同步了 {len(doc_set)} 个文档元数据")
                    state_logger.log_system_state(
                        "document_sync",
                        "completed",
                        {"synced_documents": len(doc_set)},
                    )
        except Exception as e:
            logger.warning(f"[后台任务] 文档元数据同步失败: {str(e)}")
    
    # 启动后台任务（不阻塞）
    import asyncio
    asyncio.create_task(background_init_tasks())
    
    logger.info("服务启动完成，后台任务正在执行...")
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
        "X-Request-ID",
        "X-Trace-ID",
        "X-Client-Version",
        "Accept",
        "Accept-Encoding",
        "Accept-Language",
        "Origin",
        "Referer",
        "User-Agent",
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

# 添加 OPTIONS 请求处理中间件（最后添加，确保最先处理预检请求）
app.add_middleware(OptionsMiddleware)

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

# RAG流式生成：支持SSE流式输出
app.include_router(rag_stream.router)

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
    port = settings.port if hasattr(settings, "port") else 9000
    debug = settings.debug if hasattr(settings, "debug") else False
    log_level = settings.log_level.lower() if hasattr(settings, "log_level") else "info"

    # 禁用 uvicorn 默认访问日志，使用自定义中间件处理（已过滤高频请求）
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level=log_level,
        access_log=False
    )
