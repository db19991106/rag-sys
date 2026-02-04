from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


# ========== 文档相关 ==========
class DocumentStatus(str, Enum):
    PENDING = "pending"
    SPLIT = "split"
    INDEXED = "indexed"
    ERROR = "error"


class DocumentUploadResponse(BaseModel):
    id: str
    name: str
    size: int
    status: DocumentStatus
    upload_time: datetime
    message: str


class DocumentInfo(BaseModel):
    id: str
    name: str
    size: int
    status: DocumentStatus
    upload_time: datetime
    chunk_count: Optional[int] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = []
    file_path: Optional[str] = None  # 实际保存的文件路径


# ========== 切分相关 ==========
class ChunkType(str, Enum):
    # 基础切分方式
    NAIVE = "naive"  # 朴素切分（基于分隔符）
    INTELLIGENT = "intelligent"  # 智能切分（基于文件类型）
    ENHANCED = "enhanced"  # 增强型切分（满足特定要求）
    CHAR = "char"  # 按字符切分
    SENTENCE = "sentence"  # 按句子切分
    PARAGRAPH = "paragraph"  # 按段落切分

    # 专用文档切分
    QA = "qa"  # 问答对切分
    TABLE = "table"  # 表格切分
    PICTURE = "picture"  # 图片切分
    RESUME = "resume"  # 简历切分
    MANUAL = "manual"  # 手动切分
    PAPER = "paper"  # 论文切分
    BOOK = "book"  # 书籍切分
    LAWS = "laws"  # 法律文档切分
    FINANCIAL_REPORT = "financial_report"  # 财务报告切分
    PDF = "pdf"  # PDF智能切分

    # 自定义
    CUSTOM = "custom"  # 自定义规则切分


class ChunkConfig(BaseModel):
    # 基本配置
    type: ChunkType = Field(default=ChunkType.NAIVE, description="切分方式")
    chunk_token_size: int = Field(
        default=512, ge=128, le=2048, description="每个chunk的token数量"
    )

    # 分隔符配置
    delimiters: List[str] = Field(
        default=["\n", "。", "；", "！", "？"], description="主分隔符列表"
    )
    children_delimiters: List[str] = Field(
        default=[], description="子分隔符列表（用于细粒度切分）"
    )
    enable_children: bool = Field(default=False, description="是否启用子分隔符")

    # 重叠配置
    overlapped_percent: float = Field(
        default=0.0, ge=0.0, le=0.5, description="重叠百分比(0-0.5)"
    )

    # 上下文配置
    table_context_size: int = Field(
        default=0, ge=0, le=256, description="表格上下文大小（token）"
    )
    image_context_size: int = Field(
        default=0, ge=0, le=256, description="图片上下文大小（token）"
    )

    # 兼容旧版本
    length: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="切分长度（已废弃，使用chunk_token_size）",
    )
    overlap: int = Field(
        default=50, ge=0, description="重叠长度（已废弃，使用overlapped_percent）"
    )
    custom_rule: str = Field(
        default="", description="自定义分隔符（已废弃，使用delimiters）"
    )


class ChunkInfo(BaseModel):
    id: str
    document_id: str
    num: int
    content: str
    length: int
    embedding_status: str = "pending"
    embedding_dimension: Optional[int] = None


class ChunkResponse(BaseModel):
    chunks: List[ChunkInfo]
    total: int
    auto_embedded: bool = False  # 是否自动向量化成功


# ========== 向量嵌入相关 ==========
class EmbeddingModelType(str, Enum):
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    BGE = "bge"
    OPENAI = "openai"


class EmbeddingConfig(BaseModel):
    model_type: EmbeddingModelType
    model_name: str
    batch_size: int = 32
    device: str = "cpu"


class EmbeddingResponse(BaseModel):
    model_name: str
    dimension: int
    batch_size: int
    status: str
    message: str


# ========== 向量数据库相关 ==========
class VectorDBType(str, Enum):
    FAISS = "faiss"
    MILVUS = "milvus"
    QDRANT = "qdrant"


class VectorDBConfig(BaseModel):
    db_type: VectorDBType
    dimension: int
    index_type: str = "HNSW"
    host: Optional[str] = None
    port: Optional[int] = None
    collection_name: Optional[str] = None


class VectorStatus(BaseModel):
    db_type: str
    total_vectors: int
    dimension: int
    status: str


# ========== 检索相关 ==========
class SimilarityAlgorithm(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT = "dot"


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")
    similarity_threshold: float = Field(
        default=0.4, ge=0, le=1, description="相似度阈值（降低以获取更多结果）"
    )
    algorithm: SimilarityAlgorithm = Field(default=SimilarityAlgorithm.COSINE)
    enable_rerank: bool = Field(default=False, description="是否启用重排序")
    reranker_type: str = Field(
        default="none", description="重排序器类型: none/bge/cross-encoder"
    )
    reranker_model: str = Field(default="", description="重排序模型名称")
    reranker_top_k: int = Field(default=10, description="重排序返回top_k")
    reranker_threshold: float = Field(default=0.0, description="重排序分数阈值")


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    chunk_num: int
    content: str
    similarity: float
    match_keywords: List[str]


class RetrievalResponse(BaseModel):
    query: str
    results: List[RetrievalResult]
    total: int
    latency_ms: float


# ========== RAG 生成相关 ==========
class GenerationConfig(BaseModel):
    llm_provider: str = "openai"
    llm_model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class RAGRequest(BaseModel):
    query: str
    retrieval_config: RetrievalConfig
    generation_config: GenerationConfig
    conversation_id: Optional[str] = Field(default=None, description="对话ID")


# ========== 意图识别相关 ==========
class IntentType(str, Enum):
    QUESTION = "question"  # 问题咨询
    SEARCH = "search"  # 信息搜索
    SUMMARY = "summary"  # 内容总结
    COMPARISON = "comparison"  # 对比分析
    PROCEDURE = "procedure"  # 操作流程
    DEFINITION = "definition"  # 定义说明
    GREETING = "greeting"  # 问候
    OTHER = "other"  # 其他


class IntentResult(BaseModel):
    intent: IntentType
    confidence: float
    details: Dict[str, Any]


class RAGResponse(BaseModel):
    query: str
    answer: str
    context_chunks: List[RetrievalResult]
    generation_time_ms: float
    retrieval_time_ms: float
    total_time_ms: float
    tokens_used: Optional[int] = None


# ========== 对话相关 ==========
class Message(BaseModel):
    id: str
    conversation_id: str
    role: str  # user/assistant/system
    content: str
    timestamp: datetime


class Conversation(BaseModel):
    id: str
    user_id: str
    username: str
    messages: List[Message]
    created_at: datetime
    last_updated: datetime


# ========== 版本管理相关 ==========
class DocumentVersion(BaseModel):
    version_id: str
    document_id: str
    version: str
    created_at: datetime
    changes: str
    file_path: str


class UpdateHistory(BaseModel):
    history_id: str
    action: str
    document_id: str
    document_name: str
    user_id: str
    timestamp: datetime
    details: str


# ========== 通用响应 ==========
class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
