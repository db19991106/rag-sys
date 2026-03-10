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

    # 新增文档类型切分
    PRODUCT = "product"  # 产品文档切分
    TECHNICAL = "technical"  # 技术规范切分
    COMPLIANCE = "compliance"  # 合规文件切分
    HR = "hr"  # HR文档切分
    PROJECT = "project"  # 项目管理切分
    HYBRID = "hybrid"  # 混合切分-标题切分
    LAYERED = "layered"  # 分层智能切分（三层递进式）
    LAYERED_LLM = "layered_llm"  # 分层LLM切分（正则结构切分 + Embedding语义合并）

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
        default=0.15, ge=0.0, le=0.5, description="重叠百分比(0-0.5)"
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


class LayeredLLMConfig(BaseModel):
    """分层LLM切分配置

    用于 ChunkType.LAYERED_LLM 切分策略
    核心流程: 正则结构切分 → 内容保护 → 句子级语义切分(Embedding相似度合并)
    """
    # 结构切分参数
    max_chunk_tokens: int = Field(
        default=768, ge=128, le=2048, description="最大chunk token数"
    )
    min_chunk_tokens: int = Field(
        default=80, ge=10, le=200, description="最小chunk token数"
    )

    # 语义合并参数
    similarity_threshold: float = Field(
        default=0.65, ge=0.5, le=0.95, description="语义相似度阈值(0.5-0.95)，相邻句子相似度>=此值时尝试合并（注意：只在同一章节内进行）"
    )
    min_sentences_to_merge: int = Field(
        default=2, ge=2, le=10, description="最小合并句子数"
    )
    
    # 特殊内容保护
    preserve_table: bool = Field(default=True, description="保护表格完整性")
    preserve_code: bool = Field(default=True, description="保护代码块完整性")
    preserve_flow: bool = Field(default=True, description="保护流程完整性")
    
    # 性能优化
    batch_embedding: bool = Field(default=True, description="批量计算embedding")
    embedding_batch_size: int = Field(default=32, ge=8, le=128, description="embedding批大小")
    enable_cache: bool = Field(default=True, description="启用embedding缓存")


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
    MILVUS_LITE = "milvus_lite"
    QDRANT = "qdrant"


class VectorDBConfig(BaseModel):
    db_type: VectorDBType
    dimension: int
    index_type: str = "HNSW"
    host: Optional[str] = None
    port: Optional[int] = None
    collection_name: Optional[str] = None
    index_path: Optional[str] = None


class VectorStatus(BaseModel):
    db_type: str
    total_vectors: int
    dimension: int
    status: str
    db_size: int = 0  # 数据库文件大小（字节）


# ========== 检索相关 ==========
class SimilarityAlgorithm(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT = "dot"


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=20, ge=1, le=100, description="返回结果数量")
    similarity_threshold: float = Field(
        default=0.2,
        ge=0,
        le=1,
        description="相似度阈值（降低以获取更多结果，财务制度类建议0.2-0.3）",
    )
    algorithm: SimilarityAlgorithm = Field(default=SimilarityAlgorithm.COSINE)
    enable_rerank: bool = Field(default=True, description="是否启用重排序")
    reranker_type: str = Field(
        default="bge", description="重排序器类型: none/bge/cross-encoder"
    )
    reranker_model: str = Field(default="", description="重排序模型名称")
    reranker_top_k: int = Field(default=5, description="重排序返回top_k")
    reranker_threshold: float = Field(default=0.0, description="重排序分数阈值")
    
    # RRF 融合参数
    rrf_k: int = Field(default=60, ge=1, description="RRF 平滑参数，通常取 60")
    vector_weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="向量检索权重（与 bm25_weight 之和应为 1）"
    )
    bm25_weight: float = Field(
        default=0.5, ge=0.0, le=1.0, description="BM25 检索权重（与 vector_weight 之和应为 1）"
    )
    enable_query_expansion: bool = Field(default=True, description="是否启用查询扩展")


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
    # 默认值将从config.py的settings中获取
    llm_provider: str = "vllm"  # 默认使用vLLM加速
    llm_model: str = "Qwen2.5-7B-Instruct"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    temperature: float = 0.1  # 降低随机性，使输出更确定性
    max_tokens: int = 2000
    top_p: float = 0.7  # 降低采样范围，减少随机性
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class RAGRequest(BaseModel):
    query: str
    retrieval_config: RetrievalConfig
    generation_config: GenerationConfig
    conversation_id: Optional[str] = Field(default=None, description="对话ID")


# ========== 意图识别相关 ==========
class IntentType(str, Enum):
    """意图类型枚举 - 基于业务领域分类"""
    HR = "hr"  # 人力资源：招聘、入职、离职、绩效、薪酬、考勤
    FINANCE = "finance"  # 财务管理：报销、预算、费用、财务制度
    ADMIN = "admin"  # 行政制度：办公管理、资产管理、行政流程
    COMPLIANCE = "compliance"  # 合规安全：合规、安全、保密、风控
    PROCESS = "process"  # 流程管理：审批流程、业务流程、操作规范
    TECH_REPORT = "tech_report"  # 技术报告：技术文档、系统架构、开发指南
    CASUAL_CHAT = "casual_chat"  # 闲聊：非业务问题，使用LLM回答


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

    @property
    def sources(self) -> List[RetrievalResult]:
        """兼容旧代码，sources指向context_chunks"""
        return self.context_chunks


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
    user_profile: Dict[str, Any] = Field(
        default_factory=dict, description="用户画像，存储用户身份信息、职位等"
    )  # 新增
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
