"""
离线知识库构建配置文件
用于配置文档处理、切分、向量化的参数
"""

from pathlib import Path
from typing import List, Dict, Any

# ==================== 路径配置 ====================
# 基础目录（eval_config.py所在目录）
BASE_DIR = Path(__file__).parent

# 文档目录 - 存放待处理的原始文档
DOCS_DIR = BASE_DIR / "data" / "docs"

# 向量数据库目录
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# 模型目录（共享模型）
MODELS_DIR = Path("/root/autodl-tmp/rag/backend/data/models")

# 测试数据集路径
TEST_DATASET_PATH = BASE_DIR / "test_dataset_extended.json"

# 支持的文档格式
SUPPORTED_EXTENSIONS = [".md", ".txt", ".pdf", ".docx", ".doc"]

# ==================== 文档切分配置 ====================
# 切分方法选择
# 可选值: "financial_v2" | "financial" | "intelligent" | "naive" | "enhanced"
CHUNKING_METHOD = "financial_v2"

# 财务报销制度切分器配置 (当 CHUNKING_METHOD = "financial_v2" 时使用)
FINANCIAL_CHUNKER_CONFIG = {
    "max_chunk_size": 1000,  # 最大片段长度（字符数）
    "preserve_tables": True,  # 保留表格完整性
    "extract_metadata": True,  # 提取元数据（职级、费用类型等）
}

# 通用切分器配置 (其他切分方法使用)
CHUNKER_CONFIG = {
    "chunk_size": 512,  # 片段大小
    "chunk_overlap": 50,  # 片段重叠
    "max_tokens": 1000,  # 最大token数
}

# ==================== 嵌入模型配置 ====================
# 嵌入模型类型
# 可选值: "bge" | "sentence_transformers" | "openai"
EMBEDDING_MODEL_TYPE = "bge"

# BGE 模型配置
BGE_MODEL_CONFIG = {
    "model_name": "bge-base-zh-v1.5",  # 模型名称或路径
    "model_path": "/root/autodl-tmp/rag/backend/data/models/bge-base-zh-v1.5",
    "device": "cuda",  # 运行设备: cuda | cpu
    "normalize_embeddings": True,  # 是否归一化
}

# 其他嵌入模型配置
EMBEDDING_CONFIG = {
    "batch_size": 32,  # 批处理大小
    "max_seq_length": 512,  # 最大序列长度
}

# ==================== 向量数据库配置 ====================
# 向量数据库类型
# 可选值: "faiss" | "milvus" | "qdrant"
VECTOR_DB_TYPE = "faiss"

# FAISS 配置
FAISS_CONFIG = {
    "dimension": 768,  # 向量维度
    "index_type": "HNSW",  # 索引类型: HNSW | IVF | PQ | Flat
    "hnsw_m": 16,  # HNSW图的连接数
    "hnsw_ef_construction": 200,  # HNSW构建参数
    "hnsw_ef_search": 128,  # HASS搜索参数
}

# ==================== 处理配置 ====================
# 批处理配置
BATCH_CONFIG = {
    "batch_size": 100,  # 批处理文档数量
    "embedding_batch_size": 32,  # 向量化批处理大小
}

# 错误处理配置
ERROR_HANDLING = {
    "continue_on_error": True,  # 出错时是否继续处理其他文档
    "max_retries": 3,  # 最大重试次数
    "retry_delay": 1.0,  # 重试延迟（秒）
}

# ==================== 日志配置 ====================
LOG_CONFIG = {
    "log_level": "INFO",  # 日志级别: DEBUG | INFO | WARNING | ERROR
    "log_file": BASE_DIR / "logs" / "eval_app.log",  # 日志文件路径
    "log_to_console": True,  # 是否输出到控制台
}

# ==================== 高级配置 ====================
# 是否启用增量更新
INCREMENTAL_UPDATE = False

# 文档去重配置
DEDUPLICATION = {
    "enabled": False,  # 是否启用去重
    "method": "hash",  # 去重方法: hash | content
}

# 元数据提取配置
METADATA_EXTRACTION = {
    "extract_keywords": True,  # 提取关键词
    "extract_summary": False,  # 提取摘要
    "extract_entities": False,  # 提取实体
}

# ==================== 预定义配置方案 ====================
# 财务报销制度完整配置方案
FINANCIAL_CONFIG = {
    "chunking_method": "financial_v2",
    "embedding_model_type": "bge",
    "vector_db_type": "faiss",
    "docs_dir": DOCS_DIR,
    "vector_db_dir": VECTOR_DB_DIR,
    "models_dir": MODELS_DIR,
    "test_dataset_path": TEST_DATASET_PATH,
    "supported_extensions": [".md", ".txt"],
    "chunking_config": FINANCIAL_CHUNKER_CONFIG,
    "embedding_config": BGE_MODEL_CONFIG,
    "vector_db_config": FAISS_CONFIG,
    "batch_config": BATCH_CONFIG,
    "error_handling": ERROR_HANDLING,
    "log_config": LOG_CONFIG,
}

# 通用文档配置方案
GENERAL_CONFIG = {
    "chunking_method": "intelligent",
    "embedding_model_type": "bge",
    "vector_db_type": "faiss",
    "docs_dir": DOCS_DIR,
    "vector_db_dir": VECTOR_DB_DIR,
    "models_dir": MODELS_DIR,
    "test_dataset_path": TEST_DATASET_PATH,
    "supported_extensions": SUPPORTED_EXTENSIONS,
    "chunking_config": CHUNKER_CONFIG,
    "embedding_config": BGE_MODEL_CONFIG,
    "vector_db_config": FAISS_CONFIG,
    "batch_config": BATCH_CONFIG,
    "error_handling": ERROR_HANDLING,
    "log_config": LOG_CONFIG,
}


def get_config(config_name: str = "default") -> Dict[str, Any]:
    """
    获取配置方案

    Args:
        config_name: 配置方案名称
            - "default": 使用默认配置
            - "financial": 财务报销制度配置
            - "general": 通用文档配置

    Returns:
        配置字典
    """
    configs = {
        "default": {
            "chunking_method": CHUNKING_METHOD,
            "embedding_model_type": EMBEDDING_MODEL_TYPE,
            "vector_db_type": VECTOR_DB_TYPE,
            "docs_dir": DOCS_DIR,
            "vector_db_dir": VECTOR_DB_DIR,
            "models_dir": MODELS_DIR,
            "test_dataset_path": TEST_DATASET_PATH,
            "supported_extensions": SUPPORTED_EXTENSIONS,
            "chunking_config": FINANCIAL_CHUNKER_CONFIG
            if CHUNKING_METHOD == "financial_v2"
            else CHUNKER_CONFIG,
            "embedding_config": BGE_MODEL_CONFIG
            if EMBEDDING_MODEL_TYPE == "bge"
            else EMBEDDING_CONFIG,
            "vector_db_config": FAISS_CONFIG,
            "batch_config": BATCH_CONFIG,
            "error_handling": ERROR_HANDLING,
            "log_config": LOG_CONFIG,
        },
        "financial": FINANCIAL_CONFIG,
        "general": GENERAL_CONFIG,
    }

    return configs.get(config_name, configs["default"])


# 当前使用的配置
CURRENT_CONFIG = get_config("default")
