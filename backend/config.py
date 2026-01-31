from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path


class Settings(BaseSettings):
    # 应用配置
    app_name: str = "RAG Backend"
    app_version: str = "1.0.0"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    # 跨域配置
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # 文件上传配置
    upload_dir: str = "./data/docs"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: List[str] = [".txt", ".pdf", ".docx", ".md"]

    # 向量数据库配置
    vector_db_type: str = "faiss"  # faiss, milvus, qdrant
    vector_db_dir: str = "./vector_db"
    faiss_index_type: str = "HNSW"
    faiss_dimension: int = 768

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "rag_vectors"

    # 嵌入模型配置
    embedding_model_type: str = "sentence-transformers"  # sentence-transformers, bge, openai
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"

    # LLM 配置
    llm_provider: str = "local"  # openai, anthropic, local
    # llm_model: str = "Qwen2.5-0.5B-Instruct"
    llm_model: str = "Qwen2.5-7B-Instruct"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000

    # 本地 LLM 配置
    # local_llm_model_path: str = "./data/models/Qwen2.5-0.5B-Instruct"
    local_llm_model_path: str = "./data/models/Qwen2.5-7B-Instruct"
    local_llm_device: str = "cpu"
    local_llm_load_in_8bit: bool = False

    # 日志配置
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"

    # JWT 认证配置
    jwt_secret_key: str = "your-secret-key-change-in-production-environment"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # 速率限制配置
    rate_limit_per_minute: int = 100

    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保必要的目录存在
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vector_db_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)


settings = Settings()