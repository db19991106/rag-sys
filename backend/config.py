from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path
import torch
from utils.security import secure_config, decrypt_sensitive_config


class Settings(BaseSettings):
    # 应用配置
    app_name: str = "RAG Backend"
    app_version: str = "1.0.0"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    # 跨域配置 - 生产环境应该配置具体的域名
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # 文件上传配置
    upload_dir: str = "./data/docs"
    max_file_size: int = 50 * 1024 * 1024  # 50MB (提高限制)
    allowed_extensions: List[str] = [".txt", ".pdf", ".docx", ".md", ".html", ".doc"]

    # 向量数据库配置
    vector_db_type: str = "faiss"  # faiss, milvus, qdrant
    vector_db_dir: str = "./vector_db"
    faiss_index_type: str = "HNSW"
    faiss_dimension: int = 1024  # BGE-M3 输出维度 (原来是2560用于Qwen3-Embedding-4B)

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "rag_vectors"

    # 嵌入模型配置
    embedding_model_type: str = "sentence-transformers"
    # embedding_model_name: str = "BAAI/bge-base-zh-v1.5"
    embedding_model_name: str = "/root/autodl-tmp/rag/backend/data/models/bge-m3"
    embedding_batch_size: int = 32
    # 动态检测设备
    embedding_device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # LLM 配置
    llm_provider: str = "local"  # openai, anthropic, local
    llm_model: str = "Qwen2.5-7B-Instruct"
    llm_api_key: str = ""  # 敏感信息 - 应该加密存储
    llm_base_url: str = "https://api.openai.com/v1"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000

    # 本地 LLM 配置
    local_llm_model_path: str = "./data/models/Qwen2.5-7B-Instruct"
    # 动态检测设备
    local_llm_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    local_llm_load_in_8bit: bool = False

    # 日志配置
    log_level: str = "INFO"
    log_file: str = "/root/autodl-tmp/rag/logs/app.log"

    # JWT 认证配置 - 敏感信息
    jwt_secret_key: str = "your-secret-key-change-in-production-environment"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60

    # 审计日志配置
    audit_log_file: str = "./logs/audit.log"
    state_log_file: str = "./logs/state.log"
    log_dir: str = "./logs"

    # 速率限制配置
    rate_limit_per_minute: int = 100

    # 敏感配置字段列表（用于加密/解密）
    sensitive_fields: List[str] = ["llm_api_key", "jwt_secret_key"]

    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 解密敏感配置
        decrypted_config = decrypt_sensitive_config(
            self.dict(),
            self.sensitive_fields
        )
        for key, value in decrypted_config.items():
            if key in self.sensitive_fields:
                setattr(self, key, value)
        
        # 验证敏感配置
        self._validate_sensitive_config()
        
        # 确保必要的目录存在
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vector_db_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.audit_log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.state_log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def _validate_sensitive_config(self):
        """验证敏感配置"""
        # 检查JWT密钥是否使用了默认值
        if self.jwt_secret_key == "your-secret-key-change-in-production-environment":
            if not self.debug:
                import warnings
                warnings.warn(
                    "⚠️ 警告: 使用了默认的JWT密钥！在生产环境中请设置安全的密钥！"
                )
        
        # 检查API密钥
        if self.llm_provider in ["openai", "anthropic"] and not self.llm_api_key:
            if not self.debug:
                import warnings
                warnings.warn(
                    f"⚠️ 警告: {self.llm_provider} 需要配置API密钥！"
                )
    
    def get_masked_config(self) -> dict:
        """获取掩码后的配置（用于日志输出）"""
        config_dict = self.dict()
        for key in self.sensitive_fields:
            if key in config_dict and config_dict[key]:
                config_dict[key] = secure_config.mask_sensitive(config_dict[key])
        return config_dict


settings = Settings()
