import os
# 禁用 CodeCarbonCallback 以避免版本兼容性问题
os.environ['ACCELERATE_DISABLE_CODE_CARBON'] = '1'

from typing import List, Optional
import numpy as np
from pathlib import Path
from models import EmbeddingModelType, EmbeddingConfig, EmbeddingResponse
from utils.logger import logger
from config import settings


class EmbeddingModel:
    """嵌入模型基类"""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.dimension = 0

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """编码文本为向量"""
        raise NotImplementedError

    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension


class SentenceTransformerModel(EmbeddingModel):
    """Sentence Transformers 模型"""

    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__(model_name, device)
        try:
            # 禁用 CodeCarbonCallback 以避免版本兼容性问题
            import os
            os.environ['ACCELERATE_DISABLE_CODE_CARBON'] = '1'
            
            from sentence_transformers import SentenceTransformer
            
            # 尝试从本地加载，如果失败则从huggingface下载
            cache_dir = settings.upload_dir.replace('/data/docs', '/data/models')
            cache_path = Path(cache_dir) / model_name.replace('/', '--')
            
            # 查找本地模型文件
            local_model_path = None
            if cache_path.exists():
                import os
                for root, dirs, files in os.walk(cache_path):
                    has_model = 'pytorch_model.bin' in files or any(f.endswith('.safetensors') for f in files)
                    has_config = 'config.json' in files
                    has_st_config = 'sentence_bert_config.json' in files
                    
                    if has_model and has_config and has_st_config:
                        local_model_path = root
                        logger.info(f"找到本地模型目录: {local_model_path}")
                        break
            
            if local_model_path:
                logger.info(f"从本地缓存加载模型: {local_model_path}")
                self.model = SentenceTransformer(local_model_path, device=device)
            else:
                logger.info(f"从huggingface下载模型: {model_name}")
                cache_path.mkdir(parents=True, exist_ok=True)
                self.model = SentenceTransformer(model_name, device=device, cache_folder=str(cache_path))
                
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"加载 SentenceTransformer 模型: {model_name} (维度: {self.dimension})")
        except Exception as e:
            logger.error(f"加载 SentenceTransformer 模型失败: {str(e)}")
            raise

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """编码文本为向量"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings


class BGEModel(EmbeddingModel):
    """BGE 模型 - 使用 SentenceTransformer 加载"""

    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__(model_name, device)
        try:
            import os
            
            # 禁用 CodeCarbonCallback 以避免版本兼容性问题
            os.environ['ACCELERATE_DISABLE_CODE_CARBON'] = '1'
            
            from sentence_transformers import SentenceTransformer
            
            # 检查本地缓存
            cache_dir = settings.upload_dir.replace('/data/docs', '/data/models')
            cache_path = Path(cache_dir) / model_name.replace('/', '--')
            
            # 检查本地缓存 - 递归查找模型目录
            local_model_path = None
            if cache_path.exists():
                # 查找包含模型必要文件的目录
                for root, dirs, files in os.walk(cache_path):
                    has_model = 'pytorch_model.bin' in files or any(f.endswith('.safetensors') for f in files)
                    has_config = 'config.json' in files
                    
                    if has_model and has_config:
                        local_model_path = root
                        logger.info(f"找到本地模型目录: {local_model_path}")
                        break
            
            # 设置环境变量
            os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
            os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
            
            if local_model_path:
                # 使用本地路径加载
                logger.info(f"从本地路径加载 BGE 模型: {local_model_path}")
                self.model = SentenceTransformer(local_model_path, device=device)
            else:
                logger.info(f"从HuggingFace下载 BGE 模型: {model_name}")
                cache_path.mkdir(parents=True, exist_ok=True)
                
                # 使用sentence-transformers下载模型，指定缓存目录
                os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_path)
                
                self.model = SentenceTransformer(model_name, device=device, cache_folder=str(cache_path))
                
            # 自动获取维度
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"成功加载 BGE 模型: {model_name} (维度: {self.dimension})")
            
        except Exception as e:
            logger.error(f"加载 BGE 模型失败: {str(e)}")
            logger.error(f"错误详情: {type(e).__name__}")
            raise

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """编码文本为向量"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # BGE模型建议归一化
        )
        return embeddings


class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI 嵌入模型"""

    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__(model_name, device)
        try:
            import openai
            self.client = openai.OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
            # OpenAI 嵌入模型的维度
            if "text-embedding-3-small" in model_name:
                self.dimension = 1536
            elif "text-embedding-3-large" in model_name:
                self.dimension = 3072
            elif "text-embedding-ada-002" in model_name:
                self.dimension = 1536
            else:
                self.dimension = 1536
            logger.info(f"加载 OpenAI 嵌入模型: {model_name} (维度: {self.dimension})")
        except Exception as e:
            logger.error(f"加载 OpenAI 嵌入模型失败: {str(e)}")
            raise

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """编码文本为向量"""
        import openai

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except openai.APIError as e:
                logger.error(f"OpenAI API 调用失败: {str(e)}")
                logger.error(f"失败的批次: {batch}")
                # 抛出异常而不是返回零向量，避免污染检索结果
                raise ValueError(f"OpenAI API 调用失败，无法生成向量: {str(e)}")
            except Exception as e:
                logger.error(f"OpenAI 嵌入编码失败: {str(e)}")
                raise ValueError(f"嵌入编码失败: {str(e)}")

        return np.array(embeddings, dtype=np.float32)


class EmbeddingService:
    """嵌入服务 - 管理多种嵌入模型"""

    def __init__(self):
        self.model: Optional[EmbeddingModel] = None
        self.config: Optional[EmbeddingConfig] = None

    def load_model(self, config: EmbeddingConfig) -> EmbeddingResponse:
        """
        加载嵌入模型

        Args:
            config: 嵌入配置

        Returns:
            加载响应
        """
        try:
            # 确保模型缓存目录存在
            cache_dir = settings.upload_dir.replace('/data/docs', '/data/models')
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"准备加载模型: {config.model_name} (类型: {config.model_type})")
            
            if config.model_type == EmbeddingModelType.SENTENCE_TRANSFORMERS:
                self.model = SentenceTransformerModel(config.model_name, config.device)
            elif config.model_type == EmbeddingModelType.BGE:
                self.model = BGEModel(config.model_name, config.device)
            elif config.model_type == EmbeddingModelType.OPENAI:
                self.model = OpenAIEmbeddingModel(config.model_name, config.device)
            else:
                return EmbeddingResponse(
                    model_name=config.model_name,
                    dimension=0,
                    batch_size=config.batch_size,
                    status="error",
                    message=f"不支持的模型类型: {config.model_type}"
                )

            self.config = config

            return EmbeddingResponse(
                model_name=config.model_name,
                dimension=self.model.get_dimension(),
                batch_size=config.batch_size,
                status="success",
                message=f"模型加载成功: {config.model_name}"
            )

        except Exception as e:
            logger.error(f"加载嵌入模型失败: {str(e)}")
            return EmbeddingResponse(
                model_name=config.model_name,
                dimension=0,
                batch_size=config.batch_size,
                status="error",
                message=f"模型加载失败: {str(e)}"
            )

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        编码文本为向量

        Args:
            texts: 文本列表

        Returns:
            向量数组
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用 load_model")

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.model.get_dimension())

        batch_size = self.config.batch_size if self.config else 32
        return self.model.encode(texts, batch_size)

    def get_dimension(self) -> int:
        """获取当前模型的向量维度"""
        if self.model is None:
            return 0
        return self.model.get_dimension()

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None


# 全局嵌入服务实例
embedding_service = EmbeddingService()