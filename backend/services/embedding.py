import os
# 禁用 CodeCarbonCallback 以避免版本兼容性问题
os.environ['ACCELERATE_DISABLE_CODE_CARBON'] = '1'

from typing import List, Optional, Dict, Any
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
            from pathlib import Path
            os.environ['ACCELERATE_DISABLE_CODE_CARBON'] = '1'
            
            from sentence_transformers import SentenceTransformer
            
            # 处理模型路径
            model_path = Path(model_name).resolve()
            
            if model_path.exists():
                # 本地路径存在，使用绝对路径字符串加载
                load_path = str(model_path)
                logger.info(f"从本地路径加载模型: {load_path}")
            else:
                # 可能是 Hugging Face Hub 的模型名称
                load_path = model_name
                logger.info(f"从 Hugging Face Hub 加载模型: {load_path}")
            
            # 使用 trust_remote_code=True 以支持更多模型
            self.model = SentenceTransformer(load_path, device=device, trust_remote_code=True)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"成功加载模型: {model_name} (维度: {self.dimension})")
            
        except Exception as e:
            logger.error(f"加载 SentenceTransformer 模型失败: {e}")
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
    """BGE 模型 - 使用 Transformers 库直接加载"""

    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__(model_name, device)
        try:
            import os
            from pathlib import Path
            import torch
            from transformers import AutoModel, AutoTokenizer
            
            # 禁用 CodeCarbonCallback 以避免版本兼容性问题
            os.environ['ACCELERATE_DISABLE_CODE_CARBON'] = '1'
            
            # 处理模型路径
            model_path = Path(model_name).resolve()
            
            if model_path.exists():
                # 本地路径存在，使用绝对路径字符串加载
                load_path = str(model_path)
                logger.info(f"从本地路径加载 BGE 模型: {load_path}")
                local_files_only = True
            else:
                # 可能是 Hugging Face Hub 的模型名称
                load_path = model_name
                logger.info(f"从 Hugging Face Hub 加载 BGE 模型: {load_path}")
                local_files_only = False
            
            # 使用 transformers 直接加载
            self.tokenizer = AutoTokenizer.from_pretrained(
                load_path, 
                local_files_only=local_files_only,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                load_path, 
                local_files_only=local_files_only,
                trust_remote_code=True
            )
            self.model.to(device)
            self.model.eval()
            
            # 获取维度 - BGE 模型通常是 1024 维
            # 通过一次前向传播获取实际维度
            with torch.no_grad():
                test_input = self.tokenizer("test", return_tensors="pt", padding=True, truncation=True)
                test_input = {k: v.to(device) for k, v in test_input.items()}
                test_output = self.model(**test_input)
                # 使用 [CLS] token 的表示
                self.dimension = test_output.last_hidden_state[:, 0].shape[-1]
            
            logger.info(f"成功加载 BGE 模型: {model_name} (维度: {self.dimension})")
            
        except Exception as e:
            logger.error(f"加载 BGE 模型失败: {str(e)}")
            logger.error(f"错误详情: {type(e).__name__}")
            raise

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """编码文本为向量"""
        import torch
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # 编码文本
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt",
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 获取模型输出
                outputs = self.model(**inputs)
                
                # 使用 [CLS] token 的表示作为句子嵌入
                embeddings = outputs.last_hidden_state[:, 0]
                
                # 归一化 (BGE模型建议)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)


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
        self.cache: Dict[str, np.ndarray] = {}  # 文本向量缓存
        self.cache_size = 10000  # 缓存大小限制
        self.encode_times = []  # 编码时间统计

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
            
            logger.info(f"准备加载模型: {config.model_name} (类型: {config.model_type}) ")
            
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
            # 清空缓存
            self.cache.clear()

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
        import time
        start_time = time.time()
        
        if self.model is None:
            raise ValueError("模型未加载，请先调用 load_model")

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.model.get_dimension())

        try:
            # 检查缓存
            uncached_texts = []
            cached_vectors = {}
            
            for i, text in enumerate(texts):
                if text in self.cache:
                    cached_vectors[i] = self.cache[text]
                else:
                    uncached_texts.append(text)
            
            # 对未缓存的文本进行编码
            uncached_vectors = None
            if uncached_texts:
                batch_size = self.config.batch_size if self.config else 32
                # 动态调整批处理大小
                if len(uncached_texts) > 100:
                    batch_size = min(64, len(uncached_texts) // 2)
                elif len(uncached_texts) > 1000:
                    batch_size = min(128, len(uncached_texts) // 4)
                
                try:
                    uncached_vectors = self.model.encode(uncached_texts, batch_size)
                    
                    # 更新缓存
                    for text, vector in zip(uncached_texts, uncached_vectors):
                        if len(self.cache) >= self.cache_size:
                            # 移除最早的缓存项
                            oldest_key = next(iter(self.cache))
                            del self.cache[oldest_key]
                        self.cache[text] = vector
                except Exception as e:
                    logger.error(f"编码文本失败: {str(e)}")
                    # 返回空向量数组
                    return np.array([], dtype=np.float32).reshape(0, self.model.get_dimension())
            
            # 构建完整的结果
            result = []
            uncached_idx = 0
            for i in range(len(texts)):
                if i in cached_vectors:
                    result.append(cached_vectors[i])
                else:
                    if uncached_vectors is not None and uncached_idx < len(uncached_vectors):
                        result.append(uncached_vectors[uncached_idx])
                        uncached_idx += 1
                    else:
                        # 如果编码失败，添加零向量
                        result.append(np.zeros(self.model.get_dimension(), dtype=np.float32))
            
            # 计算编码时间
            encode_time = time.time() - start_time
            self.encode_times.append(encode_time)
            # 只保留最近100次的时间记录
            if len(self.encode_times) > 100:
                self.encode_times.pop(0)
            
            if len(texts) > 10:
                logger.info(f"编码 {len(texts)} 个文本，耗时: {encode_time:.4f}s, 缓存命中率: {len(cached_vectors)/len(texts):.2f}")
            
            return np.array(result, dtype=np.float32)
        except Exception as e:
            logger.error(f"编码过程失败: {str(e)}")
            # 返回空向量数组
            return np.array([], dtype=np.float32).reshape(0, self.model.get_dimension())

    def get_dimension(self) -> int:
        """获取当前模型的向量维度"""
        if self.model is None:
            return 0
        return self.model.get_dimension()

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        logger.info("嵌入服务缓存已清空")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        avg_encode_time = sum(self.encode_times) / len(self.encode_times) if self.encode_times else 0
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "average_encode_time": avg_encode_time,
            "encode_count": len(self.encode_times)
        }


# 全局嵌入服务实例
embedding_service = EmbeddingService()
