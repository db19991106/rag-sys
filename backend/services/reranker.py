"""
重排序服务 - 使用重排序模型对检索结果进行精排
支持 BGE Reranker 等重排序模型
"""

from typing import List, Tuple, Optional
from utils.logger import logger
from config import settings


class Reranker:
    """重排序器基类"""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None

    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回top_k结果

        Returns:
            [(原始索引, 重排序分数), ...] 按分数降序排列
        """
        raise NotImplementedError


class BGEReranker(Reranker):
    """BGE Reranker - FlagEmbedding重排序模型"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"):
        super().__init__(model_name, device)
        self._load_model()

    def _load_model(self):
        """加载BGE重排序模型"""
        try:
            from FlagEmbedding import FlagReranker
            
            # 确定模型路径
            model_path = self.model_name
            
            # 尝试从本地加载
            import os
            from pathlib import Path
            
            cache_dir = settings.upload_dir.replace('/data/docs', '/data/models')
            local_model_path = Path(cache_dir) / model_name.replace('/', '--')
            
            if local_model_path.exists():
                model_path = str(local_model_path)
                logger.info(f"从本地加载BGE重排序模型: {model_path}")
            else:
                logger.info(f"从HuggingFace下载BGE重排序模型: {model_name}")
            
            # 初始化模型
            self.model = FlagReranker(
                model_path,
                use_fp16=False,  # CPU上不使用FP16
                device=self.device
            )
            
            logger.info(f"BGE重排序模型加载成功: {self.model_name}")
            
        except ImportError:
            logger.warning("FlagEmbedding未安装，重排序功能不可用。请安装: pip install -U FlagEmbedding")
            raise
        except Exception as e:
            logger.error(f"加载BGE重排序模型失败: {str(e)}")
            raise

    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回top_k结果

        Returns:
            [(原始索引, 重排序分数), ...] 按分数降序排列
        """
        if not self.model:
            logger.warning("重排序模型未加载，返回原始顺序")
            return [(i, 0.0) for i in range(len(documents))]

        if not documents:
            return []

        try:
            # 准备输入对
            pairs = [[query, doc] for doc in documents]
            
            # 计算重排序分数
            scores = self.model.compute_score(pairs)
            
            # 组合索引和分数
            indexed_scores = list(enumerate(scores))
            
            # 按分数降序排序
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 应用top_k
            if top_k and top_k > 0:
                indexed_scores = indexed_scores[:top_k]
            
            return indexed_scores

        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
            # 失败时返回原始顺序
            return [(i, 0.0) for i in range(len(documents))]


class CrossEncoderReranker(Reranker):
    """CrossEncoder重排序器"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        super().__init__(model_name, device)
        self._load_model()

    def _load_model(self):
        """加载CrossEncoder模型"""
        try:
            from sentence_transformers import CrossEncoder
            
            # 确定模型路径
            model_path = self.model_name
            
            # 尝试从本地加载
            from pathlib import Path
            cache_dir = settings.upload_dir.replace('/data/docs', '/data/models')
            local_model_path = Path(cache_dir) / model_name.replace('/', '--')
            
            if local_model_path.exists():
                model_path = str(local_model_path)
                logger.info(f"从本地加载CrossEncoder模型: {model_path}")
            else:
                logger.info(f"从HuggingFace下载CrossEncoder模型: {model_name}")
            
            # 初始化模型
            self.model = CrossEncoder(model_path, device=self.device)
            
            logger.info(f"CrossEncoder模型加载成功: {self.model_name}")
            
        except ImportError:
            logger.warning("sentence-transformers未安装，重排序功能不可用")
            raise
        except Exception as e:
            logger.error(f"加载CrossEncoder模型失败: {str(e)}")
            raise

    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回top_k结果

        Returns:
            [(原始索引, 重排序分数), ...] 按分数降序排列
        """
        if not self.model:
            logger.warning("重排序模型未加载，返回原始顺序")
            return [(i, 0.0) for i in range(len(documents))]

        if not documents:
            return []

        try:
            # 准备输入对
            pairs = [[query, doc] for doc in documents]
            
            # 计算重排序分数
            scores = self.model.predict(pairs)
            
            # 组合索引和分数
            indexed_scores = list(enumerate(scores))
            
            # 按分数降序排序
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 应用top_k
            if top_k and top_k > 0:
                indexed_scores = indexed_scores[:top_k]
            
            return indexed_scores

        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
            # 失败时返回原始顺序
            return [(i, 0.0) for i in range(len(documents))]


class NoReranker(Reranker):
    """无重排序 - 用于禁用重排序功能的占位符"""

    def __init__(self, model_name: str = "", device: str = "cpu"):
        super().__init__(model_name, device)
        logger.info("使用无重排序模式")

    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        返回原始顺序

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回top_k结果

        Returns:
            [(原始索引, 0.0), ...]
        """
        result = [(i, 0.0) for i in range(len(documents))]
        if top_k and top_k > 0:
            result = result[:top_k]
        return result


class RerankerManager:
    """重排序管理器"""

    def __init__(self):
        self.reranker: Optional[Reranker] = None
        self.reranker_type = "none"
        self.reranker_model = ""
        self.reranker_top_k = 10
        self.reranker_threshold = 0.0

    def initialize(self, reranker_type: str = "none", model_name: str = "", device: str = "cpu", top_k: int = 10, threshold: float = 0.0):
        """
        初始化重排序器

        Args:
            reranker_type: 重排序器类型 (none/bge/cross-encoder)
            model_name: 模型名称
            device: 设备 (cpu/cuda)
            top_k: 返回top_k结果
            threshold: 重排序分数阈值
        """
        self.reranker_type = reranker_type
        self.reranker_model = model_name
        self.reranker_top_k = top_k
        self.reranker_threshold = threshold

        try:
            if reranker_type == "none":
                self.reranker = NoReranker()
            elif reranker_type == "bge":
                # 提供多种BGE模型选项
                if not model_name:
                    # 根据设备选择合适的模型
                    if "cuda" in device:
                        model_name = "BAAI/bge-reranker-v2-m3"  # 更大的模型
                    else:
                        model_name = "BAAI/bge-reranker-v2-m3"  # 平衡性能和效果
                self.reranker = BGEReranker(model_name, device)
            elif reranker_type == "cross-encoder":
                # 提供多种CrossEncoder模型选项
                if not model_name:
                    # 根据设备选择合适的模型
                    if "cuda" in device:
                        model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # 更大的模型
                    else:
                        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 更快的模型
                self.reranker = CrossEncoderReranker(model_name, device)
            else:
                logger.warning(f"未知重排序器类型: {reranker_type}，使用无重排序模式")
                self.reranker = NoReranker()

            logger.info(f"重排序器初始化成功: type={reranker_type}, model={model_name}")

        except Exception as e:
            logger.error(f"重排序器初始化失败: {str(e)}，使用无重排序模式")
            self.reranker = NoReranker()

    def rerank_results(self, query: str, results: List, apply_threshold: bool = True) -> List:
        """
        重排序检索结果

        Args:
            query: 查询文本
            results: 检索结果列表
            apply_threshold: 是否应用阈值过滤

        Returns:
            重排序后的结果列表
        """
        if not self.reranker or self.reranker_type == "none":
            return results

        if not results:
            return results

        try:
            import time
            start_time = time.time()
            
            # 提取文档内容
            documents = [r.content for r in results]
            
            # 限制文档长度，避免处理过长的文档
            max_doc_length = 1000  # 最大文档长度
            documents = [doc[:max_doc_length] for doc in documents]
            
            # 执行重排序
            reranked_indices = self.reranker.rerank(query, documents, self.reranker_top_k)
            
            # 根据重排序结果重新组织
            reranked_results = []
            for original_idx, score in reranked_indices:
                # 应用阈值过滤
                if apply_threshold and score < self.reranker_threshold:
                    logger.debug(f"重排序分数 {score:.4f} 低于阈值 {self.reranker_threshold}，跳过")
                    continue
                
                # 复制结果并更新相似度分数
                result = results[original_idx]
                # 更新相似度分数为排序分数，提高后续处理的准确性
                result.similarity = score
                reranked_results.append(result)
            
            # 如果重排序后结果太少，补充原始结果
            if len(reranked_results) < min(self.reranker_top_k, 5):
                # 收集未被选中的原始结果
                selected_indices = {idx for idx, _ in reranked_indices}
                supplementary_results = [r for i, r in enumerate(results) if i not in selected_indices][:5 - len(reranked_results)]
                reranked_results.extend(supplementary_results)
            
            rerank_time = time.time() - start_time
            logger.info(f"重排序完成: {len(results)} -> {len(reranked_results)} 个结果，耗时: {rerank_time:.4f}s")
            
            return reranked_results

        except Exception as e:
            logger.error(f"重排序失败: {str(e)}，返回原始结果")
            return results

    def get_status(self) -> dict:
        """获取重排序器状态"""
        return {
            "enabled": self.reranker_type != "none",
            "type": self.reranker_type,
            "model": self.reranker_model,
            "top_k": self.reranker_top_k,
            "threshold": self.reranker_threshold
        }


# 全局重排序管理器实例
reranker_manager = RerankerManager()