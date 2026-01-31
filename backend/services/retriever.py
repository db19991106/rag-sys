from typing import List, Tuple
import numpy as np
import time
from models import RetrievalConfig, RetrievalResult, RetrievalResponse, SimilarityAlgorithm
from services.embedding import embedding_service
from services.vector_db import vector_db_manager
from services.document_manager import document_manager
from services.reranker import reranker_manager
from utils.logger import logger


class Retriever:
    """检索器 - 执行向量相似度搜索"""

    def __init__(self):
        pass

    def retrieve(self, query: str, config: RetrievalConfig) -> RetrievalResponse:
        """
        执行检索

        Args:
            query: 查询文本
            config: 检索配置

        Returns:
            检索响应
        """
        start_time = time.time()

        try:
            # 1. 将查询转换为向量
            retrieval_start = time.time()
            query_vector = embedding_service.encode([query])[0]
            retrieval_time = (time.time() - retrieval_start) * 1000

            # 2. 搜索向量数据库（获取比top_k更多的结果用于重排序）
            search_top_k = config.top_k * 2 if config.enable_rerank else config.top_k
            search_start = time.time()
            distances, metadata_list = vector_db_manager.search(query_vector, search_top_k)
            search_time = (time.time() - search_start) * 1000

            logger.info(f"向量搜索完成: distances类型={type(distances)}, metadata_list类型={type(metadata_list)}")

            # 3. 计算相似度分数
            results = self._process_results(distances, metadata_list, config)

            # 4. 应用重排序（如果启用）
            if config.enable_rerank:
                rerank_start = time.time()
                
                # 初始化重排序器（如果需要）
                if reranker_manager.reranker_type != config.reranker_type:
                    reranker_manager.initialize(
                        reranker_type=config.reranker_type,
                        model_name=config.reranker_model,
                        device="cpu",
                        top_k=config.reranker_top_k,
                        threshold=config.reranker_threshold
                    )
                
                # 执行重排序
                results = reranker_manager.rerank_results(query, results, apply_threshold=False)
                
                rerank_time = (time.time() - rerank_start) * 1000
                logger.info(f"重排序完成: 耗时{rerank_time:.2f}ms, 结果数={len(results)}")

            # 5. 应用相似度阈值过滤
            results = [r for r in results if r.similarity >= config.similarity_threshold]

            # 6. 应用top_k限制
            results = results[:config.top_k]

            total_time = (time.time() - start_time) * 1000

            logger.info(f"检索完成: 查询='{query}', 返回={len(results)}个结果, 耗时={total_time:.2f}ms")

            return RetrievalResponse(
                query=query,
                results=results,
                total=len(results),
                latency_ms=total_time
            )

        except Exception as e:
            import traceback
            logger.error(f"检索失败: {str(e)}\n{traceback.format_exc()}")
            return RetrievalResponse(
                query=query,
                results=[],
                total=0,
                latency_ms=0
            )

    def _process_results(
        self,
        distances: np.ndarray,
        metadata_list: List[List[dict]],
        config: RetrievalConfig
    ) -> List[RetrievalResult]:
        """
        处理搜索结果

        Args:
            distances: 距离数组
            metadata_list: 元数据列表
            config: 检索配置

        Returns:
            检索结果列表
        """
        results = []

        if len(distances) == 0 or len(metadata_list) == 0:
            return results

        # 取第一行的结果（单个查询）
        distances_row = distances[0]
        metadata_row = metadata_list[0]

        for i, (distance, meta) in enumerate(zip(distances_row, metadata_row)):
            # 确保 meta 是字典类型
            if not isinstance(meta, dict):
                logger.warning(f"元数据类型错误: {type(meta)}, 期望 dict")
                continue

            # 过滤空内容
            content = meta.get('content', '')
            if not content or not content.strip():
                logger.debug(f"跳过空内容片段: index={i}, vector_id={meta.get('chunk_id', 'unknown')}")
                continue

            # 根据算法计算相似度
            similarity = self._calculate_similarity(distance, config.algorithm)

            # 获取文档信息
            document_id = meta.get('document_id', '')
            document_name = meta.get('document_name', 'Unknown')

            # 提取匹配关键词
            match_keywords = self._extract_match_keywords(meta)

            result = RetrievalResult(
                chunk_id=meta.get('chunk_id', ''),
                document_id=document_id,
                document_name=document_name,
                chunk_num=meta.get('chunk_num', i + 1),
                content=content,
                similarity=similarity,
                match_keywords=match_keywords
            )

            results.append(result)

        return results

    def _calculate_similarity(self, distance: float, algorithm: SimilarityAlgorithm) -> float:
        """
        根据算法计算相似度

        Args:
            distance: 距离值
            algorithm: 相似度算法

        Returns:
            相似度分数 (0-1)
        """
        if algorithm == SimilarityAlgorithm.COSINE:
            # 余弦相似度: 距离越小，相似度越高
            # 对于归一化向量，L2距离转换为余弦相似度的正确公式
            # cosine_similarity = 1 - (l2_distance^2 / 2)
            similarity = 1 - (distance ** 2) / 2
        elif algorithm == SimilarityAlgorithm.EUCLIDEAN:
            # 欧氏距离: 使用高斯核函数转换为相似度
            # similarity = exp(-distance^2 / (2 * sigma^2))
            sigma = 1.0  # 可调节参数
            similarity = np.exp(-(distance ** 2) / (2 * sigma ** 2))
        elif algorithm == SimilarityAlgorithm.DOT:
            # 点积 (假设已经归一化)
            similarity = distance
        else:
            similarity = 0.0

        return float(max(0.0, min(1.0, similarity)))

    def _extract_match_keywords(self, metadata: dict) -> List[str]:
        """
        提取匹配关键词

        Args:
            metadata: 元数据

        Returns:
            关键词列表
        """
        keywords = metadata.get('keywords', [])
        
        # 处理各种类型
        if keywords is None:
            return []
        elif isinstance(keywords, str):
            # 如果是字符串，尝试分割或直接作为单个关键词
            if keywords.strip():
                return [keywords.strip()]
            return []
        elif isinstance(keywords, list):
            # 如果是列表，过滤掉空值和非字符串
            return [str(k).strip() for k in keywords if k and str(k).strip()]
        else:
            # 其他类型转为字符串
            return [str(keywords).strip()] if keywords else []


# 全局检索器实例
retriever = Retriever()