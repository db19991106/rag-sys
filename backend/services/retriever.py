from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import time
import re
import pickle
import os
from rank_bm25 import BM25Okapi
from models import (
    RetrievalConfig,
    RetrievalResult,
    RetrievalResponse,
    SimilarityAlgorithm,
    IntentType,
)
from services.embedding import embedding_service
from services.vector_db import vector_db_manager
from services.document_manager import document_manager
from services.reranker import reranker_manager
from services.intent_recognizer import intent_recognizer, INTENT_DOCUMENT_MAPPING
from services.context_analyzer import context_analyzer
from utils.logger import logger

# BM25 索引持久化路径
BM25_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "vector_db", "bm25_index.pkl")

# BM25 全局缓存（模块级别，跨请求共享）
_bm25_global_cache = None
_bm25_global_cache_key = None
_bm25_global_chunk_mapping = None
_bm25_cache_timestamp = 0
BM25_CACHE_TTL = 3600  # 缓存有效期：1小时（秒）

# 元数据全局缓存（避免每次都从 Milvus 获取）
_metadata_global_cache = None
_metadata_cache_timestamp = 0
METADATA_CACHE_TTL = 1800  # 缓存有效期：30分钟（秒）


def clear_global_caches():
    """
    清空所有全局缓存
    
    在清空向量数据库时调用，确保后续操作不会使用过期的缓存数据。
    """
    global _bm25_global_cache, _bm25_global_cache_key, _bm25_global_chunk_mapping
    global _bm25_cache_timestamp, _metadata_global_cache, _metadata_cache_timestamp
    
    _bm25_global_cache = None
    _bm25_global_cache_key = None
    _bm25_global_chunk_mapping = None
    _bm25_cache_timestamp = 0
    _metadata_global_cache = None
    _metadata_cache_timestamp = 0
    logger.info("BM25 和元数据全局缓存已清空")


class Retriever:
    """检索器 - 执行向量相似度搜索"""

    def __init__(self):
        pass

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig,
        context: Optional[str] = None,
        rewritten_query: Optional[str] = None,
    ) -> RetrievalResponse:
        """
        执行检索

        Args:
            query: 查询文本
            config: 检索配置
            context: 对话上下文（可选）
            rewritten_query: 改写后的查询文本（可选）

        Returns:
            检索响应
        """
        start_time = time.time()

        try:
            # 确定使用的主要查询
            main_query = rewritten_query if rewritten_query else query
            logger.info(f"使用查询: 原始='{query}', 改写='{main_query}'")

            # 1. 意图识别和查询分析
            intent_start = time.time()
            try:
                intent_info = intent_recognizer.recognize_intent(main_query)
                intent_time = (time.time() - intent_start) * 1000
                logger.info(f"✓ 意图识别成功: 类型={intent_info.get('intent')}, 置信度={intent_info.get('confidence', 0):.2f}, 耗时={intent_time:.2f}ms")
            except AttributeError as e:
                # 意图识别功能不可用，使用默认值
                intent_info = {"intent": "unknown", "confidence": 0.0}
                intent_time = (time.time() - intent_start) * 1000
                logger.warning(f"⚠️ 意图识别功能不可用 (AttributeError: {str(e)})，使用默认意图信息, 耗时={intent_time:.2f}ms")
            except Exception as e:
                # 其他意图识别错误
                intent_info = {"intent": "unknown", "confidence": 0.0}
                intent_time = (time.time() - intent_start) * 1000
                logger.warning(f"⚠️ 意图识别失败 ({type(e).__name__}: {str(e)})，使用默认意图信息, 耗时={intent_time:.2f}ms")

            # 2. 查询扩展
            expand_start = time.time()
            expanded_queries = [main_query]
            try:
                if (
                    hasattr(config, "enable_query_expansion")
                    and config.enable_query_expansion
                ):
                    expanded_queries = self._expand_query(main_query, intent_info)
                    expand_time = (time.time() - expand_start) * 1000
                    logger.info(f"查询扩展结果: {expanded_queries}, 耗时={expand_time:.2f}ms")
                else:
                    expand_time = (time.time() - expand_start) * 1000
            except Exception as e:
                expand_time = (time.time() - expand_start) * 1000
                logger.warning(f"查询扩展配置不可用，使用默认查询: {e}, 耗时={expand_time:.2f}ms")

            # 3. 上下文感知查询增强（如果提供了上下文）
            if context:
                enhanced_query = self._enhance_query_with_context(main_query, context)
                expanded_queries.append(enhanced_query)
                logger.info(f"上下文增强查询: {enhanced_query}")

            # 4. 提取核心关键词
            core_keywords = self._extract_core_keywords(main_query)
            logger.info(f"核心关键词: {core_keywords}")

            # 5. 执行向量检索
            vector_start = time.time()
            logger.info("步骤1: 执行向量检索...")
            vector_results = []

            # 确保嵌入模型已加载
            if not embedding_service.is_loaded():
                logger.info("嵌入模型未加载，尝试自动加载...")
                from models import EmbeddingConfig, EmbeddingModelType
                import torch
                
                # 动态检测设备
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"自动检测到设备: {device}")

                # 使用配置的模型
                from config import settings
                
                # 转换模型类型
                model_type = EmbeddingModelType.BGE
                if settings.embedding_model_type == "sentence-transformers":
                    model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS

                embedding_config = EmbeddingConfig(
                    model_type=model_type,
                    model_name=settings.embedding_model_name,
                    batch_size=32,
                    device=device,
                )
                
                # 加载模型
                embedding_response = embedding_service.load_model(embedding_config)
                if embedding_response.status != "success":
                    logger.error(f"自动加载嵌入模型失败: {embedding_response.message}")
                    raise ValueError(f"嵌入模型加载失败: {embedding_response.message}")
                logger.info("嵌入模型加载成功")

            # 批量编码查询向量
            query_vectors = embedding_service.encode(expanded_queries)

            # 搜索向量数据库
            search_top_k = config.top_k * 2  # 多检索一些用于 RRF 融合
            
            for i, (expanded_query, query_vector) in enumerate(
                zip(expanded_queries, query_vectors)
            ):
                distances, metadata_list = vector_db_manager.search(
                    query_vector, search_top_k
                )
                
                # 调试日志：显示原始距离值
                if len(distances) > 0 and len(distances[0]) > 0:
                    raw_distances = distances[0][:5]  # 只显示前5个
                    logger.debug(f"向量检索原始距离 (前5个): {raw_distances}")
                    
                    # 计算并显示相似度
                    for dist in raw_distances:
                        sim = self._calculate_similarity(dist, config.algorithm)
                        logger.debug(f"距离 {dist:.4f} -> 相似度 {sim:.4f}")
                
                # 处理结果
                query_results = self._process_results(distances, metadata_list, config)
                vector_results.extend(query_results)

            # 对向量检索结果去重（基于 chunk_id，保留最高相似度）
            vector_results = self._deduplicate_results(vector_results)
            vector_time = (time.time() - vector_start) * 1000
            logger.info(f"向量检索结果数: {len(vector_results)}, 耗时={vector_time:.2f}ms")
            
            # 记录向量检索的 Top 5 结果用于调试
            if vector_results:
                top5_vec = sorted(vector_results, key=lambda x: x.similarity, reverse=True)[:5]
                logger.info(f"向量检索 Top 5: {[(r.document_name, f'{r.similarity:.4f}') for r in top5_vec]}")

            # 6. 执行 BM25 关键词检索
            bm25_start = time.time()
            logger.info("步骤2: 执行 BM25 关键词检索...")
            bm25_results = self._perform_bm25_retrieval_with_query(main_query, config)
            bm25_time = (time.time() - bm25_start) * 1000
            logger.info(f"BM25 检索结果数: {len(bm25_results)}, 耗时={bm25_time:.2f}ms")
            
            # 记录 BM25 检索的 Top 5 结果用于调试
            if bm25_results:
                top5_bm25 = sorted(bm25_results, key=lambda x: x.similarity, reverse=True)[:5]
                logger.info(f"BM25 检索 Top 5: {[(r.document_name, f'{r.similarity:.4f}') for r in top5_bm25]}")

            # 7. 使用改进的 RRF 融合向量检索和 BM25 检索结果
            rrf_start = time.time()
            logger.info("步骤3: 使用 RRF 融合检索结果...")
            final_results = self._fuse_with_rrf(
                vector_results=vector_results,
                bm25_results=bm25_results,
                k=getattr(config, "rrf_k", 60),  # RRF 平滑参数
                vector_weight=getattr(config, "vector_weight", 0.6),
                bm25_weight=getattr(config, "bm25_weight", 0.4),
                similarity_threshold=config.similarity_threshold,  # 传递阈值
            )
            rrf_time = (time.time() - rrf_start) * 1000
            logger.info(f"RRF 融合后总结果数: {len(final_results)}, 耗时={rrf_time:.2f}ms")
            
            # 记录融合后的 Top 5 结果用于调试
            if final_results:
                top5_final = final_results[:5]
                logger.info(f"RRF 融合 Top 5: {[(r.document_name, f'{r.similarity:.4f}') for r in top5_final]}")

            # 8. 先应用意图-文档过滤（在重排序之前，保留更多候选结果）
            intent = intent_info.get("intent", "unknown")
            if intent and intent != "unknown" and intent != "casual_chat":
                final_results = self._filter_by_intent(final_results, intent)
                logger.info(f"意图-文档过滤后结果数: {len(final_results)} (意图: {intent})")

            # 9. 应用重排序（如果启用）
            rerank_time = 0  # 初始化重排序耗时
            if config.enable_rerank:
                rerank_start = time.time()

                try:
                    # 检查重排序器是否可用（reranker_manager 本身有 is_loaded 方法）
                    if reranker_manager.reranker_type == "none" or reranker_manager.reranker is None:
                        logger.warning("⚠️ 重排序器未初始化，跳过重排序，使用原始检索结果")
                    else:
                        # 更新 reranker 参数（每次请求都更新）
                        reranker_manager.reranker_top_k = config.reranker_top_k
                        reranker_manager.reranker_threshold = config.reranker_threshold

                        # 初始化重排序器（如果需要）
                        if reranker_manager.reranker_type != config.reranker_type:
                            logger.info(f"初始化重排序器: 类型={config.reranker_type}")
                            reranker_manager.initialize(
                                reranker_type=config.reranker_type,
                                model_name=config.reranker_model,
                                device="cuda" if config.device == "cuda" else "cpu",
                                top_k=config.reranker_top_k,
                                threshold=config.reranker_threshold,
                            )

                        # 执行重排序，使用主要查询语句
                        logger.info(f"执行重排序: 模型={config.reranker_model}, top_k={config.reranker_top_k}, threshold={config.reranker_threshold}")
                        final_results = reranker_manager.rerank_results(
                            main_query, final_results, apply_threshold=True
                        )

                        rerank_time = (time.time() - rerank_start) * 1000
                        logger.info(
                            f"✓ 重排序完成: 耗时{rerank_time:.2f}ms, 结果数={len(final_results)}"
                        )
                except Exception as e:
                    logger.error(f"❌ 重排序失败: {str(e)}")
                    logger.warning("⚠️ 使用原始检索结果")
                    rerank_time = (time.time() - rerank_start) * 1000
                    logger.info(f"重排序失败耗时: {rerank_time:.2f}ms")

            # 10. 应用相似度阈值过滤
            logger.info(f"相似度阈值过滤前结果数: {len(final_results)}, 阈值: {config.similarity_threshold}")
            for r in final_results[:5]:
                logger.debug(f"  结果分数: {r.similarity:.4f}")
            final_results = [
                r for r in final_results if r.similarity >= config.similarity_threshold
            ]
            logger.info(f"相似度阈值过滤后结果数: {len(final_results)}")

            # 11. 应用top_k限制
            final_results = final_results[: config.top_k]

            total_time = (time.time() - start_time) * 1000

            # 耗时汇总日志
            logger.info(
                f"检索完成: 查询='{query}', 改写='{main_query}', 返回={len(final_results)}个结果, 总耗时={total_time:.2f}ms"
            )
            logger.info(
                f"耗时明细: 意图识别={intent_time:.0f}ms, 查询扩展={expand_time:.0f}ms, "
                f"向量检索={vector_time:.0f}ms, BM25检索={bm25_time:.0f}ms, RRF融合={rrf_time:.0f}ms"
            )

            return RetrievalResponse(
                query=query,
                rewritten_query=main_query,
                results=final_results,
                total=len(final_results),
                latency_ms=total_time,
                intent=intent_info.get("intent", "unknown"),
                confidence=intent_info.get("confidence", 0.0),
            )

        except Exception as e:
            import traceback

            logger.error(f"检索失败: {str(e)}\n{traceback.format_exc()}")
            return RetrievalResponse(
                query=query,
                rewritten_query=main_query if "main_query" in locals() else query,
                results=[],
                total=0,
                latency_ms=0,
            )

    def _process_results(
        self,
        distances: np.ndarray,
        metadata_list: List[List[dict]],
        config: RetrievalConfig,
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

        # 调试日志：显示输入数据形状
        logger.debug(f"_process_results 输入: distances shape = {distances.shape if hasattr(distances, 'shape') else 'N/A'}, "
                    f"metadata_list len = {len(metadata_list)}")

        if len(distances) == 0 or len(metadata_list) == 0:
            logger.warning("搜索结果为空: distances 或 metadata_list 为空")
            return results
        
        # 检查第一行结果
        if len(distances) > 0 and hasattr(distances[0], '__len__') and len(distances[0]) == 0:
            logger.warning("搜索结果为空: distances[0] 为空数组")
            return results
            
        if len(metadata_list) > 0 and len(metadata_list[0]) == 0:
            logger.warning("搜索结果为空: metadata_list[0] 为空数组")
            return results

        # 取第一行的结果（单个查询）
        distances_row = distances[0]
        metadata_row = metadata_list[0]
        
        logger.debug(f"处理 {len(distances_row)} 个搜索结果")

        for i, (distance, meta) in enumerate(zip(distances_row, metadata_row)):
            # 确保 meta 是字典类型
            if not isinstance(meta, dict):
                logger.warning(f"元数据类型错误: {type(meta)}, 期望 dict")
                continue

            # 过滤空内容
            content = meta.get("content", "")
            if not content or not content.strip():
                logger.debug(
                    f"跳过空内容片段: index={i}, vector_id={meta.get('chunk_id', 'unknown')}"
                )
                continue

            # 根据算法计算相似度
            similarity = self._calculate_similarity(distance, config.algorithm)
            
            # 调试日志：显示每个结果的距离和相似度
            if i < 5:  # 只显示前5个
                logger.debug(f"结果 {i+1}: 距离={distance:.4f}, 相似度={similarity:.4f}, 阈值={config.similarity_threshold}")

            # 获取文档信息
            document_id = meta.get("document_id", "")
            document_name = meta.get("document_name", "Unknown")

            # 提取匹配关键词
            match_keywords = self._extract_match_keywords(meta)

            result = RetrievalResult(
                chunk_id=meta.get("chunk_id", ""),
                document_id=document_id,
                document_name=document_name,
                chunk_num=meta.get("chunk_num", i + 1),
                content=content,
                similarity=similarity,
                match_keywords=match_keywords,
            )

            results.append(result)

        return results

    def _calculate_similarity(
        self, distance: float, algorithm: SimilarityAlgorithm
    ) -> float:
        """
        根据算法计算相似度

        Args:
            distance: 距离值（FAISS L2 距离）
            algorithm: 相似度算法

        Returns:
            相似度分数 (0-1)
        """
        if algorithm == SimilarityAlgorithm.COSINE:
            # FAISS 使用 L2 距离，对于归一化向量：
            # L2距离 d = sqrt(2 * (1 - cos_sim))
            # 因此 cos_sim = 1 - d^2/2
            # 这个公式对于归一化向量是正确的
            # 但 FAISS 返回的距离可能是非归一化的，需要更鲁棒的处理
            similarity = 1 - (distance**2) / 2
            
            # 确保相似度在有效范围内
            similarity = max(0.0, min(1.0, similarity))
            
        elif algorithm == SimilarityAlgorithm.EUCLIDEAN:
            # 欧氏距离: 使用高斯核函数转换为相似度
            # 对于 L2 距离，使用更合适的转换公式
            # 归一化向量的 L2 距离范围是 [0, 2]
            # similarity = exp(-distance^2 / (2 * sigma^2))
            sigma = 0.5  # 调整参数，使得距离1.0时相似度约为0.135
            similarity = np.exp(-(distance**2) / (2 * sigma**2))
            
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
        keywords = metadata.get("keywords", [])

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

    def _expand_query(self, query: str, intent_info: Dict[str, Any]) -> List[str]:
        """
        扩展查询，生成相关查询变体

        Args:
            query: 原始查询
            intent_info: 意图识别结果

        Returns:
            扩展查询列表
        """
        expanded_queries = [query]

        # 1. 基于意图的查询扩展
        intent = intent_info.get("intent", "")
        if intent == "faq":
            # FAQ类型查询，添加不同表述
            expanded_queries.append(f"如何{query.replace('如何', '').replace('?', '')}")
            expanded_queries.append(f"{query.replace('?', '')}的方法")
            expanded_queries.append(f"{query.replace('?', '')}的步骤")
            expanded_queries.append(f"{query.replace('?', '')}怎么做")
        elif intent == "info":
            # 信息查询，添加相关术语
            expanded_queries.append(f"关于{query}的详细信息")
            expanded_queries.append(f"{query}的定义")
            expanded_queries.append(f"{query}的特点")
        elif intent == "troubleshooting":
            # 故障排除，添加相关表述
            expanded_queries.append(
                f"解决{query.replace('如何解决', '').replace('?', '')}"
            )
            expanded_queries.append(
                f"{query.replace('如何解决', '').replace('?', '')}的问题"
            )
            expanded_queries.append(
                f"{query.replace('如何解决', '').replace('?', '')}的解决方案"
            )

        # 2. 基于关键词的查询扩展
        keywords = self._extract_keywords(query)
        if len(keywords) > 1:
            # 添加关键词组合查询
            for i, keyword in enumerate(keywords):
                if len(keyword) > 2:
                    # 关键词前置
                    expanded_queries.append(
                        f"{keyword} {query.replace(keyword, '').strip()}"
                    )
                    # 关键词后置
                    expanded_queries.append(
                        f"{query.replace(keyword, '').strip()} {keyword}"
                    )

        # 3. 同义词扩展 (简单实现)
        synonyms = {
            "问题": ["故障", "错误", "异常"],
            "方法": ["方式", "步骤", "流程"],
            "如何": ["怎样", "怎么", "如何才能"],
            "什么": ["哪些", "什么是", "何谓"],
            "为什么": ["为何", "原因", "为何会"],
        }

        for word, syns in synonyms.items():
            if word in query:
                for syn in syns:
                    expanded_query = query.replace(word, syn)
                    if expanded_query != query:
                        expanded_queries.append(expanded_query)

        # 4. 去重并限制数量
        expanded_queries = list(set(expanded_queries))[:8]  # 最多8个扩展查询

        return expanded_queries

    def _enhance_query_with_context(self, query: str, context: str) -> str:
        """
        使用对话上下文增强查询

        Args:
            query: 原始查询
            context: 对话上下文

        Returns:
            增强后的查询
        """
        # 提取上下文中的关键信息
        context_keywords = self._extract_keywords(context)

        # 过滤掉与当前查询重复的关键词
        query_keywords = self._extract_keywords(query)
        unique_context_keywords = [
            kw for kw in context_keywords if kw not in query_keywords
        ][:5]  # 最多5个上下文关键词

        # 构建增强查询
        if unique_context_keywords:
            context_str = " ".join(unique_context_keywords)
            # 使用更自然的表达方式
            enhanced_query = f"{query}，考虑之前的对话内容：{context_str}"
        else:
            enhanced_query = query

        # 限制查询长度，避免过长
        max_length = 500
        if len(enhanced_query) > max_length:
            # 保留核心信息
            enhanced_query = enhanced_query[:max_length] + "..."

        return enhanced_query

    def _extract_keywords(self, text: str) -> List[str]:
        """
        从文本中提取关键词

        Args:
            text: 输入文本

        Returns:
            关键词列表
        """
        # 简单的关键词提取实现
        # 移除标点符号 (使用标准正则，Python的re不支持\p{Punct})
        import string

        # 构建标点符号字符类
        punct_chars = re.escape(string.punctuation + '。，、；：？！""（）【】《》')
        text = re.sub(r"[\s" + punct_chars + r"]+", " ", text)

        # 分词
        words = text.strip().split()

        # 过滤停用词
        stop_words = {
            "的",
            "了",
            "是",
            "在",
            "有",
            "和",
            "我",
            "你",
            "他",
            "她",
            "它",
            "这",
            "那",
            "并",
            "或",
            "但",
            "如果",
            "因为",
            "所以",
            "如何",
            "什么",
            "为什么",
            "怎样",
        }
        keywords = [word for word in words if word not in stop_words and len(word) > 1]

        # 去重并返回前10个关键词
        return list(set(keywords))[:10]

    def _extract_core_keywords(self, text: str) -> List[str]:
        """
        提取核心关键词

        Args:
            text: 输入文本

        Returns:
            核心关键词列表
        """
        # 提取关键词
        keywords = self._extract_keywords(text)

        # 进一步过滤，保留更有意义的关键词
        core_keywords = []
        for keyword in keywords:
            # 过滤掉过于通用的词
            if len(keyword) > 1:
                core_keywords.append(keyword)

        # 去重并返回前5个核心关键词
        return list(set(core_keywords))[:5]

    def _perform_bm25_retrieval_with_query(
        self, query: str, config: RetrievalConfig
    ) -> List[RetrievalResult]:
        """
        使用查询文本执行 BM25 检索

        Args:
            query: 查询文本
            config: 检索配置

        Returns:
            BM25 检索结果列表
        """
        global _bm25_global_cache, _bm25_global_chunk_mapping, _bm25_cache_timestamp
        
        try:
            # 先检查全局缓存是否存在（预热时已加载）
            if (_bm25_global_cache is not None and 
                _bm25_global_chunk_mapping and 
                time.time() - _bm25_cache_timestamp < BM25_CACHE_TTL):
                bm25 = _bm25_global_cache
                chunk_mapping = _bm25_global_chunk_mapping
                logger.debug("使用预热的 BM25 索引")
            else:
                # 缓存不存在或过期，获取文档片段并构建索引
                all_chunks = self._get_all_document_chunks()
                if not all_chunks:
                    logger.warning("没有可用的文档片段用于 BM25 检索")
                    return []
                bm25, chunk_mapping = self._get_or_create_bm25_index(all_chunks)
            
            if bm25 is None or not chunk_mapping:
                logger.warning("BM25 索引创建失败")
                return []

            # 对查询进行分词
            query_tokens = self._tokenize_text(query)
            if not query_tokens:
                logger.warning(f"查询分词结果为空: {query}")
                return []

            logger.debug(f"BM25 查询分词结果: {query_tokens}")

            # 执行 BM25 检索
            search_top_k = config.top_k * 2
            scores = bm25.get_scores(query_tokens)

            # 排序并获取 top 结果
            top_indices = np.argsort(scores)[::-1][:search_top_k]

            # 构建结果
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    chunk = chunk_mapping[idx]
                    result = RetrievalResult(
                        chunk_id=chunk.get("chunk_id", ""),
                        document_id=chunk.get("document_id", ""),
                        document_name=chunk.get("document_name", "Unknown"),
                        chunk_num=chunk.get("chunk_num", 0),
                        content=chunk.get("content", ""),
                        similarity=float(scores[idx]),  # BM25 分数
                        match_keywords=query_tokens[:5],  # 保存前5个查询词
                    )
                    results.append(result)

            return results
        except Exception as e:
            logger.error(f"BM25 检索失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _get_or_create_bm25_index(
        self, all_chunks: List[Dict[str, Any]]
    ) -> Tuple[Optional[BM25Okapi], List[Dict[str, Any]]]:
        """
        获取或创建 BM25 索引（带全局缓存）

        Args:
            all_chunks: 所有文档片段

        Returns:
            (BM25 索引, chunk 映射列表)
        """
        global _bm25_global_cache, _bm25_global_cache_key, _bm25_global_chunk_mapping, _bm25_cache_timestamp
        
        # 使用更可靠的缓存 key：基于总向量数
        cache_key = len(all_chunks)
        current_time = time.time()
        
        # 检查全局缓存是否有效（考虑 TTL）
        if (_bm25_global_cache is not None and 
            _bm25_global_cache_key == cache_key and
            current_time - _bm25_cache_timestamp < BM25_CACHE_TTL):
            logger.debug(f"使用全局缓存的 BM25 索引 (key={cache_key}, age={int(current_time - _bm25_cache_timestamp)}s)")
            return _bm25_global_cache, _bm25_global_chunk_mapping

        # 准备 BM25 索引
        tokenized_corpus = []
        chunk_mapping = []

        logger.info(f"构建 BM25 索引，文档片段数: {len(all_chunks)}")
        
        for chunk in all_chunks:
            content = chunk.get("content", "")
            if content and content.strip():
                # 分词
                tokens = self._tokenize_text(content)
                if tokens:
                    tokenized_corpus.append(tokens)
                    chunk_mapping.append(chunk)

        if not tokenized_corpus:
            logger.warning("没有有效的文档内容用于构建 BM25 索引")
            return None, []

        # 构建 BM25 索引
        bm25 = BM25Okapi(tokenized_corpus)
        
        # 更新全局缓存
        _bm25_global_cache = bm25
        _bm25_global_cache_key = cache_key
        _bm25_global_chunk_mapping = chunk_mapping
        _bm25_cache_timestamp = current_time
        
        logger.info(f"BM25 索引构建完成并缓存，有效文档数: {len(chunk_mapping)}")

        return bm25, chunk_mapping

    def _filter_by_intent(
        self, results: List[RetrievalResult], intent: str
    ) -> List[RetrievalResult]:
        """
        根据意图过滤检索结果，只保留与意图相关的文档

        Args:
            results: 检索结果列表
            intent: 意图类型字符串

        Returns:
            过滤后的结果列表
        """
        # 将字符串意图转换为 IntentType 枚举
        try:
            intent_type = IntentType(intent) if isinstance(intent, str) else intent
        except ValueError:
            logger.warning(f"未知意图类型: {intent}，跳过过滤")
            return results

        # 获取该意图对应的文档列表
        target_docs = INTENT_DOCUMENT_MAPPING.get(intent_type, [])
        
        if not target_docs:
            logger.debug(f"意图 {intent} 没有对应的文档映射，跳过过滤")
            return results

        # 提取文档名称（不含扩展名）用于匹配
        target_doc_names = set()
        for doc in target_docs:
            # 移除 .md 扩展名
            doc_name = doc.replace(".md", "") if doc.endswith(".md") else doc
            target_doc_names.add(doc_name)
        
        logger.debug(f"意图 {intent} 对应的目标文档: {target_doc_names}")

        # 过滤结果
        filtered_results = []
        for result in results:
            # 检查文档名称是否匹配
            doc_name = result.document_name or ""
            # 移除可能的扩展名
            if doc_name.endswith(".md"):
                doc_name = doc_name[:-3]
            
            if doc_name in target_doc_names:
                filtered_results.append(result)
            else:
                logger.debug(f"过滤掉非目标文档: {result.document_name}")

        # 如果过滤后结果为空，返回原始结果（避免过度过滤）
        if not filtered_results:
            logger.warning(f"意图过滤后结果为空，保留原始结果")
            return results

        return filtered_results

    def _deduplicate_results(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        对检索结果去重，保留最高相似度的结果

        Args:
            results: 检索结果列表

        Returns:
            去重后的结果列表
        """
        unique_results = {}
        for result in results:
            if result.chunk_id:
                if result.chunk_id not in unique_results:
                    unique_results[result.chunk_id] = result
                else:
                    # 保留相似度更高的结果
                    if result.similarity > unique_results[result.chunk_id].similarity:
                        unique_results[result.chunk_id] = result

        # 转换为列表并按相似度排序
        deduped_results = list(unique_results.values())
        deduped_results.sort(key=lambda x: x.similarity, reverse=True)
        
        return deduped_results

    def _fuse_with_rrf(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        k: int = 60,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        similarity_threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        使用改进的 RRF (Reciprocal Rank Fusion) 算法融合向量检索和 BM25 检索结果

        改进后的公式: RRF_score(d) = Σ (similarity_i * w_i / (k + rank_i(d)))
        
        其中:
        - d: 文档
        - k: 平滑参数，通常取 60
        - rank_i(d): 文档 d 在第 i 个检索结果中的排名
        - w_i: 第 i 个检索系统的权重
        - similarity_i: 文档 d 在第 i 个检索中的原始相似度分数

        Args:
            vector_results: 向量检索结果列表
            bm25_results: BM25检索结果列表
            k: RRF 平滑参数，默认 60
            vector_weight: 向量检索权重，默认 0.6
            bm25_weight: BM25 检索权重，默认 0.4
            similarity_threshold: 相似度阈值，用于过滤低质量结果

        Returns:
            融合后的检索结果列表，按 RRF 分数排序
        """
        # 存储每个文档的 RRF 分数、原始相似度和元数据
        rrf_scores: Dict[str, float] = {}
        vector_similarities: Dict[str, float] = {}  # 保存向量检索的原始相似度
        result_map: Dict[str, RetrievalResult] = {}

        # 计算 BM25 分数的最大值，用于归一化
        max_bm25_score = max((r.similarity for r in bm25_results), default=1.0)
        if max_bm25_score <= 0:
            max_bm25_score = 1.0
            
        logger.debug(f"BM25 最大分数: {max_bm25_score:.4f}")

        # 处理向量检索结果 - 结合原始相似度分数
        for rank, result in enumerate(vector_results, start=1):
            if result.chunk_id:
                # 改进的 RRF 分数: similarity * weight / (k + rank)
                # 这样高相似度的文档会获得更高的分数
                rrf_score = result.similarity * vector_weight / (k + rank)
                rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + rrf_score
                vector_similarities[result.chunk_id] = result.similarity
                # 保存结果对象
                if result.chunk_id not in result_map:
                    result_map[result.chunk_id] = result
                    
        # 记录向量检索的 Top 5 用于调试
        if vector_results:
            top5_vector = sorted(vector_results, key=lambda x: x.similarity, reverse=True)[:5]
            logger.debug(f"向量检索 Top 5: {[(r.document_name, f'{r.similarity:.4f}') for r in top5_vector]}")

        # 处理 BM25 检索结果 - 归一化分数后结合
        for rank, result in enumerate(bm25_results, start=1):
            if result.chunk_id:
                # 归一化 BM25 分数到 [0, 1]
                normalized_bm25 = result.similarity / max_bm25_score if max_bm25_score > 0 else 0
                # 改进的 RRF 分数: normalized_bm25 * weight / (k + rank)
                rrf_score = normalized_bm25 * bm25_weight / (k + rank)
                rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + rrf_score
                # 如果该文档不在结果映射中，添加它
                if result.chunk_id not in result_map:
                    result_map[result.chunk_id] = result
                    vector_similarities[result.chunk_id] = 0  # 标记没有向量相似度
                    
        # 记录 BM25 检索的 Top 5 用于调试
        if bm25_results:
            top5_bm25 = sorted(bm25_results, key=lambda x: x.similarity, reverse=True)[:5]
            logger.debug(f"BM25 检索 Top 5: {[(r.document_name, f'{r.similarity:.4f}') for r in top5_bm25]}")

        # 构建 RRF 融合后的结果
        fused_results = []
        for chunk_id, rrf_score in rrf_scores.items():
            result = result_map[chunk_id]
            
            # 计算综合相似度分数
            # 结合 RRF 分数和原始向量相似度
            vec_sim = vector_similarities.get(chunk_id, 0)
            
            # 使用加权组合：70% RRF归一化分数 + 30% 原始向量相似度
            # 这确保了高向量相似度的文档不会被完全忽略
            max_rrf_score = (vector_weight + bm25_weight) / (k + 1)
            normalized_rrf = min(rrf_score / max_rrf_score, 1.0) if max_rrf_score > 0 else 0.0
            
            # 最终相似度 = 0.7 * RRF分数 + 0.3 * 向量相似度
            final_similarity = 0.7 * normalized_rrf + 0.3 * vec_sim
            
            # 更新相似度分数
            result.similarity = final_similarity
            fused_results.append(result)

        # 按相似度降序排序
        fused_results.sort(key=lambda x: x.similarity, reverse=True)

        # 记录融合后的 Top 5 用于调试
        if fused_results:
            top5_fused = fused_results[:5]
            logger.debug(f"RRF 融合 Top 5: {[(r.document_name, f'{r.similarity:.4f}') for r in top5_fused]}")

        logger.info(
            f"RRF 融合完成: 向量结果={len(vector_results)}, BM25结果={len(bm25_results)}, "
            f"融合后={len(fused_results)}, 参数 k={k}, 权重=(向量:{vector_weight}, BM25:{bm25_weight})"
        )

        return fused_results

    def _get_all_document_chunks(self) -> List[Dict[str, Any]]:
        """
        获取所有文档片段（带缓存）

        Returns:
            文档片段列表
        """
        global _metadata_global_cache, _metadata_cache_timestamp
        
        try:
            # 检查缓存是否有效
            current_time = time.time()
            if (_metadata_global_cache is not None and 
                current_time - _metadata_cache_timestamp < METADATA_CACHE_TTL):
                logger.debug("使用缓存的元数据")
                return _metadata_global_cache
            
            # 从向量数据库获取所有元数据
            # 注意：这里需要根据实际的vector_db_manager实现来调整
            # 假设vector_db_manager有get_all_metadata方法
            if hasattr(vector_db_manager, "get_all_metadata"):
                all_metadata = vector_db_manager.get_all_metadata()
                
                # 缓存元数据
                _metadata_global_cache = all_metadata
                _metadata_cache_timestamp = current_time
                logger.info(f"元数据已缓存，共 {len(all_metadata)} 条")
                
                return all_metadata
            else:
                # 如果没有该方法，返回空列表
                logger.warning(
                    "vector_db_manager没有get_all_metadata方法，无法执行BM25检索"
                )
                return []
        except Exception as e:
            logger.error(f"获取文档片段失败: {str(e)}")
            return []

    def _tokenize_text(self, text: str) -> List[str]:
        """
        分词 - 支持中文分词

        Args:
            text: 输入文本

        Returns:
            分词结果列表
        """
        if not text or not text.strip():
            return []
        
        # 尝试使用 jieba 进行中文分词
        try:
            import jieba
            
            # 使用 jieba 进行分词
            tokens = list(jieba.cut(text))
            
            # 过滤停用词和标点
            stop_words = {
                "的", "了", "是", "在", "有", "和", "我", "你", "他", "她", "它",
                "这", "那", "并", "或", "但", "如果", "因为", "所以", "如何", "什么",
                "为什么", "怎样", "呢", "吗", "呀", "吧", "啊", "哦", "嗯", "一个",
                "这个", "那个", "哪些", "哪个", "多少", "几", "什么", "怎样", "如何",
            }
            
            # 标点符号集合
            import string
            punct_set = set(string.punctuation + '。，、；：？！""''（）【】《》\n\r\t')
            
            tokens = [
                token.strip() for token in tokens 
                if token.strip() 
                and token not in stop_words 
                and token not in punct_set
                and len(token.strip()) > 1
            ]
            
            logger.debug(f"jieba 分词结果: {tokens}")
            return tokens
            
        except ImportError:
            logger.warning("jieba 未安装，使用简单分词（对中文效果较差）")
            # 简单的分词实现（回退方案）
            import string

            punct_chars = re.escape(string.punctuation + '。，、；：？！""（）【】《》')
            text = re.sub(r"[\s" + punct_chars + r"]+", " ", text)
            # 分词
            tokens = text.strip().split()
            # 过滤停用词
            stop_words = {
                "的", "了", "是", "在", "有", "和", "我", "你", "他", "她", "它",
                "这", "那", "并", "或", "但", "如果", "因为", "所以", "如何", "什么",
                "为什么", "怎样",
            }
            tokens = [
                token for token in tokens if token not in stop_words and len(token) > 1
            ]
            return tokens

    def warmup_bm25_index(self) -> bool:
        """
        预热 BM25 索引 - 在服务启动时调用
        
        Returns:
            是否成功预热
        """
        global _bm25_global_cache, _bm25_global_cache_key, _bm25_global_chunk_mapping, _bm25_cache_timestamp
        
        logger.info("开始预热 BM25 索引...")
        warmup_start = time.time()
        
        try:
            # 尝试从磁盘加载已保存的索引
            if self._load_bm25_index_from_disk():
                warmup_time = (time.time() - warmup_start) * 1000
                logger.info(f"BM25 索引预热完成（从磁盘加载），耗时={warmup_time:.2f}ms")
                return True
            
            # 磁盘没有索引，从零构建
            all_chunks = self._get_all_chunks_for_bm25()
            if not all_chunks:
                logger.warning("没有可用的文档片段，跳过 BM25 预热")
                return False
            
            # 构建索引
            bm25, chunk_mapping = self._build_bm25_index(all_chunks)
            if bm25 is None:
                logger.warning("BM25 索引构建失败")
                return False
            
            # 保存到磁盘
            self._save_bm25_index_to_disk(bm25, chunk_mapping, len(all_chunks))
            
            # 更新全局缓存
            _bm25_global_cache = bm25
            _bm25_global_cache_key = len(all_chunks)
            _bm25_global_chunk_mapping = chunk_mapping
            _bm25_cache_timestamp = time.time()
            
            warmup_time = (time.time() - warmup_start) * 1000
            logger.info(f"BM25 索引预热完成（新建），文档数={len(chunk_mapping)}，耗时={warmup_time:.2f}ms")
            return True
            
        except Exception as e:
            logger.error(f"BM25 索引预热失败: {str(e)}")
            return False

    def _get_all_chunks_for_bm25(self) -> List[Dict[str, Any]]:
        """获取所有文档片段用于 BM25 索引构建"""
        try:
            # 从向量数据库获取所有元数据
            all_metadata = vector_db_manager.get_all_metadata()
            if not all_metadata:
                logger.warning("向量数据库中没有元数据")
                return []
            
            logger.info(f"获取到 {len(all_metadata)} 条元数据用于 BM25 构建")
            return all_metadata
        except Exception as e:
            logger.error(f"获取文档片段失败: {str(e)}")
            return []

    def _build_bm25_index(self, all_chunks: List[Dict[str, Any]]) -> Tuple[Optional[BM25Okapi], List[Dict[str, Any]]]:
        """
        构建 BM25 索引
        
        Args:
            all_chunks: 所有文档片段
            
        Returns:
            (BM25 索引, chunk 映射列表)
        """
        tokenized_corpus = []
        chunk_mapping = []
        
        for chunk in all_chunks:
            content = chunk.get("content", "")
            if content and content.strip():
                tokens = self._tokenize_text(content)
                if tokens:
                    tokenized_corpus.append(tokens)
                    chunk_mapping.append(chunk)
        
        if not tokenized_corpus:
            return None, []
        
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25, chunk_mapping

    def _save_bm25_index_to_disk(self, bm25: BM25Okapi, chunk_mapping: List[Dict], doc_count: int) -> bool:
        """
        保存 BM25 索引到磁盘
        
        Args:
            bm25: BM25 索引对象
            chunk_mapping: chunk 映射列表
            doc_count: 文档总数（用于缓存 key）
            
        Returns:
            是否保存成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
            
            # 保存索引数据
            index_data = {
                "bm25": bm25,
                "chunk_mapping": chunk_mapping,
                "doc_count": doc_count,
                "timestamp": time.time()
            }
            
            with open(BM25_INDEX_PATH, "wb") as f:
                pickle.dump(index_data, f)
            
            logger.info(f"BM25 索引已保存到磁盘: {BM25_INDEX_PATH}")
            return True
            
        except Exception as e:
            logger.error(f"保存 BM25 索引失败: {str(e)}")
            return False

    def _load_bm25_index_from_disk(self, check_doc_count: bool = False) -> bool:
        """
        从磁盘加载 BM25 索引
        
        Args:
            check_doc_count: 是否检查文档数量变化（默认不检查，加快加载速度）
        
        Returns:
            是否加载成功
        """
        global _bm25_global_cache, _bm25_global_cache_key, _bm25_global_chunk_mapping, _bm25_cache_timestamp
        
        try:
            if not os.path.exists(BM25_INDEX_PATH):
                logger.info("磁盘上没有 BM25 索引文件")
                return False
            
            with open(BM25_INDEX_PATH, "rb") as f:
                index_data = pickle.load(f)
            
            # 可选：检查文档数量是否变化（耗时操作，默认跳过）
            if check_doc_count:
                current_doc_count = len(vector_db_manager.get_all_metadata()) if hasattr(vector_db_manager, 'get_all_metadata') else 0
                
                # 如果文档数量变化超过 10%，需要重建索引
                saved_doc_count = index_data.get("doc_count", 0)
                if current_doc_count > 0 and abs(current_doc_count - saved_doc_count) > saved_doc_count * 0.1:
                    logger.info(f"文档数量变化较大 ({saved_doc_count} -> {current_doc_count})，需要重建 BM25 索引")
                    return False
            
            # 更新全局缓存
            _bm25_global_cache = index_data["bm25"]
            _bm25_global_cache_key = index_data["doc_count"]
            _bm25_global_chunk_mapping = index_data["chunk_mapping"]
            _bm25_cache_timestamp = time.time()
            
            logger.info(f"BM25 索引已从磁盘加载，文档数={len(index_data['chunk_mapping'])}")
            return True
            
        except Exception as e:
            logger.error(f"加载 BM25 索引失败: {str(e)}")
            return False
    
    def rebuild_bm25_index(self) -> bool:
        """
        强制重建 BM25 索引（当文档有较大变化时手动调用）
        
        Returns:
            是否重建成功
        """
        global _bm25_global_cache, _bm25_global_cache_key, _bm25_global_chunk_mapping, _bm25_cache_timestamp
        
        logger.info("开始重建 BM25 索引...")
        rebuild_start = time.time()
        
        try:
            # 删除旧缓存
            if os.path.exists(BM25_INDEX_PATH):
                os.remove(BM25_INDEX_PATH)
            
            # 清空内存缓存
            _bm25_global_cache = None
            _bm25_global_cache_key = 0
            _bm25_global_chunk_mapping = []
            
            # 重新构建
            all_chunks = self._get_all_chunks_for_bm25()
            if not all_chunks:
                logger.warning("没有可用的文档片段，跳过 BM25 重建")
                return False
            
            bm25, chunk_mapping = self._build_bm25_index(all_chunks)
            if bm25 is None:
                logger.warning("BM25 索引重建失败")
                return False
            
            # 保存到磁盘
            self._save_bm25_index_to_disk(bm25, chunk_mapping, len(all_chunks))
            
            # 更新全局缓存
            _bm25_global_cache = bm25
            _bm25_global_cache_key = len(all_chunks)
            _bm25_global_chunk_mapping = chunk_mapping
            _bm25_cache_timestamp = time.time()
            
            rebuild_time = (time.time() - rebuild_start) * 1000
            logger.info(f"BM25 索引重建完成，文档数={len(chunk_mapping)}，耗时={rebuild_time:.2f}ms")
            return True
            
        except Exception as e:
            logger.error(f"BM25 索引重建失败: {str(e)}")
            return False


# 全局检索器实例
retriever = Retriever()
