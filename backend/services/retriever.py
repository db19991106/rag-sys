from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import time
import re
from rank_bm25 import BM25Okapi
from models import (
    RetrievalConfig,
    RetrievalResult,
    RetrievalResponse,
    SimilarityAlgorithm,
)
from services.embedding import embedding_service
from services.vector_db import vector_db_manager
from services.document_manager import document_manager
from services.reranker import reranker_manager
from services.intent_recognizer import intent_recognizer
from services.context_analyzer import context_analyzer
from utils.logger import logger


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
            try:
                intent_info = intent_recognizer.recognize_intent(main_query)
                logger.info(f"✓ 意图识别成功: 类型={intent_info.get('intent')}, 置信度={intent_info.get('confidence', 0):.2f}")
            except AttributeError as e:
                # 意图识别功能不可用，使用默认值
                intent_info = {"intent": "unknown", "confidence": 0.0}
                logger.warning(f"⚠️ 意图识别功能不可用 (AttributeError: {str(e)})，使用默认意图信息")
            except Exception as e:
                # 其他意图识别错误
                intent_info = {"intent": "unknown", "confidence": 0.0}
                logger.warning(f"⚠️ 意图识别失败 ({type(e).__name__}: {str(e)})，使用默认意图信息")

            # 2. 查询扩展
            expanded_queries = [main_query]
            try:
                if (
                    hasattr(config, "enable_query_expansion")
                    and config.enable_query_expansion
                ):
                    expanded_queries = self._expand_query(main_query, intent_info)
                    logger.info(f"查询扩展结果: {expanded_queries}")
            except Exception as e:
                logger.warning(f"查询扩展配置不可用，使用默认查询: {e}")

            # 3. 上下文感知查询增强（如果提供了上下文）
            if context:
                enhanced_query = self._enhance_query_with_context(main_query, context)
                expanded_queries.append(enhanced_query)
                logger.info(f"上下文增强查询: {enhanced_query}")

            # 4. 提取核心关键词
            core_keywords = self._extract_core_keywords(main_query)
            logger.info(f"核心关键词: {core_keywords}")

            # 5. 执行向量检索（主检索）
            logger.info("步骤1: 执行向量检索（主检索）...")
            vector_results = []

            # 批量编码查询向量
            query_vectors = embedding_service.encode(expanded_queries)

            for i, (expanded_query, query_vector) in enumerate(
                zip(expanded_queries, query_vectors)
            ):
                # 搜索向量数据库
                search_top_k = (
                    config.top_k * 2 if config.enable_rerank else config.top_k
                )
                distances, metadata_list = vector_db_manager.search(
                    query_vector, search_top_k
                )

                # 处理结果
                query_results = self._process_results(distances, metadata_list, config)

                # 为不同查询的结果添加权重
                weight = 1.0 if i == 0 else 0.7  # 原始查询权重更高
                for result in query_results:
                    result.similarity *= weight
                    vector_results.append(result)

            # 6. 执行关键词检索（副检索）
            logger.info("步骤2: 执行关键词检索（副检索）...")
            keyword_results = []

            if core_keywords:
                # 6.1 执行BM25检索
                bm25_results = self._perform_bm25_retrieval(core_keywords, config)
                logger.info(f"BM25检索结果数: {len(bm25_results)}")

                # 6.2 执行语义关键词检索
                semantic_results = self._perform_semantic_keyword_retrieval(
                    core_keywords, config
                )
                logger.info(f"语义关键词检索结果数: {len(semantic_results)}")

                # 6.3 融合关键词检索结果
                keyword_results = self._fuse_keyword_results(
                    bm25_results, semantic_results
                )
                logger.info(f"融合后关键词检索结果数: {len(keyword_results)}")

            # 7. 融合向量检索和关键词检索结果
            logger.info("步骤3: 融合检索结果...")
            final_results = self._fuse_retrieval_results(
                vector_results, keyword_results, config
            )
            logger.info(f"融合后总结果数: {len(final_results)}")

            # 8. 应用重排序（如果启用）
            if config.enable_rerank:
                rerank_start = time.time()

                try:
                    # 检查重排序器是否可用
                    if not hasattr(reranker_manager, 'is_loaded') or not reranker_manager.is_loaded():
                        logger.warning("⚠️ 重排序器未加载，跳过重排序，使用原始检索结果")
                    else:
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
                        logger.info(f"执行重排序: 模型={config.reranker_model}, top_k={config.reranker_top_k}")
                        final_results = reranker_manager.rerank_results(
                            main_query, final_results, apply_threshold=False
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

            # 9. 应用相似度阈值过滤
            final_results = [
                r for r in final_results if r.similarity >= config.similarity_threshold
            ]

            # 10. 应用top_k限制
            final_results = final_results[: config.top_k]

            total_time = (time.time() - start_time) * 1000

            logger.info(
                f"检索完成: 查询='{query}', 改写='{main_query}', 返回={len(final_results)}个结果, 耗时={total_time:.2f}ms"
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
            content = meta.get("content", "")
            if not content or not content.strip():
                logger.debug(
                    f"跳过空内容片段: index={i}, vector_id={meta.get('chunk_id', 'unknown')}"
                )
                continue

            # 根据算法计算相似度
            similarity = self._calculate_similarity(distance, config.algorithm)

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
            distance: 距离值
            algorithm: 相似度算法

        Returns:
            相似度分数 (0-1)
        """
        if algorithm == SimilarityAlgorithm.COSINE:
            # 余弦相似度: 距离越小，相似度越高
            # 对于归一化向量，L2距离转换为余弦相似度的正确公式
            # cosine_similarity = 1 - (l2_distance^2 / 2)
            similarity = 1 - (distance**2) / 2
        elif algorithm == SimilarityAlgorithm.EUCLIDEAN:
            # 欧氏距离: 使用高斯核函数转换为相似度
            # similarity = exp(-distance^2 / (2 * sigma^2))
            sigma = 1.0  # 可调节参数
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
        text = re.sub(f"[\s{punct_chars}]+", " ", text)

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

    def _perform_bm25_retrieval(
        self, core_keywords: List[str], config: RetrievalConfig
    ) -> List[RetrievalResult]:
        """
        执行BM25检索

        Args:
            core_keywords: 核心关键词列表
            config: 检索配置

        Returns:
            BM25检索结果列表
        """
        try:
            # 获取所有文档片段
            all_chunks = self._get_all_document_chunks()
            if not all_chunks:
                return []

            # 准备BM25索引
            tokenized_corpus = []
            chunk_mapping = []

            for chunk in all_chunks:
                # 分词
                tokens = self._tokenize_text(chunk["content"])
                if tokens:
                    tokenized_corpus.append(tokens)
                    chunk_mapping.append(chunk)

            if not tokenized_corpus:
                return []

            # 构建BM25索引
            bm25 = BM25Okapi(tokenized_corpus)

            # 构建查询
            query_tokens = []
            for keyword in core_keywords:
                query_tokens.extend(self._tokenize_text(keyword))

            # 执行BM25检索
            search_top_k = config.top_k * 2
            scores = bm25.get_scores(query_tokens)

            # 排序并获取top结果
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
                        similarity=float(scores[idx]),  # BM25分数
                        match_keywords=core_keywords,
                    )
                    results.append(result)

            return results
        except Exception as e:
            logger.error(f"BM25检索失败: {str(e)}")
            return []

    def _perform_semantic_keyword_retrieval(
        self, core_keywords: List[str], config: RetrievalConfig
    ) -> List[RetrievalResult]:
        """
        执行语义关键词检索

        Args:
            core_keywords: 核心关键词列表
            config: 检索配置

        Returns:
            语义关键词检索结果列表
        """
        try:
            # 获取所有文档片段
            all_chunks = self._get_all_document_chunks()
            if not all_chunks:
                return []

            # 生成关键词向量
            keyword_text = " ".join(core_keywords)
            keyword_vector = embedding_service.encode([keyword_text])[0]

            # 为每个文档片段生成向量并计算相似度
            chunk_vectors = []
            chunk_mapping = []

            # 批量处理
            chunk_contents = [chunk["content"] for chunk in all_chunks]
            content_vectors = embedding_service.encode(chunk_contents)

            # 计算相似度
            results = []
            for i, (chunk, content_vector) in enumerate(
                zip(all_chunks, content_vectors)
            ):
                # 计算余弦相似度
                similarity = np.dot(keyword_vector, content_vector) / (
                    np.linalg.norm(keyword_vector) * np.linalg.norm(content_vector)
                )

                if similarity > 0:
                    result = RetrievalResult(
                        chunk_id=chunk.get("chunk_id", ""),
                        document_id=chunk.get("document_id", ""),
                        document_name=chunk.get("document_name", "Unknown"),
                        chunk_num=chunk.get("chunk_num", 0),
                        content=chunk.get("content", ""),
                        similarity=float(similarity),  # 语义相似度
                        match_keywords=core_keywords,
                    )
                    results.append(result)

            # 排序并返回top结果
            search_top_k = config.top_k * 2
            results.sort(key=lambda x: x.similarity, reverse=True)
            return results[:search_top_k]
        except Exception as e:
            logger.error(f"语义关键词检索失败: {str(e)}")
            return []

    def _fuse_keyword_results(
        self,
        bm25_results: List[RetrievalResult],
        semantic_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        融合关键词检索结果

        Args:
            bm25_results: BM25检索结果列表
            semantic_results: 语义关键词检索结果列表

        Returns:
            融合后的关键词检索结果列表
        """
        # 基于chunk_id去重和融合
        fused_dict = {}

        # 处理BM25结果
        for result in bm25_results:
            if result.chunk_id:
                if result.chunk_id not in fused_dict:
                    fused_dict[result.chunk_id] = result
                else:
                    # 融合相似度分数
                    existing = fused_dict[result.chunk_id]
                    existing.similarity = max(existing.similarity, result.similarity)

        # 处理语义关键词结果
        for result in semantic_results:
            if result.chunk_id:
                if result.chunk_id not in fused_dict:
                    fused_dict[result.chunk_id] = result
                else:
                    # 融合相似度分数
                    existing = fused_dict[result.chunk_id]
                    existing.similarity = max(existing.similarity, result.similarity)

        # 转换为列表并排序
        fused_results = list(fused_dict.values())
        fused_results.sort(key=lambda x: x.similarity, reverse=True)

        return fused_results

    def _fuse_retrieval_results(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        config: RetrievalConfig,
    ) -> List[RetrievalResult]:
        """
        融合向量检索和关键词检索结果

        Args:
            vector_results: 向量检索结果列表
            keyword_results: 关键词检索结果列表
            config: 检索配置

        Returns:
            融合后的最终结果列表
        """
        # 基于chunk_id去重和融合
        fused_dict = {}

        # 处理向量检索结果（权重0.6）
        for result in vector_results:
            if result.chunk_id:
                if result.chunk_id not in fused_dict:
                    # 向量相似度权重0.6
                    result.similarity *= 0.6
                    fused_dict[result.chunk_id] = result
                else:
                    # 融合相似度分数
                    existing = fused_dict[result.chunk_id]
                    existing.similarity = max(
                        existing.similarity, result.similarity * 0.6
                    )

        # 处理关键词检索结果（权重0.4）
        for result in keyword_results:
            if result.chunk_id:
                if result.chunk_id not in fused_dict:
                    # 关键词相似度权重0.4
                    result.similarity *= 0.4
                    fused_dict[result.chunk_id] = result
                else:
                    # 融合相似度分数
                    existing = fused_dict[result.chunk_id]
                    # 计算加权融合分数
                    vector_score = (
                        existing.similarity / 0.6 if existing.similarity > 0 else 0
                    )
                    keyword_score = result.similarity
                    fused_score = (vector_score * 0.6) + (keyword_score * 0.4)
                    existing.similarity = fused_score

        # 转换为列表并排序
        fused_results = list(fused_dict.values())
        fused_results.sort(key=lambda x: x.similarity, reverse=True)

        # 返回top结果
        return fused_results[: config.top_k * 2]

    def _get_all_document_chunks(self) -> List[Dict[str, Any]]:
        """
        获取所有文档片段

        Returns:
            文档片段列表
        """
        try:
            # 从向量数据库获取所有元数据
            # 注意：这里需要根据实际的vector_db_manager实现来调整
            # 假设vector_db_manager有get_all_metadata方法
            if hasattr(vector_db_manager, "get_all_metadata"):
                all_metadata = vector_db_manager.get_all_metadata()
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
        分词

        Args:
            text: 输入文本

        Returns:
            分词结果列表
        """
        # 简单的分词实现
        # 移除标点符号 (使用标准正则，Python的re不支持\p{Punct})
        import string

        punct_chars = re.escape(string.punctuation + '。，、；：？！""（）【】《》')
        text = re.sub(f"[\s{punct_chars}]+", " ", text)
        # 分词
        tokens = text.strip().split()
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
        tokens = [
            token for token in tokens if token not in stop_words and len(token) > 1
        ]
        return tokens

    def _deduplicate_and_fuse_results(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        对结果进行去重和融合

        Args:
            results: 所有查询的结果列表

        Returns:
            去重和融合后的结果
        """
        # 基于chunk_id去重
        unique_results = {}
        for result in results:
            if result.chunk_id:
                if result.chunk_id not in unique_results:
                    unique_results[result.chunk_id] = result
                else:
                    # 融合相似度分数
                    existing_result = unique_results[result.chunk_id]
                    existing_result.similarity = max(
                        existing_result.similarity, result.similarity
                    )

        # 转换为列表并按相似度排序
        fused_results = list(unique_results.values())
        fused_results.sort(key=lambda x: x.similarity, reverse=True)

        return fused_results


# 全局检索器实例
retriever = Retriever()
