"""
增强版RAG生成器
整合所有增强功能：场景标签路由、HyDE、问题分解、质量门控
"""

from typing import List, Dict, Optional, AsyncGenerator, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
from utils.logger import logger
from services.conversation_manager import conversation_manager
from services.intent_recognizer import intent_recognizer, IntentType
from services.retriever import retriever
from services.reranker import reranker_manager
from services.rag_generator import rag_generator
from services.enhanced.scene_tagger import scene_tagger, SceneTag
from services.enhanced.query_enhancer import query_enhancer, EnhancedQuery
from services.enhanced.retrieval_cache import (
    retrieval_cache,
    quality_gate,
    RetrievalResult,
    QualityTier,
)
from services.enhanced.generation_cache import generation_cache, GenerationCacheEntry
from services.vector_db import vector_db_manager
from services.embedding import embedding_service


@dataclass
class RAGResponse:
    """RAG响应"""

    answer: str
    citations: List[Dict]
    session_id: str
    trace_id: str
    scene_tags: List[str]
    metadata: Dict[str, Any]
    pending_action: Optional[str] = None


class EnhancedRAGPipeline:
    """增强版RAG流水线"""

    def __init__(self):
        self.scene_tagger = scene_tagger
        self.query_enhancer = query_enhancer
        self.retrieval_cache = retrieval_cache
        self.quality_gate = quality_gate
        self.generation_cache = generation_cache

    async def generate(
        self,
        query: str,
        session_id: str = None,
        user_id: str = None,
        trace_id: str = None,
        config: Dict = None,
    ) -> RAGResponse:
        """
        增强版RAG生成主入口

        完整流程：
        1. 意图识别 + 缓存
        2. 会话加载 + 实体追踪
        3. 场景标签识别
        4. 查询增强 (指代消解/HyDE/问题分解)
        5. 检索缓存检查
        6. 多路检索
        7. 质量门控
        8. 重排序
        9. 生成缓存检查
        10. RAG生成
        11. 会话更新

        Args:
            query: 用户查询
            session_id: 会话ID
            user_id: 用户ID
            trace_id: 追踪ID
            config: 配置参数

        Returns:
            RAGResponse: RAG响应
        """
        config = config or {}
        start_time = datetime.utcnow()

        # ========== 阶段1: 意图识别 ==========
        intent_start = datetime.utcnow()
        intent_result = await self._recognize_intent(query)
        intent_time = (datetime.utcnow() - intent_start).total_seconds() * 1000

        # ========== 阶段2: 会话管理 ==========
        session_context = await self._load_session(session_id, user_id)

        # ========== 阶段3: 场景标签识别 ==========
        tag_result = self.scene_tagger.tag(query, session_context)
        scene_tags = [t.value for t in tag_result.tags]

        # 检查是否是非检索型查询
        if SceneTag.NON_RETRIEVAL.value in scene_tags:
            # 直接生成，跳过检索
            answer = await self._direct_chat_generation(query, session_context)
            await self._update_session(session_id, query, answer, session_context)

            return RAGResponse(
                answer=answer,
                citations=[],
                session_id=session_id or "new",
                trace_id=trace_id or "unknown",
                scene_tags=scene_tags,
                metadata={
                    "type": "direct_chat",
                    "intent": intent_result.get("intent"),
                    "intent_time_ms": intent_time,
                },
            )

        # ========== 阶段4: 查询增强 ==========
        enhanced_query = self.query_enhancer.enhance(query, session_context, scene_tags)

        # ========== 阶段5: 检索缓存检查 ==========
        retrieval_start = datetime.utcnow()

        # 构建主查询 (考虑子查询)
        queries_to_retrieve = [enhanced_query.main_query]
        if enhanced_query.sub_queries:
            queries_to_retrieve.extend([sq.text for sq in enhanced_query.sub_queries])

        # 检查检索缓存
        all_cached_results = []
        queries_to_search = []

        for q in queries_to_retrieve:
            cached = await self.retrieval_cache.get(q)
            if cached:
                all_cached_results.extend(cached)
            else:
                queries_to_search.append(q)

        # ========== 阶段6: 多路检索 ==========
        retrieval_results = []

        if queries_to_search:
            for q in queries_to_search:
                results = await self._multi_way_retrieval(
                    q, enhanced_query, intent_result
                )
                retrieval_results.extend(results)

                # 缓存检索结果
                await self.retrieval_cache.set(q, results)

        # 合并缓存和新的检索结果
        all_results = all_cached_results + retrieval_results

        # 去重
        all_results = self._deduplicate_results(all_results)

        retrieval_time = (datetime.utcnow() - retrieval_start).total_seconds() * 1000

        # ========== 阶段7: 质量门控 ==========
        quality_result = self.quality_gate.evaluate(
            all_results, enhanced_query.main_query, scene_tags
        )

        # 如果质量不合格且需要澄清
        if not quality_result.passed and quality_result.feedback:
            if quality_result.feedback.get("action") == "clarify":
                clarification = await self._generate_clarification(
                    enhanced_query.main_query, quality_result.feedback
                )
                await self._update_session_pending_clarification(
                    session_id, clarification
                )

                return RAGResponse(
                    answer=clarification,
                    citations=[],
                    session_id=session_id or "new",
                    trace_id=trace_id or "unknown",
                    scene_tags=scene_tags,
                    metadata={
                        "type": "clarification_needed",
                        "quality_metrics": quality_result.metrics.__dict__,
                        "retrieval_time_ms": retrieval_time,
                    },
                    pending_action="clarification_needed",
                )

        # ========== 阶段8: 重排序 ==========
        rerank_start = datetime.utcnow()

        if quality_result.tier != QualityTier.LOW and all_results:
            # 动态Top-K
            top_k = 5 if quality_result.tier == QualityTier.HIGH else 10

            reranked = await self._rerank_results(
                enhanced_query.main_query, all_results[:50], top_k
            )
        else:
            reranked = all_results[:10]

        rerank_time = (datetime.utcnow() - rerank_start).total_seconds() * 1000

        # ========== 阶段9: 生成缓存检查 ==========
        # 构建生成Prompt
        prompt = await self._build_generation_prompt(
            enhanced_query, reranked, session_context, scene_tags
        )

        cached_generation = await self.generation_cache.get(prompt)

        if cached_generation:
            # 使用缓存的生成结果
            answer = cached_generation.answer
            citations = cached_generation.citations
            generation_time = 0  # 缓存命中，无生成时间
            from_cache = True
        else:
            # ========== 阶段10: RAG生成 ==========
            generation_start = datetime.utcnow()

            # 特殊处理复合问题
            if (
                SceneTag.MULTI_QUESTION.value in scene_tags
                and enhanced_query.sub_queries
            ):
                answer, citations = await self._handle_multi_question(
                    enhanced_query, reranked, session_context
                )
            else:
                # 标准生成
                answer, citations = await self._standard_generation(
                    prompt, enhanced_query, reranked
                )

            generation_time = (
                datetime.utcnow() - generation_start
            ).total_seconds() * 1000
            from_cache = False

            # 缓存生成结果
            await self.generation_cache.set(
                prompt,
                answer,
                citations,
                tokens_used=len(prompt.split()) + len(answer.split()),
            )

        # ========== 阶段11: 会话更新 ==========
        await self._update_session(session_id, query, answer, session_context)

        # 计算总耗时
        total_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return RAGResponse(
            answer=answer,
            citations=citations,
            session_id=session_id or "new",
            trace_id=trace_id or "unknown",
            scene_tags=scene_tags,
            metadata={
                "intent": intent_result.get("intent"),
                "intent_confidence": intent_result.get("confidence"),
                "intent_time_ms": intent_time,
                "retrieval_time_ms": retrieval_time,
                "rerank_time_ms": rerank_time,
                "generation_time_ms": generation_time,
                "total_time_ms": total_time,
                "quality_tier": quality_result.tier.value,
                "quality_score": quality_result.metrics.overall_score,
                "candidates_count": len(all_results),
                "final_candidates": len(reranked),
                "generation_cached": from_cache,
                "enhancements_applied": enhanced_query.enhancements_applied,
            },
        )

    async def _recognize_intent(self, query: str) -> Dict:
        """识别意图"""
        # 使用现有的意图识别器
        result = intent_recognizer.recognize(query)
        intent_type, confidence, strategy = result
        return {
            "intent": intent_type.value,
            "confidence": confidence,
            "strategy": strategy,
        }

    async def _load_session(self, session_id: str, user_id: str) -> Dict:
        """加载会话上下文"""
        if session_id:
            session = conversation_manager.get_conversation(session_id)
            if session:
                # 提取实体和历史
                entities = self._extract_entities_from_session(session)
                return {
                    "has_history": len(session.messages) > 0,
                    "entities": entities,
                    "history_summary": self._summarize_history(session),
                    "turn_count": len(session.messages) // 2,
                }

        return {
            "has_history": False,
            "entities": [],
            "history_summary": "",
            "turn_count": 0,
        }

    def _extract_entities_from_session(self, session) -> List[Dict]:
        """从会话中提取实体"""
        entities = []

        # 简单的实体提取：从最近的回答中提取关键词
        if session.messages:
            for msg in reversed(session.messages[-6:]):  # 看最近3轮
                if msg.role == "assistant":
                    # 简单的关键词提取逻辑
                    words = self._extract_keywords(msg.content)
                    for word in words[:5]:  # 每轮取top5
                        entities.append(
                            {
                                "name": word,
                                "confidence": 0.7,
                                "timestamp": msg.timestamp
                                if hasattr(msg, "timestamp")
                                else datetime.utcnow().isoformat(),
                            }
                        )

        return entities

    def _summarize_history(self, session) -> str:
        """总结会话历史"""
        if not session.messages:
            return ""

        # 简单的摘要：取最近一轮的问题和回答
        recent_msgs = session.messages[-2:]
        return " ".join([m.content[:100] for m in recent_msgs])

    async def _multi_way_retrieval(
        self, query: str, enhanced_query: EnhancedQuery, intent_result: Dict
    ) -> List[RetrievalResult]:
        """多路检索"""
        results = []

        # 1. 向量检索
        try:
            query_vector = embedding_service.encode(query)
            vector_results = vector_db_manager.search(
                query_vector=query_vector, top_k=30, filters=enhanced_query.filters
            )

            for r in vector_results:
                results.append(
                    RetrievalResult(
                        doc_id=r.get("id", ""),
                        content=r.get("content", ""),
                        score=r.get("score", 0),
                        source="vector",
                        metadata=r.get("metadata", {}),
                    )
                )
        except Exception as e:
            logger.warning(f"Vector retrieval error: {e}")

        # 2. HyDE检索 (如果有假设文档)
        if enhanced_query.hypotheses:
            for hypothesis in enhanced_query.hypotheses:
                try:
                    hyp_vector = embedding_service.encode(hypothesis.text)
                    hyde_results = vector_db_manager.search(
                        query_vector=hyp_vector, top_k=10
                    )

                    for r in hyde_results:
                        results.append(
                            RetrievalResult(
                                doc_id=r.get("id", ""),
                                content=r.get("content", ""),
                                score=r.get("score", 0) * 0.9,  # HyDE结果稍微降权
                                source="hyde",
                                metadata={
                                    **r.get("metadata", {}),
                                    "hyde_entity": hypothesis.entity,
                                },
                            )
                        )
                except Exception as e:
                    logger.warning(f"HyDE retrieval error: {e}")

        return results

    def _deduplicate_results(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """去重检索结果"""
        seen = {}
        deduped = []

        for r in results:
            key = r.doc_id
            if key not in seen:
                seen[key] = r
                deduped.append(r)
            else:
                # 保留更高分数的
                if r.score > seen[key].score:
                    seen[key] = r

        return deduped

    async def _rerank_results(
        self, query: str, results: List[RetrievalResult], top_k: int
    ) -> List[RetrievalResult]:
        """重排序"""
        if not results:
            return []

        try:
            # 准备reranker输入
            documents = [r.content for r in results]
            reranked_indices = reranker_manager.rerank(query, documents, top_k=top_k)

            # 根据rerank结果重新排序
            reranked = []
            for idx in reranked_indices:
                if idx < len(results):
                    reranked.append(results[idx])

            return reranked
        except Exception as e:
            logger.warning(f"Reranking error: {e}")
            # 降级：按原始分数排序
            return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]

    async def _build_generation_prompt(
        self,
        enhanced_query: EnhancedQuery,
        results: List[RetrievalResult],
        session_context: Dict,
        scene_tags: List[str],
    ) -> str:
        """构建生成Prompt"""
        # 构建上下文
        context_parts = []
        for i, r in enumerate(results[:5], 1):
            context_parts.append(f"[{i}] {r.content[:500]}")

        context = "\n\n".join(context_parts)

        # 选择Prompt模板
        if SceneTag.COMPARATIVE.value in scene_tags:
            template = self._get_comparative_template()
        elif SceneTag.TEMPORAL.value in scene_tags:
            template = self._get_temporal_template()
        else:
            template = self._get_standard_template()

        prompt = template.format(context=context, query=enhanced_query.main_query)

        return prompt

    def _get_standard_template(self) -> str:
        """标准RAG模板"""
        return """基于以下参考信息回答问题：

参考信息：
{context}

问题：{query}

请基于以上参考信息回答问题。如果参考信息不足以回答问题，请说明。
回答时请标注引用来源（如[^1^]）。"""

    def _get_comparative_template(self) -> str:
        """对比型模板"""
        return """基于以下参考信息进行对比分析：

参考信息：
{context}

对比问题：{query}

请以表格或结构化方式呈现对比结果，明确指出各方面的异同。"""

    def _get_temporal_template(self) -> str:
        """时间敏感模板"""
        return """基于以下最新参考信息回答问题：

参考信息：
{context}

问题：{query}

请注意信息的时效性，优先使用最新的参考信息。如果参考信息的时间不明确，请说明。"""

    async def _standard_generation(
        self, prompt: str, enhanced_query: EnhancedQuery, results: List[RetrievalResult]
    ) -> tuple[str, List[Dict]]:
        """标准RAG生成"""
        # 使用现有的rag_generator
        answer = rag_generator.generate(prompt, max_tokens=2000)

        # 构建引用
        citations = []
        for i, r in enumerate(results[:5], 1):
            citations.append(
                {
                    "id": i,
                    "doc_id": r.doc_id,
                    "source": r.source,
                    "preview": r.content[:100] + "...",
                }
            )

        return answer, citations

    async def _handle_multi_question(
        self,
        enhanced_query: EnhancedQuery,
        results: List[RetrievalResult],
        session_context: Dict,
    ) -> tuple[str, List[Dict]]:
        """处理复合问题"""
        # 按执行计划生成子答案
        sub_answers = []

        for sq in enhanced_query.sub_queries:
            # 为每个子查询生成答案
            sub_prompt = f"基于参考信息回答：{sq.text}\n\n参考：{[r.content[:300] for r in results[:3]]}"
            sub_answer = rag_generator.generate(sub_prompt, max_tokens=500)
            sub_answers.append({"question": sq.text, "answer": sub_answer})

        # 聚合子答案
        qa_parts = []
        for sa in sub_answers:
            qa_parts.append(f"Q: {sa['question']}")
            qa_parts.append(f"A: {sa['answer']}")
        qa_text = "\n".join(qa_parts)

        aggregation_prompt = f"""综合以下各部分的回答，给出完整的答案：

{qa_text}

请确保回答逻辑连贯，过渡自然。"""

        final_answer = rag_generator.generate(aggregation_prompt, max_tokens=1500)

        citations = [
            {"id": i, "doc_id": r.doc_id} for i, r in enumerate(results[:5], 1)
        ]

        return final_answer, citations

    async def _direct_chat_generation(self, query: str, session_context: Dict) -> str:
        """直接聊天生成 (跳过检索)"""
        prompt = f"""你是一个友好的AI助手。请回答用户的问题：

用户：{query}

请给出友好、简洁的回答。"""

        return rag_generator.generate(prompt, max_tokens=500)

    async def _generate_clarification(self, query: str, feedback: Dict) -> str:
        """生成澄清请求"""
        suggestions = feedback.get("suggestions", [])

        # 构建建议列表文本
        suggestion_lines = "\n".join([f"- {s}" for s in suggestions])

        clarification_prompt = f"""用户的问题是："{query}"

这个问题不够明确，可能的原因包括：
{suggestion_lines}

请生成一个友好的澄清请求，询问用户更具体的信息。要求：
1. 礼貌友好
2. 指出可能的不明确之处
3. 给出2-3个具体的选项或建议"""

        return rag_generator.generate(clarification_prompt, max_tokens=300)

    async def _update_session(
        self, session_id: str, query: str, answer: str, session_context: Dict
    ):
        """更新会话"""
        if session_id:
            conversation_manager.add_message(session_id, "user", query)
            conversation_manager.add_message(session_id, "assistant", answer)

    async def _update_session_pending_clarification(
        self, session_id: str, clarification: str
    ):
        """更新会话待澄清状态"""
        # 这里可以添加待澄清标记
        pass

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        import re

        words = re.findall(r"[\w\u4e00-\u9fa5]+", text.lower())
        stop_words = {
            "的",
            "了",
            "在",
            "是",
            "我",
            "有",
            "和",
            "就",
            "不",
            "人",
            "都",
            "一",
            "上",
            "也",
            "很",
            "到",
            "说",
            "要",
            "去",
            "你",
            "会",
            "着",
            "没有",
            "看",
            "好",
            "自己",
            "这",
            "中",
            "为",
            "来",
            "个",
            "能",
            "以",
            "可",
            "而",
            "及",
            "与",
            "或",
            "但",
            "如果",
            "因为",
            "所以",
            "使用",
            "进行",
            "通过",
            "根据",
            "关于",
            "需要",
            "可以",
            "表示",
            "用于",
        }
        return [w for w in words if len(w) > 1 and w not in stop_words][:10]


# 单例
enhanced_rag_pipeline = EnhancedRAGPipeline()

__all__ = ["EnhancedRAGPipeline", "RAGResponse", "enhanced_rag_pipeline"]
