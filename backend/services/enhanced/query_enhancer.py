"""
查询增强器 (Query Enhancer)
实现功能：指代消解、HyDE(假设文档嵌入)、问题分解
"""

import re
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from utils.logger import logger


class EnhancementType(str, Enum):
    """增强类型"""

    COREFERENCE_RESOLUTION = "coreference_resolution"  # 指代消解
    HYDE_SINGLE = "hyde_single"  # 单假设HyDE
    HYDE_MULTI = "hyde_multi"  # 多假设HyDE
    QUERY_DECOMPOSITION = "query_decomposition"  # 问题分解
    TEMPORAL_INJECTION = "temporal_injection"  # 时间约束注入


@dataclass
class SubQuery:
    """子查询"""

    id: str
    text: str
    dependencies: List[str] = field(default_factory=list)  # 依赖的其他子查询ID
    query_type: str = "independent"  # independent | dependent
    expected_answer_type: str = "text"


@dataclass
class HypothesisDoc:
    """假设文档"""

    text: str
    entity: Optional[str] = None
    confidence: float = 0.0


@dataclass
class EnhancedQuery:
    """增强后的查询"""

    original_query: str
    main_query: str  # 主查询(改写后)
    sub_queries: List[SubQuery] = field(default_factory=list)
    hypotheses: List[HypothesisDoc] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    scene_tags: List[str] = field(default_factory=list)
    execution_plan: Dict = field(default_factory=dict)
    enhancements_applied: List[str] = field(default_factory=list)


class QueryEnhancer:
    """查询增强器"""

    def __init__(self, llm_client=None):
        self.llm = llm_client

        # 指代词映射表
        self.coreference_patterns = {
            r"[它这那此][个]?": None,  # 需要上下文解析
            r"[该上]述[的]?": None,
            r"[之]?前[提到|说]?[的]?": None,
        }

    def enhance(
        self, query: str, session_context: Dict, scene_tags: List[str]
    ) -> EnhancedQuery:
        """
        查询增强主入口

        Args:
            query: 原始查询
            session_context: 会话上下文
            scene_tags: 场景标签

        Returns:
            EnhancedQuery: 增强后的查询包
        """
        enhanced = EnhancedQuery(
            original_query=query, main_query=query, scene_tags=scene_tags
        )

        # 1. 指代消解 (如果存在HistoryRef标签)
        if "HistoryRef" in scene_tags:
            resolved_query, entities_used = self._resolve_coreference(
                query, session_context
            )
            if resolved_query != query:
                enhanced.main_query = resolved_query
                enhanced.enhancements_applied.append("coreference_resolution")
                enhanced.filters["entities_inherited"] = entities_used
                logger.info(f"Coreference resolved: '{query}' -> '{resolved_query}'")

        # 2. HyDE增强 (如果存在Ambiguous标签)
        if "Ambiguous" in scene_tags:
            hypotheses = self._generate_hyde_hypotheses(
                enhanced.main_query, session_context, max_hypotheses=3
            )
            enhanced.hypotheses = hypotheses
            enhanced.enhancements_applied.append(
                "hyde_multi" if len(hypotheses) > 1 else "hyde_single"
            )

        # 3. 问题分解 (如果存在MultiQuestion标签)
        if "MultiQuestion" in scene_tags:
            sub_queries, execution_plan = self._decompose_query(enhanced.main_query)
            enhanced.sub_queries = sub_queries
            enhanced.execution_plan = execution_plan
            enhanced.enhancements_applied.append("query_decomposition")

        # 4. 时间约束注入 (如果存在Temporal标签)
        if "Temporal" in scene_tags:
            enhanced.filters["temporal_boost"] = True
            enhanced.filters["recency_weight"] = 0.3
            enhanced.enhancements_applied.append("temporal_injection")

        # 5. 对比处理 (如果存在Comparative标签)
        if "Comparative" in scene_tags:
            comparative_queries = self._handle_comparative(enhanced.main_query)
            if comparative_queries:
                enhanced.sub_queries.extend(comparative_queries)
                enhanced.enhancements_applied.append("comparative_expansion")

        return enhanced

    def _resolve_coreference(
        self, query: str, session_context: Dict
    ) -> Tuple[str, List[str]]:
        """
        指代消解

        Args:
            query: 包含指代词的查询
            session_context: 包含历史实体的会话上下文

        Returns:
            (消解后的查询, 使用的实体列表)
        """
        entities = session_context.get("entities", [])
        if not entities:
            return query, []

        # 按时间戳排序，取最近的实体
        sorted_entities = sorted(
            entities, key=lambda e: e.get("timestamp", ""), reverse=True
        )

        resolved_query = query
        entities_used = []

        # 检测指代词
        coreference_keywords = ["它", "这", "那", "此", "该", "上述", "之前"]

        for keyword in coreference_keywords:
            if keyword in resolved_query:
                # 找到最相关的实体
                best_entity = None
                best_score = 0

                for entity in sorted_entities:
                    entity_name = entity.get("name", "")
                    entity_confidence = entity.get("confidence", 0)

                    # 计算相关性分数
                    score = entity_confidence

                    # 如果查询中有实体名的一部分，提高分数
                    if entity_name and len(entity_name) > 2:
                        for part in entity_name.split():
                            if len(part) > 2 and part in query:
                                score += 0.2

                    if score > best_score and score > 0.5:
                        best_score = score
                        best_entity = entity

                if best_entity:
                    # 替换指代词
                    entity_name = best_entity["name"]
                    pattern = re.compile(re.escape(keyword) + r"[个]?")
                    resolved_query = pattern.sub(entity_name, resolved_query, count=1)
                    entities_used.append(entity_name)

                    logger.debug(f"Resolved '{keyword}' -> '{entity_name}'")

        return resolved_query, entities_used

    def _generate_hyde_hypotheses(
        self, query: str, session_context: Dict, max_hypotheses: int = 3
    ) -> List[HypothesisDoc]:
        """
        生成HyDE假设文档

        HyDE (Hypothetical Document Embedding):
        使用LLM生成假设的理想回答文档，然后基于这些文档进行检索

        Args:
            query: 查询
            session_context: 会话上下文
            max_hypotheses: 最大假设数量

        Returns:
            List[HypothesisDoc]: 假设文档列表
        """
        hypotheses = []

        # 检查是否有歧义实体
        ambiguous_entities = self._detect_ambiguous_entities(query)

        if ambiguous_entities and len(ambiguous_entities) <= max_hypotheses:
            # 为每个歧义实体生成假设
            for entity_info in ambiguous_entities:
                entity = entity_info["entity"]
                meanings = entity_info["meanings"]

                for meaning in meanings[:1]:  # 每个实体取最可能的含义
                    # 构建特化查询
                    specialized_query = query.replace(entity, meaning)

                    # 生成假设文档 (简化版，实际应调用LLM)
                    hypothesis_text = self._generate_hypothesis_text(
                        specialized_query, meaning
                    )

                    hypotheses.append(
                        HypothesisDoc(
                            text=hypothesis_text, entity=meaning, confidence=0.7
                        )
                    )
        else:
            # 生成单一假设
            hypothesis_text = self._generate_hypothesis_text(query)
            hypotheses.append(HypothesisDoc(text=hypothesis_text, confidence=0.8))

        return hypotheses[:max_hypotheses]

    def _generate_hypothesis_text(self, query: str, context: str = None) -> str:
        """
        生成假设文档文本

        简化版实现：基于查询构建模板化的假设文档
        实际生产环境应使用LLM生成
        """
        # 模板化假设生成
        templates = [
            f"关于'{query}'，相关信息包括：",
            f"'{query}'的详细说明如下：",
            f"以下是关于'{query}'的主要内容：",
        ]

        # 这里简化处理，实际应该调用LLM
        # 例如使用GPT-4生成一段假想的完美回答

        base_text = templates[hash(query) % len(templates)]

        # 添加查询关键词作为假设内容
        keywords = self._extract_keywords(query)
        if keywords:
            base_text += " " + ", ".join(keywords[:5]) + "等关键信息。"

        return base_text

    def _detect_ambiguous_entities(self, query: str) -> List[Dict]:
        """检测歧义实体"""
        ambiguous_dict = {
            "苹果": {
                "meanings": ["苹果公司", "苹果水果"],
                "context_hints": ["手机", "股价", "吃", "水果"],
            },
            "java": {
                "meanings": ["Java编程语言", "爪哇岛"],
                "context_hints": ["编程", "代码", "印度尼西亚", "旅游"],
            },
            "python": {
                "meanings": ["Python编程语言", "蟒蛇"],
                "context_hints": ["编程", "代码", "动物", "蛇"],
            },
            " Aurora": {
                "meanings": ["极光现象", "欧若拉(罗马女神)"],
                "context_hints": ["天文", "北欧", "神话", "女神"],
            },
            "": {
                "meanings": ["亚马逊公司", "亚马逊雨林", "亚马逊河"],
                "context_hints": ["电商", "购物", "森林", "河流"],
            },
        }

        results = []
        query_lower = query.lower()

        for entity, info in ambiguous_dict.items():
            if entity in query_lower or entity.lower() in query_lower:
                results.append({"entity": entity, **info})

        return results

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简化的关键词提取
        words = re.findall(r"[\w\u4e00-\u9fa5]+", text)
        # 过滤停用词
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
            "一个",
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
        }
        keywords = [w for w in words if len(w) > 1 and w not in stop_words]
        return keywords[:10]

    def _decompose_query(self, query: str) -> Tuple[List[SubQuery], Dict]:
        """
        问题分解

        将复合问题分解为多个子问题，构建DAG执行计划

        Returns:
            (子查询列表, 执行计划)
        """
        sub_queries = []

        # 基于标点分割句子
        sentences = re.split(r"[。！？；\n]", query)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 基于连接词识别子问题
        if len(sentences) == 1:
            # 尝试基于连接词分割
            sub_parts = self._split_by_connectors(query)
        else:
            sub_parts = sentences

        # 构建子查询
        for i, part in enumerate(sub_parts):
            sub_q = SubQuery(
                id=f"sub_{i}", text=part, dependencies=[], query_type="independent"
            )
            sub_queries.append(sub_q)

        # 识别依赖关系
        for i, sq in enumerate(sub_queries):
            # 检查是否依赖前面子问题的答案
            dep_keywords = ["上述", "前面提到", "之前", "上面"]
            for keyword in dep_keywords:
                if keyword in sq.text:
                    # 依赖所有前面的子查询
                    sq.dependencies = [f"sub_{j}" for j in range(i)]
                    sq.query_type = "dependent"
                    break

        # 构建执行计划 (DAG)
        execution_plan = self._build_execution_plan(sub_queries)

        return sub_queries, execution_plan

    def _split_by_connectors(self, query: str) -> List[str]:
        """基于连接词分割查询"""
        connectors = ["并且", "而且", "同时", "另外", "还有", "以及"]

        parts = [query]
        for connector in connectors:
            new_parts = []
            for part in parts:
                if connector in part:
                    new_parts.extend(part.split(connector))
                else:
                    new_parts.append(part)
            parts = new_parts

        return [p.strip() for p in parts if p.strip()]

    def _build_execution_plan(self, sub_queries: List[SubQuery]) -> Dict:
        """构建执行计划 (DAG)"""
        # 拓扑排序
        in_degree = {sq.id: 0 for sq in sub_queries}
        adj_list = {sq.id: [] for sq in sub_queries}

        for sq in sub_queries:
            for dep in sq.dependencies:
                adj_list[dep].append(sq.id)
                in_degree[sq.id] += 1

        # Kahn算法
        stages = []
        current_stage = [sq_id for sq_id, degree in in_degree.items() if degree == 0]

        while current_stage:
            stages.append(current_stage)
            next_stage = []

            for sq_id in current_stage:
                for neighbor in adj_list[sq_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_stage.append(neighbor)

            current_stage = next_stage

        return {
            "type": "dag",
            "stages": stages,
            "total_subqueries": len(sub_queries),
            "parallel_groups": len(stages),
        }

    def _handle_comparative(self, query: str) -> List[SubQuery]:
        """处理对比型查询"""
        entities = self._extract_comparative_entities(query)

        if len(entities) >= 2:
            sub_queries = []
            for i, entity in enumerate(entities[:2]):
                sq = SubQuery(
                    id=f"comp_{i}",
                    text=f"{entity}的相关信息",
                    dependencies=[],
                    query_type="independent",
                    expected_answer_type="entity_info",
                )
                sub_queries.append(sq)

            # 添加对比分析子查询
            analysis_sq = SubQuery(
                id="comp_analysis",
                text=f"对比分析{'和'.join(entities[:2])}的异同",
                dependencies=[f"comp_{i}" for i in range(len(entities[:2]))],
                query_type="dependent",
                expected_answer_type="comparison",
            )
            sub_queries.append(analysis_sq)

            return sub_queries

        return []

    def _extract_comparative_entities(self, query: str) -> List[str]:
        """提取对比实体"""
        patterns = [
            r"([\w\u4e00-\u9fa5]+)[和与跟].*?([\w\u4e00-\u9fa5]+).*?[区别差异不同对比]",
            r"([\w\u4e00-\u9fa5]+)\s*vs\s*([\w\u4e00-\u9fa5]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return list(match.groups())

        return []


# 单例
query_enhancer = QueryEnhancer()

__all__ = [
    "QueryEnhancer",
    "EnhancedQuery",
    "SubQuery",
    "HypothesisDoc",
    "query_enhancer",
]
