"""
场景标签识别器 (SceneTagger)
识别查询场景标签：HistoryRef, Ambiguous, MultiQuestion, NonRetrieval, Temporal, Comparative
"""

import re
from typing import List, Dict, Set, Optional
from enum import Enum
from dataclasses import dataclass
from utils.logger import logger


class SceneTag(str, Enum):
    """场景标签枚举"""

    HISTORY_REF = "HistoryRef"  # 历史引用 (指代词)
    AMBIGUOUS = "Ambiguous"  # 查询歧义 (多实体/多意图)
    MULTI_QUESTION = "MultiQuestion"  # 复合问题 (多个子问题)
    NON_RETRIEVAL = "NonRetrieval"  # 非检索型 (问候/闲聊)
    TEMPORAL = "Temporal"  # 时间敏感
    COMPARATIVE = "Comparative"  # 对比型


@dataclass
class SceneTagResult:
    """场景标签识别结果"""

    tags: List[SceneTag]
    confidence: Dict[str, float]
    details: Dict[str, any]
    routing_strategy: str


class SceneTagger:
    """场景标签识别器"""

    # 指代词模式
    HISTORY_REF_PATTERNS = [
        r"[它这那此其这那][个]?[个]?",
        r"[该上]述[的]?",
        r"[之]?前[提到|说|的]?",
        r"刚才[提到|说]?[的]?",
        r"那个[东西|问题|内容]",
        r"[这那]个[问题|东西|内容]",
    ]

    # 对比词模式
    COMPARATIVE_PATTERNS = [
        r"对比|比较|区别|差异|不同|vs|versus|pk",
        r"[和|与|跟].*?[有]?什么[区别|不同]",
        r"[哪个|哪些].*?[更好|更优|更差]",
        r"[A-Za-z].*?vs.*?[A-Za-z]",  # A vs B格式
    ]

    # 时间敏感词
    TEMPORAL_PATTERNS = [
        r"最新|最近|今年|去年|本月|本周|今天|昨天",
        r"202[0-9]|202[0-9]年",
        r"[上|下|这]个?[月|周|季度|年]",
        r"当前|现在|目前|时下|现今",
        r"[过去|未来|将来].*?[几|多少].*?[年|月|天]",
    ]

    # 非检索型模式 (问候/闲聊)
    NON_RETRIEVAL_PATTERNS = [
        r"^[你好|您好|嗨|哈喽|hi|hello]",
        r"^谢谢|感谢",
        r"^再见|拜拜",
        r"^你[是|叫]?什么名字",
        r"^你[能|可以]?做什么",
        r"^帮助|help",
        r"^\s*$",  # 空查询
    ]

    # 疑问词 (用于检测复合问题)
    QUESTION_WORDS = [
        "什么",
        "怎么",
        "如何",
        "为什么",
        "多少",
        "几",
        "哪",
        "谁",
        "何时",
        "哪里",
    ]

    def __init__(self):
        self.history_ref_regex = re.compile(
            "|".join(self.HISTORY_REF_PATTERNS), re.IGNORECASE
        )
        self.comparative_regex = re.compile(
            "|".join(self.COMPARATIVE_PATTERNS), re.IGNORECASE
        )
        self.temporal_regex = re.compile(
            "|".join(self.TEMPORAL_PATTERNS), re.IGNORECASE
        )
        self.non_retrieval_regex = re.compile(
            "|".join(self.NON_RETRIEVAL_PATTERNS), re.IGNORECASE
        )

    def tag(self, query: str, session_context: Optional[Dict] = None) -> SceneTagResult:
        """
        识别查询的场景标签

        Args:
            query: 用户查询
            session_context: 会话上下文 (包含历史实体等)

        Returns:
            SceneTagResult: 标签识别结果
        """
        tags = []
        confidence = {}
        details = {}

        # 1. 检测历史引用 (HistoryRef)
        has_history_ref, hist_confidence = self._detect_history_ref(
            query, session_context
        )
        if has_history_ref:
            tags.append(SceneTag.HISTORY_REF)
            confidence["HistoryRef"] = hist_confidence
            details["history_ref_patterns"] = self._extract_history_refs(query)

        # 2. 检测歧义 (Ambiguous)
        is_ambiguous, amb_confidence, amb_details = self._detect_ambiguity(
            query, session_context
        )
        if is_ambiguous:
            tags.append(SceneTag.AMBIGUOUS)
            confidence["Ambiguous"] = amb_confidence
            details["ambiguity"] = amb_details

        # 3. 检测复合问题 (MultiQuestion)
        is_multi, multi_confidence, multi_details = self._detect_multi_question(query)
        if is_multi:
            tags.append(SceneTag.MULTI_QUESTION)
            confidence["MultiQuestion"] = multi_confidence
            details["multi_question"] = multi_details

        # 4. 检测非检索型 (NonRetrieval)
        is_non_retrieval, non_ret_confidence = self._detect_non_retrieval(query)
        if is_non_retrieval:
            tags.append(SceneTag.NON_RETRIEVAL)
            confidence["NonRetrieval"] = non_ret_confidence

        # 5. 检测时间敏感 (Temporal)
        is_temporal, temp_confidence, temp_details = self._detect_temporal(query)
        if is_temporal:
            tags.append(SceneTag.TEMPORAL)
            confidence["Temporal"] = temp_confidence
            details["temporal_keywords"] = temp_details

        # 6. 检测对比型 (Comparative)
        is_comparative, comp_confidence, comp_details = self._detect_comparative(query)
        if is_comparative:
            tags.append(SceneTag.COMPARATIVE)
            confidence["Comparative"] = comp_confidence
            details["comparative_entities"] = comp_details

        # 确定路由策略
        routing_strategy = self._determine_routing_strategy(tags)

        result = SceneTagResult(
            tags=tags,
            confidence=confidence,
            details=details,
            routing_strategy=routing_strategy,
        )

        logger.info(
            f"Scene tagging: query='{query[:50]}...', tags={[t.value for t in tags]}"
        )

        return result

    def _detect_history_ref(
        self, query: str, session_context: Optional[Dict]
    ) -> tuple[bool, float]:
        """检测历史引用"""
        if not session_context or not session_context.get("has_history"):
            return False, 0.0

        matches = self.history_ref_regex.findall(query)
        if matches:
            # 计算置信度 (基于匹配数量和位置)
            confidence = min(0.6 + len(matches) * 0.1, 0.95)
            return True, confidence

        # 检查是否有实体指代
        if session_context.get("entities"):
            entity_names = [e["name"] for e in session_context["entities"]]
            for entity in entity_names:
                if entity in query:
                    return False, 0.0  # 有具体实体，不算指代

        return False, 0.0

    def _extract_history_refs(self, query: str) -> List[str]:
        """提取指代引用"""
        return self.history_ref_regex.findall(query)

    def _detect_ambiguity(
        self, query: str, session_context: Optional[Dict]
    ) -> tuple[bool, float, Dict]:
        """检测查询歧义"""
        details = {}

        # 检查多实体歧义
        # 例如: "苹果怎么样" -> 可能是公司或水果
        ambiguous_entities = self._extract_potential_ambiguous_entities(query)

        if ambiguous_entities:
            details["ambiguous_entities"] = ambiguous_entities
            confidence = min(0.5 + len(ambiguous_entities) * 0.15, 0.9)
            return True, confidence, details

        # 检查意图熵 (简化版：多个意图关键词)
        intent_keywords = self._count_intent_keywords(query)
        if intent_keywords > 2:
            details["multiple_intent_signals"] = intent_keywords
            return True, 0.6, details

        return False, 0.0, details

    def _extract_potential_ambiguous_entities(self, query: str) -> List[Dict]:
        """提取潜在歧义实体 (简化版)"""
        # 常见歧义实体词典
        ambiguous_dict = {
            "苹果": ["苹果公司", "苹果水果"],
            "java": ["Java编程语言", "爪哇岛"],
            "python": ["Python编程语言", "蟒蛇"],
            " Aurora": ["极光", "欧若拉"],
            "": ["亚马逊公司", "亚马逊雨林"],
        }

        results = []
        query_lower = query.lower()

        for entity, meanings in ambiguous_dict.items():
            if entity in query_lower:
                results.append({"entity": entity, "possible_meanings": meanings})

        return results

    def _count_intent_keywords(self, query: str) -> int:
        """计算意图关键词数量"""
        intent_keywords = [
            "是什么",
            "为什么",
            "怎么做",
            "多少钱",
            "在哪里",
            "对比",
            "区别",
            "推荐",
            "评价",
            "分析",
        ]
        count = 0
        for keyword in intent_keywords:
            if keyword in query:
                count += 1
        return count

    def _detect_multi_question(self, query: str) -> tuple[bool, float, Dict]:
        """检测复合问题"""
        details = {}

        # 方法1: 统计疑问词数量
        question_count = 0
        for qw in self.QUESTION_WORDS:
            question_count += query.count(qw)

        # 方法2: 检查分句数量 (基于标点)
        sentences = re.split(r"[。！？；\n]", query)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 方法3: 检查连接词
        connectors = [
            "并且",
            "而且",
            "同时",
            "另外",
            "还有",
            "以及",
            "首先",
            "其次",
            "最后",
            "第一",
            "第二",
        ]
        connector_count = sum(1 for c in connectors if c in query)

        details["question_words_count"] = question_count
        details["sentence_count"] = len(sentences)
        details["connector_count"] = connector_count

        # 判断逻辑
        is_multi = False
        confidence = 0.0

        if question_count >= 2 and len(sentences) >= 2:
            is_multi = True
            confidence = 0.8
        elif connector_count > 0 and question_count >= 1:
            is_multi = True
            confidence = 0.6
        elif len(sentences) >= 3:
            is_multi = True
            confidence = 0.5

        return is_multi, confidence, details

    def _detect_non_retrieval(self, query: str) -> tuple[bool, float]:
        """检测非检索型查询"""
        query_stripped = query.strip()

        # 检查是否匹配非检索模式
        if self.non_retrieval_regex.match(query_stripped):
            return True, 0.95

        # 检查长度
        if len(query_stripped) < 5:
            return True, 0.7

        # 检查是否是纯问候
        greeting_words = ["你好", "您好", "嗨", "哈喽", "hello", "hi"]
        if any(query_stripped.startswith(gw) for gw in greeting_words):
            return True, 0.8

        return False, 0.0

    def _detect_temporal(self, query: str) -> tuple[bool, float, List[str]]:
        """检测时间敏感查询"""
        matches = self.temporal_regex.findall(query)

        if matches:
            confidence = min(0.5 + len(matches) * 0.1, 0.95)
            return True, confidence, list(set(matches))

        return False, 0.0, []

    def _detect_comparative(self, query: str) -> tuple[bool, float, List[str]]:
        """检测对比型查询"""
        matches = self.comparative_regex.findall(query)

        if matches:
            # 尝试提取对比实体
            entities = self._extract_comparative_entities(query)
            confidence = min(0.6 + len(entities) * 0.1, 0.9) if entities else 0.6
            return True, confidence, entities

        return False, 0.0, []

    def _extract_comparative_entities(self, query: str) -> List[str]:
        """提取对比实体"""
        # 简化版：查找"A和B"、"A vs B"模式
        patterns = [
            r"([\w\u4e00-\u9fa5]+)[和与跟].*?([\w\u4e00-\u9fa5]+).*?[区别差异不同对比]",
            r"([\w\u4e00-\u9fa5]+)\s*vs\s*([\w\u4e00-\u9fa5]+)",
            r"([\w\u4e00-\u9fa5]+)\s*[和与跟]?\s*([\w\u4e00-\u9fa5]+)\s*[有]?什么[区别不同]",
        ]

        entities = []
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entities.extend(match.groups())

        return list(set(entities))

    def _determine_routing_strategy(self, tags: List[SceneTag]) -> str:
        """根据标签确定路由策略"""
        tag_values = {t.value for t in tags}

        if SceneTag.NON_RETRIEVAL.value in tag_values:
            return "direct_generation"

        if SceneTag.MULTI_QUESTION.value in tag_values:
            return "multi_question_decomposition"

        if SceneTag.COMPARATIVE.value in tag_values:
            return "comparative_analysis"

        if SceneTag.AMBIGUOUS.value in tag_values:
            return "hyde_multi_hypothesis"

        if SceneTag.HISTORY_REF.value in tag_values:
            return "contextual_rewrite"

        if SceneTag.TEMPORAL.value in tag_values:
            return "temporal_boost"

        return "standard_rag"


# 单例
scene_tagger = SceneTagger()

__all__ = ["SceneTagger", "SceneTag", "SceneTagResult", "scene_tagger"]
