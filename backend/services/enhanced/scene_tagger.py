"""
场景标签识别器 (SceneTagger)
识别查询场景标签：HistoryRef, Ambiguous, MultiQuestion, NonRetrieval, Temporal, Comparative
"""

import re
import os
import torch
from typing import List, Dict, Set, Optional
from enum import Enum
from dataclasses import dataclass
from utils.logger import logger
from config import settings


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
        self.comparative_regex = re.compile(
            "|".join(self.COMPARATIVE_PATTERNS), re.IGNORECASE
        )
        self.temporal_regex = re.compile(
            "|".join(self.TEMPORAL_PATTERNS), re.IGNORECASE
        )
        self.non_retrieval_regex = re.compile(
            "|".join(self.NON_RETRIEVAL_PATTERNS), re.IGNORECASE
        )
        
        # LLM 客户端用于判断历史引用
        self.coref_llm = None
        self.coref_tokenizer = None
        self._coref_llm_initialized = False
    
    def _init_coref_llm(self):
        """延迟初始化指代消解专用 LLM"""
        if self._coref_llm_initialized:
            return
            
        try:
            os.environ['TQDM_DISABLE'] = '1'
            os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
            os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_path = settings.coref_llm_model_path
            device = settings.coref_llm_device
            
            logger.info(f"[SceneTagger] 初始化 LLM: {model_path}")

            import sys
            from io import StringIO
            
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            try:
                sys.stdout = StringIO()
                sys.stderr = StringIO()
                
                self.coref_tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )

                model_kwargs = {
                    "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                    "device_map": "auto" if device == "cuda" else None,
                }

                self.coref_llm = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    **model_kwargs
                )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

            if device == "cpu":
                self.coref_llm = self.coref_llm.to(device)

            self._coref_llm_initialized = True
            logger.info(f"[SceneTagger] LLM 初始化完成")

        except Exception as e:
            logger.error(f"[SceneTagger] 初始化 LLM 失败: {str(e)}")
            raise
    
    def _coref_llm_chat(self, prompt: str, temperature: float = 0.0) -> str:
        """调用 LLM"""
        if not self._coref_llm_initialized:
            self._init_coref_llm()
            
        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.coref_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.coref_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.coref_llm.device)

            with torch.no_grad():
                do_sample = True
                temp = temperature
                if temperature <= 0.0:
                    do_sample = False
                    temp = 1.0
                
                outputs = self.coref_llm.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=temp,
                    top_p=0.9,
                    do_sample=do_sample,
                    pad_token_id=self.coref_tokenizer.eos_token_id
                )

            response = self.coref_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return response.strip()

        except Exception as e:
            logger.error(f"[SceneTagger] LLM 调用失败: {str(e)}")
            return ""

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
        """使用 LLM 检测历史引用"""
        if not session_context or not session_context.get("has_history"):
            return False, 0.0

        # 获取对话历史
        history = session_context.get("history", [])
        if not history:
            return False, 0.0
        
        # 格式化对话历史（最近3轮）
        recent_history = history[-3:] if len(history) > 3 else history
        history_str = ""
        for msg in recent_history:
            role = msg.get("role", "")
            content = msg.get("content", "")[:100]  # 截断过长的内容
            role_cn = "用户" if role == "user" else "助手"
            history_str += f"{role_cn}：{content}\n"
        
        # 使用 LLM 判断
        prompt = f"""任务：判断用户查询是否需要参考对话历史。

对话历史：
{history_str}
当前查询：{query}

判断标准：
- 查询有"它/这/那/该/此"等代词 → 回答"是"
- 查询省略了主语，需要历史补充 → 回答"是"
- 查询完整独立，无需历史 → 回答"否"

只回答"是"或"否"："""

        try:
            resp = self._coref_llm_chat(prompt, temperature=0.0).strip()
            is_contextual = resp == "是"
            
            if is_contextual:
                logger.info(f"[SceneTagger] LLM 判断查询依赖历史: {query}")
                return True, 0.85  # LLM 置信度
            else:
                return False, 0.0
                
        except Exception as e:
            logger.error(f"[SceneTagger] LLM 判断失败: {str(e)}")
            return False, 0.0

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
