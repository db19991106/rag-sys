"""
意图识别服务 - 基于大语言模型的识别方案
支持动态使用本地部署的7B参数规模大语言模型
"""

from typing import Dict, List, Tuple, Optional
from enum import Enum
import json
import re
import time
from utils.logger import logger


class IntentType(str, Enum):
    """意图类型枚举"""

    QUESTION = "question"  # 问题咨询
    SEARCH = "search"  # 信息搜索
    SUMMARY = "summary"  # 内容总结
    COMPARISON = "comparison"  # 对比分析
    PROCEDURE = "procedure"  # 操作流程
    DEFINITION = "definition"  # 定义说明
    GREETING = "greeting"  # 问候
    OTHER = "other"  # 其他


class IntentConfig:
    """意图对应的检索配置"""

    CONFIGS = {
        IntentType.QUESTION: {
            "top_k": 5,
            "similarity_threshold": 0.2,
            "description": "问题咨询",
            "prompt_template": "根据以下信息回答问题：",
        },
        IntentType.SEARCH: {
            "top_k": 10,
            "similarity_threshold": 0.2,
            "description": "信息搜索",
            "prompt_template": "以下是搜索到的相关信息：",
        },
        IntentType.SUMMARY: {
            "top_k": 15,
            "similarity_threshold": 0.2,
            "description": "内容总结",
            "prompt_template": "请总结以下内容：",
        },
        IntentType.COMPARISON: {
            "top_k": 8,
            "similarity_threshold": 0.2,
            "description": "对比分析",
            "prompt_template": "对比以下信息：",
        },
        IntentType.PROCEDURE: {
            "top_k": 5,
            "similarity_threshold": 0.2,
            "description": "操作流程",
            "prompt_template": "以下操作步骤：",
        },
        IntentType.DEFINITION: {
            "top_k": 5,
            "similarity_threshold": 0.2,
            "description": "定义说明",
            "prompt_template": "定义如下：",
        },
        IntentType.GREETING: {
            "top_k": 3,
            "similarity_threshold": 0.2,
            "description": "问候",
            "prompt_template": "您好！",
        },
        IntentType.OTHER: {
            "top_k": 5,
            "similarity_threshold": 0.2,
            "description": "其他",
            "prompt_template": "根据以下信息回答：",
        },
    }

    @classmethod
    def get_config(cls, intent: IntentType) -> Dict:
        """获取意图对应的配置"""
        return cls.CONFIGS.get(intent, cls.CONFIGS[IntentType.OTHER])


class IntentRecognizer:
    """
    意图识别器

    支持多种识别方法：
    1. 基于规则的快速识别
    2. 基于LLM的精确识别（当启用且模型可用时）
    """

    def __init__(self):
        self._initialized = False
        self._config = None

    def initialize_with_config(self, config):
        """使用配置初始化"""
        self._config = config
        self._initialized = True
        logger.info("意图识别器已初始化（使用配置）")

    def recognize(self, query: str) -> Tuple[IntentType, float, Dict]:
        """
        识别用户查询的意图

        Args:
            query: 用户查询文本

        Returns:
            Tuple of (意图类型, 置信度, 详细信息)
        """
        # 首先尝试基于规则的快速识别
        intent, confidence, details = self._rule_based_recognize(query)

        # 如果规则识别置信度高，直接返回
        if confidence >= 0.9:
            return intent, confidence, details

        # 如果配置了LLM且初始化成功，使用LLM进行精确识别
        if self._initialized and self._config:
            try:
                llm_intent, llm_confidence, llm_details = self._llm_based_recognize(
                    query
                )
                # 如果LLM置信度更高，使用LLM结果
                if llm_confidence > confidence:
                    return llm_intent, llm_confidence, llm_details
            except Exception as e:
                logger.warning(f"LLM意图识别失败，使用规则结果: {e}")

        return intent, confidence, details

    def _rule_based_recognize(self, query: str) -> Tuple[IntentType, float, Dict]:
        """
        基于规则的意图识别

        通过关键词匹配快速识别常见意图
        """
        query_lower = query.lower().strip()

        # 1. 问候识别
        greeting_patterns = [
            r"^(你好|您好|嗨|hello|hi|hey|早上好|下午好|晚上好)",
            r"^(在吗|在不在|有人吗)",
        ]
        for pattern in greeting_patterns:
            if re.search(pattern, query_lower):
                return (
                    IntentType.GREETING,
                    0.95,
                    {"method": "rule", "matched_pattern": "greeting"},
                )

        # 2. 总结类识别
        summary_keywords = [
            "总结",
            "概括",
            "概述",
            "汇总",
            "归纳",
            "总结下",
            "概括一下",
        ]
        if any(kw in query_lower for kw in summary_keywords):
            return (
                IntentType.SUMMARY,
                0.9,
                {"method": "rule", "matched_keywords": summary_keywords},
            )

        # 3. 对比类识别
        comparison_keywords = [
            "对比",
            "比较",
            "区别",
            "差异",
            "不同",
            "vs",
            "versus",
            "哪个更好",
            "哪个更",
        ]
        if any(kw in query_lower for kw in comparison_keywords):
            return (
                IntentType.COMPARISON,
                0.9,
                {"method": "rule", "matched_keywords": comparison_keywords},
            )

        # 4. 流程/步骤类识别
        procedure_keywords = [
            "怎么",
            "如何",
            "步骤",
            "流程",
            "怎么做",
            "如何做",
            "怎样",
            "方法",
            "教程",
            "指南",
        ]
        if any(kw in query_lower for kw in procedure_keywords):
            return (
                IntentType.PROCEDURE,
                0.85,
                {"method": "rule", "matched_keywords": procedure_keywords},
            )

        # 5. 定义类识别
        definition_keywords = ["是什么", "什么是", "定义", "概念", "意思", "含义"]
        if any(kw in query_lower for kw in definition_keywords):
            return (
                IntentType.DEFINITION,
                0.85,
                {"method": "rule", "matched_keywords": definition_keywords},
            )

        # 6. 搜索类识别（广义的查询）
        search_keywords = ["搜索", "查找", "找", "查询", "列出", "有哪些", "有什么"]
        if any(kw in query_lower for kw in search_keywords):
            return (
                IntentType.SEARCH,
                0.8,
                {"method": "rule", "matched_keywords": search_keywords},
            )

        # 7. 默认问题类
        question_patterns = [
            r".*[？?]$",  # 以问号结尾
            r"^(请问|我想知道|能否|能否告诉我)",  # 疑问词开头
        ]
        for pattern in question_patterns:
            if re.search(pattern, query_lower):
                return (
                    IntentType.QUESTION,
                    0.7,
                    {"method": "rule", "matched_pattern": "question"},
                )

        # 默认其他类型
        return IntentType.OTHER, 0.5, {"method": "rule", "fallback": True}

    def _llm_based_recognize(self, query: str) -> Tuple[IntentType, float, Dict]:
        """
        基于LLM的意图识别

        使用大语言模型进行更精确的意图识别
        """
        from services.rag_generator import rag_generator

        # 构建提示词
        prompt = f"""分析以下用户查询的意图类别。

用户查询: "{query}"

请从以下类别中选择最匹配的意图：
- question: 问题咨询（询问具体信息）
- search: 信息搜索（查找资料）
- summary: 内容总结（要求总结内容）
- comparison: 对比分析（比较差异）
- procedure: 操作流程（询问步骤、流程）
- definition: 定义说明（询问定义、概念）
- greeting: 问候（打招呼）
- other: 其他

请以JSON格式返回结果：
{{"intent": "意图类别", "confidence": 0.8, "reason": "判断理由"}}

注意：只返回JSON，不要其他内容。"""

        try:
            # 使用轻量级配置调用LLM
            from models import GenerationConfig

            llm_config = GenerationConfig(
                llm_provider="local",
                llm_model="Qwen2.5-7B-Instruct",
                temperature=0.1,  # 低温度，确定性输出
                max_tokens=200,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )

            # 获取LLM客户端并生成
            llm_client = rag_generator._get_llm_client(llm_config)
            response_data = llm_client.generate(prompt)
            
            # 提取文本响应
            response = response_data.get("text", "")

            # 尝试解析JSON
            result = json.loads(response.strip())

            intent_str = result.get("intent", "other")
            confidence = result.get("confidence", 0.5)
            reason = result.get("reason", "")

            # 转换字符串为IntentType
            try:
                intent = IntentType(intent_str)
            except ValueError:
                intent = IntentType.OTHER

            return (
                intent,
                confidence,
                {"method": "llm", "reason": reason, "raw_response": response},
            )

        except Exception as e:
            logger.error(f"LLM意图识别解析失败: {e}")
            # 如果LLM识别失败，返回其他类型
            return IntentType.OTHER, 0.3, {"method": "llm", "error": str(e)}

    def recognize_intent(self, query: str) -> Dict:
        """
        识别意图（兼容旧接口）

        Args:
            query: 用户查询文本

        Returns:
            Dict: 包含intent、confidence和details的字典
        """
        intent, confidence, details = self.recognize(query)
        return {"intent": intent.value, "confidence": confidence, "details": details}


# 全局意图识别器实例
intent_recognizer = IntentRecognizer()
