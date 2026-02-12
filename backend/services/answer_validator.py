"""
答案验证器 - 检查答案与上下文的一致性
解决答案溯源与可信度校验问题
"""

import re
from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher
from utils.logger import logger


class AnswerValidator:
    """
    答案验证器
    - 检查答案是否真实来自上下文
    - 检测幻觉（Hallucination）
    - 语义一致性检查
    """

    def __init__(self, similarity_threshold: float = 0.6):
        """
        初始化答案验证器

        Args:
            similarity_threshold: 相似度阈值，低于此值视为幻觉
        """
        self.similarity_threshold = similarity_threshold

    def validate_answer(
        self, answer: str, context_chunks: List[str], query: str = ""
    ) -> Dict[str, Any]:
        """
        验证答案的真实性和一致性

        Args:
            answer: LLM生成的答案
            context_chunks: 上下文片段列表
            query: 用户查询

        Returns:
            验证结果
        """
        validation_result = {
            "is_valid": True,
            "confidence": 1.0,
            "issues": [],
            "supporting_evidence": [],
            "unsupported_claims": [],
            "hallucination_detected": False,
        }

        if not answer or not context_chunks:
            validation_result["is_valid"] = False
            validation_result["issues"].append("答案或上下文为空")
            return validation_result

        # 1. 提取答案中的关键陈述
        claims = self._extract_claims(answer)

        # 2. 检查每个陈述是否有上下文支持
        for claim in claims:
            support_info = self._find_support(claim, context_chunks)

            if support_info["found"]:
                validation_result["supporting_evidence"].append(
                    {
                        "claim": claim,
                        "source": support_info["source"],
                        "similarity": support_info["similarity"],
                    }
                )
            else:
                validation_result["unsupported_claims"].append(
                    {"claim": claim, "max_similarity": support_info["similarity"]}
                )

                if support_info["similarity"] < self.similarity_threshold:
                    validation_result["hallucination_detected"] = True

        # 3. 计算整体可信度
        total_claims = len(claims)
        supported_claims = len(validation_result["supporting_evidence"])

        if total_claims > 0:
            validation_result["confidence"] = supported_claims / total_claims
            validation_result["is_valid"] = (
                validation_result["confidence"] >= self.similarity_threshold
            )

        # 4. 检查引用真实性
        citation_check = self._validate_citations(answer, context_chunks)
        validation_result["citation_check"] = citation_check

        if citation_check["invalid_citations"]:
            validation_result["issues"].append(
                f"发现{citation_check['invalid_citations']}个无效引用"
            )

        # 5. 检查答案是否完全偏离主题
        if self._is_off_topic(answer, query, context_chunks):
            validation_result["is_valid"] = False
            validation_result["issues"].append("答案可能偏离主题")

        return validation_result

    def _extract_claims(self, answer: str) -> List[str]:
        """
        从答案中提取关键陈述

        Args:
            answer: 答案文本

        Returns:
            关键陈述列表
        """
        claims = []

        # 按句子分割
        sentences = re.split(r"[。！？；\n]+", answer)

        for sent in sentences:
            sent = sent.strip()
            # 过滤掉过短或引用标记
            if len(sent) > 10 and not re.match(r"^\[\d+\]", sent):
                claims.append(sent)

        return claims

    def _find_support(self, claim: str, context_chunks: List[str]) -> Dict[str, Any]:
        """
        在上下文中查找支持证据

        Args:
            claim: 待验证的陈述
            context_chunks: 上下文片段

        Returns:
            支持信息
        """
        best_match = {"found": False, "source": None, "similarity": 0.0}

        for i, chunk in enumerate(context_chunks):
            # 计算文本相似度
            similarity = self._calculate_similarity(claim, chunk)

            if similarity > best_match["similarity"]:
                best_match["similarity"] = similarity
                best_match["source"] = f"片段{i + 1}"

            # 如果相似度超过阈值，认为找到支持
            if similarity >= self.similarity_threshold:
                best_match["found"] = True

        return best_match

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两段文本的相似度

        使用SequenceMatcher + 关键词匹配的组合方法

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度分数 (0-1)
        """
        # 方法1: SequenceMatcher（字符级）
        char_similarity = SequenceMatcher(None, text1, text2).ratio()

        # 方法2: 关键词匹配
        keywords1 = set(self._extract_keywords(text1))
        keywords2 = set(self._extract_keywords(text2))

        if keywords1 and keywords2:
            intersection = keywords1 & keywords2
            union = keywords1 | keywords2
            keyword_similarity = len(intersection) / len(union) if union else 0
        else:
            keyword_similarity = 0

        # 组合两种方法（权重可调整）
        combined_similarity = 0.4 * char_similarity + 0.6 * keyword_similarity

        return combined_similarity

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取：去除停用词和标点
        import re

        # 移除标点符号
        text = re.sub(r"[^\w\s]", "", text)

        # 停用词列表（简单版）
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
        }

        words = [w for w in text.split() if w not in stop_words and len(w) > 1]
        return words

    def _validate_citations(
        self, answer: str, context_chunks: List[str]
    ) -> Dict[str, Any]:
        """
        验证答案中的引用标记是否有效

        Args:
            answer: 答案文本
            context_chunks: 上下文片段

        Returns:
            引用验证结果
        """
        # 提取引用标记，如 [1], [2,3], [1-3]
        citation_pattern = r"\[(\d+(?:[-,\s]*\d+)*)\]"
        citations = re.findall(citation_pattern, answer)

        total_citations = len(citations)
        valid_citations = 0
        invalid_citations = 0

        for citation in citations:
            # 解析引用编号
            numbers = self._parse_citation_numbers(citation)

            for num in numbers:
                # 检查引用编号是否在有效范围内
                if 1 <= num <= len(context_chunks):
                    valid_citations += 1
                else:
                    invalid_citations += 1

        return {
            "total_citations": total_citations,
            "valid_citations": valid_citations,
            "invalid_citations": invalid_citations,
            "citation_list": citations,
        }

    def _parse_citation_numbers(self, citation: str) -> List[int]:
        """解析引用字符串为数字列表"""
        numbers = []
        # 移除空格
        citation = citation.replace(" ", "")

        # 处理范围，如 "1-3"
        if "-" in citation:
            parts = citation.split("-")
            if len(parts) == 2:
                try:
                    start, end = int(parts[0]), int(parts[1])
                    numbers.extend(range(start, end + 1))
                except ValueError:
                    pass
        # 处理列表，如 "1,2,3"
        elif "," in citation:
            for part in citation.split(","):
                try:
                    numbers.append(int(part))
                except ValueError:
                    pass
        # 单个数字
        else:
            try:
                numbers.append(int(citation))
            except ValueError:
                pass

        return numbers

    def _is_off_topic(self, answer: str, query: str, context_chunks: List[str]) -> bool:
        """
        检查答案是否偏离主题

        Args:
            answer: 答案
            query: 查询
            context_chunks: 上下文

        Returns:
            是否偏离主题
        """
        # 提取查询关键词
        query_keywords = set(self._extract_keywords(query))

        # 提取上下文关键词
        context_text = " ".join(context_chunks)
        context_keywords = set(self._extract_keywords(context_text))

        # 提取答案关键词
        answer_keywords = set(self._extract_keywords(answer))

        # 检查答案是否包含查询或上下文的关键词
        if not query_keywords:
            return False

        query_overlap = len(answer_keywords & query_keywords) / len(query_keywords)
        context_overlap = (
            len(answer_keywords & context_keywords) / len(answer_keywords)
            if answer_keywords
            else 0
        )

        # 如果与查询的重叠度太低，或者与上下文的重叠度太低，可能偏离主题
        if query_overlap < 0.1 and context_overlap < 0.2:
            return True

        return False


# 便捷函数
def validate_answer(
    answer: str, context_chunks: List[str], query: str = "", threshold: float = 0.6
) -> Dict[str, Any]:
    """验证答案"""
    validator = AnswerValidator(threshold)
    return validator.validate_answer(answer, context_chunks, query)


def check_hallucination(answer: str, context_chunks: List[str]) -> bool:
    """检查是否出现幻觉"""
    validator = AnswerValidator()
    result = validator.validate_answer(answer, context_chunks)
    return result.get("hallucination_detected", False)
