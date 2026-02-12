"""
Token预算管理器 - 动态管理上下文窗口
解决上下文窗口管理粗糙问题
"""

from typing import List, Dict, Any, Optional
from utils.logger import logger
from config import settings


class TokenBudgetManager:
    """
    Token预算管理器
    - 根据模型动态计算可用token预算
    - 智能分配上下文空间
    - 预留系统prompt和输出空间
    """

    # 各模型的上下文窗口大小
    MODEL_CONTEXT_LIMITS = {
        # OpenAI模型
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
        # Anthropic模型
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        # 本地模型
        "Qwen2.5-7B-Instruct": 32768,
        "Qwen2.5-0.5B-Instruct": 32768,
        "chatglm3-6b": 8192,
        "baichuan2-7b": 4096,
    }

    # 预留token数
    RESERVED_TOKENS = {
        "system_prompt": 200,  # 系统prompt
        "output": 500,  # 输出空间
        "safety_buffer": 200,  # 安全缓冲
        "query": 100,  # 查询文本
        "identity": 50,  # 身份信息
        "references": 100,  # 引用标记
    }

    def __init__(self, model_name: str = None):
        """
        初始化Token预算管理器

        Args:
            model_name: 模型名称，默认从配置读取
        """
        self.model_name = model_name or getattr(settings, "llm_model", "gpt-3.5-turbo")
        self.total_limit = self._get_context_limit()
        self.reserved = sum(self.RESERVED_TOKENS.values())
        self.available = self.total_limit - self.reserved

        logger.info(
            f"Token预算管理器初始化: 模型={self.model_name}, "
            f"总限制={self.total_limit}, 预留={self.reserved}, 可用={self.available}"
        )

    def _get_context_limit(self) -> int:
        """获取模型的上下文窗口大小"""
        # 精确匹配
        if self.model_name in self.MODEL_CONTEXT_LIMITS:
            return self.MODEL_CONTEXT_LIMITS[self.model_name]

        # 部分匹配
        for model_key, limit in self.MODEL_CONTEXT_LIMITS.items():
            if model_key.lower() in self.model_name.lower():
                return limit

        # 默认使用保守值
        logger.warning(f"未知的模型 {self.model_name}，使用默认上下文窗口 4096")
        return 4096

    def estimate_tokens(self, text: str) -> int:
        """
        估算文本的token数

        简单估算：
        - 英文：1 token ≈ 4 字符
        - 中文：1 token ≈ 1 汉字
        - 混合：取平均值

        Args:
            text: 待估算文本

        Returns:
            估算的token数
        """
        if not text:
            return 0

        # 计算字符数
        char_count = len(text)

        # 统计中文字符数
        chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        other_chars = char_count - chinese_chars

        # 估算：中文字符1:1，其他字符1:0.25
        estimated_tokens = chinese_chars + (other_chars / 4)

        # 增加10%的缓冲
        return int(estimated_tokens * 1.1)

    def allocate_context_budget(
        self, chunks: List[Dict], query: str = "", identity: str = None
    ) -> List[Dict]:
        """
        按优先级分配上下文token预算

        Args:
            chunks: 检索到的文档片段
            query: 用户查询
            identity: 用户身份信息

        Returns:
            选中的片段列表
        """
        if not chunks:
            return []

        # 计算已使用的token
        used_tokens = self.estimate_tokens(query)
        if identity:
            used_tokens += self.estimate_tokens(identity)

        # 可用预算
        remaining_budget = self.available - used_tokens

        logger.info(
            f"Token预算分配: 总可用={self.available}, 查询占用={used_tokens}, "
            f"剩余={remaining_budget}, 片段数={len(chunks)}"
        )

        # 按相似度排序（已排序）
        selected = []
        current_tokens = 0

        for chunk in chunks:
            content = chunk.get("content", "")
            chunk_tokens = self.estimate_tokens(content)

            # 预估片段占用（包含格式化标记）
            formatted_tokens = chunk_tokens + 20  # 20 tokens for labels

            if current_tokens + formatted_tokens <= remaining_budget:
                selected.append(chunk)
                current_tokens += formatted_tokens
            else:
                # 剩余空间不足，尝试截断
                remaining = remaining_budget - current_tokens
                if remaining > 100:  # 至少还能容纳100 tokens
                    # 截断内容
                    truncated_content = self._truncate_to_tokens(
                        content, remaining - 30
                    )
                    if truncated_content:
                        chunk_copy = chunk.copy()
                        chunk_copy["content"] = truncated_content + "...(已截断)"
                        selected.append(chunk_copy)
                        current_tokens += self.estimate_tokens(truncated_content) + 30
                break

        logger.info(
            f"上下文分配完成: 选中 {len(selected)}/{len(chunks)} 个片段, "
            f"预估token数={current_tokens}"
        )

        return selected

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        将文本截断到指定token数以内

        Args:
            text: 原始文本
            max_tokens: 最大token数

        Returns:
            截断后的文本
        """
        if not text:
            return ""

        # 二分查找合适的截断位置
        left, right = 0, len(text)
        result = ""

        while left <= right:
            mid = (left + right) // 2
            truncated = text[:mid]
            tokens = self.estimate_tokens(truncated)

            if tokens <= max_tokens:
                result = truncated
                left = mid + 1
            else:
                right = mid - 1

        # 尝试在句子边界截断
        if result:
            for sep in ["\n\n", "。", "；", "！", "？", "\n"]:
                last_sep = result.rfind(sep)
                if last_sep > len(result) * 0.7:  # 至少保留70%
                    result = result[: last_sep + len(sep)]
                    break

        return result

    def validate_context_window(
        self, prompt: str, max_tokens: int = None
    ) -> Dict[str, Any]:
        """
        验证上下文窗口是否足够

        Args:
            prompt: 完整的prompt
            max_tokens: 期望的输出token数

        Returns:
            验证结果
        """
        prompt_tokens = self.estimate_tokens(prompt)
        output_tokens = max_tokens or self.RESERVED_TOKENS["output"]
        total_needed = prompt_tokens + output_tokens

        is_valid = total_needed <= self.total_limit

        result = {
            "is_valid": is_valid,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_needed": total_needed,
            "context_limit": self.total_limit,
            "remaining": self.total_limit - total_needed,
        }

        if not is_valid:
            result["warning"] = (
                f"超出上下文窗口: 需要{total_needed} tokens，限制{self.total_limit} tokens"
            )
            logger.warning(result["warning"])

        return result

    def get_budget_summary(self) -> Dict[str, Any]:
        """获取预算摘要"""
        return {
            "model": self.model_name,
            "context_limit": self.total_limit,
            "reserved": self.reserved,
            "available": self.available,
            "reservation_breakdown": self.RESERVED_TOKENS.copy(),
        }


# 便捷函数
def create_token_budget_manager(model_name: str = None) -> TokenBudgetManager:
    """创建Token预算管理器"""
    return TokenBudgetManager(model_name)


def estimate_tokens_for_chunks(chunks: List[Dict], query: str = "") -> int:
    """估算chunks和query的总token数"""
    manager = TokenBudgetManager()
    total = manager.estimate_tokens(query)
    for chunk in chunks:
        total += manager.estimate_tokens(chunk.get("content", ""))
    return total
