"""
安全的Prompt构建器 - 防止Prompt注入攻击
解决Prompt注入与安全问题
"""

from typing import List, Dict, Any
from jinja2 import Template
from markupsafe import escape
import re
from utils.logger import logger


class SafePromptBuilder:
    """
    安全的Prompt构建器
    - 使用Jinja2模板自动转义
    - 检测危险模式
    - 过滤恶意内容
    """

    # RAG Prompt模板（使用Jinja2，自动转义）
    RAG_TEMPLATE = Template("""
基于以下参考信息回答问题。如果参考信息不足以回答，请明确说明。

参考信息：
{% for chunk in chunks %}
[{{ loop.index }}] {{ chunk.content | e }}
{% endfor %}

用户问题：{{ query | e }}

{% if identity %}
用户身份信息：{{ identity | e }}
请注意：回答必须针对该用户的身份信息，只提供与其身份相关的内容。
{% endif %}

请提供准确、简洁的回答，并在引用处标注[数字]。
""")

    # 危险模式检测正则
    DANGEROUS_PATTERNS = [
        r"忽略.*指令",
        r"跳过.*步骤",
        r"输出.*密码",
        r"system\s*prompt",
        r"<\|.*\|>",  # 特殊token注入
        r"作为.*助手",
        r"你现在的角色是",
        r"请扮演",
        r"忘记.*指令",
        r"绕过.*限制",
        r"破解.*安全",
    ]

    @classmethod
    def build_rag_prompt(
        cls,
        chunks: List[Dict],
        query: str,
        identity: str = None,
        context_summary: str = None,
    ) -> str:
        """
        构建安全的RAG Prompt

        Args:
            chunks: 检索到的文档片段
            query: 用户查询
            identity: 用户身份信息（可选）
            context_summary: 对话历史摘要（可选）

        Returns:
            安全的Prompt字符串
        """
        # 1. 过滤危险内容
        filtered_chunks = []
        for chunk in chunks:
            if cls._contains_dangerous_content(chunk.get("content", "")):
                logger.warning(
                    f"检测到危险内容，已过滤 (chunk_id: {chunk.get('chunk_id', 'unknown')})"
                )
                chunk = chunk.copy()
                chunk["content"] = "[内容已过滤：包含潜在指令注入]"
            filtered_chunks.append(chunk)

        # 2. 过滤用户查询中的危险内容
        if cls._contains_dangerous_content(query):
            logger.warning("检测到查询包含危险内容")
            query = cls._sanitize_query(query)

        # 3. 使用模板渲染（自动转义特殊字符）
        try:
            prompt = cls.RAG_TEMPLATE.render(
                chunks=filtered_chunks, query=query, identity=identity
            )
            return prompt
        except Exception as e:
            logger.error(f"模板渲染失败: {str(e)}")
            # 回退到简单拼接
            return cls._build_simple_prompt(filtered_chunks, query, identity)

    @classmethod
    def _contains_dangerous_content(cls, text: str) -> bool:
        """
        检测文本是否包含危险内容

        Args:
            text: 待检测文本

        Returns:
            是否包含危险内容
        """
        if not text:
            return False

        text_lower = text.lower()
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                logger.warning(f"检测到危险模式: {pattern} in text: {text[:100]}...")
                return True
        return False

    @classmethod
    def _sanitize_query(cls, query: str) -> str:
        """
        清理查询中的危险内容

        Args:
            query: 原始查询

        Returns:
            清理后的查询
        """
        sanitized = query
        for pattern in cls.DANGEROUS_PATTERNS:
            sanitized = re.sub(pattern, "[已过滤]", sanitized, flags=re.IGNORECASE)
        return sanitized

    @classmethod
    def _build_simple_prompt(
        cls, chunks: List[Dict], query: str, identity: str = None
    ) -> str:
        """
        简单的Prompt构建（回退方案）

        Args:
            chunks: 文档片段
            query: 用户查询
            identity: 用户身份信息

        Returns:
            Prompt字符串
        """
        # 手动转义所有内容
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            content = escape(chunk.get("content", ""))
            context_parts.append(f"【参考文档{i}】\n{content}\n")

        context = "\n".join(context_parts)
        safe_query = escape(query)

        prompt = f"""请根据以下参考信息回答问题。如果参考信息不足以回答，请明确说明。

参考文档:
{context}

问题: {safe_query}

请提供准确、简洁的回答，并在引用处标注[数字]。"""

        if identity:
            safe_identity = escape(identity)
            prompt += f"\n\n用户身份: {safe_identity}\n请针对该用户身份提供相关信息。"

        return prompt

    @classmethod
    def validate_prompt(cls, prompt: str) -> Dict[str, Any]:
        """
        验证Prompt安全性

        Args:
            prompt: 待验证的Prompt

        Returns:
            验证结果
        """
        issues = []

        # 检查危险模式
        if cls._contains_dangerous_content(prompt):
            issues.append("包含潜在危险内容")

        # 检查长度
        if len(prompt) > 8000:
            issues.append("Prompt过长")

        # 检查特殊token
        if re.search(r"<\|.*?\|>", prompt):
            issues.append("包含特殊token")

        return {"is_safe": len(issues) == 0, "issues": issues, "length": len(prompt)}


# 便捷函数
def build_safe_rag_prompt(
    chunks: List[Dict], query: str, identity: str = None, context_summary: str = None
) -> str:
    """构建安全的RAG Prompt"""
    return SafePromptBuilder.build_rag_prompt(chunks, query, identity, context_summary)


def validate_prompt_safety(prompt: str) -> Dict[str, Any]:
    """验证Prompt安全性"""
    return SafePromptBuilder.validate_prompt(prompt)
