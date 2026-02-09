from typing import List, Dict, Tuple, Optional, Any
import traceback
from datetime import datetime
from models import Message
from utils.logger import logger

# 导入本地LLM客户端
from services.conversation_manager import llm_client


class ContextAnalyzer:
    """
    上下文分析器 - 分析对话上下文并生成历史感知的改写查询
    """

    def __init__(self):
        self.llm = llm_client
        # 最多保留最近10轮对话（避免上下文过长）
        self.max_history_rounds = 10

    def analyze_context(
        self,
        conversation_history: List[Message],
        current_query: str,
        existing_profile: Dict[str, Any] = None,
    ) -> Dict:
        """
        对外核心接口：分析上下文并生成历史感知的改写查询
        Args:
            conversation_history: 对话历史消息列表
            current_query: 当前用户查询
            existing_profile: 已有的用户画像（累积的身份信息）
        Returns:
            分析结果字典，包含user_profile字段
        """
        # 初始化返回结果
        analysis_result = {
            "is_contextual": False,
            "main_topic": "",
            "entities": [],
            "enhanced_query": current_query,
            "rewritten_query": current_query,
            "context_summary": "",
            "user_profile": existing_profile or {},  # 返回用户画像
        }

        # 无历史对话，直接返回原始查询
        if not conversation_history:
            logger.info("无对话历史，返回原始查询")
            return analysis_result

        try:
            # 1. LLM判断：当前查询是否依赖上下文
            is_contextual = self._llm_is_contextual(conversation_history, current_query)
            analysis_result["is_contextual"] = is_contextual

            if not is_contextual:
                logger.info(f"LLM判断：当前查询「{current_query}」不依赖历史上下文")
                return analysis_result

            # 2. LLM抽取：上下文核心信息（主题/实体/摘要/用户身份）
            main_topic, entities, context_summary, user_identity = (
                self._llm_extract_context_info(conversation_history)
            )
            analysis_result["main_topic"] = main_topic
            analysis_result["entities"] = entities
            analysis_result["context_summary"] = context_summary

            # 合并用户画像（新识别的身份信息合并到已有画像）
            if user_identity:
                if existing_profile:
                    analysis_result["user_profile"] = {
                        **existing_profile,
                        **user_identity,
                    }
                else:
                    analysis_result["user_profile"] = user_identity
                logger.info(f"识别到用户身份信息: {user_identity}")

            # 3. LLM改写：生成独立可检索的查询（传入用户画像）
            rewritten_query = self._llm_rewrite_query(
                conversation_history,
                current_query,
                main_topic,
                entities,
                analysis_result["user_profile"],  # 传入用户画像
            )
            analysis_result["rewritten_query"] = rewritten_query
            analysis_result["enhanced_query"] = rewritten_query

            logger.info(
                f"上下文分析完成 | 核心主题：{main_topic} | 改写后查询：{rewritten_query}"
            )

        except Exception as e:
            logger.error(f"上下文分析异常：{str(e)}\n{traceback.format_exc()}")
            # 异常时返回原始查询，保证程序不中断
            pass

        return analysis_result

    def _format_history(self, history: List[Message]) -> str:
        """格式化对话历史为LLM可读的字符串"""
        # 只保留最近N轮，避免上下文过长
        recent_history = history[-self.max_history_rounds :]
        history_lines = []
        for msg in recent_history:
            role_cn = "用户" if msg.role == "user" else "助手"
            history_lines.append(f"{role_cn}：{msg.content}")
        return "\n".join(history_lines)

    def _llm_is_contextual(self, history: List[Message], query: str) -> bool:
        """判断：当前查询是否依赖历史对话（规则+LLM混合判断）"""
        # 首先用规则快速判断：包含明显指代词
        reference_words = [
            "它",
            "这",
            "那",
            "该",
            "此",
            "其",
            "对方",
            "他们",
            "这个",
            "那个",
            "这种",
            "那种",
        ]
        for word in reference_words:
            if word in query:
                logger.info(f"规则判断：查询中包含指代词「{word}」，依赖上下文")
                return True

        # 如果查询太短（少于10字），很可能是省略句，依赖上下文
        if len(query) < 10:
            logger.info(
                f"规则判断：查询过短（{len(query)}字），可能省略主语，依赖上下文"
            )
            return True

        # 否则使用LLM判断
        history_str = self._format_history(history)

        prompt = f"""你是对话上下文关联性判断专家，请严格按规则判断。

【对话历史】
{history_str}

【当前用户查询】
{query}

判断规则（满足任一即依赖上下文）：
1. 如果查询包含指代词（它、这、那、该、此、其、这个、那个、这种、那种等）→ 依赖上下文
2. 如果查询不完整、缺少主语或宾语 → 依赖上下文
3. 如果查询需要结合历史才能理解 → 依赖上下文
4. 只有查询本身完整独立、无需历史即可理解 → 不依赖

重要：
- "那"、"这个"、"该"等词是指代词，出现时一定依赖上下文！
- 只回答"是"或"否"，不要解释

请回答："""

        try:
            resp = self.llm.chat(prompt, temperature=0.0).strip()
            logger.info(f"LLM判断结果：{resp}")
            return resp == "是"
        except Exception as e:
            logger.error(f"LLM判断失败：{e}")
            # 失败时默认依赖上下文（安全策略）
            return True

    def _llm_extract_context_info(
        self, history: List[Message]
    ) -> Tuple[str, List[str], str, Dict[str, str]]:
        """
        LLM抽取：核心主题、关键实体、上下文摘要、用户身份信息

        Returns:
            (核心主题, 关键实体列表, 上下文摘要, 用户身份信息字典)
        """
        history_str = self._format_history(history)

        prompt = f"""你是专业的对话信息抽取专家，请严格按格式输出结果。

【对话历史】
{history_str}

抽取要求：
1. 核心主题：总结整个对话的核心议题（一句话，不超过20字）
2. 关键实体：提取对话中的核心实体（如职位、制度、产品、概念等，用中文逗号分隔）
3. 用户身份信息：识别用户的职位、部门、级别等身份信息（格式：属性=值，每行一个）
   - 特别注意识别用户自述的身份，如"我是高管"、"我是普通员工"、"我在技术部"等
   - 职位示例：职位=高管、职位=经理、职位=普通员工
   - 部门示例：部门=技术部、部门=销售部
4. 上下文摘要：简洁概括对话内容（不超过100字）

输出格式（严格遵守，不要多余内容）：
核心主题：xxx
关键实体：实体1,实体2,实体3
用户身份信息：
职位=xxx
部门=xxx
上下文摘要：xxx"""

        resp = self.llm.chat(prompt, temperature=0.0).strip()

        # 解析LLM返回结果
        main_topic = ""
        entities = []
        context_summary = ""
        user_identity = {}

        in_identity_section = False
        for line in resp.split("\n"):
            line = line.strip()
            if line.startswith("核心主题："):
                main_topic = line.replace("核心主题：", "").strip()
                in_identity_section = False
            elif line.startswith("关键实体："):
                entities_str = line.replace("关键实体：", "").strip()
                entities = [e.strip() for e in entities_str.split(",") if e.strip()]
                in_identity_section = False
            elif line.startswith("用户身份信息："):
                in_identity_section = True
            elif line.startswith("上下文摘要："):
                context_summary = line.replace("上下文摘要：", "").strip()
                in_identity_section = False
            elif in_identity_section and "=" in line:
                # 解析用户身份信息
                key_value = line.replace("-", "").strip()
                if "=" in key_value:
                    key, value = key_value.split("=", 1)
                    user_identity[key.strip()] = value.strip()

        return main_topic, entities, context_summary, user_identity

    def _llm_rewrite_query(
        self,
        history: List[Message],
        query: str,
        main_topic: str = "",
        entities: List[str] = None,
        user_profile: Dict[str, str] = None,
    ) -> str:
        """
        LLM改写：将依赖上下文的查询转为独立可检索的查询

        Returns:
            改写后的查询字符串
        """
        history_str = self._format_history(history)
        entities_str = ", ".join(entities) if entities else ""

        # 构建用户身份信息字符串
        user_profile_str = ""
        if user_profile:
            user_profile_str = "\n".join([f"{k}={v}" for k, v in user_profile.items()])

        prompt = f"""你是RAG系统的查询改写专家，必须将依赖上下文的用户查询改写为独立、完整、无歧义的检索查询。

【对话历史】
{history_str}

【已识别的核心主题】
{main_topic if main_topic else "（待识别）"}

【已识别的关键实体】
{entities_str if entities_str else "（待识别）"}

【用户身份信息】（非常重要，改写时必须考虑）
{user_profile_str if user_profile_str else "（未识别）"}

【当前用户查询（依赖上下文）】
{query}

改写要求（必须严格遵守）：
1. 找出查询中的指代词（那、该、此、这个、那种、我、本人等），用【核心主题】或【关键实体】或【用户身份信息】替换
2. 如果用户用"我"、"本人"指代自己，必须用【用户身份信息】中的具体身份替换
3. 补全省略的主语、宾语、谓语，使查询完整可理解
4. 保持用户原始意图不变，不要添加额外信息
5. 改写后的查询必须是一个完整的问句或陈述句，可用于文档检索

示例：
历史："通讯费报销标准是什么？" → "主管150元/月"
查询："那部门总监呢？"
改写："部门总监的通讯费报销标准是多少？"

【改写后查询】（只输出改写后的查询，不要解释）："""

        try:
            rewritten = self.llm.chat(prompt, temperature=0.1).strip()
            # 清理可能的提示词残留
            rewritten = (
                rewritten.replace("【改写后查询】", "")
                .replace("改写后查询：", "")
                .strip()
            )

            if rewritten and len(rewritten) > 5:
                logger.info(f"查询改写成功: '{query}' → '{rewritten}'")
                return rewritten
            else:
                logger.warning(f"LLM改写返回空或太短，使用原始查询: {query}")
                return query
        except Exception as e:
            logger.error(f"查询改写失败：{e}")
            return query


# 全局上下文分析器实例
context_analyzer = ContextAnalyzer()
