from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import uuid
import traceback
import os
import torch
from models import Conversation, Message, RAGRequest, RAGResponse
from config import settings
from utils.logger import logger


# ====================== 1. LLM 客户端封装（使用本地模型） ======================
class LLMClient:
    """LLM 客户端封装，使用本地模型"""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path or settings.local_llm_model_path
        self.device = device or settings.local_llm_device
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
    def _initialize(self):
        """延迟初始化模型"""
        if self._initialized:
            return
            
        try:
            # 设置环境变量以抑制日志和进度条
            os.environ['TQDM_DISABLE'] = '1'
            os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
            os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
            os.environ['TRANSFORMERS_SILENCE_DEPRECATION_WARNINGS'] = '1'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            logger.info(f"初始化本地 LLM 模型: {self.model_path}")

            # 临时重定向stdout和stderr以抑制进度条
            import sys
            from io import StringIO
            
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            try:
                sys.stdout = StringIO()
                sys.stderr = StringIO()
                
                # 加载 tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )

                # 加载模型
                model_kwargs = {
                    "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                    "device_map": "auto" if self.device == "cuda" else None,
                }

                if settings.local_llm_load_in_8bit:
                    model_kwargs["load_in_8bit"] = True

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    **model_kwargs
                )
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            self._initialized = True
            logger.info(f"本地 LLM 模型初始化完成，设备: {self.device}")

        except Exception as e:
            logger.error(f"初始化本地 LLM 客户端失败: {str(e)}")
            raise
    
    def chat(self, prompt: str, temperature: float = 0.1) -> str:
        """
        核心方法：调用LLM获取回复
        
        Args:
            prompt: 输入提示
            temperature: 温度参数
            
        Returns:
            LLM 返回的文本
        """
        if not self._initialized:
            self._initialize()
            
        try:
            # 构建 messages 格式
            messages = [
                {"role": "user", "content": prompt}
            ]

            # 应用 chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)

            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # 解码
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            return response.strip()

        except Exception as e:
            logger.error(f"LLM 调用失败: {str(e)}")
            logger.error(traceback.format_exc())
            return ""
    
    def unload(self):
        """卸载模型，释放显存"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                if self.device == 'cuda' and hasattr(self.model, 'to'):
                    self.model = self.model.to('cpu')
                
                del self.model
                self.model = None
                
                if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                    del self.tokenizer
                    self.tokenizer = None
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                
                self._initialized = False
                logger.info("本地 LLM 模型已卸载，显存已释放")
        except Exception as e:
            logger.error(f"卸载本地 LLM 模型失败: {str(e)}")


# 全局 LLM 客户端实例
llm_client = LLMClient()


# ====================== 2. 纯LLM驱动的上下文分析器 ======================
class ContextAnalyzer:
    """
    纯LLM驱动的上下文分析器
    无关键词/无正则/无规则，完全靠大模型判断关联性 + 改写查询
    """
    def __init__(self):
        self.llm = llm_client
        # 最多保留最近10轮对话（避免上下文过长）
        self.max_history_rounds = 10

    def analyze_context(self, conversation_history: List[Message], current_query: str) -> Dict:
        """
        对外核心接口：分析上下文并生成历史感知的改写查询
        Args:
            conversation_history: 对话历史消息列表
            current_query: 当前用户查询
        Returns:
            分析结果字典
        """
        # 初始化返回结果
        analysis_result = {
            'is_contextual': False,
            'main_topic': '',
            'entities': [],
            'enhanced_query': current_query,
            'rewritten_query': current_query,
            'context_summary': ''
        }

        # 无历史对话，直接返回原始查询
        if not conversation_history:
            logger.info("无对话历史，返回原始查询")
            return analysis_result

        try:
            # 1. LLM判断：当前查询是否依赖上下文
            is_contextual = self._llm_is_contextual(conversation_history, current_query)
            analysis_result['is_contextual'] = is_contextual

            if not is_contextual:
                logger.info(f"LLM判断：当前查询「{current_query}」不依赖历史上下文")
                return analysis_result

            # 2. LLM抽取：上下文核心信息（主题/实体/摘要）
            main_topic, entities, context_summary = self._llm_extract_context_info(conversation_history)
            analysis_result['main_topic'] = main_topic
            analysis_result['entities'] = entities
            analysis_result['context_summary'] = context_summary

            # 3. LLM改写：生成独立可检索的查询
            rewritten_query = self._llm_rewrite_query(conversation_history, current_query)
            analysis_result['rewritten_query'] = rewritten_query
            analysis_result['enhanced_query'] = rewritten_query

            logger.info(f"上下文分析完成 | 核心主题：{main_topic} | 改写后查询：{rewritten_query}")

        except Exception as e:
            logger.error(f"上下文分析异常：{str(e)}\n{traceback.format_exc()}")
            # 异常时返回原始查询，保证程序不中断
            pass

        return analysis_result

    def _format_history(self, history: List[Message]) -> str:
        """格式化对话历史为LLM可读的字符串"""
        # 只保留最近N轮，避免上下文过长
        recent_history = history[-self.max_history_rounds:]
        history_lines = []
        for msg in recent_history:
            role_cn = "用户" if msg.role == "user" else "助手"
            history_lines.append(f"{role_cn}：{msg.content}")
        return "\n".join(history_lines)

    def _llm_is_contextual(self, history: List[Message], query: str) -> bool:
        """LLM判断：当前查询是否依赖历史对话"""
        history_str = self._format_history(history)
        
        prompt = f"""你是对话上下文关联性判断专家，请严格按规则判断。

【对话历史】
{history_str}

【当前用户查询】
{query}

判断规则：
1. 如果查询包含指代（它、这、那、该、此、其、对方、他们等）→ 依赖上下文
2. 如果查询不完整、省略核心主语/主题 → 依赖上下文
3. 如果查询本身完整独立、无需上下文即可理解 → 不依赖
4. 只回答"是"或"否"，不要任何其他内容、解释或标点

请回答："""

        resp = self.llm.chat(prompt, temperature=0.0).strip()
        return resp == "是"

    def _llm_extract_context_info(self, history: List[Message]) -> Tuple[str, List[str], str]:
        """LLM抽取：核心主题、关键实体、上下文摘要"""
        history_str = self._format_history(history)
        
        prompt = f"""你是专业的对话信息抽取专家，请严格按格式输出结果。

【对话历史】
{history_str}

抽取要求：
1. 核心主题：总结整个对话的核心议题（一句话，不超过20字）
2. 关键实体：提取对话中的核心实体（如职位、制度、产品、概念等，用中文逗号分隔）
3. 上下文摘要：简洁概括对话内容（不超过100字）

输出格式（严格遵守，不要多余内容）：
核心主题：xxx
关键实体：实体1,实体2,实体3
上下文摘要：xxx"""

        resp = self.llm.chat(prompt, temperature=0.0).strip()
        
        # 解析LLM返回结果
        main_topic = ""
        entities = []
        context_summary = ""
        
        for line in resp.split("\n"):
            line = line.strip()
            if line.startswith("核心主题："):
                main_topic = line.replace("核心主题：", "").strip()
            elif line.startswith("关键实体："):
                entities_str = line.replace("关键实体：", "").strip()
                entities = [e.strip() for e in entities_str.split(",") if e.strip()]
            elif line.startswith("上下文摘要："):
                context_summary = line.replace("上下文摘要：", "").strip()

        return main_topic, entities, context_summary

    def _llm_rewrite_query(self, history: List[Message], query: str) -> str:
        """LLM改写：将依赖上下文的查询转为独立可检索的查询"""
        history_str = self._format_history(history)
        
        prompt = f"""你是RAG系统的查询改写专家，目标是将依赖上下文的用户查询改写为独立、完整、无歧义的检索查询。

【对话历史】
{history_str}

【当前用户查询】
{query}

改写规则：
1. 补全所有指代（将"它/这/那/该/此"等替换为具体实体）
2. 补全省略的核心主题、条件、范围，保持用户原始意图不变
3. 改写后的查询必须完整、清晰，可直接用于文档检索
4. 只输出改写后的查询，不要任何解释、标点或多余内容

【改写后查询】："""

        rewritten = self.llm.chat(prompt, temperature=0.1).strip()
        # 兜底：如果LLM返回空，用原始查询
        return rewritten if rewritten else query


# 全局上下文分析器实例
context_analyzer = ContextAnalyzer()


# ====================== 3. 对话管理器 ======================
class ConversationManager:
    """对话管理器"""

    def __init__(self):
        self.conversations: Dict[str, Conversation] = {}
        self.max_history_length = 20  # 每会话最大历史消息数
        self.max_context_tokens = 4000  # 最大上下文token数

    def create_conversation(self, user_id: str, username: str) -> Conversation:
        """
        创建新对话

        Args:
            user_id: 用户ID
            username: 用户名

        Returns:
            对话对象
        """
        conversation_id = str(uuid.uuid4())
        conversation = Conversation(
            id=conversation_id,
            user_id=user_id,
            username=username,
            messages=[],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        self.conversations[conversation_id] = conversation
        return conversation

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        获取对话

        Args:
            conversation_id: 对话ID

        Returns:
            对话对象，如果不存在返回None
        """
        return self.conversations.get(conversation_id)

    def add_message(self, conversation_id: str, role: str, content: str) -> Optional[Message]:
        """
        添加消息到对话

        Args:
            conversation_id: 对话ID
            role: 角色 (user/assistant/system)
            content: 消息内容

        Returns:
            消息对象，如果对话不存在返回None
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return None

        message = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
            timestamp=datetime.now()
        )
        conversation.messages.append(message)
        conversation.last_updated = datetime.now()

        # 限制历史消息数量
        if len(conversation.messages) > self.max_history_length:
            conversation.messages = conversation.messages[-self.max_history_length:]

        return message

    def get_conversation_history(self, conversation_id: str, limit: int = 10) -> List[Message]:
        """
        获取对话历史

        Args:
            conversation_id: 对话ID
            limit: 限制数量

        Returns:
            消息列表
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        return conversation.messages[-limit:]

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        删除对话

        Args:
            conversation_id: 对话ID

        Returns:
            是否删除成功
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False

    def list_conversations(self, user_id: str) -> List[Conversation]:
        """
        列出用户的所有对话

        Args:
            user_id: 用户ID

        Returns:
            对话列表
        """
        return [conv for conv in self.conversations.values() if conv.user_id == user_id]

    def get_context_from_history(self, conversation_id: str, current_query: str) -> str:
        """
        从对话历史中获取上下文

        Args:
            conversation_id: 对话ID
            current_query: 当前查询

        Returns:
            上下文字符串
        """
        history = self.get_conversation_history(conversation_id, limit=5)  # 最近5轮对话
        context_parts = []

        for message in history:
            if message.role == "user":
                context_parts.append(f"用户: {message.content}")
            elif message.role == "assistant":
                context_parts.append(f"助手: {message.content}")

        # 添加当前查询
        context_parts.append(f"用户: {current_query}")

        return "\n".join(context_parts)

    def summarize_conversation(self, conversation_id: str) -> str:
        """
        总结对话

        Args:
            conversation_id: 对话ID

        Returns:
            对话摘要
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return ""

        # 这里可以集成LLM来生成摘要
        # 暂时返回简单的摘要
        messages = conversation.messages
        if not messages:
            return ""

        first_message = messages[0].content[:100]
        last_message = messages[-1].content[:100]
        return f"对话包含 {len(messages)} 条消息，主题涉及: {first_message}... {last_message}..."

    def compress_context(self, context: str, max_length: int = 2000) -> str:
        """
        压缩上下文

        Args:
            context: 原始上下文
            max_length: 最大长度

        Returns:
            压缩后的上下文
        """
        if len(context) <= max_length:
            return context

        # 简单的压缩策略：保留开头和结尾
        half = max_length // 2
        return context[:half] + "...（上下文已压缩）..." + context[-half:]

    def analyze_intent(self, conversation_id: str, query: str) -> Dict[str, Any]:
        """
        分析用户意图

        Args:
            conversation_id: 对话ID
            query: 用户查询

        Returns:
            意图分析结果
        """
        # 这里可以集成意图识别模型
        # 暂时返回简单的意图分析
        intent_keywords = {
            "question": ["什么", "如何", "为什么", "哪里", "什么时候", "谁", "多少"],
            "request": ["请", "帮我", "给我", "需要", "想要"],
            "feedback": ["好的", "不错", "很好", "谢谢", "感谢"],
            "clarification": ["什么意思", "请解释", "再说一遍", "不太明白"]
        }

        intent = "question"  # 默认意图
        confidence = 0.5

        for intent_type, keywords in intent_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    intent = intent_type
                    confidence = 0.8
                    break

        return {
            "intent": intent,
            "confidence": confidence,
            "query": query
        }


# 全局对话管理器实例
conversation_manager = ConversationManager()
