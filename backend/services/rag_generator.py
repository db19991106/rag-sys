from typing import List
import time
import torch
from models import RAGRequest, RAGResponse, RetrievalResult, GenerationConfig
from services.retriever import retriever
from utils.logger import logger
from config import settings


class LLMClient:
    """LLM 客户端基类"""

    def __init__(self, config: GenerationConfig):
        self.config = config

    def generate(self, prompt: str) -> str:
        """生成回答"""
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """OpenAI 客户端"""

    def __init__(self, config: GenerationConfig):
        super().__init__(config)
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=config.llm_api_key or settings.llm_api_key,
                base_url=config.llm_base_url or settings.llm_base_url
            )
            logger.info(f"初始化 OpenAI 客户端: {config.llm_model}")
        except Exception as e:
            logger.error(f"初始化 OpenAI 客户端失败: {str(e)}")
            raise

    def generate(self, prompt: str) -> str:
        """生成回答"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的助手，请根据提供的上下文信息回答问题。如果上下文中没有相关信息，请明确说明。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI 生成失败: {str(e)}")
            return f"生成失败: {str(e)}"


class AnthropicClient(LLMClient):
    """Anthropic 客户端"""

    def __init__(self, config: GenerationConfig):
        super().__init__(config)
        try:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=config.llm_api_key or settings.llm_api_key
            )
            logger.info(f"初始化 Anthropic 客户端: {config.llm_model}")
        except Exception as e:
            logger.error(f"初始化 Anthropic 客户端失败: {str(e)}")
            raise

    def generate(self, prompt: str) -> str:
        """生成回答"""
        try:
            response = self.client.messages.create(
                model=self.config.llm_model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system="你是一个专业的助手，请根据提供的上下文信息回答问题。如果上下文中没有相关信息，请明确说明。",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic 生成失败: {str(e)}")
            return f"生成失败: {str(e)}"


class LocalLLMClient(LLMClient):
    """本地 LLM 客户端 (使用 transformers)"""

    def __init__(self, config: GenerationConfig):
        super().__init__(config)
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            model_path = getattr(settings, 'local_llm_model_path', None)
            device = getattr(settings, 'local_llm_device', 'cpu')

            if not model_path:
                raise ValueError("未配置本地模型路径，请在 config.py 中设置 local_llm_model_path")

            logger.info(f"加载本地 LLM 模型: {model_path}")

            # 加载 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )

            # 加载模型
            model_kwargs = {
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                "device_map": "auto" if device == "cuda" else None,
            }

            if getattr(settings, 'local_llm_load_in_8bit', False):
                model_kwargs["load_in_8bit"] = True

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                **model_kwargs
            )

            if device == "cpu":
                self.model = self.model.to(device)

            logger.info(f"本地 LLM 模型加载完成，设备: {device}")

        except Exception as e:
            logger.error(f"初始化本地 LLM 客户端失败: {str(e)}")
            logger.error("请确保已安装 transformers 库: pip install transformers torch")
            raise

    def generate(self, prompt: str) -> str:
        """生成回答"""
        try:
            # 构建 messages 格式
            messages = [
                {"role": "system", "content": "你是一个专业的助手，请根据提供的上下文信息回答问题。如果上下文中没有相关信息，请明确说明。"},
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
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
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
            logger.error(f"本地 LLM 生成失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return f"生成失败: {str(e)}"


class RAGGenerator:
    """RAG 生成器 - 结合检索和生成"""

    def __init__(self):
        pass

    def generate(self, query: str, retrieval_config, generation_config) -> RAGResponse:
        """
        执行 RAG 生成

        Args:
            query: 查询问题
            retrieval_config: 检索配置
            generation_config: 生成配置

        Returns:
            RAG 响应
        """
        start_time = time.time()

        try:
            logger.info(f"[RAG生成] 开始处理查询: '{query}'")

            # 1. 检索相关文档
            logger.info(f"[RAG生成] 步骤1/4: 开始检索相关文档...")
            retrieval_start = time.time()
            retrieval_response = retriever.retrieve(query, retrieval_config)
            retrieval_time = (time.time() - retrieval_start) * 1000

            logger.info(f"[RAG生成] 检索完成: 找到 {len(retrieval_response.results)} 个相关片段, "
                       f"耗时 {retrieval_time:.2f}ms")

            # 2. 构建上下文
            logger.info(f"[RAG生成] 步骤2/4: 构建上下文...")
            # 根据不同的LLM模型设置不同的上下文长度限制
            max_context_length = 4000  # 默认4000字符（约3000 token）
            if generation_config.llm_provider == "local":
                # 本地模型可能支持更长的上下文
                max_context_length = 6000  # 约4500 token
            
            context = self._build_context(retrieval_response.results, max_context_length)
            logger.info(f"[RAG生成] 上下文构建完成, 长度: {len(context)} 字符 (限制: {max_context_length})")

            # 3. 构建 Prompt
            logger.info(f"[RAG生成] 步骤3/4: 构建 Prompt...")
            prompt = self._build_prompt(query, context)
            logger.debug(f"[RAG生成] Prompt 内容: {prompt[:500]}...")

            # 4. 生成回答
            logger.info(f"[RAG生成] 步骤4/4: 使用 LLM 生成回答...")
            logger.info(f"[RAG生成] LLM 配置: provider={generation_config.llm_provider}, "
                       f"model={generation_config.llm_model}")
            generation_start = time.time()
            llm_client = self._get_llm_client(generation_config)
            answer = llm_client.generate(prompt)
            generation_time = (time.time() - generation_start) * 1000

            total_time = (time.time() - start_time) * 1000

            logger.info(f"[RAG生成] 生成完成: 回答长度={len(answer)}字符, "
                       f"耗时 {generation_time:.2f}ms")
            logger.info(f"[RAG生成] 总耗时: {total_time:.2f}ms "
                       f"(检索: {retrieval_time:.2f}ms, 生成: {generation_time:.2f}ms)")

            return RAGResponse(
                query=query,
                answer=answer,
                context_chunks=retrieval_response.results,
                generation_time_ms=generation_time,
                retrieval_time_ms=retrieval_time,
                total_time_ms=total_time
            )

        except Exception as e:
            logger.error(f"[RAG生成] 处理失败: {str(e)}")
            import traceback
            logger.error(f"[RAG生成] 错误堆栈:\n{traceback.format_exc()}")
            return RAGResponse(
                query=query,
                answer=f"生成失败: {str(e)}",
                context_chunks=[],
                generation_time_ms=0,
                retrieval_time_ms=0,
                total_time_ms=0
            )

    def _build_context(self, results: List[RetrievalResult], max_context_length: int = 4000) -> str:
        """
        构建上下文

        Args:
            results: 检索结果
            max_context_length: 最大上下文长度（字符数），默认4000字符（约3000 token）

        Returns:
            上下文字符串
        """
        if not results:
            return "没有找到相关文档。"

        context_parts = []
        total_length = 0

        for i, result in enumerate(results, 1):
            # 计算该片段的长度（包括标签）
            segment = f"【参考文档{i}】\n{result.content}\n"
            segment_length = len(segment)

            # 检查是否超过最大长度
            if total_length + segment_length > max_context_length:
                logger.warning(f"上下文长度超过限制 ({max_context_length} 字符)，截断到前 {i-1} 个文档")
                break

            context_parts.append(segment)
            total_length += segment_length

        context = "\n".join(context_parts)

        # 如果上下文为空（所有片段都太长），返回第一个片段的截断版本
        if not context_parts and results:
            logger.warning("所有片段都超过长度限制，使用第一个片段的截断版本")
            first_segment = f"【参考文档1】\n{results[0].content}\n"
            if len(first_segment) > max_context_length:
                # 截断到最大长度
                truncated = first_segment[:max_context_length - 50] + "...（已截断）\n"
                return truncated
            return first_segment

        return context

    def _build_prompt(self, query: str, context: str) -> str:
        """
        构建 Prompt

        Args:
            query: 查询问题
            context: 上下文

        Returns:
            Prompt 字符串
        """
        prompt = f"""请根据以下参考文档回答问题。如果文档中没有相关信息，请明确说明。

参考文档:
{context}

问题: {query}

回答:"""
        return prompt

    def _get_llm_client(self, config: GenerationConfig) -> LLMClient:
        """
        获取 LLM 客户端

        Args:
            config: 生成配置

        Returns:
            LLM 客户端
        """
        provider = config.llm_provider.lower()

        if provider == "openai":
            return OpenAIClient(config)
        elif provider == "anthropic":
            return AnthropicClient(config)
        elif provider == "local":
            return LocalLLMClient(config)
        else:
            raise ValueError(f"不支持的 LLM 提供商: {provider}")


# 全局 RAG 生成器实例
rag_generator = RAGGenerator()