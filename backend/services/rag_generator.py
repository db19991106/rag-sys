from typing import List, Optional, Dict, Any
import time

# 先导入 accelerate，确保 transformers 能检测到它
try:
    import accelerate
except ImportError:
    pass

import torch
from models import RAGRequest, RAGResponse, RetrievalResult, GenerationConfig
from services.retriever import retriever
from services.context_analyzer import context_analyzer
from services.conversation_manager import conversation_manager
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
                base_url=config.llm_base_url or settings.llm_base_url,
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
                    {
                        "role": "system",
                        "content": "你是一个专业的助手，请根据提供的上下文信息回答问题。如果上下文中没有相关信息，请明确说明。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI 生成失败: {str(e)}")
            return f"生成失败: {str(e)}"


def _check_model_exists(model_path: str) -> bool:
    """检查模型文件是否存在"""
    from pathlib import Path

    path = Path(model_path)
    if not path.exists():
        logger.error(f"模型目录不存在: {model_path}")
        return False

    # 检查必要的文件
    required_files = ["config.json"]
    model_files = ["pytorch_model.bin", "model.safetensors"]

    has_config = (path / "config.json").exists()
    has_model = any((path / f).exists() for f in model_files)

    if not has_config:
        logger.error(f"缺少配置文件: {path / 'config.json'}")

    if not has_model:
        logger.error(f"缺少模型文件，需要以下之一: {', '.join(model_files)}")

    return has_config and has_model


def _detect_device() -> str:
    """检测可用的计算设备"""
    import torch

    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (
            1024**3
        )  # GB

        logger.info(f"检测到 CUDA 设备: {device_name}, 显存: {total_memory:.2f}GB")

        # 7B模型需要约14GB显存（FP16）
        # 如果显存不足，建议使用8bit量化
        model_memory_need = 14  # GB for 7B model in FP16
        if total_memory < model_memory_need:
            logger.warning(
                f"显存不足: 需要 {model_memory_need}GB，当前只有 {total_memory:.2f}GB"
            )
            logger.warning("将尝试使用8bit量化或CPU")
            # 返回cuda，但后续会尝试8bit量化
            return "cuda"

        return "cuda"

    logger.info("未检测到CUDA设备，将使用CPU")
    return "cpu"


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
                messages=[{"role": "user", "content": prompt}],
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
            import os
            from pathlib import Path

            # 设置环境变量以抑制日志和进度条
            os.environ["TQDM_DISABLE"] = "1"
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            os.environ["TRANSFORMERS_SILENCE_DEPRECATION_WARNINGS"] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            from transformers import AutoTokenizer, AutoModelForCausalLM

            # 1. 确定模型路径
            model_path = getattr(settings, "local_llm_model_path", None)
            if not model_path:
                raise ValueError(
                    "未配置本地模型路径，请在 config.py 中设置 local_llm_model_path"
                )

            # 根据前端指定的模型名称动态调整模型路径
            # 使用相对于backend目录的路径
            backend_dir = Path(__file__).parent.parent
            if config.llm_model == "Qwen2.5-7B-Instruct":
                model_path = str(backend_dir / "data" / "models" / "Qwen2.5-7B-Instruct")
            elif config.llm_model == "Qwen2.5-0.5B-Instruct":
                model_path = str(backend_dir / "data" / "models" / "Qwen2.5-0.5B-Instruct")

            # 2. 检查模型文件是否存在
            if not self._check_model_exists(model_path):
                error_msg = f"模型文件不存在或损坏: {model_path}\n"
                error_msg += "请确保已下载模型文件到指定目录\n"
                error_msg += "下载方法:\n"
                error_msg += "  huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./data/models/Qwen2.5-7B-Instruct"
                raise FileNotFoundError(error_msg)

            # 3. 检测设备
            device_preference = getattr(settings, "local_llm_device", "auto")
            if device_preference == "auto":
                device = self._detect_device()
            else:
                device = device_preference
                logger.info(f"使用配置的设备: {device}")

            # 4. 检查内存需求
            memory_requirements = self._check_memory_requirement(model_path, device)
            logger.info(f"内存需求分析: {memory_requirements['reason']}")

            self.device = memory_requirements["device"]
            load_in_8bit = memory_requirements["load_in_8bit"]
            load_in_4bit = memory_requirements["load_in_4bit"]

            # 5. 加载模型
            logger.info(f"加载本地 LLM 模型: {model_path}")

            # 临时重定向stdout和stderr以抑制进度条
            import sys
            from io import StringIO

            # 保存原始的stdout和stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr

            try:
                # 重定向到空缓冲区
                sys.stdout = StringIO()
                sys.stderr = StringIO()

                # 加载 tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True
                )

                # 构建模型加载参数
                model_kwargs = {
                    "trust_remote_code": True,
                }

                if self.device == "cuda":
                    model_kwargs["dtype"] = torch.float16
                    if load_in_8bit:
                        model_kwargs["load_in_8bit"] = True
                        logger.info("启用8bit量化以节省显存")
                    elif load_in_4bit:
                        try:
                            from bitsandbytes import BitsAndBytesConfig

                            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                            )
                            logger.info("启用4bit量化以节省显存")
                        except ImportError:
                            logger.warning(
                                "bitsandbytes未安装，无法使用4bit量化，尝试8bit量化"
                            )
                            model_kwargs["load_in_8bit"] = True
                    else:
                        # 检查是否安装了 accelerate，如果安装了才使用 device_map
                        try:
                            import accelerate

                            # 如果显存足够，尝试使用device_map="auto"以自动分配
                            model_kwargs["device_map"] = "auto"
                            logger.info("使用 device_map='auto' 自动分配模型")
                        except ImportError:
                            logger.warning(
                                "accelerate 未安装，无法使用 device_map='auto'，模型将加载到默认GPU"
                            )
                            # 不显式指定 device，让模型自动加载到 GPU
                else:
                    # CPU设备使用float32
                    model_kwargs["dtype"] = torch.float32
                    logger.warning("使用CPU设备，生成速度较慢")

                # 加载模型
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, **model_kwargs
                )
            finally:
                # 恢复原始的stdout和stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr

            logger.info(f"本地 LLM 模型加载完成，设备: {self.device}")

        except Exception as e:
            logger.error(f"初始化本地 LLM 客户端失败: {str(e)}")
            logger.error("请确保已安装 transformers 库: pip install transformers torch")
            logger.error("如果使用CUDA设备，请确保已安装CUDA和cudatoolkit")
            raise

    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        生成回答并返回性能指标
        
        Returns:
            {
                "text": str,              # 生成的文本
                "input_tokens": int,      # 输入token数
                "output_tokens": int,     # 输出token数
                "total_tokens": int,      # 总token数
                "time_to_first_token_ms": float,  # 首token时延(ms)
                "total_time_ms": float,   # 总生成时间(ms)
                "tokens_per_second": float,  # 生成速度(token/s)
            }
        """
        import time
        
        try:
            start_time = time.time()
            
            # 构建 messages 格式
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的助手，请根据提供的上下文信息回答问题。如果上下文中没有相关信息，请明确说明。",
                },
                {"role": "user", "content": prompt},
            ]

            # 应用 chat template
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=2048
            ).to(self.model.device)
            
            input_token_count = inputs["input_ids"].shape[1]

            # 使用 streamer 来捕获首token时间
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=True
            )
            
            # 准备生成参数
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
            )
            
            first_token_received = False
            first_token_time = None
            generation_start = time.time()

            # 在单独线程中运行生成
            def generate_in_thread():
                with torch.no_grad():
                    self.model.generate(**generation_kwargs)
            
            thread = Thread(target=generate_in_thread)
            thread.start()
            
            # 收集生成的文本并测量首token时间
            generated_text = []
            for new_text in streamer:
                if not first_token_received:
                    first_token_time = time.time()
                    first_token_received = True
                generated_text.append(new_text)
            
            thread.join()
            
            end_time = time.time()
            
            # 合并生成的文本
            response = "".join(generated_text).strip()
            
            # 计算输出token数
            output_token_count = len(self.tokenizer.encode(response, add_special_tokens=False))
            
            # 计算时间指标
            total_time_ms = (end_time - start_time) * 1000
            time_to_first_token_ms = (first_token_time - generation_start) * 1000 if first_token_time else 0
            generation_time_ms = (end_time - generation_start) * 1000
            
            # 计算生成速度（基于实际生成token数的时间）
            tokens_per_second = (output_token_count / generation_time_ms * 1000) if generation_time_ms > 0 else 0
            
            return {
                "text": response,
                "input_tokens": input_token_count,
                "output_tokens": output_token_count,
                "total_tokens": input_token_count + output_token_count,
                "time_to_first_token_ms": time_to_first_token_ms,
                "total_time_ms": total_time_ms,
                "generation_time_ms": generation_time_ms,
                "tokens_per_second": tokens_per_second,
            }

        except Exception as e:
            logger.error(f"本地 LLM 生成失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "text": f"生成失败: {str(e)}",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "time_to_first_token_ms": 0,
                "total_time_ms": 0,
                "generation_time_ms": 0,
                "tokens_per_second": 0,
            }

    def _check_model_exists(self, model_path: str) -> bool:
        """检查模型文件是否存在"""
        from pathlib import Path

        path = Path(model_path)
        if not path.exists():
            logger.error(f"模型目录不存在: {model_path}")
            return False

        # 检查必要的文件（支持单文件和分片模型）
        single_model_files = ["pytorch_model.bin", "model.safetensors"]
        # 分片模型文件模式：model-00001-of-00004.safetensors
        import glob

        sharded_models = glob.glob(str(path / "model-*.safetensors"))

        has_config = (path / "config.json").exists()
        has_single_model = any((path / f).exists() for f in single_model_files)
        has_sharded_model = len(sharded_models) > 0

        if not has_config:
            logger.error(f"缺少配置文件: {path / 'config.json'}")

        if not has_single_model and not has_sharded_model:
            logger.error(
                f"缺少模型文件，需要以下之一: {', '.join(single_model_files)} 或分片模型文件"
            )

        return has_config and (has_single_model or has_sharded_model)

    def unload(self):
        """
        卸载模型，释放显存
        """
        try:
            if hasattr(self, "model") and self.model is not None:
                # 移动模型到 CPU 以释放 GPU 内存
                if self.device == "cuda" and hasattr(self.model, "to"):
                    self.model = self.model.to("cpu")

                # 删除模型和 tokenizer
                del self.model
                self.model = None

                if hasattr(self, "tokenizer") and self.tokenizer is not None:
                    del self.tokenizer
                    self.tokenizer = None

                # 清理 PyTorch 缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                logger.info("本地 LLM 模型已卸载，显存已释放")
        except Exception as e:
            logger.error(f"卸载本地 LLM 模型失败: {str(e)}")

    def _check_memory_requirement(self, model_path, device):
        """
        检查内存需求

        Args:
            model_path: 模型路径
            device: 设备

        Returns:
            内存需求信息
        """
        try:
            import torch

            # 检查设备内存
            if device == "cuda" and torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )  # GB

                # 根据模型类型估计内存需求
                model_name = self.config.llm_model
                if "7B" in model_name:
                    # 7B模型需求
                    required_memory = 14  # GB for FP16
                    if total_memory >= required_memory:
                        return {
                            "device": "cuda",
                            "load_in_8bit": False,
                            "load_in_4bit": False,
                            "reason": f"显存充足 ({total_memory:.2f}GB >= {required_memory}GB)，使用FP16精度",
                        }
                    elif total_memory >= 8:
                        return {
                            "device": "cuda",
                            "load_in_8bit": True,
                            "load_in_4bit": False,
                            "reason": f"显存不足 ({total_memory:.2f}GB < {required_memory}GB)，使用8bit量化",
                        }
                    elif total_memory >= 4:
                        return {
                            "device": "cuda",
                            "load_in_8bit": False,
                            "load_in_4bit": True,
                            "reason": f"显存严重不足 ({total_memory:.2f}GB)，使用4bit量化",
                        }
                    else:
                        return {
                            "device": "cpu",
                            "load_in_8bit": False,
                            "load_in_4bit": False,
                            "reason": f"显存严重不足 ({total_memory:.2f}GB)，切换到CPU",
                        }
                elif "0.5B" in model_name:
                    # 0.5B模型需求
                    required_memory = 2  # GB
                    return {
                        "device": "cuda",
                        "load_in_8bit": False,
                        "load_in_4bit": False,
                        "reason": f"小模型，显存充足，使用FP16精度",
                    }

            # CPU情况
            return {
                "device": "cpu",
                "load_in_8bit": False,
                "load_in_4bit": False,
                "reason": "使用CPU设备",
            }
        except Exception as e:
            logger.warning(f"内存检查失败: {str(e)}")
            # 默认为CPU
            return {
                "device": "cpu",
                "load_in_8bit": False,
                "load_in_4bit": False,
                "reason": f"内存检查失败，默认使用CPU: {str(e)}",
            }

    def _detect_device(self):
        """
        检测可用的计算设备
        """
        import torch

        # 检查CUDA是否可用
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (
                1024**3
            )  # GB

            logger.info(f"检测到 CUDA 设备: {device_name}, 显存: {total_memory:.2f}GB")

            # 7B模型需要约14GB显存（FP16）
            # 如果显存不足，建议使用8bit量化
            model_memory_need = 14  # GB for 7B model in FP16
            if total_memory < model_memory_need:
                logger.warning(
                    f"显存不足: 需要 {model_memory_need}GB，当前只有 {total_memory:.2f}GB"
                )
                logger.warning("将尝试使用8bit量化或CPU")
                # 返回cuda，但后续会尝试8bit量化
                return "cuda"

            return "cuda"

        logger.info("未检测到CUDA设备，将使用CPU")
        return "cpu"

    def _check_model_exists(self, model_path: str) -> bool:
        """
        检查模型文件是否存在
        """
        from pathlib import Path

        path = Path(model_path)
        if not path.exists():
            logger.error(f"模型目录不存在: {model_path}")
            return False

        # 检查必要的文件（支持单文件和分片模型）
        single_model_files = ["pytorch_model.bin", "model.safetensors"]
        # 分片模型文件模式：model-00001-of-00004.safetensors
        import glob

        sharded_models = glob.glob(str(path / "model-*.safetensors"))

        has_config = (path / "config.json").exists()
        has_single_model = any((path / f).exists() for f in single_model_files)
        has_sharded_model = len(sharded_models) > 0

        if not has_config:
            logger.error(f"缺少配置文件: {path / 'config.json'}")

        if not has_single_model and not has_sharded_model:
            logger.error(
                f"缺少模型文件，需要以下之一: {', '.join(single_model_files)} 或分片模型文件"
            )

        return has_config and (has_single_model or has_sharded_model)


class RAGGenerator:
    """RAG 生成器 - 结合检索和生成"""

    def __init__(self):
        pass

    def generate(
        self,
        query: str,
        retrieval_config,
        generation_config,
        conversation_id: Optional[str] = None,
    ) -> RAGResponse:
        """
        执行 RAG 生成

        Args:
            query: 查询问题
            retrieval_config: 检索配置
            generation_config: 生成配置
            conversation_id: 对话ID（可选）

        Returns:
            RAG 响应
        """
        start_time = time.time()

        try:
            logger.info(f"[RAG生成] 开始处理查询: '{query}', 对话ID: {conversation_id}")

            # 1. 获取对话历史并分析上下文
            conversation_history = []
            rewritten_query = query
            context_summary = ""
            entities = []  # 关键实体列表
            user_profile = {}  # 用户画像（累积的身份信息）

            if conversation_id:
                conversation = conversation_manager.get_conversation(conversation_id)
                if conversation:
                    conversation_history = conversation.messages
                    # 获取已有的用户画像（累积的身份信息）
                    user_profile = getattr(conversation, "user_profile", {}) or {}
                    logger.info(
                        f"[RAG生成] 找到对话历史: {len(conversation_history)} 条消息, "
                        f"已有用户画像={user_profile}"
                    )

                    # 分析上下文并重写查询
                    context_analysis = context_analyzer.analyze_context(
                        conversation_history,
                        query,
                        user_profile,  # 传入已有画像
                    )
                    rewritten_query = context_analysis["rewritten_query"]
                    context_summary = context_analysis["context_summary"]
                    entities = context_analysis.get("entities", [])  # 当前轮次的实体

                    # 更新用户画像（合并新识别的身份信息）
                    new_profile = context_analysis.get("user_profile", {})
                    if new_profile:
                        user_profile.update(new_profile)
                        conversation.user_profile = user_profile  # 保存回对话对象
                        logger.info(f"[RAG生成] 更新用户画像: {user_profile}")

                    # 合并用户画像到entities（确保身份信息被使用）
                    if user_profile:
                        identity_values = list(user_profile.values())
                        entities = list(set(entities + identity_values))  # 合并并去重

                    logger.info(
                        f"[RAG生成] 上下文分析完成: 重写查询='{rewritten_query}', "
                        f"关键实体={entities}, 用户画像={user_profile}"
                    )

            # 2. 检索相关文档（使用改写后的查询）
            logger.info(f"[RAG生成] 步骤1/4: 开始检索相关文档...")
            retrieval_start = time.time()
            retrieval_response = retriever.retrieve(
                query, retrieval_config, context_summary, rewritten_query
            )
            retrieval_time = (time.time() - retrieval_start) * 1000

            logger.info(
                f"[RAG生成] 检索完成: 找到 {len(retrieval_response.results)} 个相关片段, "
                f"耗时 {retrieval_time:.2f}ms"
            )

            # 3. 构建上下文
            logger.info(f"[RAG生成] 步骤2/4: 构建上下文...")
            # 根据不同的LLM模型设置不同的上下文长度限制
            max_context_length = 4000  # 默认4000字符（约3000 token）
            if generation_config.llm_provider == "local":
                # 本地模型可能支持更长的上下文
                max_context_length = 6000  # 约4500 token

            # 构建文档上下文
            document_context = self._build_context(
                retrieval_response.results, max_context_length
            )

            # 构建完整上下文（包含对话历史和文档上下文）
            full_context = document_context
            if context_summary:
                # 在文档上下文前添加对话历史摘要
                full_context = f"对话历史摘要: {context_summary}\n\n" + document_context

            logger.info(
                f"[RAG生成] 上下文构建完成, 长度: {len(full_context)} 字符 (限制: {max_context_length})"
            )

            # 4. 构建 Prompt（传入关键实体以生成个性化回答）
            logger.info(f"[RAG生成] 步骤3/4: 构建 Prompt...")
            prompt = self._build_prompt(
                rewritten_query, full_context, entities, user_profile
            )
            logger.debug(f"[RAG生成] Prompt 内容: {prompt[:500]}...")

            # 5. 生成回答
            logger.info(f"[RAG生成] 步骤4/4: 使用 LLM 生成回答...")
            logger.info(
                f"[RAG生成] LLM 配置: provider={generation_config.llm_provider}, "
                f"model={generation_config.llm_model}"
            )
            generation_start = time.time()
            llm_client = self._get_llm_client(generation_config)
            answer = llm_client.generate(prompt)
            generation_time = (time.time() - generation_start) * 1000

            # 生成完成后卸载模型，释放显存
            if hasattr(llm_client, "unload"):
                llm_client.unload()

            total_time = (time.time() - start_time) * 1000

            logger.info(
                f"[RAG生成] 生成完成: 回答长度={len(answer)}字符, "
                f"耗时 {generation_time:.2f}ms"
            )
            logger.info(
                f"[RAG生成] 总耗时: {total_time:.2f}ms "
                f"(检索: {retrieval_time:.2f}ms, 生成: {generation_time:.2f}ms)"
            )

            # 如果检测到本地存在对应知识库，在回答的下方加入引用
            if retrieval_response.results:
                references = "\n\n引用来源："
                for i, result in enumerate(retrieval_response.results, 1):
                    references += f"\n[{i}] 文档: {result.document_name}, 相似度: {result.similarity:.4f}"
                answer += references

            return RAGResponse(
                query=query,
                answer=answer,
                context_chunks=retrieval_response.results,
                generation_time_ms=generation_time,
                retrieval_time_ms=retrieval_time,
                total_time_ms=total_time,
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
                total_time_ms=0,
            )

    def _build_context(
        self, results: List[RetrievalResult], max_context_length: int = 4000
    ) -> str:
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
                logger.warning(
                    f"上下文长度超过限制 ({max_context_length} 字符)，截断到前 {i - 1} 个文档"
                )
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
                truncated = first_segment[: max_context_length - 50] + "...（已截断）\n"
                return truncated
            return first_segment

        return context

    def _build_prompt(
        self,
        query: str,
        context: str,
        entities: List[str] = None,
        user_profile: Dict[str, Any] = None,
    ) -> str:
        """
        构建 Prompt，支持基于关键实体的个性化回答

        Args:
            query: 查询问题
            context: 上下文
            entities: 关键实体列表（如["高管", "报销标准"]）
            user_profile: 用户画像（累积的身份信息）

        Returns:
            Prompt 字符串
        """
        # 构建用户身份信息提示（优先使用user_profile）
        identity_section = ""

        if user_profile:
            # 优先使用累积的用户画像
            identity_parts = []
            for key, value in user_profile.items():
                identity_parts.append(f"{key}: {value}")
            if identity_parts:
                identity_str = "; ".join(identity_parts)
                identity_section = f"\n【重要】用户身份信息: {identity_str}\n"
                identity_section += "注意：以下回答必须针对该用户的身份信息，只提供与其身份相关的具体内容，不要提及其他无关类别。\n"
        elif entities:
            # 备用：从entities中提取身份信息
            identity_keywords = [
                "高管",
                "经理",
                "主管",
                "员工",
                "普通员工",
                "总监",
                "CEO",
                "CTO",
                "技术部",
                "财务部",
                "销售部",
            ]
            user_identity = []
            for entity in entities:
                for keyword in identity_keywords:
                    if keyword in entity:
                        user_identity.append(entity)
                        break

            if user_identity:
                identity_str = "、".join(user_identity)
                identity_section = f"\n【重要】用户身份信息: {identity_str}\n"
                identity_section += "注意：以下回答必须针对该用户的身份信息，只提供与其身份相关的具体内容。\n"

        prompt = f"""请根据以下参考文档回答问题。{identity_section}

参考文档:
{context}

问题: {query}

回答要求: 
1. 请根据上述【用户身份信息】提供精准、个性化的回答
2. 如果文档包含多个类别/级别的信息，只回答与用户身份相关的内容
3. 不要提供与用户身份无关的其他类别信息
4. 回答开头应明确提及用户身份，如：作为普通员工、作为高管"""
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
