from typing import List, Optional, Dict, Any
import time
import asyncio

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

            # 使用配置中的模型路径
            model_path = settings.local_llm_model_path

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
                    "attn_implementation": "eager",  # 禁用 Flash Attention，解决版本兼容性问题
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

            # Tokenize - 使用模型配置的最大长度
            # Qwen2.5-7B 支持 32K 上下文，我们使用 8192 作为输入上限
            max_input_length = getattr(self.model.config, 'max_position_embeddings', 8192)
            max_input_length = min(max_input_length, 8192)  # 限制最大 8K 输入
            
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_input_length
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


class VLLMClient(LLMClient):
    """vLLM 客户端 - 通过 OpenAI 兼容 API 调用 vLLM 服务"""

    def __init__(self, config: GenerationConfig):
        super().__init__(config)
        try:
            import openai

            # vLLM 提供兼容 OpenAI 的 API
            self.client = openai.OpenAI(
                api_key="EMPTY",  # vLLM 不需要真实 API key
                base_url=f"http://{settings.vllm_host}:{settings.vllm_port}/v1",
            )
            # vLLM 使用完整模型路径作为模型名称
            self.model_name = settings.vllm_model_path
            logger.info(
                f"初始化 vLLM 客户端: {self.model_name}, "
                f"服务地址: http://{settings.vllm_host}:{settings.vllm_port}"
            )
        except Exception as e:
            logger.error(f"初始化 vLLM 客户端失败: {str(e)}")
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

            # 使用 OpenAI 兼容 API 调用 vLLM
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                stream=True,  # 使用流式输出以获取首token时间
            )
            
            # 收集流式响应并测量首token时间
            first_token_received = False
            first_token_time = None
            generation_start = time.time()
            generated_text = []
            
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        if not first_token_received:
                            first_token_time = time.time()
                            first_token_received = True
                        generated_text.append(delta.content)
            
            end_time = time.time()
            
            # 合并生成的文本
            answer = "".join(generated_text).strip()
            
            # 计算输出token数（vLLM 会返回 usage 信息）
            # 由于流式响应可能不包含 usage，我们估算
            output_token_count = len(answer) // 2  # 粗略估算中文 token 数
            input_token_count = len(prompt) // 2
            
            # 计算时间指标
            total_time_ms = (end_time - start_time) * 1000
            time_to_first_token_ms = (first_token_time - generation_start) * 1000 if first_token_time else 0
            generation_time_ms = (end_time - generation_start) * 1000
            
            # 计算生成速度
            tokens_per_second = (output_token_count / generation_time_ms * 1000) if generation_time_ms > 0 else 0
            
            return {
                "text": answer,
                "input_tokens": input_token_count,
                "output_tokens": output_token_count,
                "total_tokens": input_token_count + output_token_count,
                "time_to_first_token_ms": time_to_first_token_ms,
                "total_time_ms": total_time_ms,
                "generation_time_ms": generation_time_ms,
                "tokens_per_second": tokens_per_second,
            }

        except Exception as e:
            logger.error(f"vLLM 生成失败: {str(e)}")
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
        intent_type: Optional[str] = None,
    ) -> RAGResponse:
        """
        执行 RAG 生成

        Args:
            query: 查询问题
            retrieval_config: 检索配置
            generation_config: 生成配置
            conversation_id: 对话ID（可选）
            intent_type: 意图类型（hr, finance, admin, compliance, process, tech_report, casual_chat）

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
            max_context_length = 6000  # 默认6000字符（约4500 token）
            if generation_config.llm_provider == "local":
                # 本地模型可能支持更长的上下文
                max_context_length = 8000  # 约6000 token

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
            logger.info(f"[RAG生成] 意图类型: {intent_type}")
            prompt = self._build_prompt(
                rewritten_query, full_context, entities, user_profile, intent_type
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
            response = llm_client.generate(prompt)
            answer = response["text"]
            generation_time = response.get("generation_time_ms", (time.time() - generation_start) * 1000)

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

    def generate_without_retrieval(
        self,
        query: str,
        generation_config,
        conversation_id: Optional[str] = None,
        prompt_prefix: str = "",
    ) -> RAGResponse:
        """
        直接使用 LLM 生成回答，不检索知识库

        用于闲聊类意图，直接使用 LLM 进行对话

        Args:
            query: 查询问题
            generation_config: 生成配置
            conversation_id: 对话ID（可选）
            prompt_prefix: 提示词前缀

        Returns:
            RAG 响应
        """
        start_time = time.time()

        try:
            logger.info(f"[RAG生成-闲聊] 开始处理查询: '{query}', 对话ID: {conversation_id}")

            # 1. 获取对话历史
            conversation_history = []
            if conversation_id:
                conversation = conversation_manager.get_conversation(conversation_id)
                if conversation:
                    conversation_history = conversation.messages
                    logger.info(
                        f"[RAG生成-闲聊] 找到对话历史: {len(conversation_history)} 条消息"
                    )

            # 2. 构建对话提示词
            logger.info(f"[RAG生成-闲聊] 步骤1/2: 构建 Prompt...")

            # 构建系统提示词
            system_prompt = """你是一个友好、专业的企业内部助手。你可以与用户进行轻松的对话，回答问候、感谢等非业务相关问题。

请注意：
1. 回答要简洁友好
2. 如果用户的问题涉及公司业务（如薪酬、考勤、制度等），请引导用户使用具体的业务问题提问
3. 不要编造公司政策或制度信息"""

            # 构建用户消息
            if prompt_prefix:
                user_message = f"{prompt_prefix}\n\n用户问题：{query}"
            else:
                user_message = query

            # 3. 使用 LLM 生成回答
            logger.info(f"[RAG生成-闲聊] 步骤2/2: 使用 LLM 生成回答...")
            logger.info(f"[RAG生成-闲聊] LLM 配置: provider={generation_config.llm_provider}, "
                       f"model={generation_config.llm_model}")

            generation_start = time.time()
            llm_client = self._get_llm_client(generation_config)

            # 构建完整的 prompt
            full_prompt = f"{system_prompt}\n\n用户问题：{user_message}\n\n请回答："
            response_data = llm_client.generate(full_prompt)

            # 提取回答文本
            if isinstance(response_data, dict):
                answer = response_data.get("text", str(response_data))
            else:
                answer = str(response_data)

            generation_time = (time.time() - generation_start) * 1000
            total_time = (time.time() - start_time) * 1000

            logger.info(f"[RAG生成-闲聊] 生成完成: 回答长度={len(answer)}字符, 耗时{generation_time:.2f}ms")

            # 4. 保存对话消息
            if conversation_id:
                conversation_manager.add_message(conversation_id, "user", query)
                conversation_manager.add_message(conversation_id, "assistant", answer)

            return RAGResponse(
                query=query,
                answer=answer,
                context_chunks=[],  # 闲聊不需要上下文
                generation_time_ms=generation_time,
                retrieval_time_ms=0,  # 不检索
                total_time_ms=total_time,
            )

        except Exception as e:
            logger.error(f"[RAG生成-闲聊] 处理失败: {str(e)}")
            import traceback
            logger.error(f"[RAG生成-闲聊] 错误堆栈:\n{traceback.format_exc()}")
            return RAGResponse(
                query=query,
                answer=f"抱歉，我暂时无法回答这个问题。请稍后再试。",
                context_chunks=[],
                generation_time_ms=0,
                retrieval_time_ms=0,
                total_time_ms=0,
            )

    async def generate_stream(
        self,
        query: str,
        retrieval_config,
        generation_config,
        conversation_id: Optional[str] = None,
        context_chunks: List[RetrievalResult] = None,
        use_knowledge_base: bool = True,
        intent_type: str = None,
    ):
        """
        流式生成 RAG 回答

        Args:
            query: 查询问题
            retrieval_config: 检索配置
            generation_config: 生成配置
            conversation_id: 对话ID（可选）
            context_chunks: 已检索的上下文片段（可选，避免重复检索）
            use_knowledge_base: 是否使用知识库
            intent_type: 意图类型（用于构建针对性 prompt）

        Yields:
            str: 生成的文本片段
        """
        try:
            logger.info(f"[流式RAG] 开始处理查询: '{query}', 意图: {intent_type}")

            # 构建上下文
            if use_knowledge_base and context_chunks:
                context = self._build_context(context_chunks, max_context_length=6000)
                logger.info(f"[流式RAG] 构建上下文完成，长度: {len(context)} 字符")
            else:
                context = "没有相关参考文档。"

            # 构建 Prompt（传入意图类型）
            prompt = self._build_prompt(query, context, [], {}, intent_type=intent_type)

            # 获取 LLM 客户端
            llm_client = self._get_llm_client(generation_config)

            # 根据不同客户端类型进行流式生成
            provider = generation_config.llm_provider.lower()

            if provider == "vllm":
                # 使用 vLLM 流式生成
                async for chunk in self._stream_from_vllm(llm_client, prompt, generation_config):
                    yield chunk
            elif provider == "local":
                # 使用本地模型流式生成
                for chunk in self._stream_from_local(llm_client, prompt, generation_config):
                    yield chunk
                    await asyncio.sleep(0)  # 让出控制权
            else:
                # 其他客户端不支持流式，直接返回完整结果
                response = llm_client.generate(prompt)
                if isinstance(response, dict):
                    answer = response.get("text", "")
                else:
                    answer = str(response)
                yield answer

            logger.info(f"[流式RAG] 生成完成")

        except Exception as e:
            logger.error(f"[流式RAG] 生成失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            yield f"生成失败: {str(e)}"

    async def _stream_from_vllm(self, client, prompt: str, config):
        """从 vLLM 流式生成"""
        try:
            import openai

            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的助手，请根据提供的上下文信息回答问题。",
                },
                {"role": "user", "content": prompt},
            ]

            logger.info(f"[vLLM] 开始流式生成，模型: {client.model_name}, temperature: {config.temperature}, max_tokens: {config.max_tokens}, seed: 42")

            response = client.client.chat.completions.create(
                model=client.model_name,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                seed=42,  # 固定随机种子，确保相同输入产生相同输出
                stream=True,
            )

            chunk_count = 0
            total_content = []
            for chunk in response:
                chunk_count += 1
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        total_content.append(delta.content)
                        yield delta.content

            full_output = ''.join(total_content)
            logger.info(f"[vLLM] 流式生成完成，共 {chunk_count} 个 chunk，总长度: {len(full_output)}")
            logger.info(f"[vLLM] 输出内容: {full_output}")

        except Exception as e:
            logger.error(f"vLLM流式生成失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            yield f"生成失败: {str(e)}"

    def _stream_from_local(self, client, prompt: str, config):
        """从本地模型流式生成"""
        try:
            from transformers import TextIteratorStreamer
            from threading import Thread
            import torch

            # 构建 messages 格式
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的助手，请根据提供的上下文信息回答问题。",
                },
                {"role": "user", "content": prompt},
            ]

            # 应用 chat template
            text = client.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize
            inputs = client.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=8192
            ).to(client.model.device)

            # 创建流式输出器
            streamer = TextIteratorStreamer(
                client.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            # 生成参数
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True,
                pad_token_id=client.tokenizer.eos_token_id,
                streamer=streamer,
            )

            # 在单独线程中运行生成
            def generate_in_thread():
                with torch.no_grad():
                    client.model.generate(**generation_kwargs)

            thread = Thread(target=generate_in_thread)
            thread.start()

            # 收集生成的文本
            for new_text in streamer:
                yield new_text

            thread.join()

        except Exception as e:
            logger.error(f"本地模型流式生成失败: {str(e)}")
            yield f"生成失败: {str(e)}"

    def _build_context(
        self, results: List[RetrievalResult], max_context_length: int = 6000
    ) -> str:
        """
        构建上下文

        Args:
            results: 检索结果
            max_context_length: 最大上下文长度（字符数），默认6000字符（约4500 token）

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
        intent_type: str = None,
    ) -> str:
        """
        构建 Prompt，支持基于意图类型的针对性回答

        Args:
            query: 查询问题
            context: 上下文
            entities: 关键实体列表（如["高管", "报销标准"]）
            user_profile: 用户画像（累积的身份信息）
            intent_type: 意图类型（hr, finance, admin, compliance, process, tech_report, casual_chat）

        Returns:
            Prompt 字符串
        """
        # ==================== 意图专属提示配置 ====================
        INTENT_PROMPTS = {
            # ==================== 人力资源 ====================
            "hr": {
                "domain_hint": """【人力资源领域提示】
本领域涉及招聘、入职、薪酬、绩效、考勤、培训、晋升、福利、离职等员工全生命周期管理。

关键注意点：
1. 职级/岗位差异：薪酬标准、福利待遇、审批权限等通常按职级划分（如高管、经理、主管、普通员工）
2. 多文档关联：员工入职→劳动合同→薪酬定级→社保公积金，需串联多个制度
3. 时间节点：试用期、转正时间、年假计算等涉及具体时间规定
4. 边界界定：如涉及HR系统操作故障 → 不归此类，应转 tech_report""",
                "answer_style": """回答要求：
1. 如涉及职级/岗位差异的标准，请先明确用户身份，再提供对应标准
2. 如涉及多部门流程（如入职、离职），请按步骤说明各环节
3. 如涉及金额或天数，请明确具体数值
4. 引用制度名称，便于用户查阅原文""",
            },

            # ==================== 财务管理 ====================
            "finance": {
                "domain_hint": """【财务管理领域提示】
本领域涉及报销、预算、费用、成本、税务、资金、资产等财务相关事务。

关键注意点：
1. 城市/地区差异：差旅报销标准通常按城市类别划分
   - 一线城市：北京、上海、广州、深圳
   - 新一线/省会：成都、杭州、武汉、南京等
   - 其他城市
2. 职级差异：不同职级的报销限额、审批权限不同
3. 费用类型：交通费、住宿费、餐费、补贴等各有标准
4. 多文档关联：差旅标准→城市分类→职级标准，需综合查询
5. 边界界定：如涉及财务系统操作故障 → 不归此类，应转 tech_report""",
                "answer_style": """回答要求：
1. 如涉及报销标准，请提供完整信息：交通标准 + 住宿标准 + 补贴标准
2. 如涉及城市/职级差异，请明确说明不同情况的对应标准
3. 如涉及审批流程，请说明审批节点和责任人
4. **重要：用户未说明职级时，必须列出所有职级的对应标准，不要假设用户身份**
5. 列出标准后，可询问用户的具体职级以便提供更精准的信息
6. 金额请标注单位，如"住宿费500元/晚" """,
            },

            # ==================== 行政制度 ====================
            "admin": {
                "domain_hint": """【行政制度领域提示】
本领域涉及办公环境、会议室、办公用品、车辆、印章、档案、食堂、门禁等行政事务管理。

关键注意点：
1. 资源申请：会议室、车辆、办公用品等需关联审批流程
2. 使用规则：办公设备、车辆等有使用规范和注意事项
3. 跨部门协作：部分行政事务需财务、人力部门配合
4. 安全管理：印章、档案、门禁等涉及安全管控要求
5. 边界界定：如涉及OA/行政系统操作故障 → 不归此类，应转 tech_report""",
                "answer_style": """回答要求：
1. 如涉及资源申请，请说明申请方式、审批流程、使用规则
2. 如涉及费用（如车辆使用），请说明费用标准和承担方式
3. 如涉及安全管理，请强调注意事项和违规后果
4. 联系方式或办理地点如有提及，请一并提供""",
            },

            # ==================== 合规安全 ====================
            "compliance": {
                "domain_hint": """【合规安全领域提示】
本领域涉及信息安全、数据保护、商业秘密、合同管理、内部审计、合规培训等风险管理事务。

关键注意点：
1. 分级分类：数据、信息、保密等级有明确划分
2. 责任界定：违规行为有明确的处罚标准
3. 流程规范：数据处理、合同签署等有严格流程要求
4. 法律法规：需符合相关法律法规要求
5. 边界界定：如涉及安全系统/工具操作 → 不归此类，应转 tech_report""",
                "answer_style": """回答要求：
1. 用语严谨、准确，涉及法规条款请明确引用
2. 如涉及违规处理，请说明具体处罚标准和申诉途径
3. 如涉及保密等级，请明确说明对应的管控要求
4. 如涉及数据安全，请说明具体防护措施和操作规范""",
            },

            # ==================== 流程管理 ====================
            "process": {
                "domain_hint": """【流程管理领域提示】
本领域涉及采购、销售、项目、质量、变更、供应商等各类业务流程的规范化管理。

关键注意点：
1. 流程节点：各流程有明确的节点、责任人、时限要求
2. 审批权限：不同金额/事项对应不同审批层级
3. 跨部门协作：流程通常涉及多个部门的职责分工
4. 异常处理：流程中断、变更等异常情况有处理规范
5. 边界界定：工艺/工程类流程（生产线、制造流程）→ 转 tech_report；管理类流程（审批流、协作机制）→ 保留此类""",
                "answer_style": """回答要求：
1. 如涉及审批流程，请按步骤清晰说明：提交→审核→批准→执行
2. 请明确各节点的责任人和时限要求
3. 如涉及跨部门流程，请说明各部门的职责分工
4. 如涉及金额阈值，请明确不同层级的审批权限""",
            },

            # ==================== 技术报告 ====================
            "tech_report": {
                "domain_hint": """【技术领域提示】
本领域涵盖各行业的专业技术，包括：
- IT技术：软件开发、系统运维、AI/LLM、编程语言、框架工具等
- 工程技术：机械、电子、制造、建筑、自动化设备操作等
- 科研技术：实验方法、科研设备、数据分析、实验室安全等
- 医疗技术：医疗设备操作、临床技术、医学检验等

关键注意点：
1. 技术概念：涉及专业术语和技术原理，需准确解释
2. 系统操作：业务系统（财务/HR/OA/合规等）的使用故障、配置方法
3. 架构设计：系统架构需说明组件关系和数据流向
4. 实践应用：包含代码示例、配置方法、最佳实践
5. 故障排查：设备或系统故障的诊断步骤和解决方案""",
                "answer_style": """回答要求：
1. 技术概念请用清晰易懂的语言解释，必要时使用类比
2. 如涉及架构或流程，可使用结构化描述说明组件关系
3. 如涉及代码或配置，请使用代码块格式展示
4. 如涉及系统故障，请提供排查步骤和解决方案
5. 如文档信息不足，请诚实说明，不要编造
6. 引用文档中的关键信息，便于用户验证""",
            },

            # ==================== 闲聊 ====================
            "casual_chat": {
                "domain_hint": """【闲聊提示】
用户可能是日常问候或非业务相关的聊天。如用户实际有业务问题，可引导至专业咨询渠道。""",
                "answer_style": """回答要求：
1. 以友好、自然的方式回应
2. 如涉及公司业务问题，可引导至专业咨询渠道
3. 回答简洁、礼貌""",
            },
        }

        # ==================== 用户身份信息（简化版）====================
        # 各意图关注的关键身份字段（仅保留必要字段）
        IDENTITY_FIELDS = {
            "hr": ["职级", "部门"],
            "finance": ["职级", "部门"],
            "admin": ["部门"],
            "compliance": ["部门"],
            "process": ["职级", "部门"],
            "tech_report": [],  # 技术领域不关注身份
        }

        # 字段映射（原始字段 → 友好名称）
        FIELD_MAPPING = {
            "level": "职级",
            "grade": "职级",
            "department": "部门",
            "dept": "部门",
        }

        # 构建身份信息段落
        identity_section = ""
        if intent_type in IDENTITY_FIELDS and IDENTITY_FIELDS[intent_type]:
            relevant_fields = IDENTITY_FIELDS[intent_type]
            
            if user_profile:
                identity_parts = []
                for raw_key, value in user_profile.items():
                    friendly_key = FIELD_MAPPING.get(raw_key, raw_key)
                    if friendly_key in relevant_fields and value:
                        identity_parts.append(f"{friendly_key}: {value}")
                
                if identity_parts:
                    identity_section = f"\n【用户身份】{'；'.join(identity_parts)}\n"
                else:
                    # 仅日志记录，不放入 Prompt
                    logger.debug(f"[PromptBuilder] 缺失身份字段: {relevant_fields}, user_profile: {user_profile}")
            else:
                # 仅日志记录，不放入 Prompt
                logger.debug(f"[PromptBuilder] 无用户身份信息, 需要字段: {relevant_fields}")

        # ==================== 获取当前意图配置 ====================
        intent_config = INTENT_PROMPTS.get(intent_type)

        # 未知意图处理
        if not intent_config:
            logger.warning(f"[PromptBuilder] 未知意图类型: {intent_type}, 查询: {query[:50]}..., 使用默认配置")
            intent_config = {
                "domain_hint": """【通用领域提示】
未识别到特定领域配置，使用通用回答策略。""",
                "answer_style": """回答要求：
1. 请根据参考文档准确回答问题
2. 如文档信息不足，请诚实说明，不要编造
3. 回答简洁明了，重点突出""",
            }

        # ==================== 组装最终 Prompt ====================
        prompt = f"""请根据以下参考文档回答用户问题。

{intent_config['domain_hint']}{identity_section}
参考文档:
{context}

问题: {query}

{intent_config['answer_style']}"""

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
        elif provider == "vllm":
            return VLLMClient(config)
        else:
            raise ValueError(f"不支持的 LLM 提供商: {provider}")


# 全局 RAG 生成器实例
rag_generator = RAGGenerator()
