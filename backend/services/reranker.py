"""
重排序服务 - 使用重排序模型对检索结果进行精排
支持 BGE Reranker 等重排序模型
"""

from typing import List, Tuple, Optional
from utils.logger import logger
from config import settings


class Reranker:
    """重排序器基类"""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None

    def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回top_k结果

        Returns:
            [(原始索引, 重排序分数), ...] 按分数降序排列
        """
        raise NotImplementedError


class BGEReranker(Reranker):
    """BGE Reranker - FlagEmbedding重排序模型"""

    def __init__(
        self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"
    ):
        super().__init__(model_name, device)
        self._load_model()

    def _load_model(self):
        """加载BGE重排序模型"""
        try:
            from FlagEmbedding import FlagReranker

            # 确定模型路径
            model_path = self.model_name

            # 尝试从本地加载
            from pathlib import Path

            # 如果传入的是绝对路径且存在，直接使用
            if Path(model_path).is_absolute() and Path(model_path).exists():
                logger.info(f"从本地加载BGE重排序模型: {model_path}")
            else:
                # 尝试从缓存目录加载
                cache_dir = settings.upload_dir.replace("/data/docs", "/data/models")
                local_model_path = Path(cache_dir) / self.model_name.replace("/", "--")

                if local_model_path.exists():
                    model_path = str(local_model_path)
                    logger.info(f"从本地缓存加载BGE重排序模型: {model_path}")
                else:
                    logger.info(f"从HuggingFace下载BGE重排序模型: {self.model_name}")

            # 初始化模型
            self.model = FlagReranker(
                model_path,
                use_fp16=("cuda" in self.device),  # GPU上使用FP16减少显存占用
            )

            self.is_loaded = True
            logger.info(f"BGE重排序模型加载成功: {self.model_name}")

        except ImportError:
            logger.warning(
                "FlagEmbedding未安装，重排序功能不可用。请安装: pip install -U FlagEmbedding"
            )
            # 不抛出异常，而是设置未加载状态
            self.is_loaded = False
        except Exception as e:
            logger.error(f"加载BGE重排序模型失败: {str(e)}")
            self.is_loaded = False

    def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回top_k结果

        Returns:
            [(原始索引, 重排序分数), ...] 按分数降序排列
        """
        if not self.model:
            logger.warning("重排序模型未加载，返回原始顺序")
            return [(i, 0.0) for i in range(len(documents))]

        if not documents:
            return []

        try:
            # 准备输入对
            pairs = [[query, doc] for doc in documents]

            # 计算重排序分数
            scores = self.model.compute_score(pairs)

            # 组合索引和分数
            indexed_scores = list(enumerate(scores))

            # 按分数降序排序
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            # 应用top_k
            if top_k and top_k > 0:
                indexed_scores = indexed_scores[:top_k]

            return indexed_scores

        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
            # 失败时返回原始顺序
            return [(i, 0.0) for i in range(len(documents))]


class CrossEncoderReranker(Reranker):
    """CrossEncoder重排序器"""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ):
        super().__init__(model_name, device)
        self._load_model()

    def _load_model(self):
        """加载CrossEncoder模型"""
        try:
            from sentence_transformers import CrossEncoder

            # 确定模型路径
            model_path = self.model_name

            # 尝试从本地加载
            from pathlib import Path

            cache_dir = settings.upload_dir.replace("/data/docs", "/data/models")
            local_model_path = Path(cache_dir) / model_name.replace("/", "--")

            if local_model_path.exists():
                model_path = str(local_model_path)
                logger.info(f"从本地加载CrossEncoder模型: {model_path}")
            else:
                logger.info(f"从HuggingFace下载CrossEncoder模型: {model_name}")

            # 初始化模型
            self.model = CrossEncoder(model_path, device=self.device)

            logger.info(f"CrossEncoder模型加载成功: {self.model_name}")

        except ImportError:
            logger.warning("sentence-transformers未安装，重排序功能不可用")
            raise
        except Exception as e:
            logger.error(f"加载CrossEncoder模型失败: {str(e)}")
            raise

    def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回top_k结果

        Returns:
            [(原始索引, 重排序分数), ...] 按分数降序排列
        """
        if not self.model:
            logger.warning("重排序模型未加载，返回原始顺序")
            return [(i, 0.0) for i in range(len(documents))]

        if not documents:
            return []

        try:
            # 准备输入对
            pairs = [[query, doc] for doc in documents]

            # 计算重排序分数
            scores = self.model.predict(pairs)

            # 组合索引和分数
            indexed_scores = list(enumerate(scores))

            # 按分数降序排序
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            # 应用top_k
            if top_k and top_k > 0:
                indexed_scores = indexed_scores[:top_k]

            return indexed_scores

        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
            # 失败时返回原始顺序
            return [(i, 0.0) for i in range(len(documents))]


class Qwen3Reranker(Reranker):
    """Qwen3 Reranker - 基于 Qwen3 的重排序模型
    
    使用 LLM 的 yes/no 输出概率作为相关性分数
    """

    def __init__(
        self, 
        model_name: str = "Qwen/Qwen3-Reranker-4B", 
        device: str = "cuda",
        instruction: str = "Given a web search query, retrieve relevant passages that answer the query"
    ):
        super().__init__(model_name, device)
        self.instruction = instruction
        self.max_length = 8192
        self._load_model()

    def _load_model(self):
        """加载 Qwen3 Reranker 模型"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # 确定模型路径
            model_path = self.model_name
            from pathlib import Path
            
            # 如果是绝对路径且存在，直接使用
            if Path(model_path).is_absolute() and Path(model_path).exists():
                logger.info(f"从本地加载 Qwen3 Reranker 模型: {model_path}")
            else:
                logger.info(f"从 HuggingFace 加载 Qwen3 Reranker 模型: {self.model_name}")
            
            # 加载 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 检查 GPU 显存
            use_gpu = False
            if "cuda" in self.device and torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    allocated = torch.cuda.memory_allocated(0)
                    free_memory = gpu_memory - allocated
                    
                    # 需要约 10GB 显存
                    if free_memory > 10 * 1024 * 1024 * 1024:
                        use_gpu = True
                        logger.info(f"GPU 显存充足 ({free_memory / 1024**3:.1f}GB 空闲)，使用 GPU 加载")
                    else:
                        logger.warning(f"GPU 显存不足 ({free_memory / 1024**3:.1f}GB 空闲)，将使用 CPU 加载")
                except Exception as e:
                    logger.warning(f"检查 GPU 显存失败: {e}，将使用 CPU 加载")
            
            # 加载模型
            if use_gpu:
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    logger.info("Qwen3 Reranker 模型已加载到 GPU")
                except torch.cuda.OutOfMemoryError:
                    logger.warning("GPU 显存不足，回退到 CPU 加载")
                    torch.cuda.empty_cache()
                    use_gpu = False
            
            if not use_gpu:
                # CPU 模式 - 使用 float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                logger.info("Qwen3 Reranker 模型已加载到 CPU")
            
            self.model.eval()
            
            # 获取 yes/no token ID
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            
            # 构建后缀 tokens
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think/>\n\n</think/>\n\n"
            self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
            
            self.is_loaded = True
            logger.info(f"Qwen3 Reranker 模型加载成功: {self.model_name}")
            
        except ImportError as e:
            logger.warning(f"transformers 未安装或版本过低: {e}")
            self.is_loaded = False
        except Exception as e:
            logger.error(f"加载 Qwen3 Reranker 模型失败: {str(e)}")
            self.is_loaded = False

    def _format_instruction(self, query: str, doc: str) -> str:
        """格式化输入 prompt"""
        text = [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {self.instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
        ]
        return text

    def _process_inputs(self, pairs: List[Tuple[str, str]]):
        """处理输入对"""
        import torch
        
        messages_list = [self._format_instruction(query, doc) for query, doc in pairs]
        
        # 应用 chat template
        messages = self.tokenizer.apply_chat_template(
            messages_list, tokenize=True, add_generation_prompt=True
        )
        
        # 截断并添加后缀
        max_content_length = self.max_length - len(self.suffix_tokens)
        messages = [ele[:max_content_length] + self.suffix_tokens for ele in messages]
        
        # Padding
        inputs = self.tokenizer.pad(
            {"input_ids": messages}, 
            padding=True, 
            return_tensors="pt"
        )
        
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    @staticmethod
    def _compute_logits(model, inputs, token_true_id, token_false_id) -> List[float]:
        """计算相关性分数"""
        import torch
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_scores = outputs.logits[:, -1, :]
            
            true_vector = batch_scores[:, token_true_id]
            false_vector = batch_scores[:, token_false_id]
            
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            
            scores = batch_scores[:, 1].exp().tolist()
        
        return scores

    def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回 top_k 结果

        Returns:
            [(原始索引, 重排序分数), ...] 按分数降序排列
        """
        if not self.model:
            logger.warning("Qwen3 Reranker 模型未加载，返回原始顺序")
            return [(i, 0.0) for i in range(len(documents))]

        if not documents:
            return []

        try:
            import torch
            
            # 分批处理以避免内存问题
            batch_size = 8  # 减小批次大小
            all_scores = []
            
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                pairs = [(query, doc) for doc in batch_docs]
                
                # 处理输入
                inputs = self._process_inputs(pairs)
                
                # 计算分数
                scores = self._compute_logits(
                    self.model, inputs, self.token_true_id, self.token_false_id
                )
                all_scores.extend(scores)
                
                # 清理内存
                del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 组合索引和分数
            indexed_scores = list(enumerate(all_scores))
            
            # 按分数降序排序
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 应用 top_k
            if top_k and top_k > 0:
                indexed_scores = indexed_scores[:top_k]
            
            return indexed_scores

        except Exception as e:
            logger.error(f"Qwen3 Reranker 重排序失败: {str(e)}")
            # 失败时返回原始顺序
            return [(i, 0.0) for i in range(len(documents))]


class NoReranker(Reranker):
    """无重排序 - 用于禁用重排序功能的占位符"""

    def __init__(self, model_name: str = "", device: str = "cpu"):
        super().__init__(model_name, device)
        logger.info("使用无重排序模式")

    def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        返回原始顺序

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回top_k结果

        Returns:
            [(原始索引, 0.0), ...]
        """
        result = [(i, 0.0) for i in range(len(documents))]
        if top_k and top_k > 0:
            result = result[:top_k]
        return result


class RerankerManager:
    """重排序管理器"""

    def __init__(self):
        self.reranker: Optional[Reranker] = None
        self.reranker_type = "none"
        self.reranker_model = ""
        self.reranker_top_k = 10
        self.reranker_threshold = 0.0

    def initialize(
        self,
        reranker_type: str = "none",
        model_name: str = "",
        device: str = "cpu",
        top_k: int = 10,
        threshold: float = 0.0,
    ):
        """
        初始化重排序器

        Args:
            reranker_type: 重排序器类型 (none/bge/cross-encoder)
            model_name: 模型名称
            device: 设备 (cpu/cuda)
            top_k: 返回top_k结果
            threshold: 重排序分数阈值
        """
        self.reranker_type = reranker_type
        self.reranker_model = model_name
        self.reranker_top_k = top_k
        self.reranker_threshold = threshold

        try:
            if reranker_type == "none":
                self.reranker = NoReranker()
            elif reranker_type == "bge":
                # 提供多种BGE模型选项
                if not model_name:
                    # 根据设备选择合适的模型
                    if "cuda" in device:
                        model_name = "BAAI/bge-reranker-v2-m3"  # 更大的模型
                    else:
                        model_name = "BAAI/bge-reranker-v2-m3"  # 平衡性能和效果
                self.reranker = BGEReranker(model_name, device)
            elif reranker_type == "qwen3":
                # Qwen3 Reranker 模型
                if not model_name:
                    model_name = "Qwen/Qwen3-Reranker-4B"
                self.reranker = Qwen3Reranker(model_name, device)
            elif reranker_type == "cross-encoder":
                # 提供多种CrossEncoder模型选项
                if not model_name:
                    # 根据设备选择合适的模型
                    if "cuda" in device:
                        model_name = (
                            "cross-encoder/ms-marco-MiniLM-L-12-v2"  # 更大的模型
                        )
                    else:
                        model_name = (
                            "cross-encoder/ms-marco-MiniLM-L-6-v2"  # 更快的模型
                        )
                self.reranker = CrossEncoderReranker(model_name, device)
            else:
                logger.warning(f"未知重排序器类型: {reranker_type}，使用无重排序模式")
                self.reranker = NoReranker()

            logger.info(f"重排序器初始化成功: type={reranker_type}, model={model_name}")

        except Exception as e:
            logger.error(f"重排序器初始化失败: {str(e)}，使用无重排序模式")
            self.reranker = NoReranker()

    def rerank_results(
        self, query: str, results: List, apply_threshold: bool = True
    ) -> List:
        """
        重排序检索结果

        Args:
            query: 查询文本
            results: 检索结果列表
            apply_threshold: 是否应用阈值过滤

        Returns:
            重排序后的结果列表
        """
        if not self.reranker or self.reranker_type == "none":
            return results

        if not results:
            return results

        try:
            import time

            start_time = time.time()

            # 提取文档内容
            documents = [r.content for r in results]

            # 限制文档长度，避免处理过长的文档
            max_doc_length = 2000  # 最大文档长度（提高到 2000 以保留更多信息）
            documents = [doc[:max_doc_length] for doc in documents]

            # 执行重排序
            reranked_indices = self.reranker.rerank(
                query, documents, self.reranker_top_k
            )

            # 计算分数范围用于归一化
            all_scores = [score for _, score in reranked_indices]
            min_score = min(all_scores) if all_scores else 0
            max_score = max(all_scores) if all_scores else 1
            
            # 使用 sigmoid 函数将分数归一化到 [0, 1] 范围
            # 这样负数分数也会被转换为 (0, 0.5) 之间的值
            import math
            
            def normalize_score(score):
                """使用 sigmoid 函数归一化分数"""
                return 1 / (1 + math.exp(-score))

            # 根据重排序结果重新组织
            reranked_results = []
            for original_idx, score in reranked_indices:
                # 归一化分数到 [0, 1] 范围
                normalized_score = normalize_score(score)
                
                # 应用阈值过滤（使用归一化后的分数）
                if apply_threshold and normalized_score < self.reranker_threshold:
                    logger.debug(
                        f"重排序分数 {score:.4f} (归一化: {normalized_score:.4f}) 低于阈值 {self.reranker_threshold}，跳过"
                    )
                    continue

                # 复制结果并更新相似度分数
                result = results[original_idx]
                # 使用归一化后的分数作为相似度
                result.similarity = normalized_score
                reranked_results.append(result)
                logger.debug(f"重排序: 原始分数={score:.4f}, 归一化={normalized_score:.4f}")

            # 如果重排序后结果太少，补充原始结果
            if len(reranked_results) < min(self.reranker_top_k, 5):
                # 收集未被选中的原始结果
                selected_indices = {idx for idx, _ in reranked_indices}
                supplementary_results = [
                    r for i, r in enumerate(results) if i not in selected_indices
                ][: 5 - len(reranked_results)]
                reranked_results.extend(supplementary_results)

            rerank_time = time.time() - start_time
            logger.info(
                f"重排序完成: {len(results)} -> {len(reranked_results)} 个结果，耗时: {rerank_time:.4f}s"
            )

            return reranked_results

        except Exception as e:
            logger.error(f"重排序失败: {str(e)}，返回原始结果")
            return results

    def is_loaded(self) -> bool:
        """检查重排序器是否已加载"""
        if self.reranker is None:
            return False
        if self.reranker_type == "none":
            return False
        # 检查底层模型是否成功加载
        if hasattr(self.reranker, 'is_loaded'):
            return self.reranker.is_loaded
        return self.reranker_type != "none"

    def get_status(self) -> dict:
        """获取重排序器状态"""
        return {
            "enabled": self.reranker_type != "none",
            "type": self.reranker_type,
            "model": self.reranker_model,
            "top_k": self.reranker_top_k,
            "threshold": self.reranker_threshold,
        }


# 全局重排序管理器实例
reranker_manager = RerankerManager()
