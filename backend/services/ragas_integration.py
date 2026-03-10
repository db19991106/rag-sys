"""
RAGAS 集成模块 - 提供完整的 RAG 系统评估功能
支持 RAGAS 0.4.x 版本
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
import json
import re

# 设置环境变量避免 RAGAS 的某些依赖问题
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# RAGAS 导入 - 使用新的 collections API
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._context_entities_recall import context_entity_recall
from ragas.metrics._answer_similarity import answer_similarity
from ragas.metrics._answer_correctness import answer_correctness
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.evaluation import evaluate
from ragas.llms.base import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings
from langchain_core.outputs import LLMResult, Generation
from langchain_core.callbacks import Callbacks
from langchain_core.embeddings import Embeddings
from langchain_core.prompt_values import PromptValue

# 导入项目内部服务
from services.embedding import embedding_service
from utils.logger import logger


@dataclass
class RAGASEvaluationResult:
    """RAGAS 评估结果数据结构"""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    context_entity_recall: float
    answer_similarity: Optional[float] = None
    answer_correctness: Optional[float] = None
    overall_score: float = 0.0


class CustomRagasLLM(BaseRagasLLM):
    """
    自定义 RAGAS LLM 适配器
    使用 LangChain HuggingFacePipeline 包装本地模型
    兼容 RAGAS 0.4.3 API
    """

    def __init__(self, llm_client: Any = None, model_path: str = None, device: str = "cuda"):
        """
        初始化 RAGAS LLM 适配器
        
        Args:
            llm_client: 可选的现有 LLM 客户端（优先使用）
            model_path: 模型路径（如果 llm_client 为 None）
            device: 设备类型
        """
        self.llm_client = llm_client
        self._langchain_llm = None
        self.model_path = model_path
        self.device = device
        super().__init__()

    def _get_langchain_llm(self):
        """获取或创建 LangChain LLM 实例"""
        if self._langchain_llm is not None:
            return self._langchain_llm
        
        # 如果有现有的 LLM 客户端且有模型，尝试重用
        if self.llm_client is not None:
            try:
                # 检查 llm_client 是否有模型属性
                if hasattr(self.llm_client, 'model') and self.llm_client.model is not None:
                    from langchain_huggingface import HuggingFacePipeline
                    from transformers import pipeline
                    
                    logger.info("重用已加载的 LLM 模型创建 LangChain Pipeline")
                    
                    # 使用已加载的模型和 tokenizer
                    model = self.llm_client.model
                    tokenizer = self.llm_client.tokenizer
                    
                    # 创建 text-generation pipeline
                    pipe = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=512,
                        temperature=0.1,
                        do_sample=True,
                        top_p=0.9,
                        return_full_text=False,
                    )
                    
                    self._langchain_llm = HuggingFacePipeline(pipeline=pipe)
                    logger.info("LangChain HuggingFacePipeline 创建完成（重用模型）")
                    return self._langchain_llm
            except Exception as e:
                logger.warning(f"重用 LLM 客户端模型失败: {e}，将创建新模型")
        
        # 如果没有可重用的模型，创建新的
        if self.model_path:
            try:
                from langchain_huggingface import HuggingFacePipeline
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                import torch
                
                logger.info(f"为 RAGAS 加载新模型: {self.model_path}")
                
                # 加载 tokenizer 和模型
                tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                # 创建 text-generation pipeline
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                    return_full_text=False,
                )
                
                self._langchain_llm = HuggingFacePipeline(pipeline=pipe)
                logger.info("LangChain HuggingFacePipeline 加载完成")
                
            except Exception as e:
                logger.error(f"加载 LangChain HuggingFacePipeline 失败: {e}")
                raise
        
        return self._langchain_llm

    def _extract_text_from_prompt(self, prompt: Any) -> str:
        """从 PromptValue 或其他格式中提取文本"""
        if hasattr(prompt, 'to_string'):
            return prompt.to_string()
        elif isinstance(prompt, str):
            return prompt
        elif isinstance(prompt, tuple):
            return " ".join(str(p) for p in prompt if p)
        elif isinstance(prompt, list):
            return " ".join(str(p) for p in prompt if p)
        else:
            return str(prompt)

    def generate_text(
        self,
        prompt: Any,  # PromptValue
        n: int = 1,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """
        生成文本 - 同步版本
        使用 LangChain HuggingFacePipeline 进行生成
        """
        try:
            # 从 PromptValue 提取文本
            prompt_text = self._extract_text_from_prompt(prompt)
            
            # 使用 LangChain LLM
            lc_llm = self._get_langchain_llm()
            
            # 调用 LangChain LLM
            result = lc_llm.invoke(prompt_text)
            
            # 提取文本
            if hasattr(result, 'content'):
                text = result.content
            elif isinstance(result, str):
                text = result
            else:
                text = str(result)
            
            # 清理输出 - 尝试提取 JSON
            text = self._clean_json_output(text, prompt_text)
            
            # 处理 RAGAS 期望的输出格式
            # RAGAS faithfulness 指标期望 {"text": "..."} 格式
            # 但模型可能返回 {"statements": [...]} 格式
            text = self._ensure_ragas_output_format(text)
            
            # 生成 n 个结果（使用相同的结果，因为 pipeline 不支持 n > 1）
            generations = [Generation(text=text) for _ in range(n)]
            
            return LLMResult(generations=[generations])
            
        except Exception as e:
            logger.error(f"RAGAS LLM 生成失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return LLMResult(generations=[[Generation(text="") for _ in range(n)]])

    def _clean_json_output(self, text: str, prompt: str) -> str:
        """
        清理 LLM 输出，确保输出是有效的 JSON
        
        RAGAS 期望 JSON 格式的输出，但 LLM 可能会输出额外的文本
        """
        # 如果输出包含 JSON，提取它
        text = text.strip()
        
        # 检查是否已经是有效的 JSON
        try:
            json.loads(text)
            return text
        except:
            pass
        
        # 尝试提取 JSON 块
        # 模式 1: ```json ... ```
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            candidate = json_match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except:
                pass
        
        # 模式 2: { ... } 或 [ ... ]
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if json_match:
            candidate = json_match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except:
                pass
        
        # 模式 3: 尝试修复常见的 JSON 错误
        # 例如: 单引号变双引号
        try:
            fixed = text.replace("'", '"')
            json.loads(fixed)
            return fixed
        except:
            pass
        
        # 如果无法提取 JSON，返回原始文本
        return text

    def _ensure_ragas_output_format(self, text: str) -> str:
        """
        确保输出是 RAGAS 期望的格式
        
        RAGAS faithfulness 等指标期望 {"text": "..."} 格式
        但模型可能返回多种不同格式，需要统一转换
        """
        text = text.strip()
        
        # 首先尝试解析为 JSON
        try:
            data = json.loads(text)
            
            # 如果已经有 text 字段，直接返回
            if "text" in data:
                return text
            
            # 格式1: statements 格式（faithfulness 评估的常见输出）
            if "statements" in data and isinstance(data["statements"], list):
                statement_texts = []
                for stmt in data["statements"]:
                    if isinstance(stmt, dict):
                        statement = stmt.get("statement", "")
                        verdict = stmt.get("verdict", "")
                        reason = stmt.get("reason", "")
                        statement_texts.append(f"- {statement} (verdict: {verdict}, reason: {reason})")
                ragas_format = {"text": "\n".join(statement_texts)}
                return json.dumps(ragas_format, ensure_ascii=False)
            
            # 格式2: classifications 格式（context_recall 评估的输出）
            if "classifications" in data and isinstance(data["classifications"], list):
                classification_texts = []
                for cls in data["classifications"]:
                    if isinstance(cls, dict):
                        statement = cls.get("statement", "")
                        attributed = cls.get("attributed", "")
                        classification_texts.append(f"- {statement} (attributed: {attributed})")
                ragas_format = {"text": "\n".join(classification_texts)}
                return json.dumps(ragas_format, ensure_ascii=False)
            
            # 格式3: reason + verdict 格式
            if "reason" in data and ("verdict" in data or "attributed" in data):
                verdict = data.get("verdict", data.get("attributed", ""))
                reason = data.get("reason", "")
                ragas_format = {"text": f"verdict: {verdict}, reason: {reason}"}
                return json.dumps(ragas_format, ensure_ascii=False)
            
            # 格式4: question + noncommittal 格式
            if "question" in data and "noncommittal" in data:
                question = data.get("question", "")
                noncommittal = data.get("noncommittal", "")
                ragas_format = {"text": f"question: {question}, noncommittal: {noncommittal}"}
                return json.dumps(ragas_format, ensure_ascii=False)
            
            # 格式5: 直接是纯文本字符串
            if isinstance(data, str):
                return json.dumps({"text": data}, ensure_ascii=False)
            
            # 其他未知格式
            return json.dumps({"text": str(data)}, ensure_ascii=False)
            
        except json.JSONDecodeError:
            # 不是 JSON 格式，将纯文本包装成 text 字段
            # 这是最常见的情况：模型直接返回文本而不是 JSON
            if text:
                return json.dumps({"text": text}, ensure_ascii=False)
            return text

    async def agenerate_text(
        self,
        prompt: Any,  # PromptValue
        n: int = 1,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """
        异步生成文本
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_text(prompt, n, temperature, stop, callbacks)
        )

    def is_finished(self, response: Any) -> bool:
        """检查生成是否完成"""
        return True


class CustomRagasEmbeddings(BaseRagasEmbeddings):
    """
    自定义 RAGAS 嵌入模型适配器
    使用项目的 BGE 嵌入模型
    支持重用已加载的嵌入服务
    """

    def __init__(self, use_cpu: bool = False):
        super().__init__()
        self.use_cpu = use_cpu
        self._model_loaded = False
        self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        """确保嵌入模型已加载"""
        if self._model_loaded:
            return
            
        # 首先检查是否已有加载的模型
        if embedding_service.is_loaded():
            logger.info("重用已加载的嵌入模型")
            self._model_loaded = True
            return
        
        # 否则加载模型
        from models import EmbeddingConfig, EmbeddingModelType
        from config import settings
        
        logger.info("加载 BGE 嵌入模型用于 RAGAS...")
        
        # 确定 device
        device = "cpu" if self.use_cpu else settings.embedding_device
        
        try:
            embedding_service.load_model(
                EmbeddingConfig(
                    model_type=EmbeddingModelType.BGE,
                    model_name=settings.embedding_model_name,
                    device=device,
                )
            )
            self._model_loaded = True
        except Exception as e:
            logger.warning(f"在 {device} 上加载嵌入模型失败: {e}，尝试使用 CPU")
            if device != "cpu":
                try:
                    embedding_service.load_model(
                        EmbeddingConfig(
                            model_type=EmbeddingModelType.BGE,
                            model_name=settings.embedding_model_name,
                            device="cpu",
                        )
                    )
                    self._model_loaded = True
                except Exception as e2:
                    logger.error(f"在 CPU 上加载嵌入模型也失败: {e2}")

    def embed_text(self, text: str) -> List[float]:
        """嵌入单个文本"""
        self._ensure_model_loaded()
        try:
            embedding = embedding_service.encode([text])
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"文本嵌入失败: {e}")
            return [0.0] * 1024  # BGE-M3 维度

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本"""
        return self.embed_text(text)

    async def aembed_query(self, text: str) -> List[float]:
        """异步嵌入查询文本"""
        return self.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档"""
        self._ensure_model_loaded()
        try:
            embeddings = embedding_service.encode(texts)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"文档嵌入失败: {e}")
            return [[0.0] * 1024 for _ in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步嵌入多个文档"""
        return self.embed_documents(texts)


class RAGASEvaluator:
    """
    RAGAS 评估器
    提供完整的 RAG 系统评估功能
    """

    def __init__(self, llm_client: Optional[Any] = None, model_path: str = None, device: str = "cuda", use_cpu_embeddings: bool = True):
        """
        初始化 RAGAS 评估器

        Args:
            llm_client: LLM 客户端，用于 RAGAS 的 LLM-based 评估
            model_path: 模型路径（用于创建 LangChain HuggingFacePipeline）
            device: 设备类型
            use_cpu_embeddings: 是否使用 CPU 运行嵌入模型（避免 GPU 显存不足）
        """
        self.llm_client = llm_client
        self.model_path = model_path
        self.device = device
        
        # 创建 RAGAS LLM 适配器
        if llm_client or model_path:
            self.ragas_llm = CustomRagasLLM(
                llm_client=llm_client,
                model_path=model_path,
                device=device
            )
        else:
            self.ragas_llm = None
            
        # 使用 CPU 运行嵌入模型，避免 GPU 显存不足
        self.ragas_embeddings = CustomRagasEmbeddings(use_cpu=use_cpu_embeddings)

        # 可用的评估指标
        self.metrics = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "context_entity_recall": context_entity_recall,
            "answer_similarity": answer_similarity,
            "answer_correctness": answer_correctness,
        }

        logger.info("RAGAS 评估器初始化完成")

    def evaluate_single(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        评估单个 RAG 结果

        Args:
            query: 用户查询
            answer: 生成的答案
            contexts: 检索到的上下文列表
            ground_truth: 标准答案（可选，用于 answer_correctness）
            metrics: 要计算的指标列表，None 表示计算所有

        Returns:
            评估结果字典
        """
        try:
            # 构建样本
            sample = SingleTurnSample(
                user_input=query,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth if ground_truth else None,
            )

            # 选择要计算的指标
            if metrics is None:
                # 默认计算核心指标
                selected_metrics = [
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ]
                # 如果有 ground_truth，添加 correctness 和 similarity
                if ground_truth:
                    selected_metrics.extend([answer_correctness, answer_similarity])
            else:
                selected_metrics = [
                    self.metrics[m] for m in metrics if m in self.metrics
                ]

            # 运行评估
            result = evaluate(
                dataset=EvaluationDataset(samples=[sample]),
                metrics=selected_metrics,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings,
            )

            # 提取结果 - 使用 to_pandas() 方法
            scores = {}
            try:
                # ragas 0.4.3 返回 EvaluationResult 对象
                result_df = result.to_pandas()
                for metric_name in result_df.columns:
                    if metric_name not in ["user_input", "response", "retrieved_contexts", "reference"]:
                        score = result_df[metric_name].iloc[0]
                        # 处理 NaN
                        if hasattr(score, '__float__'):
                            scores[metric_name] = float(score) if not (isinstance(score, float) and np.isnan(score)) else 0.0
                        else:
                            scores[metric_name] = 0.0
            except Exception as extract_error:
                logger.warning(f"提取评估结果失败: {extract_error}")
                # 尝试其他方式提取
                if hasattr(result, '_scores'):
                    scores = result._scores

            # 计算综合得分
            valid_scores = [v for v in scores.values() if v > 0]
            overall_score = np.mean(valid_scores) if valid_scores else 0.0

            return {
                "scores": scores,
                "overall_score": round(overall_score, 3),
                "success": True,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"RAGAS 评估失败: {e}")
            return {
                "scores": {},
                "overall_score": 0.0,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        批量评估多个测试用例

        Args:
            test_cases: 测试用例列表，每个包含 query, answer, contexts, ground_truth
            metrics: 要计算的指标列表

        Returns:
            批量评估结果
        """
        results = []

        for i, case in enumerate(test_cases):
            logger.info(f"RAGAS 评估 [{i+1}/{len(test_cases)}]: {case['query'][:30]}...")

            result = self.evaluate_single(
                query=case["query"],
                answer=case["answer"],
                contexts=case["contexts"],
                ground_truth=case.get("ground_truth"),
                metrics=metrics,
            )

            results.append({
                "query": case["query"],
                "answer": case["answer"][:100] + "..." if len(case["answer"]) > 100 else case["answer"],
                "ground_truth": case.get("ground_truth", "")[:100] + "..." if case.get("ground_truth") and len(case["ground_truth"]) > 100 else case.get("ground_truth", ""),
                "evaluation": result,
            })

        # 汇总统计
        all_scores = {}
        for result in results:
            if result["evaluation"]["success"]:
                for metric, score in result["evaluation"]["scores"].items():
                    if metric not in all_scores:
                        all_scores[metric] = []
                    all_scores[metric].append(score)

        statistics = {}
        for metric, scores in all_scores.items():
            if scores:
                statistics[metric] = {
                    "mean": round(np.mean(scores), 3),
                    "std": round(np.std(scores), 3),
                    "min": round(np.min(scores), 3),
                    "max": round(np.max(scores), 3),
                    "median": round(np.median(scores), 3),
                }

        # 计算总体得分
        overall_scores = [r["evaluation"]["overall_score"] for r in results if r["evaluation"]["success"]]
        overall_mean = round(np.mean(overall_scores), 3) if overall_scores else 0.0

        return {
            "total_cases": len(test_cases),
            "successful_evaluations": sum(1 for r in results if r["evaluation"]["success"]),
            "failed_evaluations": sum(1 for r in results if not r["evaluation"]["success"]),
            "overall_score": overall_mean,
            "statistics": statistics,
            "detailed_results": results,
        }

    def evaluate_with_llm_judge(
        self,
        query: str,
        answer: str,
        ground_truth: str,
        criteria: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        使用 LLM 作为裁判进行评估

        Args:
            query: 用户查询
            answer: 生成的答案
            ground_truth: 标准答案
            criteria: 评估标准列表

        Returns:
            LLM 评估结果
        """
        if not self.llm_client:
            return {
                "error": "未提供 LLM 客户端",
                "success": False,
            }

        if criteria is None:
            criteria = [
                "准确性 (Accuracy): 答案是否包含正确信息",
                "完整性 (Completeness): 答案是否涵盖 ground_truth 的所有要点",
                "简洁性 (Conciseness): 答案是否简洁，无冗余信息",
                "相关性 (Relevance): 答案是否直接回答查询问题",
            ]

        criteria_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(criteria)])

        prompt = f"""你是一位严格的评估专家。请评估以下答案的质量。

【查询】
{query}

【标准答案】
{ground_truth}

【待评估答案】
{answer}

【评估标准】
{criteria_text}

请对每条标准给出 0-10 分的评分，并给出简要说明。最后计算平均分。

请以 JSON 格式输出：
{{
    "criteria_scores": [
        {{"criterion": "准确性", "score": 8, "reason": "..."}},
        ...
    ],
    "average_score": 7.5,
    "overall_feedback": "总体评价..."
}}
"""

        try:
            response = self.llm_client.generate(prompt)
            # 尝试解析 JSON
            import json
            import re

            # 提取 JSON 部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result["success"] = True
                return result
            else:
                return {
                    "raw_response": response,
                    "success": False,
                    "error": "无法解析 LLM 输出",
                }

        except Exception as e:
            logger.error(f"LLM 裁判评估失败: {e}")
            return {
                "error": str(e),
                "success": False,
            }


def create_ragas_evaluator(
    llm_client: Optional[Any] = None,
    model_path: Optional[str] = None,
    device: str = "cuda",
    use_cpu_embeddings: bool = True
) -> RAGASEvaluator:
    """
    工厂函数：创建 RAGAS 评估器

    Args:
        llm_client: LLM 客户端
        model_path: 模型路径（用于 LangChain HuggingFacePipeline）
        device: 设备类型
        use_cpu_embeddings: 是否使用 CPU 运行嵌入模型（默认 True，避免 GPU 显存不足）

    Returns:
        RAGASEvaluator 实例
    """
    return RAGASEvaluator(
        llm_client=llm_client,
        model_path=model_path,
        device=device,
        use_cpu_embeddings=use_cpu_embeddings
    )