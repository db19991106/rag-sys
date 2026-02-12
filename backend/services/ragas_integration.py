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
from ragas.dataset_schema import SingleTurnSample
from ragas.evaluation import evaluate
from ragas.llms.base import BaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings
from langchain_core.outputs import LLMResult, Generation
from langchain_core.callbacks import Callbacks
from langchain_core.embeddings import Embeddings

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
    用于将本地 LLM 集成到 RAGAS 评估中
    """

    def __init__(self, llm_client: Any):
        self.llm_client = llm_client
        super().__init__()

    def generate(
        self,
        prompts: List[str],
        n: int = 1,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        callbacks: Optional[Callbacks] = None,
    ) -> LLMResult:
        """生成文本 - 同步版本"""
        generations = []
        for prompt in prompts:
            try:
                # 使用本地 LLM 客户端生成
                text = self.llm_client.generate(prompt)
                generations.append([Generation(text=text)])
            except Exception as e:
                logger.error(f"RAGAS LLM 生成失败: {e}")
                generations.append([Generation(text="")])

        return LLMResult(generations=generations)

    async def agenerate(
        self,
        prompts: List[str],
        n: int = 1,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        callbacks: Optional[Callbacks] = None,
    ) -> LLMResult:
        """异步生成文本"""
        # 使用同步版本
        return self.generate(prompts, n, temperature, stop, callbacks)


class CustomRagasEmbeddings(BaseRagasEmbeddings):
    """
    自定义 RAGAS 嵌入模型适配器
    使用项目的 BGE 嵌入模型
    """

    def __init__(self):
        super().__init__()
        self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        """确保嵌入模型已加载"""
        if not embedding_service.is_loaded():
            from models import EmbeddingConfig, EmbeddingModelType
            logger.info("加载 BGE 嵌入模型用于 RAGAS...")
            embedding_service.load_model(
                EmbeddingConfig(
                    model_type=EmbeddingModelType.BGE,
                    model_name="BAAI/bge-base-zh-v1.5",
                    device="cpu",
                )
            )

    def embed_text(self, text: str) -> List[float]:
        """嵌入单个文本"""
        self._ensure_model_loaded()
        try:
            embedding = embedding_service.encode([text])
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"文本嵌入失败: {e}")
            return [0.0] * 768  # BGE base 维度

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本"""
        return self.embed_text(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档"""
        self._ensure_model_loaded()
        try:
            embeddings = embedding_service.encode(texts)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"文档嵌入失败: {e}")
            return [[0.0] * 768 for _ in texts]


class RAGASEvaluator:
    """
    RAGAS 评估器
    提供完整的 RAG 系统评估功能
    """

    def __init__(self, llm_client: Optional[Any] = None):
        """
        初始化 RAGAS 评估器

        Args:
            llm_client: LLM 客户端，用于 RAGAS 的 LLM-based 评估
        """
        self.llm_client = llm_client
        self.ragas_llm = CustomRagasLLM(llm_client) if llm_client else None
        self.ragas_embeddings = CustomRagasEmbeddings()

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
                dataset=[sample],
                metrics=selected_metrics,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings,
            )

            # 提取结果
            scores = {}
            for metric_name in result.columns:
                if metric_name not in ["user_input", "response", "retrieved_contexts", "reference"]:
                    score = result[metric_name].iloc[0]
                    # 处理 NaN
                    scores[metric_name] = float(score) if not np.isnan(score) else 0.0

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


def create_ragas_evaluator(llm_client: Optional[Any] = None) -> RAGASEvaluator:
    """
    工厂函数：创建 RAGAS 评估器

    Args:
        llm_client: LLM 客户端

    Returns:
        RAGASEvaluator 实例
    """
    return RAGASEvaluator(llm_client=llm_client)