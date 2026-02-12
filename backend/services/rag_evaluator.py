from typing import List, Dict, Tuple, Optional, Any
import time
import numpy as np
import json
from datetime import datetime
from models import RetrievalResult, RAGResponse, RetrievalConfig, GenerationConfig
from utils.logger import logger
from config import settings


class RAGEvaluator:
    """RAG系统评估器"""

    def __init__(self):
        self.evaluation_history = []
        self.baseline_metrics = {}
        self.thresholds = {
            "precision_at_k": 0.7,
            "recall_at_k": 0.6,
            "mrr": 0.6,
            "context_relevance": 0.7,
            "info_completeness": 0.6,
            "fidelity": 0.7,
            "answer_relevance": 0.7,
            "readability": 0.6,
        }

    def evaluate_retrieval(
        self,
        query: str,
        results: List[RetrievalResult],
        ground_truth: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        评估检索阶段

        Args:
            query: 查询文本
            results: 检索结果
            ground_truth: 真实相关文档列表（可选）

        Returns:
            检索评估指标
        """
        metrics = {}

        # 计算precision@k
        for k in [1, 3, 5, 10]:
            precision = self._calculate_precision_at_k(results, k)
            metrics[f"precision_at_{k}"] = precision

        # 计算recall@k
        if ground_truth:
            for k in [1, 3, 5, 10]:
                recall = self._calculate_recall_at_k(results, ground_truth, k)
                metrics[f"recall_at_{k}"] = recall

        # 计算MRR
        if ground_truth:
            mrr = self._calculate_mrr(results, ground_truth)
            metrics["mrr"] = mrr

        # 计算相关性得分分布
        similarity_scores = [r.similarity for r in results]
        if similarity_scores:
            metrics["similarity_distribution"] = {
                "mean": float(np.mean(similarity_scores)),
                "std": float(np.std(similarity_scores)),
                "min": float(min(similarity_scores)),
                "max": float(max(similarity_scores)),
                "median": float(np.median(similarity_scores)),
            }

        # 计算相关性阈值过滤后的结果数
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        for threshold in thresholds:
            filtered_count = len([r for r in results if r.similarity >= threshold])
            metrics[f"count_above_{threshold}"] = filtered_count

        return metrics

    def evaluate_enhancement(
        self, query: str, results: List[RetrievalResult], context: str
    ) -> Dict[str, Any]:
        """
        评估增强阶段

        Args:
            query: 查询文本
            results: 检索结果
            context: 构建的上下文

        Returns:
            增强阶段评估指标
        """
        metrics = {}

        # 上下文相关性
        context_relevance = self._calculate_context_relevance(query, results)
        metrics["context_relevance"] = context_relevance

        # 信息完整性
        info_completeness = self._calculate_info_completeness(results)
        metrics["info_completeness"] = info_completeness

        # 噪声过滤效果
        noise_filtering = self._calculate_noise_filtering(results)
        metrics["noise_filtering"] = noise_filtering

        # 上下文长度分析
        metrics["context_length"] = len(context)
        metrics["average_chunk_length"] = (
            np.mean([len(r.content) for r in results]) if results else 0
        )
        metrics["chunk_count"] = len(results)

        return metrics

    def evaluate_generation(
        self, query: str, response: RAGResponse, ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        评估生成阶段

        Args:
            query: 查询文本
            response: RAG响应
            ground_truth: 真实答案（可选）

        Returns:
            生成阶段评估指标
        """
        metrics = {}

        # 答案准确率
        if ground_truth:
            accuracy = self._calculate_accuracy(response.answer, ground_truth)
            metrics["accuracy"] = accuracy

        # 对源材料的忠实度
        fidelity = self._calculate_fidelity(response.answer, response.context_chunks)
        metrics["fidelity"] = fidelity

        # 与查询的相关性
        relevance = self._calculate_answer_relevance(query, response.answer)
        metrics["answer_relevance"] = relevance

        # 可读性
        readability = self._calculate_readability(response.answer)
        metrics["readability"] = readability

        # 生成时间分析
        metrics["generation_time_ms"] = response.generation_time_ms
        metrics["retrieval_time_ms"] = response.retrieval_time_ms
        metrics["total_time_ms"] = response.total_time_ms

        # 回答长度分析
        metrics["answer_length"] = len(response.answer)
        metrics["tokens_estimated"] = len(response.answer) / 0.75  # 粗略估算token数

        return metrics

    def evaluate_full_rag(
        self,
        query: str,
        response: RAGResponse,
        ground_truth: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        评估完整的RAG流程

        Args:
            query: 查询文本
            response: RAG响应
            ground_truth: 真实数据（可选）

        Returns:
            完整RAG评估指标
        """
        start_time = time.time()

        # 构建上下文
        context = "\n".join(
            [
                f"【参考文档{i + 1}】\n{r.content}\n"
                for i, r in enumerate(response.context_chunks)
            ]
        )

        # 评估各个阶段
        retrieval_metrics = self.evaluate_retrieval(
            query,
            response.context_chunks,
            ground_truth.get("relevant_docs") if ground_truth else None,
        )

        enhancement_metrics = self.evaluate_enhancement(
            query, response.context_chunks, context
        )

        generation_metrics = self.evaluate_generation(
            query, response, ground_truth.get("answer") if ground_truth else None
        )

        # 整合所有指标
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "evaluation_time_ms": (time.time() - start_time) * 1000,
            "retrieval": retrieval_metrics,
            "enhancement": enhancement_metrics,
            "generation": generation_metrics,
            "overall": {
                "success": len(response.context_chunks) > 0
                and len(response.answer) > 0,
                "total_context_chunks": len(response.context_chunks),
                "answer_length": len(response.answer),
                "total_time_ms": response.total_time_ms,
            },
        }

        # 保存评估历史
        self.evaluation_history.append(metrics)

        return metrics

    def run_ab_test(
        self,
        query: str,
        config_a: Dict[str, Any],
        config_b: Dict[str, Any],
        rag_generator: Any,
    ) -> Dict[str, Any]:
        """
        运行A/B测试

        Args:
            query: 查询文本
            config_a: A配置
            config_b: B配置
            rag_generator: RAG生成器实例

        Returns:
            A/B测试结果
        """
        # 构建配置
        retrieval_config_a = RetrievalConfig(**config_a["retrieval"])
        generation_config_a = GenerationConfig(**config_a["generation"])

        retrieval_config_b = RetrievalConfig(**config_b["retrieval"])
        generation_config_b = GenerationConfig(**config_b["generation"])

        # 运行A配置
        logger.info(f"运行A/B测试 - 配置A")
        response_a = rag_generator.generate(
            query, retrieval_config_a, generation_config_a
        )
        metrics_a = self.evaluate_full_rag(query, response_a)

        # 运行B配置
        logger.info(f"运行A/B测试 - 配置B")
        response_b = rag_generator.generate(
            query, retrieval_config_b, generation_config_b
        )
        metrics_b = self.evaluate_full_rag(query, response_b)

        # 计算差异
        comparison = self._compare_metrics(metrics_a, metrics_b)

        # 生成测试洞察
        test_insights = {
            "summary": f"A/B测试完成，胜者: {'A' if comparison['overall_winner'] == 'A' else 'B'}",
            "key_improvements": [],
            "recommended_config": config_a
            if comparison["overall_winner"] == "A"
            else config_b,
        }

        # 分析关键改进点
        for category in ["retrieval", "enhancement", "generation"]:
            cat_comparison = comparison.get(category, {})
            for metric, data in cat_comparison.items():
                if isinstance(data, dict) and "percent_change" in data:
                    change = data["percent_change"]
                    if abs(change) > 10:  # 大于10%的变化视为显著
                        winner = "B" if change > 0 else "A"
                        test_insights["key_improvements"].append(
                            f"{category} - {metric}: {'提升' if change > 0 else '下降'} {abs(change):.1f}% ({winner}配置更好)"
                        )

        return {
            "query": query,
            "config_a": config_a,
            "config_b": config_b,
            "metrics_a": metrics_a,
            "metrics_b": metrics_b,
            "comparison": comparison,
            "winner": "A" if comparison["overall_winner"] == "A" else "B",
            "insights": test_insights,
        }

    def build_optimization_recommendation(
        self, evaluation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        基于评估历史构建RAG系统优化推荐

        Args:
            evaluation_history: 评估历史记录

        Returns:
            优化推荐
        """
        if not evaluation_history:
            return {
                "status": "insufficient_data",
                "message": "评估历史数据不足，无法生成推荐",
                "recommendations": [],
            }

        # 分析历史趋势
        trends = self._analyze_evaluation_trends(evaluation_history)

        # 生成优化建议
        recommendations = []

        # 基于检索趋势的建议
        retrieval_trends = trends.get("retrieval", {})
        if retrieval_trends:
            precision_trend = retrieval_trends.get("precision_at_1", {})
            if precision_trend.get("slope", 0) < -0.01:
                recommendations.append(
                    {
                        "category": "retrieval",
                        "priority": "high",
                        "title": "检索精确率下降",
                        "description": "最近的评估显示检索精确率呈下降趋势",
                        "actions": [
                            "检查向量数据库索引状态",
                            "考虑更新嵌入模型或调整其参数",
                            "优化检索参数，如top_k和similarity_threshold",
                        ],
                    }
                )

        # 基于增强趋势的建议
        enhancement_trends = trends.get("enhancement", {})
        if enhancement_trends:
            context_relevance_trend = enhancement_trends.get("context_relevance", {})
            if context_relevance_trend.get("slope", 0) < -0.01:
                recommendations.append(
                    {
                        "category": "enhancement",
                        "priority": "medium",
                        "title": "上下文相关性下降",
                        "description": "最近的评估显示上下文相关性呈下降趋势",
                        "actions": [
                            "调整上下文构建策略",
                            "优化文档切分参数",
                            "考虑使用更细粒度的文档切分方法",
                        ],
                    }
                )

        # 基于生成趋势的建议
        generation_trends = trends.get("generation", {})
        if generation_trends:
            fidelity_trend = generation_trends.get("fidelity", {})
            if fidelity_trend.get("slope", 0) < -0.01:
                recommendations.append(
                    {
                        "category": "generation",
                        "priority": "high",
                        "title": "生成忠实度下降",
                        "description": "最近的评估显示生成忠实度呈下降趋势",
                        "actions": [
                            "优化提示词模板，强调忠实于源材料",
                            "调整LLM生成参数，如降低temperature",
                            "考虑使用更适合的LLM模型",
                        ],
                    }
                )

        # 基于时间性能的建议
        time_trends = trends.get("time", {})
        if time_trends:
            total_time_trend = time_trends.get("total_time_ms", {})
            if total_time_trend.get("slope", 0) > 100:
                recommendations.append(
                    {
                        "category": "performance",
                        "priority": "medium",
                        "title": "响应时间增加",
                        "description": "最近的评估显示响应时间呈上升趋势",
                        "actions": [
                            "优化检索性能，如使用缓存",
                            "考虑使用更快的嵌入模型",
                            "调整LLM生成参数以提高速度",
                        ],
                    }
                )

        return {
            "status": "success",
            "message": f"基于{len(evaluation_history)}次评估生成的优化建议",
            "trends": trends,
            "recommendations": recommendations,
            "next_steps": [
                "优先实施高优先级建议",
                "定期运行评估以监控改进效果",
                "考虑进行A/B测试验证优化效果",
            ],
        }

    def _analyze_evaluation_trends(
        self, evaluation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        分析评估历史趋势

        Args:
            evaluation_history: 评估历史记录

        Returns:
            趋势分析结果
        """
        trends = {"retrieval": {}, "enhancement": {}, "generation": {}, "time": {}}

        # 提取关键指标的时间序列数据
        metrics_series = {}
        for i, eval_data in enumerate(evaluation_history):
            timestamp = i  # 使用索引作为时间点

            # 提取检索指标
            retrieval = eval_data.get("retrieval", {})
            for metric in ["precision_at_1", "precision_at_3", "mrr"]:
                if metric in retrieval:
                    if metric not in metrics_series:
                        metrics_series[metric] = []
                    metrics_series[metric].append((timestamp, retrieval[metric]))

            # 提取增强指标
            enhancement = eval_data.get("enhancement", {})
            for metric in ["context_relevance", "info_completeness"]:
                if metric in enhancement:
                    if metric not in metrics_series:
                        metrics_series[metric] = []
                    metrics_series[metric].append((timestamp, enhancement[metric]))

            # 提取生成指标
            generation = eval_data.get("generation", {})
            for metric in ["fidelity", "answer_relevance", "readability"]:
                if metric in generation:
                    if metric not in metrics_series:
                        metrics_series[metric] = []
                    metrics_series[metric].append((timestamp, generation[metric]))

            # 提取时间指标
            overall = eval_data.get("overall", {})
            if "total_time_ms" in overall:
                if "total_time_ms" not in metrics_series:
                    metrics_series["total_time_ms"] = []
                metrics_series["total_time_ms"].append(
                    (timestamp, overall["total_time_ms"])
                )

        # 计算每个指标的趋势
        for metric, series in metrics_series.items():
            if len(series) < 2:
                continue

            # 提取x和y值
            x = np.array([point[0] for point in series])
            y = np.array([point[1] for point in series])

            # 线性回归计算趋势
            if len(x) > 1:
                slope, intercept = np.polyfit(x, y, 1)
                r_squared = np.corrcoef(x, y)[0, 1] ** 2 if len(x) > 1 else 0

                # 确定指标类别
                category = "retrieval"
                if metric in ["context_relevance", "info_completeness"]:
                    category = "enhancement"
                elif metric in ["fidelity", "answer_relevance", "readability"]:
                    category = "generation"
                elif metric in ["total_time_ms"]:
                    category = "time"

                trends[category][metric] = {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "r_squared": float(r_squared),
                    "current_value": float(y[-1]),
                    "previous_value": float(y[0]),
                    "change_percent": float((y[-1] - y[0]) / y[0] * 100)
                    if y[0] != 0
                    else 0,
                }

        return trends

    def generate_insights(self, evaluation_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成可落地洞察

        Args:
            evaluation_metrics: 评估指标

        Returns:
            洞察结果
        """
        insights = {
            "strengths": [],
            "weaknesses": [],
            "recommendations": [],
            "root_causes": [],
            "action_items": [],
            "visualization_data": {},
        }

        # 分析检索性能
        retrieval = evaluation_metrics.get("retrieval", {})
        if retrieval:
            # 检查precision@k
            for k in [1, 3, 5]:
                precision = retrieval.get(f"precision_at_{k}", 0)
                if precision >= self.thresholds["precision_at_k"]:
                    insights["strengths"].append(
                        f"检索精确率@k={k}表现良好: {precision:.2f}"
                    )
                else:
                    insights["weaknesses"].append(
                        f"检索精确率@k={k}需要改进: {precision:.2f}"
                    )
                    insights["recommendations"].append(
                        f"考虑调整检索参数以提高精确率@k={k}"
                    )

            # 检查相关性分布
            similarity_dist = retrieval.get("similarity_distribution", {})
            if similarity_dist:
                mean_similarity = similarity_dist.get("mean", 0)
                if mean_similarity < 0.7:
                    insights["weaknesses"].append(
                        f"平均相关性得分较低: {mean_similarity:.2f}"
                    )
                    insights["recommendations"].append(
                        "考虑使用更适合的嵌入模型或调整检索参数"
                    )

            # 构建检索性能可视化数据
            precision_data = []
            for k in [1, 3, 5, 10]:
                precision = retrieval.get(f"precision_at_{k}", 0)
                precision_data.append({"k": k, "value": precision})
            insights["visualization_data"]["precision_at_k"] = precision_data

            # 构建相关性分布可视化数据
            if similarity_dist:
                insights["visualization_data"]["similarity_distribution"] = (
                    similarity_dist
                )

        # 分析增强性能
        enhancement = evaluation_metrics.get("enhancement", {})
        if enhancement:
            context_relevance = enhancement.get("context_relevance", 0)
            if context_relevance < self.thresholds["context_relevance"]:
                insights["weaknesses"].append(
                    f"上下文相关性较低: {context_relevance:.2f}"
                )
                insights["recommendations"].append("优化上下文构建策略")

            info_completeness = enhancement.get("info_completeness", 0)
            if info_completeness < self.thresholds["info_completeness"]:
                insights["weaknesses"].append(
                    f"信息完整性较低: {info_completeness:.2f}"
                )
                insights["recommendations"].append("增加检索结果数量或优化检索策略")

            # 构建增强性能可视化数据
            insights["visualization_data"]["enhancement_metrics"] = {
                "context_relevance": context_relevance,
                "info_completeness": info_completeness,
                "noise_filtering": enhancement.get("noise_filtering", 0),
            }

        # 分析生成性能
        generation = evaluation_metrics.get("generation", {})
        if generation:
            fidelity = generation.get("fidelity", 0)
            if fidelity < self.thresholds["fidelity"]:
                insights["weaknesses"].append(f"对源材料的忠实度较低: {fidelity:.2f}")
                insights["recommendations"].append("优化提示词或调整生成参数")

            answer_relevance = generation.get("answer_relevance", 0)
            if answer_relevance < self.thresholds["answer_relevance"]:
                insights["weaknesses"].append(
                    f"回答与查询的相关性较低: {answer_relevance:.2f}"
                )
                insights["recommendations"].append("优化检索策略或提示词")

            # 构建生成性能可视化数据
            insights["visualization_data"]["generation_metrics"] = {
                "fidelity": fidelity,
                "answer_relevance": answer_relevance,
                "readability": generation.get("readability", 0),
                "accuracy": generation.get("accuracy", 0),
            }

            # 构建时间性能可视化数据
            insights["visualization_data"]["time_metrics"] = {
                "generation_time_ms": generation.get("generation_time_ms", 0),
                "retrieval_time_ms": generation.get("retrieval_time_ms", 0),
                "total_time_ms": generation.get("total_time_ms", 0),
            }

        # 分析时间性能
        total_time = evaluation_metrics.get("overall", {}).get("total_time_ms", 0)
        if total_time > 5000:  # 超过5秒
            insights["weaknesses"].append(f"响应时间较长: {total_time:.2f}ms")
            insights["recommendations"].append("优化检索或生成性能")

        # 执行根因分析
        insights["root_causes"] = self._analyze_root_causes(evaluation_metrics)

        # 生成具体的行动项
        if insights["weaknesses"]:
            insights["action_items"] = self._generate_action_items(
                insights["weaknesses"]
            )

        return insights

    def _analyze_root_causes(self, evaluation_metrics: Dict[str, Any]) -> List[str]:
        """
        分析性能下降的根因

        Args:
            evaluation_metrics: 评估指标

        Returns:
            根因分析结果
        """
        root_causes = []

        # 分析检索问题根因
        retrieval = evaluation_metrics.get("retrieval", {})
        if retrieval:
            similarity_dist = retrieval.get("similarity_distribution", {})
            if similarity_dist:
                mean_similarity = similarity_dist.get("mean", 0)
                std_similarity = similarity_dist.get("std", 0)

                if mean_similarity < 0.6:
                    if std_similarity > 0.2:
                        root_causes.append(
                            "嵌入模型可能不适合当前数据分布，导致相似度评分波动较大"
                        )
                    else:
                        root_causes.append("嵌入模型整体表现不佳，需要更换或微调")

            # 检查检索结果数量
            chunk_count = evaluation_metrics.get("enhancement", {}).get(
                "chunk_count", 0
            )
            if chunk_count < 3:
                root_causes.append(
                    "检索结果数量不足，可能是向量数据库索引问题或查询表述问题"
                )

        # 分析增强问题根因
        enhancement = evaluation_metrics.get("enhancement", {})
        if enhancement:
            context_relevance = enhancement.get("context_relevance", 0)
            info_completeness = enhancement.get("info_completeness", 0)

            if context_relevance < 0.6 and info_completeness < 0.6:
                root_causes.append("文档切分策略可能不合理，导致片段质量较差")
            elif context_relevance < 0.6:
                root_causes.append("上下文构建策略需要优化，可能是片段选择或排序问题")
            elif info_completeness < 0.6:
                root_causes.append("检索策略需要调整，可能需要增加top_k或优化查询扩展")

        # 分析生成问题根因
        generation = evaluation_metrics.get("generation", {})
        if generation:
            fidelity = generation.get("fidelity", 0)
            answer_relevance = generation.get("answer_relevance", 0)

            if fidelity < 0.6:
                root_causes.append("提示词模板可能需要优化，强调忠实于源材料")

            if answer_relevance < 0.6:
                if fidelity >= 0.6:
                    root_causes.append(
                        "回答相关性低但忠实度高，可能是检索结果与查询不匹配"
                    )
                else:
                    root_causes.append(
                        "回答质量整体较差，可能是LLM模型选择或参数配置问题"
                    )

        # 分析时间性能根因
        total_time = evaluation_metrics.get("overall", {}).get("total_time_ms", 0)
        retrieval_time = generation.get("retrieval_time_ms", 0)
        generation_time = generation.get("generation_time_ms", 0)

        if total_time > 5000:
            if retrieval_time > 3000:
                root_causes.append("检索时间过长，可能是向量数据库性能问题或网络延迟")
            if generation_time > 3000:
                root_causes.append("生成时间过长，可能是LLM模型响应慢或参数配置问题")

        return root_causes

    def _calculate_precision_at_k(
        self, results: List[RetrievalResult], k: int
    ) -> float:
        """计算precision@k"""
        if not results:
            return 0.0
        top_k_results = results[:k]
        # 这里使用相似度分数作为相关性的近似
        # 在实际应用中，应该使用人工标注或更复杂的相关性评估
        relevant_count = len([r for r in top_k_results if r.similarity >= 0.7])
        return relevant_count / min(k, len(results))

    def _calculate_recall_at_k(
        self, results: List[RetrievalResult], ground_truth: List[str], k: int
    ) -> float:
        """计算recall@k"""
        if not ground_truth:
            return 0.0
        top_k_results = results[:k]
        # 这里简化处理，实际应该使用更准确的匹配方法
        retrieved_docs = set([r.document_id for r in top_k_results])
        relevant_docs = set(ground_truth)
        intersection = retrieved_docs.intersection(relevant_docs)
        return len(intersection) / len(relevant_docs)

    def _calculate_mrr(
        self, results: List[RetrievalResult], ground_truth: List[str]
    ) -> float:
        """计算平均倒数排名"""
        if not ground_truth:
            return 0.0

        for i, result in enumerate(results, 1):
            if result.document_id in ground_truth:
                return 1.0 / i
        return 0.0

    def _calculate_context_relevance(
        self, query: str, results: List[RetrievalResult]
    ) -> float:
        """计算上下文相关性"""
        if not results:
            return 0.0
        # 简单计算：平均相似度分数
        return float(np.mean([r.similarity for r in results]))

    def _calculate_info_completeness(self, results: List[RetrievalResult]) -> float:
        """计算信息完整性"""
        if not results:
            return 0.0
        # 考虑因素：结果数量、内容长度、相似度分布
        count_score = min(len(results) / 5, 1.0)  # 期望至少5个结果
        length_score = min(
            np.mean([len(r.content) for r in results]) / 200, 1.0
        )  # 期望平均长度200字符
        similarity_score = np.mean([r.similarity for r in results])
        return (count_score + length_score + similarity_score) / 3

    def _calculate_noise_filtering(self, results: List[RetrievalResult]) -> float:
        """计算噪声过滤效果"""
        if not results:
            return 0.0
        # 计算高相似度结果的比例
        high_quality_count = len([r for r in results if r.similarity >= 0.7])
        return high_quality_count / len(results)

    def _calculate_accuracy(self, answer: str, ground_truth: str) -> float:
        """计算答案准确率"""
        if not answer or not ground_truth:
            return 0.0
        # 简单实现：计算词重叠率
        answer_tokens = set(answer.lower().split())
        ground_truth_tokens = set(ground_truth.lower().split())
        intersection = answer_tokens.intersection(ground_truth_tokens)
        if not ground_truth_tokens:
            return 0.0
        return len(intersection) / len(ground_truth_tokens)

    def _calculate_fidelity(
        self, answer: str, context_chunks: List[RetrievalResult]
    ) -> float:
        """计算对源材料的忠实度"""
        if not answer or not context_chunks:
            return 0.0
        # 检查答案中的信息是否来自上下文
        context_text = " ".join([chunk.content for chunk in context_chunks])
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context_text.lower().split())
        intersection = answer_tokens.intersection(context_tokens)
        if not answer_tokens:
            return 0.0
        return len(intersection) / len(answer_tokens)

    def _calculate_answer_relevance(self, query: str, answer: str) -> float:
        """计算回答与查询的相关性"""
        if not query or not answer:
            return 0.0
        # 简单实现：计算词重叠率
        query_tokens = set(query.lower().split())
        answer_tokens = set(answer.lower().split())
        intersection = query_tokens.intersection(answer_tokens)
        if not query_tokens:
            return 0.0
        return len(intersection) / len(query_tokens)

    def _calculate_readability(self, answer: str) -> float:
        """计算可读性"""
        if not answer:
            return 0.0
        # 简单实现：考虑句子长度、段落结构等
        sentences = answer.split("。")
        avg_sentence_length = (
            np.mean([len(s) for s in sentences if s.strip()]) if sentences else 0
        )
        paragraphs = answer.split("\n")
        paragraph_count = len([p for p in paragraphs if p.strip()])
        # 理想句子长度：15-20字符
        sentence_length_score = max(0, 1 - abs(avg_sentence_length - 17.5) / 35)
        # 理想段落数：至少1
        paragraph_score = min(paragraph_count / 3, 1.0)
        return (sentence_length_score + paragraph_score) / 2

    def _compare_metrics(
        self, metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """比较两个配置的指标"""
        comparison = {
            "retrieval": {},
            "enhancement": {},
            "generation": {},
            "overall": {},
            "overall_winner": "A",
        }

        # 比较检索指标
        retrieval_a = metrics_a.get("retrieval", {})
        retrieval_b = metrics_b.get("retrieval", {})
        for key in set(retrieval_a.keys()) | set(retrieval_b.keys()):
            val_a = retrieval_a.get(key, 0)
            val_b = retrieval_b.get(key, 0)
            if isinstance(val_a, dict) or isinstance(val_b, dict):
                continue
            comparison["retrieval"][key] = {
                "a": val_a,
                "b": val_b,
                "difference": val_b - val_a,
                "percent_change": (val_b - val_a) / (val_a if val_a != 0 else 1) * 100,
            }

        # 比较增强指标
        enhancement_a = metrics_a.get("enhancement", {})
        enhancement_b = metrics_b.get("enhancement", {})
        for key in set(enhancement_a.keys()) | set(enhancement_b.keys()):
            val_a = enhancement_a.get(key, 0)
            val_b = enhancement_b.get(key, 0)
            comparison["enhancement"][key] = {
                "a": val_a,
                "b": val_b,
                "difference": val_b - val_a,
                "percent_change": (val_b - val_a) / (val_a if val_a != 0 else 1) * 100,
            }

        # 比较生成指标
        generation_a = metrics_a.get("generation", {})
        generation_b = metrics_b.get("generation", {})
        for key in set(generation_a.keys()) | set(generation_b.keys()):
            val_a = generation_a.get(key, 0)
            val_b = generation_b.get(key, 0)
            comparison["generation"][key] = {
                "a": val_a,
                "b": val_b,
                "difference": val_b - val_a,
                "percent_change": (val_b - val_a) / (val_a if val_a != 0 else 1) * 100,
            }

        # 计算总体赢家
        scores = {"A": 0, "B": 0}
        # 对关键指标进行评分
        key_metrics = [
            "precision_at_1",
            "precision_at_3",
            "context_relevance",
            "info_completeness",
            "fidelity",
            "answer_relevance",
        ]

        for metric in key_metrics:
            # 检查retrieval指标
            if metric in comparison["retrieval"]:
                diff = comparison["retrieval"][metric]["difference"]
                if diff > 0.05:
                    scores["B"] += 1
                elif diff < -0.05:
                    scores["A"] += 1
            # 检查enhancement指标
            elif metric in comparison["enhancement"]:
                diff = comparison["enhancement"][metric]["difference"]
                if diff > 0.05:
                    scores["B"] += 1
                elif diff < -0.05:
                    scores["A"] += 1
            # 检查generation指标
            elif metric in comparison["generation"]:
                diff = comparison["generation"][metric]["difference"]
                if diff > 0.05:
                    scores["B"] += 1
                elif diff < -0.05:
                    scores["A"] += 1

        # 比较响应时间
        time_a = metrics_a.get("overall", {}).get("total_time_ms", 0)
        time_b = metrics_b.get("overall", {}).get("total_time_ms", 0)
        if time_b < time_a * 0.9:
            scores["B"] += 1
        elif time_a < time_b * 0.9:
            scores["A"] += 1

        comparison["overall"]["score_a"] = scores["A"]
        comparison["overall"]["score_b"] = scores["B"]
        comparison["overall_winner"] = "A" if scores["A"] >= scores["B"] else "B"

        return comparison

    def _generate_action_items(self, weaknesses: List[str]) -> List[str]:
        """生成行动项"""
        action_items = []

        if any("precision" in w for w in weaknesses):
            action_items.append("调整检索参数，如top_k、similarity_threshold等")
            action_items.append("考虑使用更适合的嵌入模型")
            action_items.append("优化向量数据库索引配置")

        if any("context" in w for w in weaknesses):
            action_items.append("改进上下文构建策略，如调整上下文长度限制")
            action_items.append("优化文档切分参数，提高片段质量")

        if any("fidelity" in w for w in weaknesses):
            action_items.append("优化提示词模板，强调忠实于源材料")
            action_items.append("调整生成参数，如temperature、top_p等")

        if any("relevance" in w for w in weaknesses):
            action_items.append("改进检索策略，如使用更复杂的查询扩展")
            action_items.append("考虑启用重排序功能")

        if any("time" in w for w in weaknesses):
            action_items.append("优化检索性能，如使用更快的嵌入模型")
            action_items.append("考虑使用缓存机制")

        return action_items


# 全局RAG评估器实例
rag_evaluator = RAGEvaluator()
