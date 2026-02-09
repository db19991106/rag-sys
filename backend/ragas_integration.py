#!/usr/bin/env python3
"""
RAGAS评估集成模块 - 修复版
与你的RAG系统无缝集成
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_entity_recall,
        answer_similarity,
        answer_correctness,
    )
    from datasets import Dataset

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("⚠️  RAGAS未安装，请先运行: pip install ragas")
    print("   或使用简化版评估器: python ragas_integration.py --mode simple")

# 导入RAG服务
from services.rag_generator import rag_generator
from services.embedding import embedding_service
from services.vector_db import vector_db_manager
from services.retriever import retriever
from models import RetrievalConfig, GenerationConfig, EmbeddingConfig, VectorDBConfig
from models import EmbeddingModelType, VectorDBType
from utils.logger import logger


class RAGASIntegration:
    """RAGAS评估集成类"""

    def __init__(self, use_ground_truth: bool = True):
        self.use_ground_truth = use_ground_truth
        self.evaluation_history = []
        self.initialized = False

        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS未安装，请先运行: pip install ragas")

        # 基础指标
        self.basic_metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        # 高级指标（需要ground truth）
        self.advanced_metrics = [
            answer_correctness,
            answer_similarity,
            context_entity_recall,
        ]

    def initialize_services(self):
        """初始化RAG服务（嵌入模型和向量数据库）"""
        if self.initialized:
            return True

        print("🚀 初始化RAG服务...")

        try:
            # 1. 初始化嵌入服务
            if not embedding_service.is_loaded():
                print("   📥 加载嵌入模型...")
                config = EmbeddingConfig(
                    model_type=EmbeddingModelType.BGE,
                    model_name="BAAI/bge-small-zh-v1.5",
                    device="cpu",
                    batch_size=8,
                )
                response = embedding_service.load_model(config)
                if response.status != "success":
                    print(f"   ❌ 嵌入模型加载失败: {response.message}")
                    return False
                print(f"   ✅ 嵌入模型加载成功 (维度: {response.dimension})")
            else:
                print("   ✅ 嵌入模型已加载")

            # 2. 初始化向量数据库
            if vector_db_manager.db is None:
                print("   📥 初始化向量数据库...")
                dimension = embedding_service.get_dimension()
                config = VectorDBConfig(
                    db_type=VectorDBType.FAISS, dimension=dimension, index_type="HNSW"
                )
                success = vector_db_manager.initialize(config)
                if not success:
                    print("   ❌ 向量数据库初始化失败")
                    return False
                print("   ✅ 向量数据库初始化成功")
            else:
                print("   ✅ 向量数据库已初始化")

            # 3. 检查向量库状态
            status = vector_db_manager.get_status()
            print(f"   📊 向量库状态: {status.total_vectors} 个向量")

            if status.total_vectors == 0:
                print("   ⚠️  警告: 向量库为空，请先上传文档")
                return False

            self.initialized = True
            print("✅ RAG服务初始化完成\n")
            return True

        except Exception as e:
            print(f"   ❌ 初始化失败: {e}")
            import traceback

            traceback.print_exc()
            return False

    def evaluate_single_query(
        self,
        query: str,
        ground_truth: str = None,
        retrieval_config: RetrievalConfig = None,
        generation_config: GenerationConfig = None,
    ) -> Dict[str, Any]:
        """
        评估单个查询

        Args:
            query: 查询问题
            ground_truth: 期望答案（可选）
            retrieval_config: 检索配置
            generation_config: 生成配置

        Returns:
            评估结果字典
        """
        # 确保服务已初始化
        if not self.initialize_services():
            return {
                "query": query,
                "error": "RAG服务初始化失败",
                "timestamp": datetime.now().isoformat(),
            }

        print(f"🔍 评估查询: {query[:50]}...")

        # 使用默认配置
        if retrieval_config is None:
            retrieval_config = RetrievalConfig(top_k=5)
        if generation_config is None:
            generation_config = GenerationConfig(temperature=0.7, max_tokens=500)

        # 运行RAG系统
        start_time = time.time()
        try:
            response = rag_generator.generate(
                query=query,
                retrieval_config=retrieval_config,
                generation_config=generation_config,
            )
            rag_time = time.time() - start_time

            # 提取信息（修复：使用正确的属性名）
            answer = response.answer
            # RAGResponse中使用context_chunks而不是sources
            contexts = (
                [chunk.content for chunk in response.context_chunks]
                if response.context_chunks
                else []
            )

            print(f"   ✓ RAG执行完成 ({rag_time:.2f}s)")
            print(f"   📄 检索到 {len(contexts)} 个上下文")
            print(f"   💬 回答长度: {len(answer)} 字符")

        except Exception as e:
            print(f"   ❌ RAG执行失败: {e}")
            import traceback

            traceback.print_exc()
            return {
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

        # 如果没有上下文，跳过评估
        if not contexts:
            print("   ⚠️  无检索上下文，跳过评估")
            return {
                "query": query,
                "answer": answer,
                "error": "无检索上下文",
                "timestamp": datetime.now().isoformat(),
            }

        # 准备RAGAS数据
        data_dict = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        }

        if ground_truth:
            data_dict["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data_dict)

        # 选择评估指标
        metrics = self.basic_metrics.copy()
        if ground_truth and self.use_ground_truth:
            metrics.extend(self.advanced_metrics)

        # 运行评估
        print("   🧪 运行RAGAS评估...")
        try:
            eval_start = time.time()
            result = evaluate(dataset=dataset, metrics=metrics, raise_exceptions=False)
            eval_time = time.time() - eval_start

            # 转换结果
            scores = {
                k: float(v[0]) if hasattr(v, "__getitem__") else float(v)
                for k, v in result.items()
            }

            print(f"   ✓ 评估完成 ({eval_time:.2f}s)")

            # 构建结果
            evaluation_result = {
                "query": query,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "scores": scores,
                "rag_time": rag_time,
                "eval_time": eval_time,
                "timestamp": datetime.now().isoformat(),
            }

            # 添加到历史
            self.evaluation_history.append(evaluation_result)

            # 打印结果
            self._print_scores(scores)

            return evaluation_result

        except Exception as e:
            print(f"   ❌ 评估失败: {e}")
            import traceback

            traceback.print_exc()
            return {
                "query": query,
                "answer": answer,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def evaluate_test_dataset(
        self, test_file: str, max_samples: int = None
    ) -> Dict[str, Any]:
        """
        评估测试数据集

        Args:
            test_file: 测试数据JSON文件路径
            max_samples: 最大评估样本数（可选）

        Returns:
            批量评估结果
        """
        # 确保服务已初始化
        if not self.initialize_services():
            return {
                "error": "RAG服务初始化失败",
                "timestamp": datetime.now().isoformat(),
            }

        print(f"📂 加载测试数据集: {test_file}")

        # 加载测试数据
        test_path = Path(test_file)
        if not test_path.exists():
            # 尝试在test_data目录下查找
            test_path = Path(__file__).parent / "test_data" / test_file

        if not test_path.exists():
            print(f"❌ 测试文件不存在: {test_file}")
            return {
                "error": f"测试文件不存在: {test_file}",
                "timestamp": datetime.now().isoformat(),
            }

        try:
            with open(test_path, "r", encoding="utf-8") as f:
                test_data = json.load(f)
        except Exception as e:
            print(f"❌ 加载测试文件失败: {e}")
            return {
                "error": f"加载测试文件失败: {e}",
                "timestamp": datetime.now().isoformat(),
            }

        # 获取测试用例
        test_cases = test_data.get("end_to_end_test_cases", [])
        if not test_cases:
            test_cases = test_data.get("retrieval_test_cases", [])

        if not test_cases:
            print("❌ 测试数据为空")
            return {"error": "测试数据为空", "timestamp": datetime.now().isoformat()}

        if max_samples:
            test_cases = test_cases[:max_samples]

        print(f"🧪 开始评估 {len(test_cases)} 个测试用例\n")

        # 批量评估
        results = []
        for i, case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] ", end="")

            query = case.get("query")
            expected = case.get("expected_answer_contains", [])
            ground_truth = (
                " ".join(expected) if isinstance(expected, list) else str(expected)
            )

            result = self.evaluate_single_query(
                query=query, ground_truth=ground_truth if ground_truth else None
            )
            results.append(result)

            print()

        # 计算统计
        successful_evals = [r for r in results if "scores" in r]
        if successful_evals:
            avg_scores = {}
            for metric in successful_evals[0]["scores"].keys():
                values = [
                    r["scores"][metric]
                    for r in successful_evals
                    if metric in r["scores"]
                ]
                avg_scores[metric] = sum(values) / len(values) if values else 0

            summary = {
                "total_cases": len(test_cases),
                "successful_evals": len(successful_evals),
                "failed_evals": len(test_cases) - len(successful_evals),
                "average_scores": avg_scores,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            summary = {
                "total_cases": len(test_cases),
                "successful_evals": 0,
                "failed_evals": len(test_cases),
                "timestamp": datetime.now().isoformat(),
            }

        # 保存结果
        self._save_results(results, summary)

        # 打印总结
        self._print_summary(summary)

        return {"results": results, "summary": summary}

    def _print_scores(self, scores: Dict[str, float]):
        """打印评分"""
        for metric, score in scores.items():
            # 根据分数显示颜色/表情
            if score >= 0.8:
                status = "🟢"
            elif score >= 0.6:
                status = "🟡"
            else:
                status = "🔴"

            print(f"   {status} {metric}: {score:.3f}")

    def _print_summary(self, summary: Dict):
        """打印总结"""
        print("\n" + "=" * 70)
        print("📊 评估总结")
        print("=" * 70)
        print(f"总测试用例: {summary['total_cases']}")
        print(f"成功评估: {summary['successful_evals']} ✅")
        print(f"失败: {summary['failed_evals']} ❌")

        if "average_scores" in summary and summary["average_scores"]:
            print("\n平均指标得分:")
            for metric, score in summary["average_scores"].items():
                if score >= 0.8:
                    status = "🟢 优秀"
                elif score >= 0.6:
                    status = "🟡 良好"
                else:
                    status = "🔴 需优化"

                print(f"  {metric}: {score:.3f} {status}")

        print("=" * 70)

    def _save_results(self, results: List[Dict], summary: Dict):
        """保存结果到文件"""
        output_dir = Path(__file__).parent / "evaluation_results"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"ragas_eval_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {"summary": summary, "results": results},
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\n💾 结果已保存: {output_file}")


class SimpleRAGEvaluator:
    """
    简化版RAG评估器（无需RAGAS依赖）
    使用基于规则的评估
    """

    def __init__(self):
        self.results = []

    def evaluate_retrieval_quality(
        self, query: str, expected_keywords: List[str], retrieved_contexts: List[str]
    ) -> Dict[str, float]:
        """评估检索质量（基于关键词匹配）"""
        if not retrieved_contexts:
            return {
                "keyword_hit_rate": 0.0,
                "context_coverage": 0.0,
                "avg_context_length": 0,
            }

        # 关键词命中检查
        hits = 0
        for keyword in expected_keywords:
            if any(keyword in ctx for ctx in retrieved_contexts):
                hits += 1

        keyword_hit_rate = hits / len(expected_keywords) if expected_keywords else 0

        # 计算平均上下文长度
        avg_length = sum(len(ctx) for ctx in retrieved_contexts) / len(
            retrieved_contexts
        )

        return {
            "keyword_hit_rate": keyword_hit_rate,
            "num_contexts": len(retrieved_contexts),
            "avg_context_length": avg_length,
        }


# ==================== 命令行入口 ====================


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="RAGAS评估工具")
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "simple"],
        default="single",
        help="评估模式",
    )
    parser.add_argument("--query", "-q", type=str, help="单个查询")
    parser.add_argument("--ground-truth", "-g", type=str, help="期望答案")
    parser.add_argument(
        "--test-file",
        "-f",
        type=str,
        default="test_data/test_dataset.json",
        help="测试数据文件",
    )
    parser.add_argument("--max-samples", "-n", type=int, help="最大样本数")

    args = parser.parse_args()

    if args.mode == "single":
        if not args.query:
            print("❌ 请提供查询: --query '你的问题'")
            return

        if not RAGAS_AVAILABLE:
            print("⚠️  RAGAS未安装，使用简化评估器")
            print("   安装命令: pip install ragas")
            evaluator = SimpleRAGEvaluator()
            print("简化评估模式 - 仅支持基础功能")
        else:
            evaluator = RAGASIntegration()
            result = evaluator.evaluate_single_query(
                query=args.query, ground_truth=args.ground_truth
            )
            if evaluator.evaluation_history:
                print("\n" + evaluator.get_evaluation_report())

    elif args.mode == "batch":
        if not RAGAS_AVAILABLE:
            print("❌ 批量评估需要RAGAS，请先安装: pip install ragas")
            return

        evaluator = RAGASIntegration()
        evaluator.evaluate_test_dataset(
            test_file=args.test_file, max_samples=args.max_samples
        )

    elif args.mode == "simple":
        print("使用简化评估器（无需RAGAS）")
        evaluator = SimpleRAGEvaluator()
        print("简化评估模式 - 执行基础关键词匹配评估")


if __name__ == "__main__":
    main()
