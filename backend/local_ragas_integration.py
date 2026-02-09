#!/usr/bin/env python3
"""
本地模型RAGAS评估集成模块
使用Ollama或本地vLLM/LLaMA.cpp等本地LLM
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
    from ragas.llms import LangchainLLMWrapper

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("⚠️  RAGAS未安装，请先运行: pip install ragas")

# 尝试导入LangChain的本地模型支持
try:
    from langchain_community.llms import Ollama
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️  langchain-community未安装，运行: pip install langchain-community")

# 导入RAG服务
from services.rag_generator import rag_generator
from services.embedding import embedding_service
from services.vector_db import vector_db_manager
from services.retriever import retriever
from models import RetrievalConfig, GenerationConfig, EmbeddingConfig, VectorDBConfig
from models import EmbeddingModelType, VectorDBType


class LocalLLMConfig:
    """本地LLM配置"""

    # 支持的本地模型提供商
    PROVIDERS = {
        "ollama": {
            "name": "Ollama",
            "description": "本地Ollama服务",
            "default_model": "qwen2.5:7b",
            "base_url": "http://localhost:11434",
        },
        "vllm": {
            "name": "vLLM",
            "description": "vLLM推理服务",
            "default_model": "Qwen/Qwen2.5-7B-Instruct",
            "base_url": "http://localhost:8000",
        },
        "llamacpp": {
            "name": "llama.cpp",
            "description": "llama.cpp本地模型",
            "default_model": "qwen2.5-7b-q4_k_m.gguf",
            "base_url": "http://localhost:8080",
        },
    }

    @classmethod
    def get_provider(cls, name: str):
        return cls.PROVIDERS.get(name, cls.PROVIDERS["ollama"])


class LocalRAGASIntegration:
    """使用本地LLM的RAGAS评估集成类"""

    def __init__(
        self,
        provider: str = "ollama",
        model_name: str = None,
        base_url: str = None,
        temperature: float = 0.0,
        use_ground_truth: bool = True,
    ):
        """
        初始化本地RAGAS评估器

        Args:
            provider: 本地模型提供商 (ollama/vllm/llamacpp)
            model_name: 模型名称，默认使用配置中的默认模型
            base_url: API地址，默认使用配置中的地址
            temperature: 生成温度，评估建议用0.0确保一致性
            use_ground_truth: 是否使用标准答案评估
        """
        self.provider = provider
        self.config = LocalLLMConfig.get_provider(provider)
        self.model_name = model_name or self.config["default_model"]
        self.base_url = base_url or self.config["base_url"]
        self.temperature = temperature
        self.use_ground_truth = use_ground_truth
        self.evaluation_history = []
        self.initialized = False
        self.local_llm = None
        self.ragas_llm = None

        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS未安装，请先运行: pip install ragas")

        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-community未安装，运行: pip install langchain-community"
            )

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

    def initialize_local_llm(self):
        """初始化本地LLM"""
        if self.local_llm is not None:
            return True

        print(f"🤖 初始化本地LLM: {self.config['name']}")
        print(f"   模型: {self.model_name}")
        print(f"   地址: {self.base_url}")

        try:
            if self.provider == "ollama":
                self.local_llm = Ollama(
                    model=self.model_name,
                    base_url=self.base_url,
                    temperature=self.temperature,
                    callback_manager=CallbackManager(
                        [StreamingStdOutCallbackHandler()]
                    ),
                )
            elif self.provider == "vllm":
                from langchain_community.llms import VLLM

                self.local_llm = VLLM(
                    model=self.model_name,
                    openai_api_base=self.base_url,
                    temperature=self.temperature,
                )
            else:
                print(f"❌ 不支持的提供商: {self.provider}")
                return False

            # 包装为RAGAS LLM
            self.ragas_llm = LangchainLLMWrapper(self.local_llm)

            print(f"   ✅ 本地LLM初始化成功\n")
            return True

        except Exception as e:
            print(f"   ❌ 本地LLM初始化失败: {e}")
            print(f"\n💡 请确保:")
            print(f"   1. {self.config['name']} 服务已启动")
            print(f"   2. 模型 {self.model_name} 已下载")
            if self.provider == "ollama":
                print(f"   3. 运行: ollama pull {self.model_name}")
            return False

    def test_local_llm(self):
        """测试本地LLM是否可用"""
        if not self.initialize_local_llm():
            return False

        print("🧪 测试本地LLM...")
        try:
            # 简单测试
            response = self.local_llm.invoke("你好，请用一句话介绍自己。")
            print(f"   ✅ LLM响应正常")
            print(f"   响应: {response[:100]}...")
            return True
        except Exception as e:
            print(f"   ❌ LLM测试失败: {e}")
            return False

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
        """
        # 确保服务已初始化
        if not self.initialize_services():
            return {
                "query": query,
                "error": "RAG服务初始化失败",
                "timestamp": datetime.now().isoformat(),
            }

        # 确保LLM已初始化
        if not self.initialize_local_llm():
            return {
                "query": query,
                "error": "本地LLM初始化失败",
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

            # 提取信息
            answer = response.answer
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

        # 运行评估（使用本地LLM）
        print("   🧪 运行RAGAS评估（本地LLM）...")
        try:
            eval_start = time.time()
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.ragas_llm,
                raise_exceptions=False,
            )
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
        """评估测试数据集"""
        if not self.initialize_services():
            return {"error": "RAG服务初始化失败"}

        if not self.initialize_local_llm():
            return {"error": "本地LLM初始化失败"}

        print(f"📂 加载测试数据集: {test_file}")

        # 加载测试数据
        test_path = Path(test_file)
        if not test_path.exists():
            test_path = Path(__file__).parent / "test_data" / test_file

        if not test_path.exists():
            print(f"❌ 测试文件不存在: {test_file}")
            return {"error": f"测试文件不存在: {test_file}"}

        try:
            with open(test_path, "r", encoding="utf-8") as f:
                test_data = json.load(f)
        except Exception as e:
            print(f"❌ 加载测试文件失败: {e}")
            return {"error": f"加载测试文件失败: {e}"}

        # 获取测试用例
        test_cases = test_data.get("end_to_end_test_cases", [])
        if not test_cases:
            test_cases = test_data.get("retrieval_test_cases", [])

        if not test_cases:
            print("❌ 测试数据为空")
            return {"error": "测试数据为空"}

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
                query, ground_truth if ground_truth else None
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
                "llm_provider": self.provider,
                "llm_model": self.model_name,
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
        self._print_summary(summary)

        return {"results": results, "summary": summary}

    def _print_scores(self, scores: Dict[str, float]):
        """打印评分"""
        for metric, score in scores.items():
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
        provider_name = self.provider if self.provider else "local"
        output_file = output_dir / f"ragas_eval_{provider_name}_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {"summary": summary, "results": results},
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\n💾 结果已保存: {output_file}")


# ==================== 命令行入口 ====================


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="本地LLM RAGAS评估工具")
    parser.add_argument(
        "--provider",
        "-p",
        choices=["ollama", "vllm", "llamacpp"],
        default="ollama",
        help="本地LLM提供商",
    )
    parser.add_argument(
        "--model", "-m", type=str, help="模型名称（默认使用配置中的默认模型）"
    )
    parser.add_argument(
        "--base-url", "-u", type=str, help="API地址（默认使用配置中的地址）"
    )
    parser.add_argument("--test", "-t", action="store_true", help="测试LLM连接")
    parser.add_argument(
        "--mode", choices=["single", "batch"], default="batch", help="评估模式"
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

    if not RAGAS_AVAILABLE or not LANGCHAIN_AVAILABLE:
        print("❌ 缺少必要的依赖，请安装:")
        print("   pip install ragas langchain-community")
        return

    # 创建评估器
    evaluator = LocalRAGASIntegration(
        provider=args.provider,
        model_name=args.model,
        base_url=args.base_url,
        temperature=0.0,  # 评估时保持确定性
    )

    if args.test:
        # 仅测试LLM连接
        print("🔧 测试本地LLM连接")
        print("=" * 70)
        if evaluator.test_local_llm():
            print("\n✅ 本地LLM测试通过，可以进行评估")
        else:
            print("\n❌ 本地LLM测试失败，请检查配置")

    elif args.mode == "single":
        if not args.query:
            print("❌ 请提供查询: --query '你的问题'")
            return

        result = evaluator.evaluate_single_query(
            query=args.query, ground_truth=args.ground_truth
        )

        if evaluator.evaluation_history:
            print("\n" + evaluator.get_evaluation_report())

    elif args.mode == "batch":
        evaluator.evaluate_test_dataset(
            test_file=args.test_file, max_samples=args.max_samples
        )


if __name__ == "__main__":
    main()
