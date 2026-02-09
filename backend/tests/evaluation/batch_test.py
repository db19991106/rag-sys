#!/usr/bin/env python3
"""
RAG系统批量测试脚本 - 使用扩展测试数据集(125条)

使用方法:
  python3 batch_test.py --mode retrieval       # 运行检索测试
  python3 batch_test.py --mode e2e            # 运行端到端测试
  python3 batch_test.py --mode all            # 运行所有测试
  python3 batch_test.py --category 住宿标准    # 按分类测试
  python3 batch_test.py --difficulty easy     # 按难度测试
  python3 batch_test.py --limit 10            # 只测前10条
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.rag_generator import rag_generator
from services.embedding import embedding_service
from services.vector_db import vector_db_manager
from models import RetrievalConfig, GenerationConfig, EmbeddingConfig, VectorDBConfig
from models import EmbeddingModelType, VectorDBType
from config import settings


class BatchTester:
    """批量测试器"""

    def __init__(self):
        self.test_data = None
        self.results = []
        self.load_test_data()
        self.init_services()

    def load_test_data(self):
        """加载测试数据"""
        test_file = Path(__file__).parent / "test_data" / "test_dataset_extended.json"
        with open(test_file, "r", encoding="utf-8") as f:
            self.test_data = json.load(f)
        print(f"✅ 加载测试数据: {test_file}")

    def init_services(self):
        """初始化服务"""
        print("📦 初始化服务...")

        if not embedding_service.is_loaded():
            config = EmbeddingConfig(
                model_type=EmbeddingModelType.BGE,
                model_name="BAAI/bge-small-zh-v1.5",
                device="cpu",
            )
            embedding_service.load_model(config)

        dimension = embedding_service.get_dimension()
        vdb_config = VectorDBConfig(
            db_type=VectorDBType.FAISS, dimension=dimension, index_type="HNSW"
        )
        vector_db_manager.initialize(vdb_config)

        status = vector_db_manager.get_status()
        print(f"   ✅ 向量库: {status.total_vectors} 个向量\n")

    def test_retrieval(self, test_cases: List[Dict], limit: int = None) -> Dict:
        """测试检索功能"""
        if limit:
            test_cases = test_cases[:limit]

        print(f"\n🔍 运行检索测试 ({len(test_cases)} 条)...")
        print("=" * 80)

        results = []
        passed = 0

        for i, case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] {case['id']}")
            print(f"查询: {case['query']}")
            print(f"分类: {case['category']} | 难度: {case['difficulty']}")

            # 执行检索
            start = time.time()
            try:
                from services.retriever import retriever

                retrieval_config = RetrievalConfig(top_k=5)
                response = retriever.retrieve(case["query"], retrieval_config)
                elapsed = (time.time() - start) * 1000

                # 检查关键词命中
                retrieved_text = " ".join([r.content for r in response.results])
                keywords_hit = sum(
                    1 for kw in case["expected_keywords"] if kw in retrieved_text
                )
                keyword_rate = (
                    keywords_hit / len(case["expected_keywords"])
                    if case["expected_keywords"]
                    else 1.0
                )

                # 评估结果
                if keyword_rate >= 0.6:
                    status = "✅ 通过"
                    passed += 1
                else:
                    status = "⚠️  警告"

                print(f"   检索结果: {len(response.results)} 个片段")
                print(
                    f"   关键词命中: {keywords_hit}/{len(case['expected_keywords'])} ({keyword_rate * 100:.0f}%)"
                )
                print(f"   响应时间: {elapsed:.1f}ms")
                print(f"   状态: {status}")

                results.append(
                    {
                        "id": case["id"],
                        "query": case["query"],
                        "status": "passed" if keyword_rate >= 0.6 else "warning",
                        "keyword_rate": keyword_rate,
                        "response_time_ms": elapsed,
                        "retrieved_count": len(response.results),
                    }
                )

            except Exception as e:
                print(f"   ❌ 错误: {str(e)}")
                results.append(
                    {
                        "id": case["id"],
                        "query": case["query"],
                        "status": "error",
                        "error": str(e),
                    }
                )

        # 统计
        print("\n" + "=" * 80)
        print(
            f"📊 检索测试完成: {passed}/{len(test_cases)} 通过 ({passed / len(test_cases) * 100:.1f}%)"
        )

        return {
            "total": len(test_cases),
            "passed": passed,
            "failed": len(test_cases) - passed,
            "pass_rate": passed / len(test_cases) if test_cases else 0,
            "results": results,
        }

    def test_end_to_end(self, test_cases: List[Dict], limit: int = None) -> Dict:
        """测试端到端功能"""
        if limit:
            test_cases = test_cases[:limit]

        print(f"\n🎯 运行端到端测试 ({len(test_cases)} 条)...")
        print("=" * 80)

        results = []
        passed = 0

        for i, case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] {case['id']}")
            print(f"查询: {case['query'][:60]}...")

            # 执行RAG
            start = time.time()
            try:
                response = rag_generator.generate(
                    query=case["query"],
                    retrieval_config=RetrievalConfig(top_k=3),
                    generation_config=GenerationConfig(
                        llm_provider=settings.llm_provider,
                        llm_model=settings.llm_model,
                        temperature=0.7,
                        max_tokens=500,
                    ),
                )
                elapsed = (time.time() - start) * 1000

                answer = response.answer

                # 检查必含关键词
                contains_hit = sum(
                    1 for kw in case["expected_answer_contains"] if kw in answer
                )
                contains_rate = (
                    contains_hit / len(case["expected_answer_contains"])
                    if case["expected_answer_contains"]
                    else 1.0
                )

                # 检查不应含有的关键词
                not_contains_hit = sum(
                    1 for kw in case["expected_answer_not_contains"] if kw in answer
                )

                # 检查响应时间
                time_ok = elapsed <= case["max_response_time_ms"]

                # 检查长度
                length_ok = len(answer) >= case["min_answer_length"]

                # 综合评估
                if contains_rate >= 0.7 and not_contains_hit == 0 and length_ok:
                    status = "✅ 通过"
                    passed += 1
                elif contains_rate >= 0.5:
                    status = "⚠️  部分通过"
                else:
                    status = "❌ 未通过"

                print(f"   回答长度: {len(answer)} 字符")
                print(
                    f"   必含关键词: {contains_hit}/{len(case['expected_answer_contains'])}"
                )
                print(f"   应排除关键词违规: {not_contains_hit}")
                print(f"   响应时间: {elapsed / 1000:.1f}s")
                print(f"   状态: {status}")

                results.append(
                    {
                        "id": case["id"],
                        "query": case["query"],
                        "status": "passed"
                        if status == "✅ 通过"
                        else "partial"
                        if status == "⚠️  部分通过"
                        else "failed",
                        "contains_rate": contains_rate,
                        "response_time_ms": elapsed,
                        "answer_length": len(answer),
                    }
                )

            except Exception as e:
                print(f"   ❌ 错误: {str(e)}")
                results.append(
                    {
                        "id": case["id"],
                        "query": case["query"],
                        "status": "error",
                        "error": str(e),
                    }
                )

        # 统计
        print("\n" + "=" * 80)
        print(
            f"📊 端到端测试完成: {passed}/{len(test_cases)} 通过 ({passed / len(test_cases) * 100:.1f}%)"
        )

        return {
            "total": len(test_cases),
            "passed": passed,
            "failed": len(test_cases) - passed,
            "pass_rate": passed / len(test_cases) if test_cases else 0,
            "results": results,
        }

    def run_all_tests(self, limit: int = None):
        """运行所有测试"""
        print("\n" + "=" * 80)
        print("🚀 RAG系统批量测试 - 扩展数据集 (125条)")
        print("=" * 80)

        # 合并检索测试
        retrieval_cases = self.test_data["retrieval_test_cases"] + self.test_data.get(
            "retrieval_test_cases_part2", []
        )
        e2e_cases = self.test_data["end_to_end_test_cases"]

        # 运行测试
        retrieval_results = self.test_retrieval(retrieval_cases, limit)
        e2e_results = self.test_end_to_end(e2e_cases, limit)

        # 生成报告
        self.generate_report(retrieval_results, e2e_results)

    def filter_by_category(self, category: str) -> List[Dict]:
        """按分类筛选"""
        retrieval_cases = self.test_data["retrieval_test_cases"] + self.test_data.get(
            "retrieval_test_cases_part2", []
        )
        return [c for c in retrieval_cases if c["category"] == category]

    def filter_by_difficulty(self, difficulty: str) -> List[Dict]:
        """按难度筛选"""
        retrieval_cases = self.test_data["retrieval_test_cases"] + self.test_data.get(
            "retrieval_test_cases_part2", []
        )
        return [c for c in retrieval_cases if c["difficulty"] == difficulty]

    def generate_report(self, retrieval_results: Dict, e2e_results: Dict):
        """生成测试报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = (
            Path(__file__).parent / f"test_reports/batch_test_report_{timestamp}.json"
        )
        report_file.parent.mkdir(exist_ok=True)

        report = {
            "timestamp": timestamp,
            "summary": {
                "retrieval": {
                    "total": retrieval_results["total"],
                    "passed": retrieval_results["passed"],
                    "pass_rate": retrieval_results["pass_rate"],
                },
                "end_to_end": {
                    "total": e2e_results["total"],
                    "passed": e2e_results["passed"],
                    "pass_rate": e2e_results["pass_rate"],
                },
                "overall_pass_rate": (
                    retrieval_results["passed"] + e2e_results["passed"]
                )
                / (retrieval_results["total"] + e2e_results["total"]),
            },
            "retrieval_results": retrieval_results,
            "e2e_results": e2e_results,
        }

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n📝 测试报告已保存: {report_file}")
        print("\n" + "=" * 80)
        print("📊 总体测试结果")
        print("=" * 80)
        print(
            f"检索测试: {retrieval_results['passed']}/{retrieval_results['total']} 通过 ({retrieval_results['pass_rate'] * 100:.1f}%)"
        )
        print(
            f"端到端测试: {e2e_results['passed']}/{e2e_results['total']} 通过 ({e2e_results['pass_rate'] * 100:.1f}%)"
        )
        print(f"总体通过率: {report['summary']['overall_pass_rate'] * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="RAG系统批量测试")
    parser.add_argument(
        "--mode", choices=["retrieval", "e2e", "all"], default="all", help="测试模式"
    )
    parser.add_argument("--category", type=str, help="按分类筛选(如: 住宿标准)")
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard"], help="按难度筛选"
    )
    parser.add_argument("--limit", type=int, help="限制测试数量")
    parser.add_argument("--list-categories", action="store_true", help="列出所有分类")

    args = parser.parse_args()

    tester = BatchTester()

    if args.list_categories:
        print("\n📋 可用分类:")
        categories = set()
        for case in tester.test_data["retrieval_test_cases"]:
            categories.add(case["category"])
        for cat in sorted(categories):
            count = len(tester.filter_by_category(cat))
            print(f"  • {cat}: {count} 条")
        return

    if args.category:
        cases = tester.filter_by_category(args.category)
        print(f"\n筛选分类 '{args.category}': {len(cases)} 条测试")
        tester.test_retrieval(cases, args.limit)
    elif args.difficulty:
        cases = tester.filter_by_difficulty(args.difficulty)
        print(f"\n筛选难度 '{args.difficulty}': {len(cases)} 条测试")
        tester.test_retrieval(cases, args.limit)
    elif args.mode == "retrieval":
        cases = tester.test_data["retrieval_test_cases"] + tester.test_data.get(
            "retrieval_test_cases_part2", []
        )
        tester.test_retrieval(cases, args.limit)
    elif args.mode == "e2e":
        tester.test_end_to_end(tester.test_data["end_to_end_test_cases"], args.limit)
    else:
        tester.run_all_tests(args.limit)


if __name__ == "__main__":
    main()
