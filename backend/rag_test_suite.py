#!/usr/bin/env python3
"""
RAG系统综合测试套件
一站式测试框架，涵盖功能、性能、效果和端到端测试
"""

import sys
import os
import time
import json
import asyncio
import unittest
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from services.retriever import retriever
from services.embedding import embedding_service
from services.vector_db import vector_db_manager
from services.reranker import reranker_manager
from services.intent_recognizer import intent_recognizer
from services.evaluation import evaluator
from services.rag_generator import rag_generator
from services.chunker import RAGFlowChunker
from services.document_parser import DocumentParser
from models import (
    RetrievalConfig,
    EmbeddingConfig,
    VectorDBConfig,
    GenerationConfig,
    EmbeddingModelType,
    VectorDBType,
    SimilarityAlgorithm,
    ChunkConfig,
    ChunkType,
)


class RAGTestSuite:
    """RAG系统测试套件"""

    def __init__(self):
        self.results = {}
        self.test_data_dir = Path(__file__).parent / "test_data"
        self.test_data_dir.mkdir(exist_ok=True)

    # ==================== 1. 功能测试 ====================

    def test_document_parser(self):
        """测试文档解析功能"""
        print("\n" + "=" * 70)
        print("【测试1】文档解析功能测试")
        print("=" * 70)

        parser = DocumentParser()
        test_cases = []

        # 创建测试文件
        test_files = {
            "txt": ("test.txt", "这是一个测试文本文件。\n包含多行内容。"),
            "md": ("test.md", "# 标题\n\n这是正文内容。"),
            "json": ("test.json", '{"key": "value", "number": 123}'),
        }

        results = []
        for ext, (filename, content) in test_files.items():
            test_file = self.test_data_dir / filename
            test_file.write_text(content, encoding="utf-8")

            try:
                result = parser.parse(str(test_file))
                success = result is not None and len(result) > 0
                results.append(
                    {
                        "format": ext,
                        "success": success,
                        "content_length": len(result) if result else 0,
                    }
                )
                status = "✅" if success else "❌"
                print(f"{status} {ext.upper()}解析: {'成功' if success else '失败'}")
            except Exception as e:
                results.append({"format": ext, "success": False, "error": str(e)})
                print(f"❌ {ext.upper()}解析: 异常 - {e}")
            finally:
                test_file.unlink(missing_ok=True)

        success_rate = sum(1 for r in results if r["success"]) / len(results)
        self.results["document_parser"] = {
            "success_rate": success_rate,
            "details": results,
        }
        print(f"\n解析成功率: {success_rate * 100:.1f}%")
        return success_rate >= 0.8

    def test_chunking_strategies(self):
        """测试文档切分策略"""
        print("\n" + "=" * 70)
        print("【测试2】文档切分策略测试")
        print("=" * 70)

        chunker = RAGFlowChunker()

        # 测试内容
        test_content = """
# 测试文档

## 第一章 总则

这是第一章的内容。

### 1.1 概述
详细说明内容。

### 1.2 规定
具体规定内容。

## 第二章 实施细则

这是第二章的内容。

| 项目 | 标准 |
|------|------|
| A | 100 |
| B | 200 |
"""

        strategies = [
            (
                "INTELLIGENT",
                ChunkConfig(type=ChunkType.INTELLIGENT, chunk_token_size=512),
            ),
            ("NAIVE", ChunkConfig(type=ChunkType.NAIVE, chunk_token_size=512)),
        ]

        results = []
        for name, config in strategies:
            try:
                chunks = chunker.chunk(test_content, "test_doc", config)
                results.append(
                    {
                        "strategy": name,
                        "success": len(chunks) > 0,
                        "chunk_count": len(chunks),
                        "avg_chunk_size": sum(len(c.content) for c in chunks)
                        / len(chunks)
                        if chunks
                        else 0,
                    }
                )
                print(f"✅ {name}策略: 生成{len(chunks)}个chunk")
            except Exception as e:
                results.append({"strategy": name, "success": False, "error": str(e)})
                print(f"❌ {name}策略: 失败 - {e}")

        success = all(r["success"] for r in results)
        self.results["chunking"] = {"success": success, "details": results}
        return success

    def test_embedding_service(self):
        """测试嵌入服务"""
        print("\n" + "=" * 70)
        print("【测试3】嵌入服务测试")
        print("=" * 70)

        if not embedding_service.is_loaded():
            print("⚠️ 嵌入服务未加载，尝试初始化...")
            try:
                config = EmbeddingConfig(
                    model_type=EmbeddingModelType.BGE,
                    model_name="BAAI/bge-small-zh-v1.5",
                    device="cpu",
                    batch_size=8,
                )
                response = embedding_service.load_model(config)
                if response.status != "success":
                    print(f"❌ 嵌入服务初始化失败: {response.message}")
                    return False
            except Exception as e:
                print(f"❌ 嵌入服务初始化异常: {e}")
                return False

        # 测试编码
        test_texts = ["这是一个测试句子", "RAG系统测试", "向量嵌入测试"]

        try:
            start_time = time.time()
            embeddings = embedding_service.encode(test_texts)
            encode_time = time.time() - start_time

            success = embeddings.shape[0] == len(test_texts)
            dimension = embeddings.shape[1] if success else 0

            self.results["embedding"] = {
                "success": success,
                "dimension": dimension,
                "encode_time": encode_time,
                "avg_time_per_text": encode_time / len(test_texts),
            }

            print(f"✅ 编码成功: {len(test_texts)}个文本, 维度{dimension}")
            print(
                f"✅ 编码耗时: {encode_time * 1000:.2f}ms ({encode_time * 1000 / len(test_texts):.2f}ms/文本)"
            )
            return success
        except Exception as e:
            print(f"❌ 编码失败: {e}")
            self.results["embedding"] = {"success": False, "error": str(e)}
            return False

    def test_vector_db(self):
        """测试向量数据库"""
        print("\n" + "=" * 70)
        print("【测试4】向量数据库测试")
        print("=" * 70)

        if not embedding_service.is_loaded():
            print("⚠️ 嵌入服务未加载，跳过向量数据库测试")
            return False

        # 初始化向量数据库
        try:
            dimension = embedding_service.get_dimension()
            config = VectorDBConfig(
                db_type=VectorDBType.FAISS, dimension=dimension, index_type="HNSW"
            )
            success = vector_db_manager.initialize(config)
            if not success:
                print("❌ 向量数据库初始化失败")
                return False
        except Exception as e:
            print(f"❌ 向量数据库初始化异常: {e}")
            return False

        # 测试添加向量
        test_texts = ["测试文档1", "测试文档2", "测试文档3"]
        try:
            vectors = embedding_service.encode(test_texts)
            metadata = [
                {"text": t, "id": f"test_{i}"} for i, t in enumerate(test_texts)
            ]
            vector_db_manager.add_vectors(vectors, metadata)
            print(f"✅ 添加向量成功: {len(test_texts)}个")
        except Exception as e:
            print(f"❌ 添加向量失败: {e}")
            return False

        # 测试搜索
        try:
            query_vector = embedding_service.encode(["测试查询"])
            distances, results = vector_db_manager.search(query_vector, top_k=3)

            self.results["vector_db"] = {
                "success": True,
                "total_vectors": vector_db_manager.get_status().total_vectors,
                "search_results_count": len(results[0]) if results else 0,
            }

            print(f"✅ 搜索成功: 返回{len(results[0]) if results else 0}个结果")
            return True
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            self.results["vector_db"] = {"success": False, "error": str(e)}
            return False

    # ==================== 2. 效果测试 ====================

    def test_retrieval_quality(self):
        """测试检索质量"""
        print("\n" + "=" * 70)
        print("【测试5】检索质量测试（需要预设测试集）")
        print("=" * 70)

        # 准备测试查询和期望结果
        test_cases = [
            {
                "query": "8-9级员工出差住宿标准",
                "expected_keywords": ["住宿", "标准", "员工"],
            },
            {"query": "差旅费报销流程", "expected_keywords": ["报销", "差旅", "流程"]},
            {
                "query": "经理级别交通费用",
                "expected_keywords": ["经理", "交通", "费用"],
            },
        ]

        results = []
        config = RetrievalConfig(top_k=5)

        for case in test_cases:
            try:
                response = retriever.retrieve(case["query"], config)
                retrieved_texts = [r.content for r in response.results]

                # 检查关键词命中率
                hits = 0
                for keyword in case["expected_keywords"]:
                    if any(keyword in text for text in retrieved_texts):
                        hits += 1

                hit_rate = hits / len(case["expected_keywords"])
                results.append(
                    {
                        "query": case["query"],
                        "hit_rate": hit_rate,
                        "results_count": len(response.results),
                    }
                )

                status = "✅" if hit_rate >= 0.6 else "⚠️"
                print(
                    f"{status} 查询: {case['query'][:30]}... 命中率: {hit_rate * 100:.0f}%"
                )
            except Exception as e:
                results.append({"query": case["query"], "error": str(e)})
                print(f"❌ 查询失败: {case['query'][:30]}... - {e}")

        avg_hit_rate = np.mean([r["hit_rate"] for r in results if "hit_rate" in r])
        self.results["retrieval_quality"] = {
            "avg_hit_rate": avg_hit_rate,
            "details": results,
        }

        print(f"\n平均关键词命中率: {avg_hit_rate * 100:.1f}%")
        return avg_hit_rate >= 0.5

    def test_end_to_end(self):
        """端到端测试"""
        print("\n" + "=" * 70)
        print("【测试6】端到端RAG测试")
        print("=" * 70)

        test_queries = [
            "什么是RAG技术",
            "如何优化检索性能",
        ]

        results = []
        for query in test_queries:
            try:
                start_time = time.time()

                # 执行完整RAG流程
                retrieval_config = RetrievalConfig(top_k=3)
                generation_config = GenerationConfig(
                    llm_provider="local", temperature=0.7, max_tokens=300
                )

                response = rag_generator.generate(
                    query, retrieval_config, generation_config
                )
                total_time = time.time() - start_time

                success = len(response.answer) > 50  # 回答长度检查
                results.append(
                    {
                        "query": query,
                        "success": success,
                        "total_time": total_time,
                        "retrieval_time": response.retrieval_time_ms / 1000,
                        "generation_time": response.generation_time_ms / 1000,
                        "answer_length": len(response.answer),
                        "sources_count": len(response.sources),
                    }
                )

                status = "✅" if success else "❌"
                print(f"{status} 查询: {query[:30]}...")
                print(
                    f"   总耗时: {total_time * 1000:.0f}ms (检索{response.retrieval_time_ms:.0f}ms + 生成{response.generation_time_ms:.0f}ms)"
                )
                print(
                    f"   回答长度: {len(response.answer)}字符, 来源: {len(response.sources)}个"
                )
            except Exception as e:
                results.append({"query": query, "success": False, "error": str(e)})
                print(f"❌ 查询失败: {query[:30]}... - {e}")

        success_rate = sum(1 for r in results if r.get("success")) / len(results)
        self.results["end_to_end"] = {"success_rate": success_rate, "details": results}

        return success_rate >= 0.5

    # ==================== 3. 性能测试 ====================

    def test_retrieval_performance(self):
        """测试检索性能"""
        print("\n" + "=" * 70)
        print("【测试7】检索性能测试")
        print("=" * 70)

        test_queries = [
            "什么是人工智能",
            "如何使用Python",
            "机器学习算法",
            "深度学习原理",
            "数据分析方法",
        ] * 4  # 20次查询

        config = RetrievalConfig(top_k=5)
        times = []

        # 预热
        retriever.retrieve("预热查询", config)

        for query in test_queries:
            start = time.time()
            retriever.retrieve(query, config)
            times.append((time.time() - start) * 1000)

        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        p99_time = np.percentile(times, 99)

        self.results["retrieval_performance"] = {
            "avg_time_ms": avg_time,
            "p95_time_ms": p95_time,
            "p99_time_ms": p99_time,
            "throughput_qps": 1000 / avg_time if avg_time > 0 else 0,
        }

        print(f"✅ 平均响应时间: {avg_time:.2f}ms")
        print(f"✅ P95响应时间: {p95_time:.2f}ms")
        print(f"✅ P99响应时间: {p99_time:.2f}ms")
        print(f"✅ 估算吞吐量: {1000 / avg_time:.1f} QPS")

        return avg_time < 1000  # 平均1秒内

    def test_concurrent_performance(self):
        """测试并发性能"""
        print("\n" + "=" * 70)
        print("【测试8】并发性能测试")
        print("=" * 70)

        import concurrent.futures

        test_queries = ["查询" + str(i) for i in range(20)]
        concurrent_users = 5

        def worker_task(queries):
            times = []
            config = RetrievalConfig(top_k=3)
            for query in queries:
                start = time.time()
                try:
                    retriever.retrieve(query, config)
                    times.append((time.time() - start) * 1000)
                except:
                    times.append(-1)
            return times

        # 分配查询给每个worker
        queries_per_worker = len(test_queries) // concurrent_users
        worker_queries = [
            test_queries[i * queries_per_worker : (i + 1) * queries_per_worker]
            for i in range(concurrent_users)
        ]

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrent_users
        ) as executor:
            futures = [executor.submit(worker_task, q) for q in worker_queries]
            all_times = []
            for future in concurrent.futures.as_completed(futures):
                all_times.extend(future.result())

        total_time = time.time() - start_time
        valid_times = [t for t in all_times if t > 0]

        throughput = len(test_queries) / total_time
        avg_time = np.mean(valid_times) if valid_times else 0

        self.results["concurrent_performance"] = {
            "concurrent_users": concurrent_users,
            "total_requests": len(test_queries),
            "total_time": total_time,
            "throughput_qps": throughput,
            "avg_response_time_ms": avg_time,
        }

        print(f"✅ 并发用户数: {concurrent_users}")
        print(f"✅ 总请求数: {len(test_queries)}")
        print(f"✅ 总耗时: {total_time:.2f}s")
        print(f"✅ 吞吐量: {throughput:.1f} QPS")
        print(f"✅ 平均响应时间: {avg_time:.2f}ms")

        return throughput > 5  # 至少5 QPS

    # ==================== 4. 运行所有测试 ====================

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "=" * 70)
        print("RAG系统综合测试套件")
        print("=" * 70)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        test_methods = [
            ("文档解析", self.test_document_parser),
            ("文档切分", self.test_chunking_strategies),
            ("嵌入服务", self.test_embedding_service),
            ("向量数据库", self.test_vector_db),
            ("检索质量", self.test_retrieval_quality),
            ("端到端测试", self.test_end_to_end),
            ("检索性能", self.test_retrieval_performance),
            ("并发性能", self.test_concurrent_performance),
        ]

        passed = 0
        failed = 0

        for name, test_func in test_methods:
            try:
                result = test_func()
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"\n❌ {name}测试异常: {e}")
                failed += 1

        # 生成报告
        self._generate_report(passed, failed)

        return passed, failed

    def _generate_report(self, passed, failed):
        """生成测试报告"""
        print("\n" + "=" * 70)
        print("测试报告")
        print("=" * 70)
        print(f"通过: {passed}")
        print(f"失败: {failed}")
        print(f"成功率: {passed / (passed + failed) * 100:.1f}%")
        print("=" * 70)

        # 保存详细报告
        report_file = self.test_data_dir / f"test_report_{int(time.time())}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "summary": {"passed": passed, "failed": failed},
                    "results": self.results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\n详细报告已保存: {report_file}")


# ==================== 快速测试函数 ====================


def quick_test():
    """快速测试 - 只测试核心功能"""
    suite = RAGTestSuite()

    print("\n🚀 快速测试模式（核心功能）\n")

    tests = [
        ("文档解析", suite.test_document_parser),
        ("嵌入服务", suite.test_embedding_service),
        ("向量数据库", suite.test_vector_db),
        ("检索质量", suite.test_retrieval_quality),
    ]

    passed = 0
    for name, test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {name}测试异常: {e}")

    print(f"\n快速测试完成: {passed}/{len(tests)}通过")
    return passed == len(tests)


def benchmark_test():
    """基准测试 - 重点测试性能"""
    suite = RAGTestSuite()

    print("\n⚡ 基准测试模式（性能测试）\n")

    # 确保服务已初始化
    suite.test_embedding_service()
    suite.test_vector_db()

    # 性能测试
    suite.test_retrieval_performance()
    suite.test_concurrent_performance()

    print("\n基准测试完成")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG系统测试套件")
    parser.add_argument(
        "--mode",
        choices=["full", "quick", "benchmark"],
        default="full",
        help="测试模式",
    )
    args = parser.parse_args()

    if args.mode == "full":
        suite = RAGTestSuite()
        suite.run_all_tests()
    elif args.mode == "quick":
        quick_test()
    elif args.mode == "benchmark":
        benchmark_test()
