#!/usr/bin/env python3
"""
性能测试模块
"""

import sys
import time
import asyncio
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.retriever import retriever
from services.rag_generator import rag_generator
from services.document_manager import document_manager
from models import RetrievalConfig, GenerationConfig


class TestPerformance:
    """测试系统性能"""

    def test_retrieval_performance(self):
        """测试检索性能"""
        print("=" * 70)
        print("测试检索性能")
        print("=" * 70)
        
        config = RetrievalConfig(
            top_k=5,
            similarity_threshold=0.6
        )
        
        # 预热
        print("预热中...")
        retriever.retrieve("测试预热", config)
        
        # 测试多次检索
        test_queries = [
            "什么是RAG技术？",
            "如何优化检索性能？",
            "向量数据库的工作原理是什么？",
            "如何提高生成质量？",
            "系统架构设计"
        ]
        
        times = []
        for i, query in enumerate(test_queries, 1):
            start_time = time.time()
            response = retriever.retrieve(query, config)
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            times.append(duration)
            
            print(f"查询 {i}/{len(test_queries)}: {query}")
            print(f"  耗时: {duration:.2f}ms")
            print(f"  结果数: {len(response.results)}")
            print()
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        print("=" * 70)
        print("检索性能测试结果:")
        print(f"平均响应时间: {avg_time:.2f}ms")
        print(f"最大响应时间: {max_time:.2f}ms")
        print(f"最小响应时间: {min_time:.2f}ms")
        print()
        
        # 性能评估
        if avg_time < 500:
            print("✅ 检索性能优秀")
        elif avg_time < 1000:
            print("✅ 检索性能良好")
        else:
            print("⚠️  检索性能需要优化")
        print("=" * 70)
        print()

    def test_generation_performance(self):
        """测试生成性能"""
        print("=" * 70)
        print("测试生成性能")
        print("=" * 70)
        
        retrieval_config = RetrievalConfig(
            top_k=3,
            similarity_threshold=0.6
        )
        
        generation_config = GenerationConfig(
            llm_provider="local",
            llm_model="Qwen2.5-7B-Instruct",
            temperature=0.7,
            max_tokens=500
        )
        
        # 预热
        print("预热中...")
        rag_generator.generate("测试预热", retrieval_config, generation_config)
        
        # 测试多次生成
        test_queries = [
            "什么是RAG技术？",
            "如何优化检索性能？",
            "向量数据库的工作原理是什么？"
        ]
        
        times = []
        for i, query in enumerate(test_queries, 1):
            start_time = time.time()
            response = rag_generator.generate(query, retrieval_config, generation_config)
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            times.append(duration)
            
            print(f"查询 {i}/{len(test_queries)}: {query}")
            print(f"  总耗时: {duration:.2f}ms")
            print(f"  检索耗时: {response.retrieval_time_ms:.2f}ms")
            print(f"  生成耗时: {response.generation_time_ms:.2f}ms")
            print(f"  回答长度: {len(response.answer)}字符")
            print()
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        print("=" * 70)
        print("生成性能测试结果:")
        print(f"平均响应时间: {avg_time:.2f}ms")
        print(f"最大响应时间: {max_time:.2f}ms")
        print(f"最小响应时间: {min_time:.2f}ms")
        print()
        
        # 性能评估
        if avg_time < 5000:
            print("✅ 生成性能优秀")
        elif avg_time < 10000:
            print("✅ 生成性能良好")
        else:
            print("⚠️  生成性能需要优化")
        print("=" * 70)
        print()

    async def test_concurrent_performance(self):
        """测试并发性能"""
        print("=" * 70)
        print("测试并发性能")
        print("=" * 70)
        
        retrieval_config = RetrievalConfig(
            top_k=3,
            similarity_threshold=0.6
        )
        
        generation_config = GenerationConfig(
            llm_provider="local",
            llm_model="Qwen2.5-7B-Instruct",
            temperature=0.7,
            max_tokens=300
        )
        
        # 预热
        print("预热中...")
        rag_generator.generate("测试预热", retrieval_config, generation_config)
        
        # 并发测试
        concurrency_levels = [5, 10, 20]
        
        for level in concurrency_levels:
            print(f"测试并发数: {level}")
            print("-" * 70)
            
            async def test_task(i):
                query = f"测试并发查询 {i}"
                start_time = time.time()
                response = rag_generator.generate(query, retrieval_config, generation_config)
                end_time = time.time()
                return (end_time - start_time) * 1000
            
            tasks = [test_task(i) for i in range(level)]
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = (time.time() - start_time) * 1000
            
            avg_time = sum(results) / len(results)
            max_time = max(results)
            min_time = min(results)
            
            print(f"总耗时: {total_time:.2f}ms")
            print(f"平均响应时间: {avg_time:.2f}ms")
            print(f"最大响应时间: {max_time:.2f}ms")
            print(f"最小响应时间: {min_time:.2f}ms")
            print(f"吞吐量: {level / (total_time / 1000):.2f} QPS")
            print()
        
        print("=" * 70)
        print("并发性能测试完成")
        print("=" * 70)
        print()

    async def test_document_upload_performance(self):
        """测试文档上传性能"""
        print("=" * 70)
        print("测试文档上传性能")
        print("=" * 70)
        
        # 测试不同大小的文档
        test_sizes = [
            (10, "10KB"),
            (100, "100KB"),
            (500, "500KB")
        ]
        
        for size_kb, size_label in test_sizes:
            test_content = b"x" * (size_kb * 1024)
            test_filename = f"test_{size_label}.txt"
            
            print(f"测试文档大小: {size_label}")
            
            start_time = time.time()
            response = await document_manager.upload_document(test_filename, test_content)
            end_time = time.time()
            duration = (end_time - start_time) * 1000
            
            print(f"  耗时: {duration:.2f}ms")
            print(f"  状态: {response.status.value}")
            print()
            
            # 清理
            document_manager.delete_document(response.id)
        
        print("=" * 70)
        print("文档上传性能测试完成")
        print("=" * 70)
        print()


if __name__ == "__main__":
    tester = TestPerformance()
    
    # 运行所有测试
    tester.test_retrieval_performance()
    tester.test_generation_performance()
    
    # 运行异步测试
    async def run_async_tests():
        await tester.test_document_upload_performance()
        await tester.test_concurrent_performance()
    
    asyncio.run(run_async_tests())
    
    print("=" * 70)
    print("🎉 所有性能测试完成!")
    print("=" * 70)
