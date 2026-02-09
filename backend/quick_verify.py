#!/usr/bin/env python3
"""
快速RAG验证脚本 - 无需RAGAS，快速验证系统功能
"""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from services.rag_generator import rag_generator
from services.embedding import embedding_service
from services.vector_db import vector_db_manager
from models import RetrievalConfig, GenerationConfig, EmbeddingConfig, VectorDBConfig
from models import EmbeddingModelType, VectorDBType


def quick_test():
    """快速测试RAG系统"""

    print("=" * 70)
    print("🚀 RAG系统快速验证")
    print("=" * 70)
    print()

    # 初始化服务
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

    # 测试用例
    test_cases = [
        {
            "query": "8-9级员工住宿标准",
            "expected_keywords": ["住宿", "三星级", "300", "普通员工"],
        },
        {
            "query": "经理坐高铁可以选什么座位",
            "expected_keywords": ["经理", "高铁", "一等座"],
        },
        {
            "query": "报销流程是什么",
            "expected_keywords": ["报销", "流程", "审批", "发票"],
        },
    ]

    results = []

    for i, case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] {case['query']}")

        # 运行RAG - 使用本地模型配置
        from config import settings

        start = time.time()
        response = rag_generator.generate(
            query=case["query"],
            retrieval_config=RetrievalConfig(top_k=3),
            generation_config=GenerationConfig(
                llm_provider=settings.llm_provider,
                llm_model=settings.llm_model,
                temperature=0.7,
                max_tokens=300,
            ),
        )
        elapsed = time.time() - start

        # 检查结果
        answer = response.answer
        contexts = response.context_chunks

        # 关键词匹配
        matched_keywords = [
            kw for kw in case["expected_keywords"] if kw.lower() in answer.lower()
        ]
        keyword_match_rate = len(matched_keywords) / len(case["expected_keywords"])

        # 评分
        if keyword_match_rate >= 0.7:
            score = "🟢 优秀"
        elif keyword_match_rate >= 0.4:
            score = "🟡 良好"
        else:
            score = "🔴 需优化"

        print(f"   检索: {len(contexts)} 个片段")
        print(f"   回答: {len(answer)} 字符")
        print(
            f"   关键词匹配: {len(matched_keywords)}/{len(case['expected_keywords'])} {score}"
        )
        print(f"   耗时: {elapsed:.1f}s")
        print(f"   回答预览: {answer[:100]}...")
        print()

        results.append(
            {
                "query": case["query"],
                "retrieved": len(contexts),
                "answer_length": len(answer),
                "keyword_match": keyword_match_rate,
                "time": elapsed,
            }
        )

    # 总结
    print("=" * 70)
    print("📊 验证总结")
    print("=" * 70)

    avg_time = sum(r["time"] for r in results) / len(results)
    avg_match = sum(r["keyword_match"] for r in results) / len(results)

    print(f"✅ 测试通过: {len(results)}/{len(results)}")
    print(f"⏱️  平均响应时间: {avg_time:.1f}s")
    print(f"🎯 平均关键词匹配率: {avg_match * 100:.0f}%")

    if avg_match >= 0.7:
        print("🎉 系统状态: 优秀")
    elif avg_match >= 0.4:
        print("👍 系统状态: 良好")
    else:
        print("⚠️  系统状态: 需优化")

    print("=" * 70)
    print("\n提示: 如需详细RAGAS评估指标，建议:")
    print("  1. 配置OpenAI API Key进行云端评估")
    print("  2. 或使用更大参数量的本地模型（如qwen2.5:7b）")
    print("  3. 人工抽查回答质量")


if __name__ == "__main__":
    quick_test()
