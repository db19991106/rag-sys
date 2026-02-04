"""
深度诊断检索问题
"""

import requests
import json
import numpy as np


def debug_retrieval():
    """深度调试检索过程"""
    print("=" * 70)
    print("🔍 深度诊断检索问题")
    print("=" * 70)

    # 1. 测试简单的关键词检索
    test_queries = [
        "通讯费",
        "报销",
        "通讯费报销",
        "主管 报销 标准",
        "150元",
        "手机费",
        "通信补贴",
    ]

    print("\n📊 测试不同查询词:")
    for query in test_queries:
        resp = requests.post(
            "http://localhost:8000/retrieval/search",
            json={
                "query": query,
                "config": {
                    "top_k": 5,
                    "similarity_threshold": 0.0,  # 不设阈值，看所有结果
                    "algorithm": "cosine",
                },
            },
        )
        result = resp.json()
        print(f"   '{query}': {result['total']} 条结果")
        if result["results"]:
            for r in result["results"][:2]:
                print(f"      → {r['document_name']}: {r['similarity']:.3f}")

    # 2. 检查baoxiao.md的内容
    print("\n📄 检查baoxiao.md的实际内容:")
    try:
        with open("/root/autodl-tmp/rag/backend/data/docs/baoxiao.md", "r") as f:
            content = f.read()
            # 查找通讯费相关内容
            import re

            matches = re.findall(
                r"[#\*\-].*?(?:通讯|通信|手机|电话).*?(?:费|补贴|报销).*",
                content,
                re.IGNORECASE,
            )
            print(f"   找到 {len(matches)} 处通讯费相关内容:")
            for i, m in enumerate(matches[:5], 1):
                print(f"      {i}. {m[:80]}...")
    except Exception as e:
        print(f"   ❌ 读取失败: {e}")

    # 3. 直接测试向量搜索
    print("\n🔎 测试底层向量搜索:")
    from services.vector_db import vector_db_manager
    from services.embedding import embedding_service

    if embedding_service.is_loaded() and vector_db_manager.db:
        query = "通讯费报销标准"
        query_vector = embedding_service.encode([query])[0]

        print(f"   查询向量维度: {len(query_vector)}")
        print(f"   查询向量前5个值: {query_vector[:5]}")
        print(f"   查询向量范数: {np.linalg.norm(query_vector):.4f}")

        # 执行搜索
        distances, metadata_list = vector_db_manager.search(query_vector, top_k=5)
        print(f"\n   FAISS返回结果:")
        print(f"   - 距离值: {distances[0][:5] if len(distances) > 0 else 'N/A'}")
        print(f"   - 结果数: {len(distances[0]) if len(distances) > 0 else 0}")

        if len(distances) > 0 and len(distances[0]) > 0:
            print(f"\n   原始距离值分析:")
            for i, (dist, meta) in enumerate(
                zip(distances[0][:3], metadata_list[0][:3])
            ):
                print(f"      结果{i + 1}: 距离={dist:.4f}")
                # 计算理论相似度
                cosine_sim = 1 - (dist**2) / 2
                print(f"              余弦相似度={max(0, min(1, cosine_sim)):.4f}")
                if isinstance(meta, dict):
                    print(
                        f"              文档: {meta.get('document_name', 'N/A')[:30]}"
                    )
    else:
        print("   ⚠️ 服务未加载，跳过向量测试")

    print("\n" + "=" * 70)
    print("💡 诊断结论:")
    print("=" * 70)


if __name__ == "__main__":
    debug_retrieval()
