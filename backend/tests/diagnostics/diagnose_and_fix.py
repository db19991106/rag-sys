"""
诊断和修复检索问题
"""

import requests
import json


def check_vector_db():
    """检查向量数据库状态"""
    print("=" * 60)
    print("🔍 向量数据库诊断")
    print("=" * 60)

    # 1. 检查状态
    resp = requests.get("http://localhost:9000/vector-db/status")
    status = resp.json()
    print(f"\n📊 当前状态:")
    print(f"   总向量数: {status['total_vectors']}")
    print(f"   维度: {status['dimension']}")
    print(f"   类型: {status['db_type']}")

    # 2. 检查文档
    resp = requests.get("http://localhost:9000/vector-db/documents")
    docs = resp.json()
    print(f"\n📄 已索引文档:")
    if docs.get("data", {}).get("documents"):
        for doc in docs["data"]["documents"]:
            print(f"   - {doc['document_name']}: {doc['chunk_count']}个片段")
    else:
        print("   ⚠️ 没有文档")

    return status["total_vectors"]


def test_retrieval(query):
    """测试检索"""
    print(f"\n🔎 测试检索: '{query}'")
    resp = requests.post(
        "http://localhost:9000/retrieval/search",
        json={
            "query": query,
            "config": {
                "top_k": 5,
                "similarity_threshold": 0.3,  # 降低阈值
                "algorithm": "cosine",
            },
        },
    )
    result = resp.json()
    print(f"   找到 {result['total']} 条结果")
    if result["results"]:
        for r in result["results"]:
            print(f"   - {r['document_name']}: 相似度{r['similarity']:.2f}")
    return result["total"]


def reindex_document(doc_id, doc_name):
    """重新索引文档"""
    print(f"\n🔄 重新索引: {doc_name}")
    resp = requests.post(f"http://localhost:9000/chunking/embed?doc_id={doc_id}")
    if resp.status_code == 200:
        print(f"   ✅ 成功: {resp.json().get('message', '')}")
        return True
    else:
        print(f"   ❌ 失败: {resp.status_code}")
        return False


def main():
    # 检查当前状态
    total_vectors = check_vector_db()

    # 测试检索
    result1 = test_retrieval("通讯费报销")
    result2 = test_retrieval("通讯费补贴")

    if result1 == 0 and result2 == 0:
        print("\n" + "=" * 60)
        print("⚠️ 发现问题：检索返回空结果，需要重新索引文档")
        print("=" * 60)

        # 获取所有已索引文档
        resp = requests.get("http://localhost:9000/documents/list")
        docs = resp.json()

        indexed_docs = [
            (d["id"], d["name"]) for d in docs if d.get("status") == "indexed"
        ]

        print(f"\n📋 发现 {len(indexed_docs)} 个已标记为indexed的文档")
        print("   这些文档需要重新向量化...")

        # 重新索引
        success_count = 0
        for doc_id, doc_name in indexed_docs:
            if reindex_document(doc_id, doc_name):
                success_count += 1

        print(f"\n✅ 重新索引完成: {success_count}/{len(indexed_docs)} 个文档")

        # 再次检查
        print("\n" + "=" * 60)
        print("🧪 验证修复结果")
        print("=" * 60)
        check_vector_db()
        test_retrieval("通讯费报销")
        test_retrieval("主管报销标准")


if __name__ == "__main__":
    main()
