#!/usr/bin/env python3
"""
深度分析FAISS索引问题
"""

import sys

sys.path.insert(0, "/root/autodl-tmp/rag/backend")

import faiss
import numpy as np
from services.vector_db import vector_db_manager
from services.embedding import embedding_service

print("=" * 70)
print("🔬 FAISS索引深度分析")
print("=" * 70)

# 1. 检查FAISS索引类型
print("\n1. FAISS索引信息:")
if vector_db_manager.db and vector_db_manager.db.index:
    index = vector_db_manager.db.index
    print(f"   索引类型: {type(index).__name__}")
    print(f"   是否训练: {index.is_trained}")
    print(f"   维度: {index.d}")
    print(f"   向量数: {index.ntotal}")

    # 检查HNSW参数
    if hasattr(index, "hnsw"):
        print(f"   HNSW参数:")
        print(f"     - M: {index.hnsw.M}")
        print(f"     - efConstruction: {index.hnsw.efConstruction}")
        print(f"     - efSearch: {index.hnsw.efSearch}")
else:
    print("   ❌ 索引未初始化")
    sys.exit(1)

# 2. 获取数据库中的一个向量样本
print("\n2. 向量样本分析:")
if vector_db_manager.db.metadata:
    sample_key = list(vector_db_manager.db.metadata.keys())[0]
    sample_meta = vector_db_manager.db.metadata[sample_key]
    print(f"   样本Key: {sample_key}")
    print(
        f"   样本文档: {sample_meta.get('document_name', 'Unknown') if isinstance(sample_meta, dict) else 'N/A'}"
    )

    # 从FAISS中重建向量
    try:
        sample_vector = index.reconstruct(int(sample_key))
        print(f"   向量维度: {len(sample_vector)}")
        print(f"   向量范数: {np.linalg.norm(sample_vector):.6f}")
        print(f"   向量前5个值: {sample_vector[:5]}")

        # 检查是否归一化
        norm = np.linalg.norm(sample_vector)
        is_normalized = abs(norm - 1.0) < 0.01
        print(f"   是否归一化: {is_normalized} (范数={norm:.6f})")
    except Exception as e:
        print(f"   ❌ 重建向量失败: {e}")

# 3. 测试查询向量
print("\n3. 查询向量分析:")
query = "通讯费报销"
query_vector = embedding_service.encode([query])[0]
print(f"   查询: '{query}'")
print(f"   向量维度: {len(query_vector)}")
print(f"   向量范数: {np.linalg.norm(query_vector):.6f}")
print(f"   向量前5个值: {query_vector[:5]}")

norm = np.linalg.norm(query_vector)
is_normalized = abs(norm - 1.0) < 0.01
print(f"   是否归一化: {is_normalized} (范数={norm:.6f})")

# 4. 执行搜索并分析距离
print("\n4. 搜索结果分析:")
if query_vector.ndim == 1:
    query_vector = query_vector.reshape(1, -1)

distances, indices = index.search(query_vector.astype("float32"), k=5)
print(f"   返回距离: {distances[0]}")
print(f"   返回索引: {indices[0]}")

# 5. 计算相似度
print("\n5. 相似度计算:")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    if idx >= 0:
        # 假设是归一化向量，使用余弦相似度公式
        cosine_sim = 1 - (dist**2) / 2
        cosine_sim = max(0.0, min(1.0, cosine_sim))
        print(
            f"   结果{i + 1}: 索引={idx}, 距离={dist:.6f}, 余弦相似度={cosine_sim:.6f}"
        )

        # 检查原始向量
        try:
            doc_vector = index.reconstruct(int(idx))
            doc_norm = np.linalg.norm(doc_vector)
            query_norm = np.linalg.norm(query_vector[0])
            actual_cosine = np.dot(doc_vector, query_vector[0]) / (
                doc_norm * query_norm
            )
            print(f"            实际余弦相似度: {actual_cosine:.6f}")
            print(f"            文档范数: {doc_norm:.6f}")
        except Exception as e:
            print(f"            无法重建: {e}")

# 6. 检查元数据
print("\n6. 检查元数据对应:")
for idx in indices[0][:3]:
    if idx >= 0:
        meta = vector_db_manager.db.metadata.get(str(idx), {})
        if isinstance(meta, dict):
            print(f"   索引{idx}: {meta.get('document_name', 'Unknown')[:40]}")
        else:
            print(f"   索引{idx}: 元数据格式错误 - {type(meta)}")

print("\n" + "=" * 70)
print("💡 诊断结论:")
print("=" * 70)
