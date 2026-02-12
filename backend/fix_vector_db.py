#!/usr/bin/env python3
"""
修复向量数据库 - 重新索引已切分的文档
"""

import sys

sys.path.insert(0, "/root/autodl-tmp/rag/backend")

import json
import numpy as np
from pathlib import Path
from services.embedding import embedding_service
from services.vector_db import vector_db_manager
from models import EmbeddingConfig, EmbeddingModelType, VectorDBConfig, VectorDBType

print("=" * 70)
print("🔧 修复向量数据库")
print("=" * 70)

# 1. 加载已有的chunks
vector_db_path = Path("/root/autodl-tmp/rag/backend/vector_db")
chunks_files = list(vector_db_path.glob("chunks_*.json"))

if not chunks_files:
    print("❌ 没有找到切分后的文档")
    sys.exit(1)

print(f"\n📄 找到 {len(chunks_files)} 个chunks文件")

# 2. 初始化嵌入服务
print("\n🤖 初始化嵌入服务...")
config = EmbeddingConfig(
    model_type=EmbeddingModelType.BGE,
    model_name="BAAI/bge-small-zh-v1.5",
    device="cuda",
)

embedding_service.load_model(config)
print("✅ 嵌入服务初始化完成")

# 3. 初始化向量数据库
print("\n💾 初始化向量数据库...")
db_config = VectorDBConfig(
    db_type=VectorDBType.FAISS,
    dimension=512,  # BGE small模型维度
    index_type="HNSW",
)
vector_db_manager.initialize(db_config)
print("✅ 向量数据库初始化完成")

# 4. 重新索引所有chunks
total_chunks = 0
for chunks_file in chunks_files:
    with open(chunks_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    document_id = data.get("document_id", "")
    chunks = data.get("chunks", [])

    if not chunks:
        print(f"⚠️  {document_id} 没有chunks")
        continue

    print(f"\n📑 处理文档: {document_id}")
    print(f"   共 {len(chunks)} 个chunks")

    # 准备文本和元数据
    texts = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        if not content:
            continue

        texts.append(content)
        metadatas.append(
            {
                "document_id": document_id,
                "chunk_id": chunk.get("id", f"chunk_{i}"),
                "chunk_index": i,
                "content": content[:200],  # 存储前200字符作为预览
            }
        )

    if not texts:
        print(f"⚠️  没有有效文本内容")
        continue

    # 编码文本
    print(f"   编码 {len(texts)} 个文本片段...")
    try:
        vectors = embedding_service.encode(texts)
        print(f"   ✅ 生成 {len(vectors)} 个向量 (维度: {len(vectors[0])})")
    except Exception as e:
        print(f"   ❌ 编码失败: {e}")
        continue

    # 添加到向量库
    try:
        vector_db_manager.add_vectors(vectors=vectors, metadata=metadatas)
        total_chunks += len(vectors)
        print(f"   ✅ 成功添加到向量库")
    except Exception as e:
        print(f"   ❌ 添加失败: {e}")

# 5. 保存索引
print("\n💾 保存向量索引...")
vector_db_manager.save()

# 6. 检查最终状态
status = vector_db_manager.get_status()
print("\n" + "=" * 70)
print("📊 修复完成")
print("=" * 70)
print(f"✅ 总向量数: {status.total_vectors}")
print(f"✅ 维度: {status.dimension}")
print(f"✅ 状态: {status.status}")

if status.total_vectors > 0:
    print("\n🎉 向量数据库已修复！现在可以运行测试了")
else:
    print("\n⚠️  向量库仍然为空，请检查日志")
