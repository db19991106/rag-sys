#!/usr/bin/env python3
"""
手动处理文档并添加到向量库
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from services.document_manager import document_manager
from services.document_parser import DocumentParser
from services.chunker import RAGFlowChunker
from services.embedding import embedding_service
from services.vector_db import vector_db_manager
from models import ChunkConfig, ChunkType, EmbeddingConfig, VectorDBConfig
from models import EmbeddingModelType, VectorDBType


async def process_baoxiao_document():
    """处理报销文档并添加到向量库"""

    print("=" * 60)
    print("📄 处理差旅费报销文档")
    print("=" * 60)
    print()

    # 1. 读取文档
    doc_path = Path("./data/docs/baoxiao.md")
    print(f"1️⃣ 读取文档: {doc_path}")
    content = doc_path.read_text(encoding="utf-8")
    print(f"   文档大小: {len(content)} 字符")
    print()

    # 2. 解析文档（已经是文本，直接读取）
    print("2️⃣ 解析文档")
    parser = DocumentParser()
    parsed_content = parser.parse(str(doc_path))
    print(f"   解析后大小: {len(parsed_content)} 字符")
    print()

    # 3. 文档切分
    print("3️⃣ 文档切分")
    chunker = RAGFlowChunker()
    config = ChunkConfig(type=ChunkType.INTELLIGENT, chunk_token_size=512)
    doc_id = "baoxiao_001"
    chunks = chunker.chunk(parsed_content, doc_id, config)
    print(f"   生成 {len(chunks)} 个chunks")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"   Chunk {i}: {len(chunk.content)} 字符 - {chunk.content[:50]}...")
    if len(chunks) > 3:
        print(f"   ... 还有 {len(chunks) - 3} 个chunks")
    print()

    # 4. 初始化嵌入服务
    print("4️⃣ 初始化嵌入服务")
    if not embedding_service.is_loaded():
        emb_config = EmbeddingConfig(
            model_type=EmbeddingModelType.BGE,
            model_name="BAAI/bge-small-zh-v1.5",
            device="cpu",
        )
        response = embedding_service.load_model(emb_config)
        print(f"   ✅ 嵌入模型加载成功: {response.dimension} 维")
    else:
        print("   ✅ 嵌入模型已加载")
    print()

    # 5. 生成向量
    print("5️⃣ 生成向量嵌入")
    chunk_texts = [chunk.content for chunk in chunks]
    vectors = embedding_service.encode(chunk_texts)
    print(f"   ✅ 生成 {vectors.shape[0]} 个向量，维度 {vectors.shape[1]}")
    print()

    # 6. 初始化向量数据库
    print("6️⃣ 初始化向量数据库")
    dimension = embedding_service.get_dimension()
    vdb_config = VectorDBConfig(
        db_type=VectorDBType.FAISS, dimension=dimension, index_type="HNSW"
    )
    vector_db_manager.initialize(vdb_config)
    print("   ✅ 向量数据库初始化完成")
    print()

    # 7. 添加向量到数据库
    print("7️⃣ 添加向量到数据库")
    metadata = []
    for i, chunk in enumerate(chunks):
        meta = {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "chunk_num": chunk.num,
            "content": chunk.content[:200],  # 存储前200字符用于展示
            "length": chunk.length,
        }
        metadata.append(meta)

    vector_db_manager.add_vectors(vectors, metadata)
    print(f"   ✅ 成功添加 {len(vectors)} 个向量")
    print()

    # 8. 保存索引
    print("8️⃣ 保存向量索引")
    vector_db_manager.save()
    print("   ✅ 索引已保存")
    print()

    # 9. 验证
    print("9️⃣ 验证向量库")
    status = vector_db_manager.get_status()
    print(f"   总向量数: {status.total_vectors}")
    print(f"   维度: {status.dimension}")
    print()

    # 10. 测试检索
    print("🔟 测试检索")
    test_query = "8-9级员工住宿标准"
    query_vector = embedding_service.encode([test_query])
    distances, results = vector_db_manager.search(query_vector, top_k=3)
    print(f"   查询: '{test_query}'")
    print(f"   返回 {len(results[0])} 个结果:")
    for i, (dist, meta) in enumerate(zip(distances[0], results[0]), 1):
        if isinstance(meta, dict) and "content" in meta:
            print(
                f"   {i}. 相似度: {1 / (1 + dist):.3f}, 内容: {meta['content'][:80]}..."
            )
    print()

    print("=" * 60)
    print("✅ 文档处理完成！")
    print("=" * 60)
    print()
    print("现在可以运行RAGAS评估了：")
    print("  python project_local_ragas.py --mode batch")


if __name__ == "__main__":
    asyncio.run(process_baoxiao_document())
