#!/usr/bin/env python3
"""
添加补充文档到向量库
"""

import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, ".")

from services.embedding import embedding_service
from services.vector_db import vector_db_manager
from services.document_parser import DocumentParser
from models import EmbeddingConfig, VectorDBConfig, VectorDBType, EmbeddingModelType
from utils.logger import logger


def process_document_to_chunks(file_path: str) -> list:
    """将文档处理为chunks"""
    try:
        # 读取文档内容
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 简单分块策略
        chunk_size = 500
        chunk_overlap = 50

        chunks = []
        for i in range(0, len(content), chunk_size - chunk_overlap):
            chunk_text = content[i : i + chunk_size]
            if chunk_text.strip():
                chunks.append(
                    {
                        "content": chunk_text,
                        "chunk_id": f"{Path(file_path).stem}_{i // chunk_size}",
                        "document_id": Path(file_path).stem,
                        "source": file_path,
                        "chunk_index": i // chunk_size,
                    }
                )

        logger.info(f"文档 {file_path} 处理完成，生成 {len(chunks)} 个chunks")
        return chunks

    except Exception as e:
        logger.error(f"处理文档 {file_path} 失败: {e}")
        return []


def add_chunks_to_vector_db(chunks: list):
    """将chunks添加到向量库"""
    if not chunks:
        return

    try:
        # 提取文本内容
        texts = [chunk["content"] for chunk in chunks]

        # 生成embeddings
        embeddings = embedding_service.encode(texts)

        # 生成元数据
        metadatas = []
        for i, chunk in enumerate(chunks):
            metadatas.append(
                {
                    "content": chunk["content"],
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                }
            )

        # 添加到向量库
        vector_db_manager.add_vectors(embeddings, metadatas)

        logger.info(f"成功添加 {len(chunks)} 个chunks到向量库")

    except Exception as e:
        logger.error(f"添加chunks到向量库失败: {e}")


def main():
    """主函数"""
    logger.info("开始添加补充文档...")

    try:
        # 初始化嵌入服务
        logger.info("初始化嵌入服务...")
        embedding_service.load_model(
            EmbeddingConfig(
                model_type=EmbeddingModelType.BGE,
                model_name="BAAI/bge-base-zh-v1.5",
                device="cpu",
            )
        )

        # 初始化向量数据库
        logger.info("初始化向量数据库...")
        vector_db_manager.initialize(
            VectorDBConfig(
                db_type=VectorDBType.FAISS,
                dimension=embedding_service.get_dimension(),
                index_type="HNSW",
            )
        )

        # 处理补充文档
        supplement_docs = [
            "data/docs/15_酒店级别标准补充.md",
            "data/docs/16_城市分类地区差异.md",
        ]

        total_chunks_added = 0
        for doc_path in supplement_docs:
            if Path(doc_path).exists():
                logger.info(f"处理文档: {doc_path}")
                chunks = process_document_to_chunks(doc_path)
                if chunks:
                    add_chunks_to_vector_db(chunks)
                    total_chunks_added += len(chunks)
            else:
                logger.warning(f"文档不存在: {doc_path}")

        # 显示更新后的状态
        status = vector_db_manager.get_status()
        logger.info(f"向量库更新完成: {status}")
        logger.info(f"总共添加了 {total_chunks_added} 个新的chunks")

        # 保存更新后的索引
        vector_db_manager.save()
        logger.info("向量库已保存")

    except Exception as e:
        logger.error(f"添加补充文档失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
