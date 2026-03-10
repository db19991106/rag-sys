#!/usr/bin/env python3
"""
填充 Ground Truth 中的 chunk_ids

使用向量搜索来找到相关的 chunks
"""

import json
import sys
from pathlib import Path

# 添加 backend 目录到路径
BACKEND_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from config import settings
from models import VectorDBType, VectorDBConfig, EmbeddingConfig, EmbeddingModelType
from services.vector_db import vector_db_manager
from services.embedding import embedding_service


def fill_chunk_ids(input_file: str, output_file: str, top_k: int = 3):
    """填充 chunk_ids
    
    Args:
        input_file: 输入的 Ground Truth 文件
        output_file: 输出文件
        top_k: 每个问题查找的 chunk 数量
    """
    # 加载 Ground Truth
    with open(input_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    print(f"加载了 {len(ground_truth['ground_truth'])} 个测试用例")
    
    # 初始化向量数据库 - 使用 Milvus Lite
    print("初始化向量数据库 (Milvus Lite)...")
    db_config = VectorDBConfig(
        db_type=VectorDBType.MILVUS_LITE,
        dimension=settings.faiss_dimension,
        index_path=settings.milvus_lite_db_path,  # 直接使用 .db 文件路径
        collection_name=settings.milvus_collection_name
    )
    vector_db_manager.initialize(db_config)
    
    # 初始化 embedding 服务
    print("初始化嵌入服务...")
    embedding_config = EmbeddingConfig(
        model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
        model_name=settings.embedding_model_name,
        device=settings.embedding_device
    )
    embedding_service.load_model(embedding_config)
    
    success_count = 0
    
    for case_id, case_data in ground_truth['ground_truth'].items():
        query = case_data['query']
        
        # 获取查询向量
        query_vector = embedding_service.encode(query)
        
        # 搜索相关 chunks
        results = vector_db_manager.search(
            query_vector=query_vector,
            top_k=top_k
        )
        
        if results:
            # 提取 chunk_ids
            chunk_ids = [r.get('id', r.get('chunk_id', '')) for r in results]
            case_data['relevant_chunks'] = chunk_ids
            success_count += 1
            print(f"  {case_id}: {query[:30]}... -> {len(chunk_ids)} chunks")
        else:
            print(f"  {case_id}: {query[:30]}... -> 未找到 chunks")
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成！成功填充 {success_count}/{len(ground_truth['ground_truth'])} 个用例")
    print(f"保存到: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="填充 Ground Truth 中的 chunk_ids")
    parser.add_argument("--input", type=str, required=True, help="输入文件")
    parser.add_argument("--output", type=str, required=True, help="输出文件")
    parser.add_argument("--top-k", type=int, default=3, help="每个问题查找的 chunk 数量")
    
    args = parser.parse_args()
    
    fill_chunk_ids(args.input, args.output, args.top_k)
