#!/usr/bin/env python3
"""
填充 Ground Truth 中的 chunk_ids

使用简单的文本匹配来找到相关的 chunks
"""

import json
import sys
import re
from pathlib import Path
from typing import List, Dict

# 添加 backend 目录到路径
BACKEND_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))


def load_chunks_from_db() -> Dict[str, List[Dict]]:
    """从 FAISS 备份数据库导出 chunks"""
    import json
    
    # 使用备份的 FAISS 元数据
    metadata_path = "/root/autodl-tmp/rag/backend/vector_db_backup_20260303_005345/faiss_metadata.json"
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        all_chunks = {}
        
        for chunk_id, chunk_data in metadata.items():
            if isinstance(chunk_data, dict) and 'content' in chunk_data:
                all_chunks[chunk_id] = {
                    'id': chunk_id,
                    'content': chunk_data.get('content', ''),
                    'doc_id': chunk_data.get('doc_id', ''),
                    'chunk_index': chunk_data.get('chunk_index', 0)
                }
        
        print(f"从 FAISS 备份加载了 {len(all_chunks)} 个 chunks")
        return all_chunks
        
    except Exception as e:
        print(f"加载 FAISS 元数据失败: {e}")
        return {}


def find_relevant_chunks(query: str, all_chunks: Dict, top_k: int = 3) -> List[str]:
    """使用文本匹配找到相关 chunks"""
    
    # 提取查询关键词
    query_words = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+|\d+', query)
    query_words = [w for w in query_words if len(w) > 1]
    
    # 计算每个 chunk 的得分
    scored_chunks = []
    
    for chunk_id, chunk_data in all_chunks.items():
        content = chunk_data.get('content', '')
        
        # 计算关键词匹配得分
        score = 0
        for word in query_words:
            if word in content:
                score += 1
        
        if score > 0:
            scored_chunks.append((chunk_id, score, content))
    
    # 按得分排序
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # 返回得分最高的 chunks
    return [chunk[0] for chunk in scored_chunks[:top_k]]


def fill_chunk_ids(input_file: str, output_file: str, top_k: int = 3):
    """填充 chunk_ids"""
    
    # 加载 Ground Truth
    with open(input_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    print(f"加载了 {len(ground_truth['ground_truth'])} 个测试用例")
    
    # 从数据库加载 chunks
    print("从 Milvus Lite 加载 chunks...")
    all_chunks = load_chunks_from_db()
    
    if not all_chunks:
        print("未能加载 chunks，尝试使用 API...")
        return
    
    print(f"共加载 {len(all_chunks)} 个 chunks")
    
    success_count = 0
    
    for case_id, case_data in ground_truth['ground_truth'].items():
        query = case_data['query']
        
        # 查找相关 chunks
        chunk_ids = find_relevant_chunks(query, all_chunks, top_k)
        
        if chunk_ids:
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
