#!/usr/bin/env python3
"""
填充 Ground Truth 中的 chunk_ids

使用后端 API 进行向量搜索
"""

import json
import sys
import requests
from pathlib import Path
from typing import List, Dict

# 添加 backend 目录到路径
BACKEND_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))


def search_chunks_via_api(query: str, top_k: int = 3, base_url: str = "http://localhost:9000") -> List[str]:
    """通过 API 搜索 chunks"""
    
    try:
        response = requests.post(
            f"{base_url}/retrieval/search",  # 正确的 API 路径
            json={
                "query": query,
                "config": {
                    "top_k": top_k,
                    "similarity_threshold": 0.0,
                    "enable_reranker": False
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', [])
            return [r.get('id', r.get('chunk_id', '')) for r in results]
        else:
            print(f"API 请求失败: {response.status_code} - {response.text[:200]}")
            return []
            
    except requests.exceptions.ConnectionError:
        print("无法连接到后端服务，请确保服务正在运行")
        return []
    except Exception as e:
        print(f"搜索失败: {e}")
        return []


def fill_chunk_ids(input_file: str, output_file: str, top_k: int = 3, base_url: str = "http://localhost:9000"):
    """填充 chunk_ids"""
    
    # 加载 Ground Truth
    with open(input_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    print(f"加载了 {len(ground_truth['ground_truth'])} 个测试用例")
    
    success_count = 0
    
    for case_id, case_data in ground_truth['ground_truth'].items():
        query = case_data['query']
        
        # 通过 API 搜索
        chunk_ids = search_chunks_via_api(query, top_k, base_url)
        
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
    parser.add_argument("--base-url", type=str, default="http://localhost:9000", help="后端 API 地址")
    
    args = parser.parse_args()
    
    fill_chunk_ids(args.input, args.output, args.top_k, args.base_url)
