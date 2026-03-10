"""
快速评估测试 - 只测试带 Reranker 的情况
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# API 基础路径
BASE_URL = "http://localhost:9000"

# 测试配置
TOP_K = 100


def load_ground_truth() -> Dict[str, Dict]:
    """加载测试用例 Ground Truth 数据"""
    file_path = project_root.parent / "retrieval_test_cases_ground_truth.json"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('ground_truth', data)


def search_with_rerank(query: str, top_k: int = 100) -> Dict:
    """使用 Reranker 进行检索"""
    response = requests.post(
        f"{BASE_URL}/retrieval/search",
        json={
            "query": query,
            "config": {
                "top_k": top_k,
                "enable_rerank": True,
                "reranker_type": "bge",
                "reranker_top_k": 15,
                "similarity_threshold": 0.0
            }
        },
        timeout=60
    )
    return response.json()


def check_content_match(retrieved_content: str, expected_content: str, threshold: float = 0.15) -> bool:
    """检查检索内容是否匹配期望内容"""
    if not retrieved_content or not expected_content:
        return False
    
    import re
    
    # 提取数字
    expected_numbers = set(re.findall(r'\d+', expected_content))
    retrieved_numbers = set(re.findall(r'\d+', retrieved_content))
    
    # 如果有数字，检查数字是否匹配
    if expected_numbers:
        common_numbers = expected_numbers & retrieved_numbers
        if len(common_numbers) / len(expected_numbers) >= 0.2:
            return True
    
    # 关键词匹配
    expected_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', expected_content))
    retrieved_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', retrieved_content))
    
    if expected_keywords:
        common_keywords = expected_keywords & retrieved_keywords
        if len(common_keywords) / len(expected_keywords) >= threshold:
            return True
    
    return False


def run_quick_test():
    """运行快速评估测试"""
    print("="*70)
    print("快速评估测试 (仅测试带 Reranker)")
    print("="*70)
    
    # 加载数据
    ground_truth = load_ground_truth()
    test_queries = []
    for query_id, query_data in ground_truth.items():
        test_queries.append({
            'id': query_id,
            'query': query_data.get('query', ''),
            'relevant_chunks': query_data.get('relevant_chunks', [])
        })
    
    print(f"加载了 {len(test_queries)} 个测试查询")
    
    # 检查服务状态
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"后端服务状态: {response.json().get('status', 'unknown')}")
    except Exception as e:
        print(f"后端服务连接失败: {e}")
        return
    
    # 运行测试
    hit_count = 0
    total_count = 0
    
    print("\n开始测试...")
    
    for i, query_data in enumerate(test_queries):
        query_id = query_data.get('id')
        query_text = query_data.get('query')
        expected_chunks = set(query_data.get('relevant_chunks', []))
        expected_content = ground_truth.get(query_id, {}).get('relevant_content', '')
        
        print(f"\r[{i+1}/{len(test_queries)}] 测试中... 当前 Hit Rate: {hit_count}/{total_count}", end="")
        
        try:
            result = search_with_rerank(query_text, TOP_K)
            retrieved = result.get('results', [])
            
            total_count += 1
            hit = False
            
            for item in retrieved:
                chunk_id = item.get('id', item.get('chunk_id', ''))
                content = item.get('content', '')
                
                # 优先使用 chunk_id 匹配
                if chunk_id in expected_chunks:
                    hit = True
                    break
                # 如果 chunk_id 不匹配，使用内容匹配
                elif expected_content and check_content_match(content, expected_content):
                    hit = True
                    break
            
            if hit:
                hit_count += 1
                
        except Exception as e:
            print(f"\n  测试失败: {e}")
        
        time.sleep(0.3)  # 减少延迟
    
    # 计算结果
    hit_rate = hit_count / total_count if total_count > 0 else 0
    
    print(f"\n\n{'='*70}")
    print(f"测试结果")
    print(f"{'='*70}")
    print(f"Hit Rate: {hit_rate:.4f} ({hit_rate*100:.2f}%)")
    print(f"命中数: {hit_count}/{total_count}")
    
    # 保存结果
    results = {
        'hit_rate': hit_rate,
        'hit_count': hit_count,
        'total_count': total_count,
        'config': {
            'top_k': TOP_K,
            'reranker_top_k': 15,
            'similarity_threshold': 0.0
        },
        'timestamp': datetime.now().isoformat()
    }
    
    output_dir = project_root.parent / "test_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "final_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    return hit_rate


if __name__ == "__main__":
    run_quick_test()
