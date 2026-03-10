"""
迷你评估测试 - 只测试 10 个样本，快速验证效果
"""

import requests
import json
import re
from pathlib import Path

BASE_URL = "http://localhost:9000"

def load_ground_truth():
    file_path = Path("/root/autodl-tmp/rag/retrieval_test_cases_ground_truth.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('ground_truth', data)

def check_content_match(retrieved_content, expected_content, threshold=0.15):
    if not retrieved_content or not expected_content:
        return False
    expected_numbers = set(re.findall(r'\d+', expected_content))
    retrieved_numbers = set(re.findall(r'\d+', retrieved_content))
    if expected_numbers:
        common_numbers = expected_numbers & retrieved_numbers
        if len(common_numbers) / len(expected_numbers) >= 0.2:
            return True
    expected_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', expected_content))
    retrieved_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', retrieved_content))
    if expected_keywords:
        common_keywords = expected_keywords & retrieved_keywords
        if len(common_keywords) / len(expected_keywords) >= threshold:
            return True
    return False

def run_mini_test():
    ground_truth = load_ground_truth()
    queries = list(ground_truth.items())[:10]  # 只测试前 10 个
    
    hit_count = 0
    print("迷你测试 (10 个样本)")
    print("="*50)
    
    for query_id, query_data in queries:
        query = query_data.get('query', '')
        expected_chunks = set(query_data.get('relevant_chunks', []))
        expected_content = query_data.get('relevant_content', '')
        
        try:
            response = requests.post(
                f"{BASE_URL}/retrieval/search",
                json={
                    "query": query,
                    "config": {
                        "top_k": 100,
                        "enable_rerank": True,
                        "reranker_top_k": 20,
                        "similarity_threshold": 0.0
                    }
                },
                timeout=60
            )
            results = response.json().get('results', [])
            
            hit = False
            for r in results:
                chunk_id = r.get('chunk_id', '')
                content = r.get('content', '')
                if chunk_id in expected_chunks or (expected_content and check_content_match(content, expected_content)):
                    hit = True
                    break
            
            hit_count += hit
            status = "✓" if hit else "✗"
            print(f"{status} [{hit_count}/10] {query[:30]}... -> {len(results)} 个结果")
            
        except Exception as e:
            print(f"✗ 测试失败: {e}")
    
    hit_rate = hit_count / 10
    print(f"\n{'='*50}")
    print(f"Hit Rate: {hit_rate:.2%}")
    return hit_rate

if __name__ == "__main__":
    run_mini_test()
