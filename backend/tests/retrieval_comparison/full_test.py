"""
完整评估测试 - 测试所有 90 个样本
"""

import requests
import json
import re
import time
from pathlib import Path
from datetime import datetime

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

def run_full_test():
    ground_truth = load_ground_truth()
    queries = list(ground_truth.items())
    
    hit_count = 0
    total_latency = 0
    print(f"完整测试 ({len(queries)} 个样本)")
    print("="*60)
    
    for i, (query_id, query_data) in enumerate(queries):
        query = query_data.get('query', '')
        expected_chunks = set(query_data.get('relevant_chunks', []))
        expected_content = query_data.get('relevant_content', '')
        
        print(f"\r[{i+1}/{len(queries)}] Hit Rate: {hit_count}/{i} ({100*hit_count/(i or 1):.1f}%)", end="")
        
        try:
            start = time.time()
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
            latency = (time.time() - start) * 1000
            total_latency += latency
            
            results = response.json().get('results', [])
            
            hit = False
            for r in results:
                chunk_id = r.get('chunk_id', '')
                content = r.get('content', '')
                if chunk_id in expected_chunks or (expected_content and check_content_match(content, expected_content)):
                    hit = True
                    break
            
            if hit:
                hit_count += 1
                
        except Exception as e:
            print(f"\n测试失败: {e}")
        
        time.sleep(0.2)
    
    hit_rate = hit_count / len(queries)
    avg_latency = total_latency / len(queries)
    
    print(f"\n\n{'='*60}")
    print(f"最终结果")
    print(f"{'='*60}")
    print(f"Hit Rate: {hit_rate:.2%} ({hit_count}/{len(queries)})")
    print(f"平均延迟: {avg_latency:.0f}ms")
    
    # 保存结果
    results = {
        'hit_rate': hit_rate,
        'hit_count': hit_count,
        'total_queries': len(queries),
        'avg_latency_ms': avg_latency,
        'config': {
            'top_k': 100,
            'reranker_top_k': 20,
            'similarity_threshold': 0.0,
            'reranker': 'bge-reranker-large'
        },
        'timestamp': datetime.now().isoformat()
    }
    
    output_dir = Path("/root/autodl-tmp/rag/test_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "final_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    return hit_rate

if __name__ == "__main__":
    run_full_test()
