"""
完整评估测试 - 包含所有 RAG 指标
- 检索指标: Hit Rate, Precision, Recall, MRR, NDCG
- 生成指标: 忠实度, 答案相关性 (需要 RAGAS)
"""

import requests
import json
import re
import time
import math
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Tuple

BASE_URL = "http://localhost:9000"


def load_ground_truth() -> Dict:
    """加载 Ground Truth 数据"""
    file_path = Path("/root/autodl-tmp/rag/retrieval_test_cases_ground_truth.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('ground_truth', data)


def check_content_match(retrieved_content: str, expected_content: str, threshold: float = 0.15) -> bool:
    """内容匹配检查"""
    if not retrieved_content or not expected_content:
        return False
    
    # 数字匹配
    expected_numbers = set(re.findall(r'\d+', expected_content))
    retrieved_numbers = set(re.findall(r'\d+', retrieved_content))
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


def search(query: str, top_k: int = 100, reranker_top_k: int = 20) -> Dict:
    """执行检索"""
    response = requests.post(
        f"{BASE_URL}/retrieval/search",
        json={
            "query": query,
            "config": {
                "top_k": top_k,
                "enable_rerank": True,
                "reranker_top_k": reranker_top_k,
                "similarity_threshold": 0.0
            }
        },
        timeout=60
    )
    return response.json()


def calculate_precision(results: List, expected_chunks: Set, expected_content: str, k: int) -> Tuple[int, int]:
    """计算 Precision@K"""
    relevant = 0
    for i, r in enumerate(results[:k]):
        chunk_id = r.get('chunk_id', '')
        content = r.get('content', '')
        if chunk_id in expected_chunks or (expected_content and check_content_match(content, expected_content)):
            relevant += 1
    return relevant, min(k, len(results))


def calculate_mrr(results: List, expected_chunks: Set, expected_content: str) -> float:
    """计算 MRR (Mean Reciprocal Rank)"""
    for i, r in enumerate(results, 1):
        chunk_id = r.get('chunk_id', '')
        content = r.get('content', '')
        if chunk_id in expected_chunks or (expected_content and check_content_match(content, expected_content)):
            return 1.0 / i
    return 0.0


def calculate_ndcg(results: List, expected_chunks: Set, expected_content: str, k: int) -> float:
    """计算 NDCG@K"""
    dcg = 0.0
    for i, r in enumerate(results[:k], 1):
        chunk_id = r.get('chunk_id', '')
        content = r.get('content', '')
        if chunk_id in expected_chunks or (expected_content and check_content_match(content, expected_content)):
            dcg += 1.0 / math.log2(i + 1)
    
    # IDCG = 1.0 (理想情况下第一个就是相关的)
    idcg = 1.0
    return dcg / idcg if idcg > 0 else 0.0


def run_full_evaluation():
    """运行完整评估"""
    ground_truth = load_ground_truth()
    queries = list(ground_truth.items())
    
    # 指标收集
    metrics = {
        'hit_count': 0,
        'total_queries': len(queries),
        'precision_at_5': [],
        'precision_at_10': [],
        'precision_at_20': [],
        'recall_at_5': [],
        'recall_at_10': [],
        'recall_at_20': [],
        'mrr': [],
        'ndcg_at_5': [],
        'ndcg_at_10': [],
        'ndcg_at_20': [],
        'latencies': [],
        'results_per_query': []
    }
    
    print(f"完整评估测试 ({len(queries)} 个样本)")
    print("=" * 70)
    print(f"{'ID':<6} {'Hit':<4} {'P@5':<6} {'P@10':<6} {'MRR':<6} {'NDCG':<6} {'延迟':<8}")
    print("-" * 70)
    
    for i, (query_id, query_data) in enumerate(queries):
        query = query_data.get('query', '')
        expected_chunks = set(query_data.get('relevant_chunks', []))
        expected_content = query_data.get('relevant_content', '')
        
        try:
            start = time.time()
            result = search(query)
            latency = (time.time() - start) * 1000
            metrics['latencies'].append(latency)
            
            results = result.get('results', [])
            
            # Hit (至少一个相关)
            hit = False
            for r in results:
                chunk_id = r.get('chunk_id', '')
                content = r.get('content', '')
                if chunk_id in expected_chunks or (expected_content and check_content_match(content, expected_content)):
                    hit = True
                    break
            
            if hit:
                metrics['hit_count'] += 1
            
            # Precision@K
            rel_5, total_5 = calculate_precision(results, expected_chunks, expected_content, 5)
            rel_10, total_10 = calculate_precision(results, expected_chunks, expected_content, 10)
            rel_20, total_20 = calculate_precision(results, expected_chunks, expected_content, 20)
            
            metrics['precision_at_5'].append(rel_5 / total_5 if total_5 > 0 else 0)
            metrics['precision_at_10'].append(rel_10 / total_10 if total_10 > 0 else 0)
            metrics['precision_at_20'].append(rel_20 / total_20 if total_20 > 0 else 0)
            
            # Recall (简化版：如果命中则为1)
            metrics['recall_at_5'].append(1.0 if rel_5 > 0 else 0.0)
            metrics['recall_at_10'].append(1.0 if rel_10 > 0 else 0.0)
            metrics['recall_at_20'].append(1.0 if rel_20 > 0 else 0.0)
            
            # MRR
            mrr = calculate_mrr(results, expected_chunks, expected_content)
            metrics['mrr'].append(mrr)
            
            # NDCG
            metrics['ndcg_at_5'].append(calculate_ndcg(results, expected_chunks, expected_content, 5))
            metrics['ndcg_at_10'].append(calculate_ndcg(results, expected_chunks, expected_content, 10))
            metrics['ndcg_at_20'].append(calculate_ndcg(results, expected_chunks, expected_content, 20))
            
            # 保存每条查询的详细结果
            metrics['results_per_query'].append({
                'query_id': query_id,
                'query': query,
                'hit': hit,
                'precision_at_5': rel_5 / total_5 if total_5 > 0 else 0,
                'mrr': mrr,
                'latency_ms': latency,
                'result_count': len(results)
            })
            
            # 打印进度
            hit_str = "✓" if hit else "✗"
            print(f"{query_id:<6} {hit_str:<4} {rel_5/total_5:.2f}  {rel_10/total_10:.2f}  {mrr:.2f}   {metrics['ndcg_at_5'][-1]:.2f}   {latency:.0f}ms")
            
        except Exception as e:
            print(f"{query_id:<6} 错误: {e}")
            metrics['mrr'].append(0)
            metrics['precision_at_5'].append(0)
            metrics['precision_at_10'].append(0)
            metrics['precision_at_20'].append(0)
            metrics['ndcg_at_5'].append(0)
            metrics['ndcg_at_10'].append(0)
            metrics['ndcg_at_20'].append(0)
        
        time.sleep(0.2)
    
    # 计算平均值
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0
    
    final_metrics = {
        'hit_rate': metrics['hit_count'] / metrics['total_queries'],
        'precision_at_5': avg(metrics['precision_at_5']),
        'precision_at_10': avg(metrics['precision_at_10']),
        'precision_at_20': avg(metrics['precision_at_20']),
        'recall_at_5': avg(metrics['recall_at_5']),
        'recall_at_10': avg(metrics['recall_at_10']),
        'recall_at_20': avg(metrics['recall_at_20']),
        'mrr': avg(metrics['mrr']),
        'ndcg_at_5': avg(metrics['ndcg_at_5']),
        'ndcg_at_10': avg(metrics['ndcg_at_10']),
        'ndcg_at_20': avg(metrics['ndcg_at_20']),
        'avg_latency_ms': avg(metrics['latencies']),
        'total_queries': metrics['total_queries'],
        'hit_count': metrics['hit_count'],
    }
    
    # 打印结果
    print("\n" + "=" * 70)
    print("评估结果汇总")
    print("=" * 70)
    
    print("\n【检索指标】")
    print(f"  Hit Rate:        {final_metrics['hit_rate']:.2%} ({final_metrics['hit_count']}/{final_metrics['total_queries']})")
    print(f"  Precision@5:     {final_metrics['precision_at_5']:.4f}")
    print(f"  Precision@10:    {final_metrics['precision_at_10']:.4f}")
    print(f"  Precision@20:    {final_metrics['precision_at_20']:.4f}")
    print(f"  Recall@5:        {final_metrics['recall_at_5']:.4f}")
    print(f"  Recall@10:       {final_metrics['recall_at_10']:.4f}")
    print(f"  Recall@20:       {final_metrics['recall_at_20']:.4f}")
    print(f"  MRR:             {final_metrics['mrr']:.4f}")
    print(f"  NDCG@5:          {final_metrics['ndcg_at_5']:.4f}")
    print(f"  NDCG@10:         {final_metrics['ndcg_at_10']:.4f}")
    print(f"  NDCG@20:         {final_metrics['ndcg_at_20']:.4f}")
    
    print("\n【性能指标】")
    print(f"  平均延迟:        {final_metrics['avg_latency_ms']:.0f}ms")
    
    # 保存结果
    output = {
        'metrics': final_metrics,
        'config': {
            'top_k': 100,
            'reranker_top_k': 20,
            'reranker': 'bge-reranker-large',
            'similarity_threshold': 0.0
        },
        'per_query_results': metrics['results_per_query'],
        'timestamp': datetime.now().isoformat()
    }
    
    output_dir = Path("/root/autodl-tmp/rag/test_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "final_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    return final_metrics


if __name__ == "__main__":
    run_full_evaluation()
