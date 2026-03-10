"""
最终配置评估测试

配置：
- 向量数据库：Milvus Lite
- 切分方式：分层智能切分
- Reranker：bge-reranker-large
"""

import os
import sys
import json
import time
import requests
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# API 基础路径
BASE_URL = "http://localhost:9000"

# 测试配置
TOP_K = 100  # 增加到 100，提高召回率
ENABLE_RERANK = True


def load_ground_truth() -> Dict[str, Dict]:
    """加载测试用例 Ground Truth 数据"""
    file_path = project_root.parent / "retrieval_test_cases_ground_truth.json"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('ground_truth', data)


def load_test_queries() -> List[Dict]:
    """加载测试查询 - 直接从 Ground Truth 加载"""
    ground_truth = load_ground_truth()
    
    # 转换为列表格式
    queries = []
    for query_id, query_data in ground_truth.items():
        queries.append({
            'id': query_id,
            'query': query_data.get('query', ''),
            'relevant_chunks': query_data.get('relevant_chunks', [])
        })
    
    return queries


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
                "reranker_top_k": 20,  # 精排返回前20个结果
                "similarity_threshold": 0.0,
                "reranker_threshold": 0.15  # 重排序阈值
            }
        },
        timeout=60
    )
    return response.json()


def search_without_rerank(query: str, top_k: int = 100) -> Dict:
    """不使用 Reranker 进行检索"""
    response = requests.post(
        f"{BASE_URL}/retrieval/search",
        json={
            "query": query,
            "config": {
                "top_k": top_k,
                "enable_rerank": False,
                "similarity_threshold": 0.0
            }
        },
        timeout=60
    )
    return response.json()


def check_content_match(retrieved_content: str, expected_content: str, threshold: float = 0.6) -> bool:
    """使用语义相似度匹配检查检索内容是否匹配期望内容
    
    Args:
        retrieved_content: 检索到的内容
        expected_content: 期望的内容
        threshold: 语义相似度阈值，默认0.6
    
    Returns:
        bool: 是否匹配
    """
    if not retrieved_content or not expected_content:
        return False
    
    try:
        # 使用 embedding 服务计算语义相似度
        from services.embedding import embedding_service
        
        # 获取嵌入向量
        emb1 = embedding_service.embed_single(retrieved_content)
        emb2 = embedding_service.embed_single(expected_content)
        
        if emb1 is None or emb2 is None:
            # 如果嵌入失败，回退到关键词匹配
            import re
            expected_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', expected_content))
            retrieved_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', retrieved_content))
            if expected_keywords:
                common_keywords = expected_keywords & retrieved_keywords
                return len(common_keywords) / len(expected_keywords) >= 0.15
            return False
        
        # 计算余弦相似度
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return similarity >= threshold
    except Exception as e:
        # 异常时回退到关键词匹配
        import re
        expected_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', expected_content))
        retrieved_keywords = set(re.findall(r'[\u4e00-\u9fa5]{2,}', retrieved_content))
        if expected_keywords:
            common_keywords = expected_keywords & retrieved_keywords
            return len(common_keywords) / len(expected_keywords) >= 0.15
        return False


def calculate_metrics(results: List[Dict], ground_truth: Dict, test_queries: List[Dict]) -> Dict:
    """计算评估指标 - 使用内容匹配"""
    metrics = {
        'precision_at_k': [],
        'recall_at_k': [],
        'mrr': [],
        'ndcg': [],
        'hit_rate': [],
        'latency_ms': []
    }
    
    for query_data in test_queries:
        query_id = query_data.get('id')
        query_text = query_data.get('query')
        expected_content = ground_truth.get(query_id, {}).get('relevant_content', '')
        expected_chunks = set(query_data.get('relevant_chunks', []))
        
        # 查找对应的检索结果
        result = None
        for r in results:
            if r.get('query_id') == query_id:
                result = r
                break
        
        if not result:
            continue
        
        retrieved = result.get('results', [])
        latency = result.get('latency_ms', 0)
        metrics['latency_ms'].append(latency)
        
        if not retrieved:
            metrics['precision_at_k'].append(0)
            metrics['recall_at_k'].append(0)
            metrics['mrr'].append(0)
            metrics['ndcg'].append(0)
            metrics['hit_rate'].append(0)
            continue
        
        # 计算命中率 - 先用 chunk_id 匹配，再用内容匹配
        hit = False
        first_relevant_rank = 0
        relevant_count = 0
        
        for i, item in enumerate(retrieved):
            chunk_id = item.get('id', item.get('chunk_id', ''))
            content = item.get('content', '')
            
            # 优先使用 chunk_id 匹配
            if chunk_id in expected_chunks:
                if not hit:
                    first_relevant_rank = i + 1
                    hit = True
                relevant_count += 1
            # 如果 chunk_id 不匹配，使用内容匹配
            elif expected_content and check_content_match(content, expected_content):
                if not hit:
                    first_relevant_rank = i + 1
                    hit = True
                relevant_count += 1
        
        # Precision@K
        precision = relevant_count / len(retrieved) if retrieved else 0
        metrics['precision_at_k'].append(precision)
        
        # Recall@K - 至少命中一个相关文档
        recall = 1.0 if hit else 0.0
        metrics['recall_at_k'].append(recall)
        
        # MRR
        mrr = 1.0 / first_relevant_rank if first_relevant_rank > 0 else 0
        metrics['mrr'].append(mrr)
        
        # NDCG@K
        dcg = 0
        for i, item in enumerate(retrieved):
            chunk_id = item.get('id', item.get('chunk_id', ''))
            content = item.get('content', '')
            if chunk_id in expected_chunks or (expected_content and check_content_match(content, expected_content)):
                dcg += 1.0 / (i + 1)
        
        idcg = 1.0  # 理想情况下第一个就是相关的
        ndcg = dcg / idcg if idcg > 0 else 0
        metrics['ndcg'].append(ndcg)
        
        # Hit Rate
        metrics['hit_rate'].append(1.0 if hit else 0.0)
    
    # 计算平均值
    return {
        'precision_at_k': sum(metrics['precision_at_k']) / len(metrics['precision_at_k']) if metrics['precision_at_k'] else 0,
        'recall_at_k': sum(metrics['recall_at_k']) / len(metrics['recall_at_k']) if metrics['recall_at_k'] else 0,
        'mrr': sum(metrics['mrr']) / len(metrics['mrr']) if metrics['mrr'] else 0,
        'ndcg': sum(metrics['ndcg']) / len(metrics['ndcg']) if metrics['ndcg'] else 0,
        'hit_rate': sum(metrics['hit_rate']) / len(metrics['hit_rate']) if metrics['hit_rate'] else 0,
        'avg_latency_ms': sum(metrics['latency_ms']) / len(metrics['latency_ms']) if metrics['latency_ms'] else 0,
        'total_queries': len(test_queries)
    }


def run_evaluation():
    """运行评估测试"""
    print("="*70)
    print("最终配置评估测试")
    print("="*70)
    print(f"向量数据库: Milvus Lite")
    print(f"切分方式: 分层智能切分")
    print(f"Reranker: bge-reranker-large")
    print(f"Top-K: {TOP_K}")
    print("="*70)
    
    # 加载数据
    ground_truth = load_ground_truth()
    test_queries = load_test_queries()
    
    print(f"\n加载了 {len(ground_truth)} 个 ground truth 条目")
    print(f"加载了 {len(test_queries)} 个测试查询")
    
    # 检查服务状态
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"后端服务状态: {response.json().get('status', 'unknown')}")
    except Exception as e:
        print(f"后端服务连接失败: {e}")
        return
    
    # 检查 reranker 状态
    try:
        response = requests.get(f"{BASE_URL}/retrieval/reranker/status", timeout=5)
        reranker_status = response.json()
        print(f"Reranker 状态: {reranker_status}")
    except Exception as e:
        print(f"Reranker 状态检查失败: {e}")
    
    # 运行测试
    results_with_rerank = []
    results_without_rerank = []
    
    print("\n" + "-"*70)
    print("开始评估测试...")
    print("-"*70)
    
    for i, query_data in enumerate(test_queries):
        query_id = query_data.get('id')
        query_text = query_data.get('query')
        
        print(f"\n[{i+1}/{len(test_queries)}] 查询: {query_text[:50]}...")
        
        # 带 Reranker 的检索
        try:
            result = search_with_rerank(query_text, TOP_K)
            results_with_rerank.append({
                'query_id': query_id,
                'query': query_text,
                'results': result.get('results', []),
                'total': result.get('total', 0),
                'latency_ms': result.get('latency_ms', 0)
            })
            print(f"  带 Reranker: {result.get('total', 0)} 个结果, 延迟: {result.get('latency_ms', 0):.1f}ms")
        except Exception as e:
            print(f"  带 Reranker 检索失败: {e}")
        
        # 不带 Reranker 的检索
        try:
            result = search_without_rerank(query_text, TOP_K)
            results_without_rerank.append({
                'query_id': query_id,
                'query': query_text,
                'results': result.get('results', []),
                'total': result.get('total', 0),
                'latency_ms': result.get('latency_ms', 0)
            })
            print(f"  不带 Reranker: {result.get('total', 0)} 个结果, 延迟: {result.get('latency_ms', 0):.1f}ms")
        except Exception as e:
            print(f"  不带 Reranker 检索失败: {e}")
        
        time.sleep(0.5)  # 避免过载
    
    # 计算指标
    print("\n" + "-"*70)
    print("计算评估指标...")
    print("-"*70)
    
    metrics_with_rerank = calculate_metrics(results_with_rerank, ground_truth, test_queries)
    metrics_without_rerank = calculate_metrics(results_without_rerank, ground_truth, test_queries)
    
    # 打印结果
    print("\n" + "="*70)
    print("评估结果")
    print("="*70)
    
    print("\n【带 Reranker】")
    print(f"  Precision@{TOP_K}: {metrics_with_rerank['precision_at_k']:.4f}")
    print(f"  Recall@{TOP_K}: {metrics_with_rerank['recall_at_k']:.4f}")
    print(f"  MRR@{TOP_K}: {metrics_with_rerank['mrr']:.4f}")
    print(f"  NDCG@{TOP_K}: {metrics_with_rerank['ndcg']:.4f}")
    print(f"  Hit Rate@{TOP_K}: {metrics_with_rerank['hit_rate']:.4f}")
    print(f"  平均延迟: {metrics_with_rerank['avg_latency_ms']:.2f}ms")
    
    print("\n【不带 Reranker】")
    print(f"  Precision@{TOP_K}: {metrics_without_rerank['precision_at_k']:.4f}")
    print(f"  Recall@{TOP_K}: {metrics_without_rerank['recall_at_k']:.4f}")
    print(f"  MRR@{TOP_K}: {metrics_without_rerank['mrr']:.4f}")
    print(f"  NDCG@{TOP_K}: {metrics_without_rerank['ndcg']:.4f}")
    print(f"  Hit Rate@{TOP_K}: {metrics_without_rerank['hit_rate']:.4f}")
    print(f"  平均延迟: {metrics_without_rerank['avg_latency_ms']:.2f}ms")
    
    # 计算提升
    print("\n【Reranker 提升效果】")
    precision_lift = (metrics_with_rerank['precision_at_k'] - metrics_without_rerank['precision_at_k']) / metrics_without_rerank['precision_at_k'] * 100 if metrics_without_rerank['precision_at_k'] > 0 else 0
    ndcg_lift = (metrics_with_rerank['ndcg'] - metrics_without_rerank['ndcg']) / metrics_without_rerank['ndcg'] * 100 if metrics_without_rerank['ndcg'] > 0 else 0
    mrr_lift = (metrics_with_rerank['mrr'] - metrics_without_rerank['mrr']) / metrics_without_rerank['mrr'] * 100 if metrics_without_rerank['mrr'] > 0 else 0
    
    print(f"  Precision 提升: {precision_lift:+.2f}%")
    print(f"  NDCG 提升: {ndcg_lift:+.2f}%")
    print(f"  MRR 提升: {mrr_lift:+.2f}%")
    
    # 保存结果
    results = {
        'config': {
            'vector_db': 'Milvus Lite',
            'chunking': '分层智能切分',
            'reranker': 'bge-reranker-large',
            'top_k': TOP_K,
            'total_vectors': 993
        },
        'with_rerank': metrics_with_rerank,
        'without_rerank': metrics_without_rerank,
        'improvement': {
            'precision': precision_lift,
            'ndcg': ndcg_lift,
            'mrr': mrr_lift
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # 保存到文件
    output_dir = project_root.parent / "test_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "final_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
    
    return results


if __name__ == "__main__":
    run_evaluation()
