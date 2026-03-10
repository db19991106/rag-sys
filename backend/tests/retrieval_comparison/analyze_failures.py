"""
分析检索失败案例
按 RAG 流程步骤逐一检查问题
"""

import os
import sys
import json
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

BASE_URL = "http://localhost:9000"


def load_ground_truth() -> Dict[str, Dict]:
    """加载 Ground Truth"""
    file_path = project_root.parent / "retrieval_test_cases_ground_truth.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('ground_truth', data)


def semantic_similarity(text1: str, text2: str) -> float:
    """计算语义相似度"""
    try:
        from services.embedding import embedding_service
        emb1 = embedding_service.embed_single(text1)
        emb2 = embedding_service.embed_single(text2)
        if emb1 is None or emb2 is None:
            return 0.0
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    except:
        return 0.0


def analyze_query(query_id: str, query_data: Dict) -> Dict:
    """分析单个查询的检索结果"""
    query = query_data.get('query', '')
    expected_chunks = set(query_data.get('relevant_chunks', []))
    expected_content = query_data.get('relevant_content', '')
    
    print(f"\n{'='*70}")
    print(f"查询 ID: {query_id}")
    print(f"查询: {query}")
    print(f"期望 chunk_ids: {expected_chunks}")
    
    # 1. 意图识别分析
    print(f"\n--- 步骤1: 意图识别 ---")
    try:
        resp = requests.post(f"{BASE_URL}/rag/recognize-intent", json={"query": query}, timeout=30)
        intent_result = resp.json()
        print(f"  意图: {intent_result.get('intent', 'N/A')}")
        print(f"  置信度: {intent_result.get('confidence', 0):.2f}")
    except Exception as e:
        print(f"  意图识别失败: {e}")
    
    # 2. 向量检索分析（不带重排序）
    print(f"\n--- 步骤2: 向量检索 (无重排序) ---")
    try:
        resp = requests.post(
            f"{BASE_URL}/retrieval/search",
            json={
                "query": query,
                "config": {
                    "top_k": 30,
                    "enable_rerank": False,
                    "similarity_threshold": 0.0
                }
            },
            timeout=60
        )
        result = resp.json()
        results = result.get('results', [])
        
        # 检查期望的 chunk 是否在前30个结果中
        found_in_vector = []
        for i, r in enumerate(results[:30]):
            chunk_id = r.get('id', r.get('chunk_id', ''))
            if chunk_id in expected_chunks:
                found_in_vector.append((i+1, chunk_id, r.get('similarity', 0)))
        
        print(f"  检索结果数: {len(results)}")
        print(f"  期望 chunk 在前30中的位置: {found_in_vector if found_in_vector else '未找到'}")
        
        if not found_in_vector and expected_content:
            # 检查语义相似度
            print(f"  检查前5个结果的语义相似度:")
            for i, r in enumerate(results[:5]):
                sim = semantic_similarity(r.get('content', ''), expected_content)
                print(f"    结果{i+1}: chunk_id={r.get('id', 'N/A')}, 语义相似度={sim:.3f}")
                
    except Exception as e:
        print(f"  向量检索失败: {e}")
        results = []
    
    # 3. 重排序分析
    print(f"\n--- 步骤3: 重排序 ---")
    try:
        resp = requests.post(
            f"{BASE_URL}/retrieval/search",
            json={
                "query": query,
                "config": {
                    "top_k": 30,
                    "enable_rerank": True,
                    "reranker_type": "bge",
                    "reranker_top_k": 5,
                    "similarity_threshold": 0.0
                }
            },
            timeout=60
        )
        result = resp.json()
        rerank_results = result.get('results', [])
        
        # 检查期望的 chunk 是否在重排序结果中
        found_in_rerank = []
        for i, r in enumerate(rerank_results):
            chunk_id = r.get('id', r.get('chunk_id', ''))
            if chunk_id in expected_chunks:
                found_in_rerank.append((i+1, chunk_id, r.get('similarity', 0)))
        
        print(f"  重排序结果数: {len(rerank_results)}")
        print(f"  期望 chunk 在重排序结果中的位置: {found_in_rerank if found_in_rerank else '未找到'}")
        
        if not found_in_rerank and expected_content:
            print(f"  检查重排序结果的语义相似度:")
            for i, r in enumerate(rerank_results):
                sim = semantic_similarity(r.get('content', ''), expected_content)
                print(f"    结果{i+1}: chunk_id={r.get('id', 'N/A')}, 语义相似度={sim:.3f}")
                
    except Exception as e:
        print(f"  重排序失败: {e}")
    
    return {
        'query_id': query_id,
        'query': query,
        'expected_chunks': list(expected_chunks),
        'found_in_vector': len(found_in_vector) > 0 if 'found_in_vector' in dir() else False,
        'found_in_rerank': len(found_in_rerank) > 0 if 'found_in_rerank' in dir() else False,
    }


def main():
    print("="*70)
    print("RAG 检索失败案例分析")
    print("="*70)
    
    ground_truth = load_ground_truth()
    print(f"加载 {len(ground_truth)} 个测试查询")
    
    # 收集失败案例
    failures = []
    
    for query_id, query_data in ground_truth.items():
        result = analyze_query(query_id, query_data)
        if not result.get('found_in_rerank'):
            failures.append(result)
    
    # 统计
    print(f"\n{'='*70}")
    print("统计结果")
    print("="*70)
    print(f"总查询数: {len(ground_truth)}")
    print(f"失败数: {len(failures)}")
    print(f"成功数: {len(ground_truth) - len(failures)}")
    print(f"成功率: {(len(ground_truth) - len(failures)) / len(ground_truth) * 100:.2f}%")
    
    # 失败案例汇总
    if failures:
        print(f"\n失败案例汇总:")
        for f in failures[:10]:  # 只显示前10个
            print(f"  - {f['query_id']}: {f['query'][:50]}...")
            print(f"    期望: {f['expected_chunks']}")
            print(f"    向量检索找到: {f['found_in_vector']}, 重排序找到: {f['found_in_rerank']}")


if __name__ == "__main__":
    main()
