"""
实验一：纯向量检索 vs RRF融合检索准确率对比
"""

import sys
import os
import json
import time
import gc
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.retrieval_comparison.evaluation_utils import (
    clear_gpu_memory,
    wait_for_memory,
    load_ground_truth,
    evaluate_retrieval_results,
    generate_comparison_report
)

# 常量
TOP_K = 5
CANDIDATE_K = 20
OUTPUT_PATH = project_root.parent / "test_reports" / "纯向量与融合向量准确率对比.md"


def run_vector_only_experiment(ground_truth: Dict) -> Dict[str, Any]:
    """
    小实验1：纯向量检索评估
    
    配置：
    - vector_weight = 1.0
    - bm25_weight = 0.0
    - 关闭Reranker
    """
    print("\n" + "="*60)
    print("小实验1：纯向量检索评估")
    print("="*60)
    
    config = {
        'top_k': TOP_K,
        'similarity_threshold': 0.1,
        'rrf_k': 60,
        'vector_weight': 1.0,
        'bm25_weight': 0.0,
        'enable_rerank': False
    }
    
    print(f"配置: {config}")
    
    # 初始化向量数据库
    from services.vector_db import VectorDatabaseManager
    from models import VectorDBConfig
    from services.embedding import embedding_service
    from models import EmbeddingConfig, EmbeddingModelType
    from config import settings
    
    vector_db_config = VectorDBConfig(
        db_type='faiss',
        dimension=settings.faiss_dimension,
        index_type="HNSW"
    )
    
    vector_db_manager = VectorDatabaseManager()
    vector_db_manager.initialize(vector_db_config)
    
    # 初始化嵌入模型
    embedding_config = EmbeddingConfig(
        model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
        model_name=settings.embedding_model_name,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device
    )
    
    embedding_service.load_model(embedding_config)
    
    # 获取所有文档元数据（用于内容匹配）
    all_metadata = vector_db_manager.get_all_metadata()
    chunk_content_map = {m.get('chunk_id', m.get('id', '')): m.get('content', '') for m in all_metadata}
    
    print(f"获取到 {len(all_metadata)} 个文档片段元数据")
    print(f"开始评估 {len(ground_truth)} 个查询...")
    
    results = []
    response_times = []
    
    for i, (query_id, data) in enumerate(ground_truth.items()):
        query_text = data.get('query', data.get('relevant_content', ''))
        
        try:
            start_time = time.time()
            
            # 生成查询向量
            query_vector = embedding_service.encode([query_text])[0]
            
            # 执行向量检索
            distances, metadata_list = vector_db_manager.search(
                query_vector=query_vector,
                top_k=TOP_K
            )
            
            # 提取检索的chunk ID和内容
            retrieved_chunks = []
            retrieved_contents = []
            for m in metadata_list[0]:
                chunk_id = m.get('chunk_id', m.get('id', ''))
                content = m.get('content', chunk_content_map.get(chunk_id, ''))
                retrieved_chunks.append(chunk_id)
                retrieved_contents.append(content)
            
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            results.append({
                'query_id': query_id,
                'retrieved_chunks': retrieved_chunks,
                'retrieved_contents': retrieved_contents
            })
            
            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{len(ground_truth)}")
                
        except Exception as e:
            print(f"  [错误] 查询 {query_id} 失败: {e}")
            results.append({
                'query_id': query_id,
                'retrieved_chunks': [],
                'retrieved_contents': []
            })
    
    # 计算评估指标
    metrics = evaluate_retrieval_results(results, ground_truth, TOP_K)
    metrics['avg_response_time'] = sum(response_times) / len(response_times) if response_times else 0
    
    print(f"\n纯向量检索评估结果:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    # 清理资源
    del vector_db_manager
    clear_gpu_memory()
    
    return metrics


def run_rrf_fusion_experiment(ground_truth: Dict) -> Dict[str, Any]:
    """
    小实验2：RRF融合检索评估
    
    配置：
    - 向量检索权重: 0.6
    - BM25权重: 0.4
    - RRF平滑参数: 60
    - 关闭Reranker
    """
    print("\n" + "="*60)
    print("小实验2：RRF融合检索评估")
    print("="*60)
    
    config = {
        'top_k': TOP_K,
        'similarity_threshold': 0.1,
        'rrf_k': 60,
        'vector_weight': 0.6,
        'bm25_weight': 0.4,
        'enable_rerank': False
    }
    
    print(f"配置: {config}")
    
    # 初始化向量数据库
    from services.vector_db import VectorDatabaseManager
    from models import VectorDBConfig
    from services.embedding import embedding_service
    from models import EmbeddingConfig, EmbeddingModelType
    from config import settings
    from rank_bm25 import BM25Okapi
    import jieba
    
    vector_db_config = VectorDBConfig(
        db_type='faiss',
        dimension=settings.faiss_dimension,
        index_type="HNSW"
    )
    
    vector_db_manager = VectorDatabaseManager()
    vector_db_manager.initialize(vector_db_config)
    
    # 初始化嵌入模型
    embedding_config = EmbeddingConfig(
        model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
        model_name=settings.embedding_model_name,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device
    )
    
    embedding_service.load_model(embedding_config)
    
    # 获取所有文档元数据
    all_metadata = vector_db_manager.get_all_metadata()
    
    # 构建BM25语料库
    corpus = []
    chunk_ids = []
    chunk_content_map = {}
    for meta in all_metadata:
        content = meta.get('content', '')
        chunk_id = meta.get('chunk_id', meta.get('id', ''))
        corpus.append(content)
        chunk_ids.append(chunk_id)
        chunk_content_map[chunk_id] = content
    
    # 中文分词
    tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    print(f"BM25索引构建完成，文档数: {len(corpus)}")
    print(f"开始评估 {len(ground_truth)} 个查询...")
    
    results = []
    response_times = []
    
    for i, (query_id, data) in enumerate(ground_truth.items()):
        query_text = data.get('query', data.get('relevant_content', ''))
        
        try:
            start_time = time.time()
            
            # 1. 向量检索
            query_vector = embedding_service.encode([query_text])[0]
            distances, metadata_list = vector_db_manager.search(
                query_vector=query_vector,
                top_k=CANDIDATE_K
            )
            
            vector_results = {}
            for j, (dist, meta) in enumerate(zip(distances[0], metadata_list[0])):
                chunk_id = meta.get('chunk_id', meta.get('id', ''))
                vector_results[chunk_id] = {
                    'score': float(dist),
                    'rank': j + 1,
                    'content': meta.get('content', chunk_content_map.get(chunk_id, ''))
                }
            
            # 2. BM25检索
            tokenized_query = list(jieba.cut(query_text))
            bm25_scores = bm25.get_scores(tokenized_query)
            
            top_indices = np.argsort(bm25_scores)[::-1][:CANDIDATE_K]
            bm25_results = {}
            for j, idx in enumerate(top_indices):
                chunk_id = chunk_ids[idx]
                bm25_results[chunk_id] = {
                    'score': float(bm25_scores[idx]),
                    'rank': j + 1
                }
            
            # 3. RRF融合
            rrf_k = 60
            vector_weight = 0.6
            bm25_weight = 0.4
            
            rrf_scores = {}
            chunk_contents = {}
            
            for chunk_id, data in vector_results.items():
                rank = data['rank']
                score = vector_weight / (rrf_k + rank)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + score
                chunk_contents[chunk_id] = data['content']
            
            for chunk_id, data in bm25_results.items():
                rank = data['rank']
                score = bm25_weight / (rrf_k + rank)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + score
                if chunk_id not in chunk_contents:
                    chunk_contents[chunk_id] = chunk_content_map.get(chunk_id, '')
            
            # 按RRF分数排序
            sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 获取Top-K结果
            retrieved_chunks = []
            retrieved_contents = []
            for chunk_id, score in sorted_results[:TOP_K]:
                retrieved_chunks.append(chunk_id)
                retrieved_contents.append(chunk_contents.get(chunk_id, ''))
            
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            results.append({
                'query_id': query_id,
                'retrieved_chunks': retrieved_chunks,
                'retrieved_contents': retrieved_contents
            })
            
            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{len(ground_truth)}")
                
        except Exception as e:
            print(f"  [错误] 查询 {query_id} 失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'query_id': query_id,
                'retrieved_chunks': [],
                'retrieved_contents': []
            })
    
    # 计算评估指标
    metrics = evaluate_retrieval_results(results, ground_truth, TOP_K)
    metrics['avg_response_time'] = sum(response_times) / len(response_times) if response_times else 0
    
    print(f"\nRRF融合检索评估结果:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    # 清理资源
    del vector_db_manager, bm25
    clear_gpu_memory()
    
    return metrics


def main():
    """主函数"""
    print("="*60)
    print("实验一：纯向量检索 vs RRF融合检索准确率对比")
    print("="*60)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"开始时间: {timestamp}")
    
    # 加载测试数据
    ground_truth_path = project_root / "tests" / "retrieval_comparison" / "test_queries_with_questions.json"
    if not ground_truth_path.exists():
        ground_truth_path = project_root.parent / "retrieval_test_cases_ground_truth.json"
    ground_truth = load_ground_truth(str(ground_truth_path))
    print(f"加载 {len(ground_truth)} 个测试用例")
    
    # 小实验1：纯向量检索
    vector_metrics = run_vector_only_experiment(ground_truth)
    
    # 等待资源释放
    wait_for_memory(5)
    
    # 小实验2：RRF融合检索
    rrf_metrics = run_rrf_fusion_experiment(ground_truth)
    
    # 生成对比报告
    experiment_config = {
        '向量检索模型': 'BGE-M3',
        '向量维度': '1024',
        'RRF参数K': '60',
        '向量权重': '0.6',
        'BM25权重': '0.4',
        'Top-K': str(TOP_K),
        '测试用例数': str(len(ground_truth))
    }
    
    generate_comparison_report(
        experiment_name="纯向量检索与RRF融合检索准确率对比",
        baseline_name="纯向量检索",
        optimized_name="RRF融合检索",
        baseline_metrics=vector_metrics,
        optimized_metrics=rrf_metrics,
        output_path=str(OUTPUT_PATH),
        experiment_config=experiment_config
    )
    
    print("\n" + "="*60)
    print("实验一完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
