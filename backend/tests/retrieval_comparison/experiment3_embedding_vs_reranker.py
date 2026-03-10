"""
实验三：Qwen3-Embedding-4B vs BGE-Reranker-Large 精排对比

对比两种精排方式：
1. Qwen3-Embedding-4B: Embedding-based Reranking (计算query和doc的embedding，用cosine similarity重排)
2. BGE-Reranker-Large: Cross-Encoder Reranking (直接对query-doc文本对打分)
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
OUTPUT_PATH = project_root.parent / "test_reports" / "Qwen3-Embedding-4B&bge-reranker-large对比.md"

# 模型路径
QWEN3_EMBEDDING_PATH = "/root/autodl-tmp/models/Qwen3-Embedding-4B"
BGE_RERANKER_PATH = "/root/autodl-tmp/models/bge-reranker-large"
BGE_M3_PATH = "/root/autodl-tmp/models/bge-m3"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def run_embedding_rerank_experiment(ground_truth: Dict) -> Dict[str, Any]:
    """
    小实验1：Qwen3-Embedding-4B Embedding-based Reranking
    
    工作原理：
    1. 用BGE-M3进行RRF融合检索获取候选
    2. 用Qwen3-Embedding-4B计算query和每个候选doc的embedding
    3. 用cosine similarity重新排序
    """
    print("\n" + "="*60)
    print("小实验1：Qwen3-Embedding-4B Embedding-based Reranking")
    print("="*60)
    
    config = {
        'top_k': TOP_K,
        'candidate_k': CANDIDATE_K,
        'rrf_k': 60,
        'vector_weight': 0.6,
        'bm25_weight': 0.4,
        'rerank_method': 'embedding',
        'rerank_model': 'Qwen3-Embedding-4B'
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
    
    # 初始化BGE-M3嵌入模型（用于RRF检索）
    embedding_config = EmbeddingConfig(
        model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
        model_name="BAAI/bge-m3",
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device
    )
    
    embedding_service.load_model(embedding_config)
    
    # 初始化Qwen3-Embedding-4B用于精排
    print(f"加载Qwen3-Embedding-4B用于精排...")
    import torch
    from transformers import AutoModel, AutoTokenizer
    
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN3_EMBEDDING_PATH, trust_remote_code=True)
    qwen_model = AutoModel.from_pretrained(QWEN3_EMBEDDING_PATH, trust_remote_code=True)
    qwen_model.to('cuda')
    qwen_model.eval()
    qwen_dimension = qwen_model.config.hidden_size
    print(f"Qwen3-Embedding-4B加载成功，维度: {qwen_dimension}")
    
    def get_qwen_embedding(texts):
        """使用Qwen3-Embedding-4B获取embedding"""
        with torch.no_grad():
            inputs = qwen_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            outputs = qwen_model(**inputs)
            # 使用last_hidden_state的mean pooling
            attention_mask = inputs['attention_mask']
            embeddings = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            masked_embeddings = embeddings * mask
            summed = torch.sum(masked_embeddings, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = summed / counts
            # Normalize
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            return normalized.cpu().numpy()
    
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
            
            # 1. 用BGE-M3进行向量检索
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
            
            # 3. RRF融合获取候选
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
            
            # 按RRF分数排序获取候选
            sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 获取候选文档
            candidate_chunks = []
            candidate_contents = []
            for chunk_id, score in sorted_results[:CANDIDATE_K]:
                candidate_chunks.append(chunk_id)
                candidate_contents.append(chunk_contents.get(chunk_id, ''))
            
            # 4. 用Qwen3-Embedding-4B进行精排
            # 计算query的embedding
            query_embedding = get_qwen_embedding([query_text])[0]
            
            # 计算候选文档的embedding
            doc_embeddings = get_qwen_embedding(candidate_contents)
            
            # 计算cosine similarity（因为已经normalize，直接点积即可）
            similarities = np.dot(doc_embeddings, query_embedding)
            
            # 按相似度排序
            ranked_indices = np.argsort(similarities)[::-1]
            
            # 获取Top-K结果
            retrieved_chunks = []
            retrieved_contents = []
            for idx in ranked_indices[:TOP_K]:
                retrieved_chunks.append(candidate_chunks[idx])
                retrieved_contents.append(candidate_contents[idx])
            
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
    
    print(f"\nQwen3-Embedding-4B精排评估结果:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    # 清理资源
    del vector_db_manager, bm25, qwen_model, qwen_tokenizer
    clear_gpu_memory()
    
    return metrics


def run_reranker_experiment(ground_truth: Dict) -> Dict[str, Any]:
    """
    小实验2：BGE-Reranker-Large Cross-Encoder Reranking
    
    工作原理：
    1. 用BGE-M3进行RRF融合检索获取候选
    2. 用BGE-Reranker-Large直接对(query, doc)文本对打分
    3. 按分数重新排序
    """
    print("\n" + "="*60)
    print("小实验2：BGE-Reranker-Large Cross-Encoder Reranking")
    print("="*60)
    
    config = {
        'top_k': TOP_K,
        'candidate_k': CANDIDATE_K,
        'rrf_k': 60,
        'vector_weight': 0.6,
        'bm25_weight': 0.4,
        'rerank_method': 'cross-encoder',
        'rerank_model': 'BGE-Reranker-Large'
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
    
    # 初始化BGE-M3嵌入模型（用于RRF检索）
    embedding_config = EmbeddingConfig(
        model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
        model_name="BAAI/bge-m3",
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device
    )
    
    embedding_service.load_model(embedding_config)
    
    # 初始化BGE-Reranker-Large
    print(f"加载BGE-Reranker-Large用于精排...")
    from FlagEmbedding import FlagReranker
    
    reranker = FlagReranker(BGE_RERANKER_PATH, use_fp16=False)
    print(f"BGE-Reranker-Large加载成功")
    
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
    print(f"Reranker初始化完成")
    print(f"开始评估 {len(ground_truth)} 个查询...")
    
    results = []
    response_times = []
    
    for i, (query_id, data) in enumerate(ground_truth.items()):
        query_text = data.get('query', data.get('relevant_content', ''))
        
        try:
            start_time = time.time()
            
            # 1. 用BGE-M3进行向量检索
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
            
            # 3. RRF融合获取候选
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
            
            # 按RRF分数排序获取候选
            sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 获取候选文档
            candidate_chunks = []
            candidate_contents = []
            for chunk_id, score in sorted_results[:CANDIDATE_K]:
                candidate_chunks.append(chunk_id)
                candidate_contents.append(chunk_contents.get(chunk_id, ''))
            
            # 4. 用BGE-Reranker-Large进行精排
            # 准备输入对
            pairs = [[query_text, doc] for doc in candidate_contents]
            
            # 计算重排序分数
            rerank_scores = reranker.compute_score(pairs)
            
            # 按分数排序
            if isinstance(rerank_scores, list):
                ranked_indices = np.argsort(rerank_scores)[::-1]
            else:
                ranked_indices = [0]  # 单个结果的情况
            
            # 获取Top-K结果
            retrieved_chunks = []
            retrieved_contents = []
            for idx in ranked_indices[:TOP_K]:
                retrieved_chunks.append(candidate_chunks[idx])
                retrieved_contents.append(candidate_contents[idx])
            
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
    
    print(f"\nBGE-Reranker-Large精排评估结果:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    # 清理资源
    del vector_db_manager, bm25, reranker
    clear_gpu_memory()
    
    return metrics


def main():
    """主函数"""
    print("="*60)
    print("实验三：Qwen3-Embedding-4B vs BGE-Reranker-Large 精排对比")
    print("="*60)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"开始时间: {timestamp}")
    
    # 加载测试数据
    ground_truth_path = project_root / "tests" / "retrieval_comparison" / "test_queries_with_questions.json"
    if not ground_truth_path.exists():
        ground_truth_path = project_root.parent / "retrieval_test_cases_ground_truth.json"
    ground_truth = load_ground_truth(str(ground_truth_path))
    print(f"加载 {len(ground_truth)} 个测试用例")
    
    # 小实验1：Qwen3-Embedding-4B Embedding-based Reranking
    embedding_metrics = run_embedding_rerank_experiment(ground_truth)
    
    # 等待资源释放
    wait_for_memory(5)
    
    # 小实验2：BGE-Reranker-Large Cross-Encoder Reranking
    reranker_metrics = run_reranker_experiment(ground_truth)
    
    # 生成对比报告
    experiment_config = {
        '检索模型': 'BGE-M3',
        '向量维度': '1024',
        '精排模型1': 'Qwen3-Embedding-4B (Embedding-based)',
        '精排模型2': 'BGE-Reranker-Large (Cross-Encoder)',
        'RRF参数K': '60',
        '向量权重': '0.6',
        'BM25权重': '0.4',
        'Top-K': str(TOP_K),
        '候选数量': str(CANDIDATE_K),
        '测试用例数': str(len(ground_truth))
    }
    
    generate_comparison_report(
        experiment_name="Qwen3-Embedding-4B&bge-reranker-large对比",
        baseline_name="Qwen3-Embedding-4B (Embedding-based)",
        optimized_name="BGE-Reranker-Large (Cross-Encoder)",
        baseline_metrics=embedding_metrics,
        optimized_metrics=reranker_metrics,
        output_path=str(OUTPUT_PATH),
        experiment_config=experiment_config
    )
    
    print("\n" + "="*60)
    print("实验三完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
