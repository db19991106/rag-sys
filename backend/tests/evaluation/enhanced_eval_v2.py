#!/usr/bin/env python3
"""
RAG系统增强测评脚本 V2 - 完整测评指标体系
支持：NDCG、Recall、F1、语义相似度、主题覆盖率等
"""

import sys
import json
import time
import argparse
import math
import numpy as np
from pathlib import Path
from datetime import datetime
import statistics
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.embedding import embedding_service
from services.vector_db import vector_db_manager
from services.rag_evaluator import rag_evaluator
from models import (
    RetrievalConfig,
    EmbeddingConfig,
    VectorDBConfig,
    EmbeddingModelType,
    VectorDBType,
)
from config import settings


class EnhancedRAGEvaluator:
    """增强版RAG系统测评器 - 完整指标体系"""

    def __init__(self, output_dir: str = "test_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.evaluation_time = datetime.now()

    def init_services(self) -> bool:
        """初始化所有服务"""
        print("🔧 初始化服务...")

        try:
            # 加载768维嵌入模型（与向量库匹配）
            print("   加载嵌入模型(bge-base-zh-v1.5, 768维)...")
            embedding_service.load_model(
                EmbeddingConfig(
                    model_type=EmbeddingModelType.BGE,
                    model_name="BAAI/bge-base-zh-v1.5",
                    device="cpu",
                )
            )
            print(f"   ✅ 模型维度: {embedding_service.get_dimension()}")

            # 初始化向量数据库
            print("   初始化向量数据库...")
            vector_db_manager.initialize(
                VectorDBConfig(
                    db_type=VectorDBType.FAISS,
                    dimension=embedding_service.get_dimension(),
                    index_type="HNSW",
                )
            )
            status = vector_db_manager.get_status()
            print(f"   ✅ 向量库: {status.total_vectors} 个向量")

            return True

        except Exception as e:
            print(f"   ❌ 初始化失败: {e}")
            return False

    def load_test_data(self, test_file: str) -> Dict[str, Any]:
        """加载测试数据"""
        if not Path(test_file).exists():
            # 尝试在test_data目录查找
            test_file = Path(__file__).parent.parent.parent / "test_data" / test_file

        if not Path(test_file).exists():
            raise FileNotFoundError(f"测试文件不存在: {test_file}")

        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        print(f"✅ 加载测试数据: {test_file}")
        return test_data

    def calculate_ndcg_at_k(
        self, results: List[Any], ground_truth: List[str], k: int = 5
    ) -> float:
        """计算NDCG@K (归一化折损累积增益)"""
        if not ground_truth:
            return 0.0

        # 计算DCG
        dcg = 0.0
        for i, result in enumerate(results[:k]):
            # 检查这个结果是否在ground_truth中
            relevance = 0
            for gt in ground_truth:
                if gt in result.content or result.content in gt:
                    relevance = 1
                    break
            # 折损因子: log2(i+2)，因为i从0开始
            dcg += relevance / math.log2(i + 2)

        # 计算理想DCG (IDCG)
        idcg = 0.0
        for i in range(min(len(ground_truth), k)):
            idcg += 1.0 / math.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def calculate_recall_at_k(
        self, results: List[Any], ground_truth: List[str], k: int = 5
    ) -> float:
        """计算Recall@K"""
        if not ground_truth:
            return 0.0

        # 统计在top-k结果中找到的ground_truth数量
        found = 0
        for gt in ground_truth:
            for result in results[:k]:
                if gt in result.content or result.content in gt:
                    found += 1
                    break

        return found / len(ground_truth)

    def calculate_f1_at_k(
        self, precision: float, recall: float
    ) -> float:
        """计算F1@K (精确率和召回率的调和平均)"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def calculate_semantic_similarity(
        self, text1: str, text2: str
    ) -> float:
        """计算两段文本的语义相似度 (使用embedding)"""
        try:
            # 编码两段文本
            embedding1 = embedding_service.encode([text1])
            embedding2 = embedding_service.encode([text2])

            # 计算余弦相似度
            similarity = np.dot(embedding1[0], embedding2[0]) / (
                np.linalg.norm(embedding1[0]) * np.linalg.norm(embedding2[0])
            )

            return float(similarity)
        except Exception as e:
            print(f"   ⚠️ 语义相似度计算失败: {e}")
            return 0.0

    def calculate_topic_coverage(
        self, results: List[Any], expected_topics: List[str]
    ) -> Dict[str, Any]:
        """计算主题覆盖率"""
        if not expected_topics:
            return {"coverage_rate": 0.0, "covered_topics": [], "missed_topics": []}

        # 合并所有检索结果文本
        retrieved_text = " ".join([r.content for r in results]).lower()

        # 检查每个期望主题是否被覆盖
        covered_topics = []
        missed_topics = []

        for topic in expected_topics:
            # 简化的主题匹配（实际应用中可能需要更复杂的语义匹配）
            if topic.lower() in retrieved_text:
                covered_topics.append(topic)
            else:
                missed_topics.append(topic)

        coverage_rate = len(covered_topics) / len(expected_topics) if expected_topics else 0.0

        return {
            "coverage_rate": coverage_rate,
            "covered_topics": covered_topics,
            "missed_topics": missed_topics,
            "total_topics": len(expected_topics),
            "covered_count": len(covered_topics),
        }

    def run_retrieval_test(
        self, query: str, expected_keywords: list, case_info: dict
    ) -> Dict[str, Any]:
        """运行单个检索测试（增强版，包含完整指标）"""
        # 保存查询到evaluator用于MRR估算
        rag_evaluator._last_query = query

        # 向量化查询（使用本地BGE模型）
        query_vector = embedding_service.encode([query])

        # 检索
        start = time.time()
        scores, metadatas = vector_db_manager.search(query_vector, top_k=10)  # 检索更多结果用于计算NDCG
        elapsed = (time.time() - start) * 1000

        # 构建结果对象
        class FakeResult:
            def __init__(self, content, similarity, document_id, chunk_id):
                self.content = content
                self.similarity = similarity
                self.document_id = document_id
                self.chunk_id = chunk_id

        results = []
        for i, (score, meta) in enumerate(zip(scores[0], metadatas[0])):
            results.append(
                FakeResult(
                    content=meta.get("content", ""),
                    similarity=float(score),
                    document_id=meta.get("document_id", ""),
                    chunk_id=meta.get("chunk_id", f"chunk_{i}"),
                )
            )

        # ========== 关键词命中统计 ==========
        retrieved_text = " ".join([r.content for r in results])
        hits = sum(1 for kw in expected_keywords if kw in retrieved_text)
        hit_rate = hits / len(expected_keywords) if expected_keywords else 0

        matched_keywords = [kw for kw in expected_keywords if kw in retrieved_text]
        missed_keywords = [kw for kw in expected_keywords if kw not in retrieved_text]

        # ========== 基础评估指标 ==========
        ground_truth = case_info.get("ground_truth", [])
        eval_result = rag_evaluator.evaluate_retrieval(query, results[:5], ground_truth)

        # ========== 新增：NDCG@K ==========
        ndcg_at_5 = self.calculate_ndcg_at_k(results, [ground_truth] if isinstance(ground_truth, str) else ground_truth, k=5)

        # ========== 新增：Recall@K ==========
        recall_at_5 = self.calculate_recall_at_k(results, [ground_truth] if isinstance(ground_truth, str) else ground_truth, k=5)

        # ========== 新增：F1@K ==========
        precision_at_5 = eval_result.get("precision_at_5", 0)
        f1_at_5 = self.calculate_f1_at_k(precision_at_5, recall_at_5)

        # ========== 新增：语义相似度（如果有ground_truth）==========
        semantic_similarity = 0.0
        if ground_truth and isinstance(ground_truth, str) and results:
            # 计算查询与top1结果的语义相似度
            semantic_similarity = self.calculate_semantic_similarity(query, results[0].content)

        # ========== 新增：主题覆盖率 ==========
        expected_topics = case_info.get("expected_topics", [])
        topic_coverage = self.calculate_topic_coverage(results, expected_topics)

        return {
            "case_info": case_info,
            "query": query,
            "response_time_ms": elapsed,
            "results_count": len(results),
            "results": [
                {
                    "rank": i + 1,
                    "similarity": r.similarity,
                    "content": r.content[:80] + "...",
                    "chunk_id": r.chunk_id,
                    "document_id": r.document_id,
                }
                for i, r in enumerate(results[:5])
            ],
            "keyword_analysis": {
                "hit_rate": hit_rate,
                "hits": hits,
                "total_keywords": len(expected_keywords),
                "matched": matched_keywords,
                "missed": missed_keywords,
            },
            "topic_coverage": topic_coverage,
            "metrics": {
                # 基础指标
                "precision_at_1": eval_result.get("precision_at_1", 0),
                "precision_at_3": eval_result.get("precision_at_3", 0),
                "precision_at_5": eval_result.get("precision_at_5", 0),
                "recall_at_5": recall_at_5,
                "f1_at_5": f1_at_5,
                "ndcg_at_5": ndcg_at_5,
                "mrr": eval_result.get("mrr", 0),
                "context_precision": eval_result.get("context_precision", 0),
                "context_recall": eval_result.get("context_recall", 0),
                # 新增语义指标
                "semantic_similarity": semantic_similarity,
            },
            "model_info": {
                "embedding_model": "BAAI/bge-base-zh-v1.5 (本地)",
                "vector_db": "FAISS (本地)",
                "llm_provider": "local (Qwen2.5-7B-Instruct)",
            },
        }
