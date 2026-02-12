#!/usr/bin/env python3
"""
RAG系统增强测评脚本 - AutoDL适配版
支持本地模型路径和GPU加速
"""

import os
# 禁用 stdout 重定向，避免与脚本自身的 logging 冲突
os.environ['RAG_DISABLE_STDOUT_REDIRECT'] = 'true'

import sys
import json
import time
import math
import logging
import argparse
from pathlib import Path
from datetime import datetime
import statistics
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
import numpy as np

# 配置日志 - 使用eval_config中的日志配置
import sys
import os
from pathlib import Path

# 获取eval_config中的日志配置
from eval_config import LOG_CONFIG

# 确保日志目录存在
log_file = LOG_CONFIG.get("log_file")
if log_file:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_CONFIG.get("log_level", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), encoding="utf-8")
        if log_file
        else logging.NullHandler(),
    ],
    force=True,  # 强制重新配置
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 导入eval_config配置
from eval_config import (
    VECTOR_DB_DIR,
    MODELS_DIR,
    TEST_DATASET_PATH,
)

# 导入RAG生成器
from services.rag_generator import rag_generator
from models import RetrievalConfig, GenerationConfig

# ==================== 文本相似度评估指标 ====================

def calculate_bleu(reference: str, candidate: str, max_n: int = 4) -> Dict[str, float]:
    """
    计算BLEU分数（基于n-gram精确率的几何平均）
    
    Args:
        reference: 参考文本（Ground Truth）
        candidate: 候选文本（LLM生成）
        max_n: 最大n-gram阶数
    
    Returns:
        BLEU-1到BLEU-4的分数
    """
    import re
    from collections import Counter
    
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def tokenize(text):
        # 简单的中文分词（按字符分词）
        text = re.sub(r'[^\w\s]', ' ', text)
        return list(text.replace(' ', ''))
    
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    
    if len(cand_tokens) == 0:
        return {f'bleu_{n}': 0.0 for n in range(1, max_n+1)}
    
    results = {}
    for n in range(1, max_n+1):
        ref_ngrams = Counter(get_ngrams(ref_tokens, n))
        cand_ngrams = Counter(get_ngrams(cand_tokens, n))
        
        matches = sum((cand_ngrams & ref_ngrams).values())
        total = sum(cand_ngrams.values())
        
        if total == 0:
            results[f'bleu_{n}'] = 0.0
        else:
            # 简化版BLEU（无短句惩罚）
            results[f'bleu_{n}'] = matches / total
    
    return results

def calculate_rouge(reference: str, candidate: str) -> Dict[str, float]:
    """
    计算ROUGE分数（基于召回率的n-gram重叠）
    
    Args:
        reference: 参考文本（Ground Truth）
        candidate: 候选文本（LLM生成）
    
    Returns:
        ROUGE-1, ROUGE-2, ROUGE-L分数
    """
    import re
    
    def tokenize(text):
        text = re.sub(r'[^\w\s]', ' ', text)
        return list(text.replace(' ', ''))
    
    def lcs_length(X, Y):
        """计算最长公共子序列长度"""
        m, n = len(X), len(Y)
        if m == 0 or n == 0:
            return 0
        
        # 使用滚动数组优化空间
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i-1] == Y[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev, curr = curr, prev
        
        return prev[n]
    
    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)
    
    results = {}
    
    # ROUGE-N (N-gram recall)
    for n in [1, 2]:
        ref_ngrams = set()
        cand_ngrams = set()
        
        for i in range(len(ref_tokens) - n + 1):
            ref_ngrams.add(tuple(ref_tokens[i:i+n]))
        for i in range(len(cand_tokens) - n + 1):
            cand_ngrams.add(tuple(cand_tokens[i:i+n]))
        
        if len(ref_ngrams) == 0:
            results[f'rouge_{n}'] = 0.0
        else:
            overlap = len(ref_ngrams & cand_ngrams)
            results[f'rouge_{n}'] = overlap / len(ref_ngrams)
    
    # ROUGE-L (最长公共子序列)
    lcs = lcs_length(ref_tokens, cand_tokens)
    if len(ref_tokens) == 0:
        results['rouge_l'] = 0.0
    else:
        results['rouge_l'] = lcs / len(ref_tokens)
    
    return results

def calculate_answer_metrics(reference: str, candidate: str) -> Dict[str, float]:
    """
    计算答案质量综合指标
    
    Args:
        reference: 参考文本（Ground Truth）
        candidate: 候选文本（LLM生成）
    
    Returns:
        包含BLEU、ROUGE、语义相似度等的综合指标
    """
    metrics = {}
    
    # 1. BLEU分数
    bleu_scores = calculate_bleu(reference, candidate)
    metrics.update(bleu_scores)
    # 计算平均BLEU
    metrics['bleu_avg'] = sum(bleu_scores.values()) / len(bleu_scores)
    
    # 2. ROUGE分数
    rouge_scores = calculate_rouge(reference, candidate)
    metrics.update(rouge_scores)
    # 计算平均ROUGE
    metrics['rouge_avg'] = sum(rouge_scores.values()) / len(rouge_scores)
    
    # 3. 字符级精确率和召回率
    ref_set = set(reference)
    cand_set = set(candidate)
    
    if len(cand_set) > 0:
        metrics['char_precision'] = len(ref_set & cand_set) / len(cand_set)
    else:
        metrics['char_precision'] = 0.0
    
    if len(ref_set) > 0:
        metrics['char_recall'] = len(ref_set & cand_set) / len(ref_set)
    else:
        metrics['char_recall'] = 0.0
    
    if metrics['char_precision'] + metrics['char_recall'] > 0:
        metrics['char_f1'] = 2 * metrics['char_precision'] * metrics['char_recall'] / (metrics['char_precision'] + metrics['char_recall'])
    else:
        metrics['char_f1'] = 0.0
    
    # 4. 答案长度比
    metrics['length_ratio'] = len(candidate) / len(reference) if len(reference) > 0 else 0.0
    
    return metrics

from services.embedding import embedding_service
from services.vector_db import vector_db_manager
from services.rag_evaluator import rag_evaluator
from services.retriever import Retriever
from services.reranker import reranker_manager
from models import (
    RetrievalConfig,
    EmbeddingConfig,
    VectorDBConfig,
    EmbeddingModelType,
    VectorDBType,
    ChunkInfo,
)
# from config import settings  # 注释掉，避免日志冲突


@dataclass
class FakeResult:
    """模拟检索结果对象"""

    content: str
    similarity: float
    document_id: str
    chunk_id: str
    rank: int = 0
    rerank_score: Optional[float] = None


class RAGEvaluator:
    """RAG系统测评器"""

    def __init__(
        self,
        output_dir: str = "test_reports",
        model_base_path: Optional[str] = None,
        vector_db_path: Optional[str] = None,
        keep_llm_loaded: bool = True,  # 是否保持LLM模型常驻显存
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.evaluation_time = datetime.now()
        # 使用传入的路径或eval_config中的默认路径
        self.model_base_path = Path(model_base_path) if model_base_path else MODELS_DIR
        self.vector_db_path = Path(vector_db_path) if vector_db_path else VECTOR_DB_DIR
        # LLM常驻显存配置
        self.keep_llm_loaded = keep_llm_loaded
        self._llm_client = None  # 缓存LLM客户端
        self._llm_config = None  # 缓存LLM配置

    def calculate_ndcg_at_k(
        self, results: List[FakeResult], ground_truth: List[str], k: int = 5
    ) -> float:
        """计算NDCG@K - 归一化折损累积增益"""
        if not ground_truth or not results:
            return 0.0

        dcg = 0.0
        for i, result in enumerate(results[:k]):
            relevance = 0.0
            for gt in ground_truth:
                # 使用统一的匹配检查
                if self._check_text_match(result.content, gt, use_semantic=False):
                    relevance = 1.0
                    break

                # 语义相似度（用于计算相关性分数）
                try:
                    sim = self.calculate_semantic_similarity(gt, result.content[:500])
                    relevance = max(relevance, sim)
                except (RuntimeError, ValueError, TypeError):
                    pass

            # 限制relevance在[0,1]范围内，防止NDCG>1
            relevance = min(relevance, 1.0)
            if relevance > 0:
                dcg += (2**relevance - 1) / math.log2(i + 2)

        # 计算理想DCG（前k个结果都完全相关）
        ideal_relevances = [1.0] * k

        idcg = sum(
            (2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_relevances)
        )

        return dcg / idcg if idcg > 0 else 0.0

    def calculate_recall_at_k(
        self, results: List[FakeResult], ground_truth: List[str], k: int = 5
    ) -> float:
        """计算Recall@K - 召回率@K"""
        if not ground_truth:
            return 0.0

        covered_ground_truths = set()
        for gt in ground_truth:
            for result in results[:k]:
                if self._check_text_match(
                    result.content, gt, use_semantic=True, semantic_threshold=0.6
                ):
                    covered_ground_truths.add(gt)
                    break

        return len(covered_ground_truths) / len(ground_truth)

    def calculate_precision_at_k(
        self, results: List[FakeResult], ground_truth: List[str], k: int
    ) -> float:
        """计算Precision@K"""
        if not results or not ground_truth or k <= 0:
            return 0.0

        relevant_count = 0
        for result in results[:k]:
            for gt in ground_truth:
                if self._check_text_match(
                    result.content, gt, use_semantic=True, semantic_threshold=0.6
                ):
                    relevant_count += 1
                    break

        return relevant_count / k

    def calculate_mrr(
        self, results: List[FakeResult], ground_truth: List[str]
    ) -> float:
        """计算MRR - Mean Reciprocal Rank"""
        if not ground_truth or not results:
            return 0.0

        for i, result in enumerate(results[:5], 1):
            for gt in ground_truth:
                if self._check_text_match(
                    result.content, gt, use_semantic=True, semantic_threshold=0.6
                ):
                    return 1.0 / i
        return 0.0

    def calculate_f1_at_k(self, precision: float, recall: float) -> float:
        """计算F1@K - F1分数"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _check_text_match(
        self,
        text: str,
        ground_truth: str,
        use_semantic: bool = False,
        semantic_threshold: float = 0.6,
    ) -> bool:
        """统一的文本匹配检查逻辑

        Args:
            text: 待检查的文本
            ground_truth: 基准文本
            use_semantic: 是否使用语义相似度匹配
            semantic_threshold: 语义相似度阈值

        Returns:
            是否匹配
        """
        text_lower = text.lower()
        gt_lower = ground_truth.lower()

        # 1. 完全包含匹配
        if gt_lower in text_lower or text_lower in gt_lower:
            return True

        # 2. 部分匹配（对于较长的ground_truth）
        if len(gt_lower) > 4:
            gt_parts = gt_lower.split()
            if len(gt_parts) > 1:
                match_count = sum(
                    1 for part in gt_parts if len(part) > 2 and part in text_lower
                )
                if match_count >= len(gt_parts) * 0.5:
                    return True

        # 3. 语义相似度匹配
        if use_semantic:
            try:
                sim = self.calculate_semantic_similarity(ground_truth, text[:500])
                if sim > semantic_threshold:
                    return True
            except (RuntimeError, ValueError, TypeError) as e:
                logger.debug(f"语义匹配检查失败: {e}")

        return False

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度 - 使用embedding模型"""
        # 检查模型是否已加载
        if not embedding_service.is_loaded():
            logger.warning("嵌入模型未加载，无法计算语义相似度")
            return 0.0

        try:
            embeddings = embedding_service.encode([text1, text2])

            norm1 = np.linalg.norm(embeddings[0])
            norm2 = np.linalg.norm(embeddings[1])

            if norm1 == 0 or norm2 == 0:
                return 0.0

            sim = np.dot(embeddings[0], embeddings[1]) / (norm1 * norm2)
            return float(sim)
        except (RuntimeError, ValueError, TypeError) as e:
            logger.warning(f"语义相似度计算失败: {e}")
            return 0.0

    def calculate_topic_coverage(
        self, results: List[FakeResult], expected_topics: List[str]
    ) -> Dict[str, Any]:
        """计算主题覆盖率"""
        if not expected_topics:
            return {
                "coverage_rate": 0.0,
                "covered_topics": [],
                "missed_topics": [],
                "total_topics": 0,
                "covered_count": 0,
            }

        retrieved_text = " ".join([r.content for r in results]).lower()

        covered_topics = []
        missed_topics = []

        for topic in expected_topics:
            if topic.lower() in retrieved_text:
                covered_topics.append(topic)
            else:
                missed_topics.append(topic)

        coverage_rate = (
            len(covered_topics) / len(expected_topics) if expected_topics else 0.0
        )

        return {
            "coverage_rate": coverage_rate,
            "covered_topics": covered_topics,
            "missed_topics": missed_topics,
            "total_topics": len(expected_topics),
            "covered_count": len(covered_topics),
        }

    def apply_reranking(
        self, results: List[FakeResult], query: str, expected_keywords: List[str]
    ) -> List[FakeResult]:
        """应用基于关键词匹配和语义相似度的重排序算法"""

        def calculate_rerank_score(
            result: FakeResult, query: str, keywords: List[str]
        ) -> float:
            """计算重排序分数"""
            score = result.similarity * 0.4

            content_lower = result.content.lower()
            if keywords:
                keyword_match_count = sum(
                    1 for kw in keywords if kw.lower() in content_lower
                )
                keyword_score = keyword_match_count / len(keywords)
                score += keyword_score * 0.3

            if "酒店" in query or "住宿" in query:
                hotel_keywords = ["三星级", "四星级", "五星级", "快捷酒店", "经济型"]
                hotel_match_count = sum(
                    1 for hk in hotel_keywords if hk in result.content
                )
                if hotel_match_count > 0:
                    score += min(hotel_match_count * 0.05, 0.15)

            level_keywords = {
                "8-9级": ["8-9级", "普通员工", "工程师", "专员"],
                "10-11级": ["10-11级", "经理", "主管"],
                "12级": ["12级", "总监", "专家", "高级"],
            }
            for level_key, level_words in level_keywords.items():
                if level_key in query:
                    level_match_count = sum(
                        1 for lw in level_words if lw in result.content
                    )
                    if level_match_count > 0:
                        score += min(level_match_count * 0.03, 0.1)
                        break

            city_keywords = {
                "一线城市": ["上海", "北京", "广州", "深圳", "一线城市", "北上广深"],
                "新一线": [
                    "成都",
                    "杭州",
                    "武汉",
                    "西安",
                    "南京",
                    "重庆",
                    "新一线",
                    "新一线城市",
                ],
            }
            for city_type, cities in city_keywords.items():
                if any(c in query for c in cities):
                    if any(c in result.content for c in cities):
                        score += 0.05
                        break

            return score

        scored_results = []
        for result in results:
            rerank_score = calculate_rerank_score(result, query, expected_keywords)
            new_result = FakeResult(
                content=result.content,
                similarity=result.similarity,
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                rank=result.rank,
                rerank_score=rerank_score,
            )
            scored_results.append((new_result, rerank_score))

        scored_results.sort(key=lambda x: x[1], reverse=True)

        final_results = []
        for i, (result, _) in enumerate(scored_results):
            result.rank = i + 1
            final_results.append(result)

        return final_results

    def init_services(
        self, enable_rerank: bool = True, reranker_type: str = "bge"
    ) -> bool:
        """初始化所有服务（AutoDL本地路径版）"""
        print("🔧 初始化服务...")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            # 加载本地 Embedding 模型
            embedding_model_path = self.model_base_path / "bge-base-zh-v1.5"
            print(f"   加载本地嵌入模型: {embedding_model_path}")
            print(f"   使用设备: {device}")

            if not embedding_model_path.exists():
                print(f"   ❌ 模型路径不存在: {embedding_model_path}")
                print(
                    f"   请从 ModelScope 下载: modelscope download --model BAAI/bge-base-zh-v1.5 --local_dir {embedding_model_path}"
                )
                return False

            embedding_service.load_model(
                EmbeddingConfig(
                    model_type=EmbeddingModelType.BGE,
                    model_name=str(embedding_model_path),
                    device=device,
                )
            )
            print(f"   ✅ 模型维度: {embedding_service.get_dimension()}")

            # 加载本地向量数据库
            print(f"   加载向量数据库: {self.vector_db_path}")
            if not self.vector_db_path.exists():
                print(f"   ❌ 向量库路径不存在: {self.vector_db_path}")
                return False

            vector_db_manager.initialize(
                VectorDBConfig(
                    db_type=VectorDBType.FAISS,
                    dimension=embedding_service.get_dimension(),
                    index_type="HNSW",
                    index_path=str(self.vector_db_path),
                )
            )
            status = vector_db_manager.get_status()
            if status.total_vectors == 0:
                print(f"   ❌ 向量库为空，请先向量化文档")
                return False
            print(f"   ✅ 向量库: {status.total_vectors} 个向量")

            # 初始化重排序器（如果启用）
            if enable_rerank and reranker_type != "none":
                reranker_model_path = self.model_base_path / "bge-reranker-base"
                print(f"   初始化重排序器: {reranker_type}")

                if reranker_model_path.exists():
                    print(f"   使用本地模型: {reranker_model_path}")
                    reranker_manager.initialize(
                        reranker_type=reranker_type,
                        model_name=str(reranker_model_path),
                        device=device,
                        top_k=10,
                        threshold=0.0,
                    )
                else:
                    print(f"   ⚠️  本地重排序模型不存在: {reranker_model_path}")
                    print(
                        f"   请从 ModelScope 下载: modelscope download --model BAAI/bge-reranker-base --local_dir {reranker_model_path}"
                    )
                    print(f"   暂时使用基础重排序（无模型）...")
                    reranker_manager.initialize(
                        reranker_type="none",  # 使用规则重排序
                        device=device,
                        top_k=10,
                        threshold=0.0,
                    )
            else:
                print("   ⚠️  重排序器: 已禁用")

            return True

        except Exception as e:
            print(f"   ❌ 初始化失败: {e}")
            logger.error(f"服务初始化失败: {e}", exc_info=True)
            return False

    def load_test_data(self, test_file: str) -> Dict[str, Any]:
        """加载测试数据"""
        if not Path(test_file).exists():
            test_file = Path(__file__).parent.parent.parent / "test_data" / test_file

        if not Path(test_file).exists():
            raise FileNotFoundError(f"测试文件不存在: {test_file}")

        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        print(f"✅ 加载测试数据: {test_file}")
        return test_data

    def enhance_query(self, query: str, expected_topics: List[str] = None) -> str:
        """查询增强：添加相关词汇提升召回率"""
        enhanced_query = query
        additions = set()

        if "住宿" in query or "酒店" in query:
            additions.update(["酒店星级", "三星级", "四星级", "五星级", "快捷酒店"])

        level_mappings = {
            "8-9级": [
                "8-9级",
                "普通员工",
                "软件研发工程师",
                "机械研发工程师",
                "工艺工程师",
                "实施工程师",
            ],
            "10-11级": ["10-11级", "经理", "中层管理", "主管"],
            "12级": ["12级", "总监", "专家级", "高级管理"],
        }

        for level_key, level_terms in level_mappings.items():
            if level_key in query or any(term in query for term in level_terms[:2]):
                additions.update(level_terms)
                break

        city_mappings = {
            "一线城市": ["上海", "北京", "广州", "深圳"],
            "新一线": ["成都", "杭州", "武汉", "西安", "南京", "重庆", "苏州", "天津"],
        }

        for city_type, cities in city_mappings.items():
            if any(city in query for city in cities):
                additions.add(city_type)
                if city_type == "一线城市":
                    additions.add("北上广深")
                break

        if expected_topics:
            topic_mappings = {
                "住宿标准": ["出差住宿", "报销标准", "住宿费用", "酒店标准"],
                "职级差异": ["等级标准", "职位级别", "对应关系", "职级划分"],
                "地区差异": ["城市分级", "地区分类", "一线二线", "城市级别"],
            }
            for topic in expected_topics:
                if topic in topic_mappings:
                    additions.update(topic_mappings[topic])

        for add in additions:
            if add not in enhanced_query:
                enhanced_query += " " + add

        return enhanced_query

    def run_retrieval_test(
        self,
        query: str,
        expected_keywords: List[str],
        case_info: Dict[str, Any],
        enable_rerank: bool = True,
        reranker_type: str = "bge",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """运行单个检索测试"""
        case_id = case_info.get('id', 'unknown')
        
        # 记录测试开始
        logger.info(f"=" * 80)
        logger.info(f"【测试用例】{case_id}")
        logger.info(f"=" * 80)

        # 步骤1: 查询增强
        enhanced_query = self.enhance_query(query, case_info.get("expected_topics", []))
        if verbose:
            print(f"\n📝 原始查询: {query}")
            if enhanced_query != query:
                print(f"🔧 增强查询: {enhanced_query}")
        # 写入日志
        logger.info(f"原始查询: {query}")
        if enhanced_query != query:
            logger.info(f"增强查询: {enhanced_query}")

        # 步骤2: 向量编码
        query_vector = embedding_service.encode([enhanced_query])
        if verbose:
            print(f"🔢 查询向量维度: {query_vector.shape}")
        logger.info(f"查询向量维度: {query_vector.shape}")

        # 步骤3: 向量检索
        start = time.time()
        try:
            scores, metadatas = vector_db_manager.search(query_vector, top_k=15)
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            raise
        elapsed = (time.time() - start) * 1000

        if verbose:
            print(f"\n🔍 向量检索 (耗时: {elapsed:.1f}ms)")
            print(f"   检索到 {len(metadatas[0])} 个结果")
        logger.info(f"向量检索耗时: {elapsed:.1f}ms, 检索到 {len(metadatas[0])} 个结果")

        # 步骤4: 构建结果列表
        results = []
        for i, (score, meta) in enumerate(zip(scores[0], metadatas[0])):
            document_id = meta.get("document_id", "")
            chunk_id = (
                meta.get("chunk_id") or f"{document_id}_chunk_{i}"
                if document_id
                else f"chunk_{i}"
            )
            results.append(
                FakeResult(
                    content=meta.get("content", ""),
                    similarity=float(score),
                    document_id=document_id,
                    chunk_id=chunk_id,
                    rank=i + 1,
                )
            )

        if verbose:
            print(f"\n📄 原始检索结果 (Top 5):")
            for i, r in enumerate(results[:5], 1):
                content_preview = r.content[:100] + "..." if len(r.content) > 100 else r.content
                print(f"   [{i}] 相似度: {r.similarity:.3f} | {content_preview}")
        # 写入日志
        logger.info("原始检索结果 (Top 5):")
        for i, r in enumerate(results[:5], 1):
            content_log = r.content[:200] + "..." if len(r.content) > 200 else r.content
            logger.info(f"  [{i}] 相似度: {r.similarity:.3f} | 文档: {r.document_id} | {content_log}")

        # 步骤5: 重排序
        if enable_rerank:
            try:
                reranked_results = self.apply_reranking(
                    results[:10], query, expected_keywords
                )
                if verbose:
                    print(f"\n🔄 重排序完成 ({reranker_type})")
                    print(f"   重排序前 Top3: {[r.similarity for r in results[:3]]}")
                    print(f"   重排序后 Top3: {[r.rerank_score for r in reranked_results[:3]]}")
                # 写入日志
                logger.info(f"重排序完成 ({reranker_type})")
                logger.info(f"  重排序前 Top3: {[r.similarity for r in results[:3]]}")
                logger.info(f"  重排序后 Top3: {[r.rerank_score for r in reranked_results[:3]]}")
                results = reranked_results
            except (RuntimeError, ValueError, TypeError) as e:
                logger.warning(f"重排序失败，使用原始排序: {e}")
                if verbose:
                    print(f"   ⚠️ 重排序失败: {e}")

        # 步骤6: 准备 Ground Truth
        ground_truth_raw = case_info.get("ground_truth", [])
        if isinstance(ground_truth_raw, str):
            ground_truth = [ground_truth_raw]
        elif isinstance(ground_truth_raw, list) and ground_truth_raw:
            ground_truth = ground_truth_raw
        else:
            ground_truth = expected_keywords if expected_keywords else []
            if not ground_truth and verbose:
                print(f"   ⚠️ 用例 {case_info.get('id', 'unknown')} 缺少ground_truth和keywords")

        if verbose and ground_truth:
            print(f"\n🎯 Ground Truth / 期望关键词: {ground_truth[:5]}")
        if ground_truth:
            logger.info(f"Ground Truth / 期望关键词: {ground_truth}")

        # 步骤7: 关键词匹配分析
        retrieved_text = " ".join([r.content for r in results[:5]])
        hits = sum(1 for kw in expected_keywords if kw in retrieved_text)
        hit_rate = hits / len(expected_keywords) if expected_keywords else 0

        matched_keywords = [kw for kw in expected_keywords if kw in retrieved_text]
        missed_keywords = [kw for kw in expected_keywords if kw not in retrieved_text]

        if verbose and expected_keywords:
            print(f"\n🔑 关键词分析:")
            print(f"   总关键词: {len(expected_keywords)} ({expected_keywords})")
            print(f"   命中: {hits} ({matched_keywords})")
            print(f"   未命中: {len(missed_keywords)} ({missed_keywords})")
            print(f"   命中率: {hit_rate:.1%}")
        # 写入日志
        if expected_keywords:
            logger.info(f"关键词分析:")
            logger.info(f"  总关键词: {len(expected_keywords)} - {expected_keywords}")
            logger.info(f"  命中: {hits} - {matched_keywords}")
            logger.info(f"  未命中: {len(missed_keywords)} - {missed_keywords}")
            logger.info(f"  命中率: {hit_rate:.1%}")

        # 步骤8: 计算各项指标
        precision_at_1 = self.calculate_precision_at_k(results, ground_truth, 1)
        precision_at_3 = self.calculate_precision_at_k(results, ground_truth, 3)
        precision_at_5 = self.calculate_precision_at_k(results, ground_truth, 5)

        recall_at_5 = self.calculate_recall_at_k(results, ground_truth, k=5)
        f1_at_5 = self.calculate_f1_at_k(precision_at_5, recall_at_5)
        ndcg_at_5 = self.calculate_ndcg_at_k(results, ground_truth, k=5)
        mrr = self.calculate_mrr(results, ground_truth)

        semantic_similarity = 0.0
        if ground_truth and results:
            semantic_similarity = self.calculate_semantic_similarity(
                query, results[0].content
            )

        expected_topics = case_info.get("expected_topics", [])
        topic_coverage = self.calculate_topic_coverage(results, expected_topics)

        if verbose:
            print(f"\n📊 评估指标:")
            print(f"   P@1: {precision_at_1:.3f} | P@3: {precision_at_3:.3f} | P@5: {precision_at_5:.3f}")
            print(f"   Recall@5: {recall_at_5:.3f} | F1@5: {f1_at_5:.3f}")
            print(f"   NDCG@5: {ndcg_at_5:.3f} | MRR: {mrr:.3f}")
            print(f"   语义相似度: {semantic_similarity:.3f}")
            if topic_coverage.get('coverage_rate', 0) > 0:
                print(f"   主题覆盖率: {topic_coverage['coverage_rate']:.1%}")
        # 写入日志
        logger.info(f"评估指标:")
        logger.info(f"  P@1: {precision_at_1:.3f} | P@3: {precision_at_3:.3f} | P@5: {precision_at_5:.3f}")
        logger.info(f"  Recall@5: {recall_at_5:.3f} | F1@5: {f1_at_5:.3f}")
        logger.info(f"  NDCG@5: {ndcg_at_5:.3f} | MRR: {mrr:.3f}")
        logger.info(f"  语义相似度: {semantic_similarity:.3f}")
        if topic_coverage.get('coverage_rate', 0) > 0:
            logger.info(f"  主题覆盖率: {topic_coverage['coverage_rate']:.1%}")

        # 步骤9: 模型信息
        model_info = {
            "embedding_model": "BAAI/bge-base-zh-v1.5 (本地)",
            "vector_db": "FAISS (本地)",
            "llm_provider": "local (GPU)" if torch.cuda.is_available() else "local (CPU)",
            "reranker_enabled": enable_rerank,
            "reranker_type": reranker_type if enable_rerank else None,
            "reranker_top_k": 5 if enable_rerank else None,
            "query_enhanced": enhanced_query != query,
        }

        retrieved_text = " ".join([r.content for r in results[:5]])
        hits = sum(1 for kw in expected_keywords if kw in retrieved_text)
        hit_rate = hits / len(expected_keywords) if expected_keywords else 0

        matched_keywords = [kw for kw in expected_keywords if kw in retrieved_text]
        missed_keywords = [kw for kw in expected_keywords if kw not in retrieved_text]

        precision_at_1 = self.calculate_precision_at_k(results, ground_truth, 1)
        precision_at_3 = self.calculate_precision_at_k(results, ground_truth, 3)
        precision_at_5 = self.calculate_precision_at_k(results, ground_truth, 5)

        recall_at_5 = self.calculate_recall_at_k(results, ground_truth, k=5)
        f1_at_5 = self.calculate_f1_at_k(precision_at_5, recall_at_5)
        ndcg_at_5 = self.calculate_ndcg_at_k(results, ground_truth, k=5)
        mrr = self.calculate_mrr(results, ground_truth)

        semantic_similarity = 0.0
        if ground_truth and results:
            semantic_similarity = self.calculate_semantic_similarity(
                query, results[0].content
            )

        expected_topics = case_info.get("expected_topics", [])
        topic_coverage = self.calculate_topic_coverage(results, expected_topics)

        model_info = {
            "embedding_model": "BAAI/bge-base-zh-v1.5 (本地)",
            "vector_db": "FAISS (本地)",
            "llm_provider": "local (GPU)"
            if torch.cuda.is_available()
            else "local (CPU)",
            "reranker_enabled": enable_rerank,
            "reranker_type": reranker_type if enable_rerank else None,
            "reranker_top_k": 5 if enable_rerank else None,
            "query_enhanced": enhanced_query != query,
        }

        # 步骤10: LLM生成答案和答案质量评估
        llm_result = None
        answer_metrics = None
        ground_truth_text = ground_truth[0] if isinstance(ground_truth, list) and ground_truth else ""
        
        if ground_truth_text and len(results) > 0:
            # 准备上下文
            context = "\n\n".join([f"[{i+1}] {r.content}" for i, r in enumerate(results[:5])])
            
            # 生成LLM答案
            llm_result = self.generate_llm_answer(query, context, verbose)
            
            # 评估答案质量
            if llm_result.get("success") and llm_result.get("answer"):
                answer_metrics = self.evaluate_answer_quality(
                    ground_truth_text, 
                    llm_result["answer"], 
                    llm_result,  # 传入完整的性能指标
                    verbose
                )
                # 更新model_info
                model_info['llm_generation_time_ms'] = llm_result.get('generation_time_ms', 0)
                model_info['llm_tokens_per_second'] = llm_result.get('tokens_per_second', 0)
                model_info['llm_input_tokens'] = llm_result.get('input_tokens', 0)
                model_info['llm_output_tokens'] = llm_result.get('output_tokens', 0)

        return {
            "case_info": case_info,
            "query": query,
            "enhanced_query": enhanced_query,
            "response_time_ms": elapsed,
            "results_count": len(results[:5]),
            "results": [
                {
                    "rank": r.rank,
                    "similarity": r.similarity,
                    "rerank_score": r.rerank_score,
                    "content": r.content[:80] + "..."
                    if len(r.content) > 80
                    else r.content,
                    "chunk_id": r.chunk_id,
                    "document_id": r.document_id,
                }
                for r in results[:5]
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
                "precision_at_1": precision_at_1,
                "precision_at_3": precision_at_3,
                "precision_at_5": precision_at_5,
                "recall_at_5": recall_at_5,
                "f1_at_5": f1_at_5,
                "ndcg_at_5": ndcg_at_5,
                "mrr": mrr,
                "context_precision": precision_at_5,
                "context_recall": recall_at_5,
                "semantic_similarity": semantic_similarity,
            },
            "llm_answer": llm_result.get("answer") if llm_result else None,
            "llm_generation": llm_result,
            "answer_metrics": answer_metrics,
            "model_info": model_info,
        }

    def generate_llm_answer(self, query: str, context: str, verbose: bool = True) -> Dict[str, Any]:
        """
        使用LLM生成答案
        
        Args:
            query: 用户查询
            context: 检索上下文
            verbose: 是否显示详细输出
            
        Returns:
            包含答案和元数据的字典
        """
        try:
            # 构建RAG提示
            prompt = f"""基于以下参考资料，回答问题：

参考资料：
{context}

问题：{query}

请根据参考资料回答，如果参考资料中没有相关信息，请说明无法回答。"""
            
            if verbose:
                print(f"\n🤖 LLM生成答案...")
                print(f"   使用上下文长度: {len(context)} 字符")
            logger.info(f"LLM生成答案 - 上下文长度: {len(context)} 字符")
            
            # 配置生成参数
            generation_config = GenerationConfig(
                llm_provider="local",
                llm_model="Qwen2.5-7B-Instruct",
                temperature=0.7,
                max_tokens=512
            )
            
            # 获取LLM客户端（使用缓存或新建）
            llm_start = time.time()
            if self.keep_llm_loaded and self._llm_client is not None:
                # 使用已缓存的客户端
                llm_client = self._llm_client
                if verbose:
                    print(f"   使用常驻显存的LLM模型")
                logger.info("使用常驻显存的LLM模型")
            else:
                # 新建客户端
                llm_client = rag_generator._get_llm_client(generation_config)
                if self.keep_llm_loaded:
                    self._llm_client = llm_client
                    self._llm_config = generation_config
                    if verbose:
                        print(f"   LLM模型已加载到显存（将保持常驻）")
                    logger.info("LLM模型已加载到显存（将保持常驻）")
            
            generation_result = llm_client.generate(prompt)
            llm_elapsed = (time.time() - llm_start) * 1000
            
            # 提取生成的文本和性能指标
            answer = generation_result.get("text", "")
            input_tokens = generation_result.get("input_tokens", 0)
            output_tokens = generation_result.get("output_tokens", 0)
            total_tokens = generation_result.get("total_tokens", 0)
            time_to_first_token_ms = generation_result.get("time_to_first_token_ms", 0)
            total_time_ms = generation_result.get("total_time_ms", 0)
            generation_time_ms = generation_result.get("generation_time_ms", 0)
            tokens_per_second = generation_result.get("tokens_per_second", 0)
            
            # 根据配置决定是否卸载模型
            if not self.keep_llm_loaded:
                # 生成完成后卸载模型，释放显存
                if hasattr(llm_client, "unload"):
                    llm_client.unload()
            else:
                logger.info("LLM模型保持常驻显存（未卸载）")
            
            if verbose:
                print(f"\n📊 LLM性能指标:")
                print(f"   输入Token: {input_tokens} | 输出Token: {output_tokens} | 总计: {total_tokens}")
                print(f"   首Token时延: {time_to_first_token_ms:.1f}ms")
                print(f"   总生成时间: {total_time_ms:.1f}ms")
                print(f"   ⚡ 生成速度: {tokens_per_second:.2f} tokens/s")
                print(f"   答案长度: {len(answer)} 字符")
                print(f"\n💬 LLM回答:\n{answer[:200]}..." if len(answer) > 200 else f"\n💬 LLM回答:\n{answer}")
            logger.info(f"LLM性能 - 输入Token: {input_tokens}, 输出Token: {output_tokens}, "
                       f"首Token时延: {time_to_first_token_ms:.1f}ms, "
                       f"生成速度: {tokens_per_second:.2f} tokens/s")
            logger.info(f"LLM回答: {answer[:500]}..." if len(answer) > 500 else f"LLM回答: {answer}")
            
            return {
                "answer": answer,
                "generation_time_ms": llm_elapsed,
                "answer_length": len(answer),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "time_to_first_token_ms": time_to_first_token_ms,
                "total_time_ms": total_time_ms,
                "generation_time_ms": generation_time_ms,
                "tokens_per_second": tokens_per_second,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            if verbose:
                print(f"   ⚠️ LLM生成失败: {e}")
            return {
                "answer": "",
                "generation_time_ms": 0,
                "answer_length": 0,
                "success": False,
                "error": str(e)
            }
    
    def unload_llm_model(self):
        """手动卸载LLM模型，释放显存"""
        if self._llm_client is not None:
            if hasattr(self._llm_client, "unload"):
                self._llm_client.unload()
                logger.info("LLM模型已手动卸载")
            self._llm_client = None
            self._llm_config = None

    def evaluate_answer_quality(self, ground_truth: str, llm_answer: str, 
                                   llm_performance: Dict[str, Any] = None,
                                   verbose: bool = True) -> Dict[str, float]:
        """
        评估LLM答案质量
        
        Args:
            ground_truth: 标准答案
            llm_answer: LLM生成的答案
            llm_performance: LLM生成性能指标
            verbose: 是否显示详细输出
            
        Returns:
            包含各项评估指标的字典
        """
        try:
            # 计算各项指标
            metrics = calculate_answer_metrics(ground_truth, llm_answer)
            
            # 计算语义相似度
            semantic_sim = self.calculate_semantic_similarity(ground_truth, llm_answer)
            metrics['semantic_similarity'] = semantic_sim
            
            # 添加性能指标到metrics中
            if llm_performance:
                metrics['input_tokens'] = llm_performance.get('input_tokens', 0)
                metrics['output_tokens'] = llm_performance.get('output_tokens', 0)
                metrics['total_tokens'] = llm_performance.get('total_tokens', 0)
                metrics['time_to_first_token_ms'] = llm_performance.get('time_to_first_token_ms', 0)
                metrics['total_time_ms'] = llm_performance.get('total_time_ms', 0)
                metrics['generation_time_ms'] = llm_performance.get('generation_time_ms', 0)
                metrics['tokens_per_second'] = llm_performance.get('tokens_per_second', 0)
            
            if verbose:
                print(f"\n📊 答案质量评估:")
                print(f"   BLEU-1: {metrics['bleu_1']:.3f} | BLEU-2: {metrics['bleu_2']:.3f} | BLEU-avg: {metrics['bleu_avg']:.3f}")
                print(f"   ROUGE-1: {metrics['rouge_1']:.3f} | ROUGE-2: {metrics['rouge_2']:.3f} | ROUGE-L: {metrics['rouge_l']:.3f}")
                print(f"   ROUGE-avg: {metrics['rouge_avg']:.3f}")
                print(f"   字符精确率: {metrics['char_precision']:.3f} | 召回率: {metrics['char_recall']:.3f} | F1: {metrics['char_f1']:.3f}")
                print(f"   语义相似度: {semantic_sim:.3f}")
                print(f"   长度比: {metrics['length_ratio']:.2f}")
                
                # 显示性能指标
                if llm_performance:
                    print(f"\n⚡ LLM性能指标:")
                    print(f"   输入Token: {llm_performance.get('input_tokens', 0)} | 输出Token: {llm_performance.get('output_tokens', 0)}")
                    print(f"   首Token时延: {llm_performance.get('time_to_first_token_ms', 0):.1f}ms")
                    print(f"   总生成时间: {llm_performance.get('total_time_ms', 0):.1f}ms")
                    print(f"   有效生成速度: {llm_performance.get('tokens_per_second', 0):.2f} tokens/s")
            
            logger.info(f"答案质量评估:")
            logger.info(f"  BLEU: {metrics['bleu_avg']:.3f} (BLEU-1: {metrics['bleu_1']:.3f}, BLEU-2: {metrics['bleu_2']:.3f})")
            logger.info(f"  ROUGE: {metrics['rouge_avg']:.3f} (ROUGE-1: {metrics['rouge_1']:.3f}, ROUGE-2: {metrics['rouge_2']:.3f}, ROUGE-L: {metrics['rouge_l']:.3f})")
            logger.info(f"  字符级: 精确率={metrics['char_precision']:.3f}, 召回率={metrics['char_recall']:.3f}, F1={metrics['char_f1']:.3f}")
            logger.info(f"  语义相似度: {semantic_sim:.3f}")
            logger.info(f"  长度比: {metrics['length_ratio']:.2f}")
            if llm_performance:
                logger.info(f"  LLM性能: 输入Token={llm_performance.get('input_tokens', 0)}, "
                           f"输出Token={llm_performance.get('output_tokens', 0)}, "
                           f"首Token时延={llm_performance.get('time_to_first_token_ms', 0):.1f}ms, "
                           f"生成速度={llm_performance.get('tokens_per_second', 0):.2f} tokens/s")
            
            return metrics
            
        except Exception as e:
            logger.error(f"答案评估失败: {e}")
            return {
                'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_avg': 0.0,
                'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0, 'rouge_avg': 0.0,
                'char_precision': 0.0, 'char_recall': 0.0, 'char_f1': 0.0,
                'semantic_similarity': 0.0, 'length_ratio': 0.0,
                'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0,
                'time_to_first_token_ms': 0, 'total_time_ms': 0, 
                'generation_time_ms': 0, 'tokens_per_second': 0
            }

    def evaluate_retrieval_cases(
        self,
        test_cases: List[Dict],
        limit: Optional[int] = None,
        enable_rerank: bool = True,
        reranker_type: str = "bge",
        verbose: bool = True,
    ) -> List[Dict]:
        """评估检索测试用例
        
        Args:
            test_cases: 测试用例列表
            limit: 限制测试数量
            enable_rerank: 是否启用重排序
            reranker_type: 重排序器类型
            verbose: 是否显示详细过程
        """
        if limit:
            test_cases = test_cases[:limit]

        print(f"\n🧪 运行检索测试 ({len(test_cases)} 条):")
        if enable_rerank:
            print(f"📈 重排序: {reranker_type.upper()}")
        else:
            print("📊 重排序: 禁用")
        print("-" * 80)

        results = []
        for i, case in enumerate(test_cases, 1):
            query = case["query"]
            keywords = case.get("expected_keywords", [])

            print(f"\n[{i:2d}/{len(test_cases)}] {query[:45]}...", end=" ")
            print(f"[{case.get('difficulty', 'unknown')}]")
            print("=" * 80)

            case_info = {
                "id": case["id"],
                "category": case.get("category", "unknown"),
                "difficulty": case.get("difficulty", "unknown"),
                "description": case.get("description", ""),
                "ground_truth": case.get("ground_truth", []),
            }

            try:
                result = self.run_retrieval_test(
                    query, keywords, case_info, enable_rerank, reranker_type, verbose
                )
                results.append(result)

                metrics = result["metrics"]
                keyword_analysis = result["keyword_analysis"]
                model_info = result["model_info"]

                status = (
                    "✅"
                    if keyword_analysis["hit_rate"] >= 0.6
                    and metrics["ndcg_at_5"] >= 0.5
                    else "⚠️"
                    if keyword_analysis["hit_rate"] >= 0.4
                    and metrics["ndcg_at_5"] >= 0.3
                    else "❌"
                )

                rerank_indicator = (
                    "🔄" if model_info.get("reranker_enabled", False) else "📊"
                )

                print(
                    f"\n📊 结果摘要: {rerank_indicator}{status} {result['response_time_ms']:.1f}ms | "
                    f"P@1:{metrics['precision_at_1']:.2f} | "
                    f"NDCG:{metrics['ndcg_at_5']:.2f} | "
                    f"关键词:{keyword_analysis['hit_rate']:.0%}"
                )

                if len(keyword_analysis["missed"]) > 0:
                    print(f"     未命中关键词: {', '.join(keyword_analysis['missed'][:3])}")

            except Exception as e:
                print(f"\n     ❌ 测试失败: {str(e)[:50]}")
                logger.error(f"测试失败 {case['id']}: {e}", exc_info=True)
                results.append(
                    {"case_info": case_info, "query": query, "error": str(e)}
                )
            
            print("-" * 80)

        return results

    def analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """分析测试结果"""
        valid_results = [r for r in results if "metrics" in r]

        if not valid_results:
            return {"error": "无有效测试结果"}

        times = [r["response_time_ms"] for r in valid_results]
        p1s = [r["metrics"]["precision_at_1"] for r in valid_results]
        p3s = [r["metrics"]["precision_at_3"] for r in valid_results]
        p5s = [r["metrics"]["precision_at_5"] for r in valid_results]
        mrrs = [r["metrics"]["mrr"] for r in valid_results]
        hit_rates = [r["keyword_analysis"]["hit_rate"] for r in valid_results]

        recalls = [r["metrics"]["recall_at_5"] for r in valid_results]
        f1s = [r["metrics"]["f1_at_5"] for r in valid_results]
        ndcgs = [r["metrics"]["ndcg_at_5"] for r in valid_results]
        semantic_sims = [
            r["metrics"]["semantic_similarity"]
            for r in valid_results
            if r["metrics"]["semantic_similarity"] > 0
        ]
        topic_coverage_rates = [
            r["topic_coverage"]["coverage_rate"]
            for r in valid_results
            if r["topic_coverage"]["coverage_rate"] > 0
        ]
        
        # 收集 LLM 性能指标
        llm_input_tokens = [
            r["answer_metrics"]["input_tokens"]
            for r in valid_results
            if r.get("answer_metrics") and r["answer_metrics"].get("input_tokens", 0) > 0
        ]
        llm_output_tokens = [
            r["answer_metrics"]["output_tokens"]
            for r in valid_results
            if r.get("answer_metrics") and r["answer_metrics"].get("output_tokens", 0) > 0
        ]
        llm_time_to_first_token = [
            r["answer_metrics"]["time_to_first_token_ms"]
            for r in valid_results
            if r.get("answer_metrics") and r["answer_metrics"].get("time_to_first_token_ms", 0) > 0
        ]
        llm_generation_time = [
            r["answer_metrics"]["generation_time_ms"]
            for r in valid_results
            if r.get("answer_metrics") and r["answer_metrics"].get("generation_time_ms", 0) > 0
        ]
        llm_tokens_per_second = [
            r["answer_metrics"]["tokens_per_second"]
            for r in valid_results
            if r.get("answer_metrics") and r["answer_metrics"].get("tokens_per_second", 0) > 0
        ]

        by_difficulty = {}
        by_category = {}

        for r in valid_results:
            diff = r["case_info"]["difficulty"]
            cat = r["case_info"]["category"]

            if diff not in by_difficulty:
                by_difficulty[diff] = {
                    "p1": [],
                    "hit": [],
                    "mrr": [],
                    "time": [],
                    "recall": [],
                    "f1": [],
                    "ndcg": [],
                    "llm_input_tokens": [],
                    "llm_output_tokens": [],
                    "llm_ttft": [],
                    "llm_generation_time": [],
                    "llm_tokens_per_second": [],
                }
            by_difficulty[diff]["p1"].append(r["metrics"]["precision_at_1"])
            by_difficulty[diff]["hit"].append(r["keyword_analysis"]["hit_rate"])
            by_difficulty[diff]["mrr"].append(r["metrics"]["mrr"])
            by_difficulty[diff]["time"].append(r["response_time_ms"])
            by_difficulty[diff]["recall"].append(r["metrics"]["recall_at_5"])
            by_difficulty[diff]["f1"].append(r["metrics"]["f1_at_5"])
            by_difficulty[diff]["ndcg"].append(r["metrics"]["ndcg_at_5"])
            
            # 收集LLM性能数据
            if r.get("answer_metrics"):
                am = r["answer_metrics"]
                if am.get("input_tokens", 0) > 0:
                    by_difficulty[diff]["llm_input_tokens"].append(am["input_tokens"])
                if am.get("output_tokens", 0) > 0:
                    by_difficulty[diff]["llm_output_tokens"].append(am["output_tokens"])
                if am.get("time_to_first_token_ms", 0) > 0:
                    by_difficulty[diff]["llm_ttft"].append(am["time_to_first_token_ms"])
                if am.get("generation_time_ms", 0) > 0:
                    by_difficulty[diff]["llm_generation_time"].append(am["generation_time_ms"])
                if am.get("tokens_per_second", 0) > 0:
                    by_difficulty[diff]["llm_tokens_per_second"].append(am["tokens_per_second"])

            if cat not in by_category:
                by_category[cat] = {
                    "p1": [],
                    "hit": [],
                    "mrr": [],
                    "recall": [],
                    "f1": [],
                    "ndcg": [],
                    "llm_input_tokens": [],
                    "llm_output_tokens": [],
                    "llm_ttft": [],
                    "llm_generation_time": [],
                    "llm_tokens_per_second": [],
                }
            by_category[cat]["p1"].append(r["metrics"]["precision_at_1"])
            by_category[cat]["hit"].append(r["keyword_analysis"]["hit_rate"])
            by_category[cat]["mrr"].append(r["metrics"]["mrr"])
            by_category[cat]["recall"].append(r["metrics"]["recall_at_5"])
            by_category[cat]["f1"].append(r["metrics"]["f1_at_5"])
            by_category[cat]["ndcg"].append(r["metrics"]["ndcg_at_5"])
            
            # 收集LLM性能数据
            if r.get("answer_metrics"):
                am = r["answer_metrics"]
                if am.get("input_tokens", 0) > 0:
                    by_category[cat]["llm_input_tokens"].append(am["input_tokens"])
                if am.get("output_tokens", 0) > 0:
                    by_category[cat]["llm_output_tokens"].append(am["output_tokens"])
                if am.get("time_to_first_token_ms", 0) > 0:
                    by_category[cat]["llm_ttft"].append(am["time_to_first_token_ms"])
                if am.get("generation_time_ms", 0) > 0:
                    by_category[cat]["llm_generation_time"].append(am["generation_time_ms"])
                if am.get("tokens_per_second", 0) > 0:
                    by_category[cat]["llm_tokens_per_second"].append(am["tokens_per_second"])

        poor_cases = [
            r
            for r in valid_results
            if r["keyword_analysis"]["hit_rate"] < 0.4
            or r["metrics"]["ndcg_at_5"] < 0.3
        ]
        good_cases = [
            r
            for r in valid_results
            if r["keyword_analysis"]["hit_rate"] >= 0.8
            and r["metrics"]["ndcg_at_5"] >= 0.6
        ]
        failed_cases = [r for r in results if "error" in r]

        return {
            "total_tests": len(results),
            "valid_tests": len(valid_results),
            "failed_tests": len(failed_cases),
            "statistics": {
                "avg_response_time_ms": round(statistics.mean(times), 1)
                if times
                else 0,
                "p95_response_time_ms": round(sorted(times)[int(len(times) * 0.95)])
                if times
                else 0,
                "avg_precision_at_1": round(statistics.mean(p1s), 3) if p1s else 0,
                "avg_precision_at_3": round(statistics.mean(p3s), 3) if p3s else 0,
                "avg_precision_at_5": round(statistics.mean(p5s), 3) if p5s else 0,
                "avg_mrr": round(statistics.mean(mrrs), 3) if mrrs else 0,
                "avg_keyword_hit_rate": round(statistics.mean(hit_rates), 3)
                if hit_rates
                else 0,
                "avg_recall_at_5": round(statistics.mean(recalls), 3) if recalls else 0,
                "avg_f1_at_5": round(statistics.mean(f1s), 3) if f1s else 0,
                "avg_ndcg_at_5": round(statistics.mean(ndcgs), 3) if ndcgs else 0,
                "avg_semantic_similarity": round(statistics.mean(semantic_sims), 3)
                if semantic_sims
                else 0,
                "avg_topic_coverage": round(statistics.mean(topic_coverage_rates), 3)
                if topic_coverage_rates
                else 0,
                # LLM 性能指标
                "avg_llm_input_tokens": round(statistics.mean(llm_input_tokens), 1)
                if llm_input_tokens
                else 0,
                "avg_llm_output_tokens": round(statistics.mean(llm_output_tokens), 1)
                if llm_output_tokens
                else 0,
                "avg_time_to_first_token_ms": round(statistics.mean(llm_time_to_first_token), 1)
                if llm_time_to_first_token
                else 0,
                "avg_generation_time_ms": round(statistics.mean(llm_generation_time), 1)
                if llm_generation_time
                else 0,
                "avg_tokens_per_second": round(statistics.mean(llm_tokens_per_second), 2)
                if llm_tokens_per_second
                else 0,
            },
            "by_difficulty": {
                diff: {
                    "count": len(stats["p1"]),
                    "avg_precision_at_1": round(
                        statistics.mean(stats["p1"]) if stats["p1"] else 0, 3
                    ),
                    "avg_keyword_hit_rate": round(
                        statistics.mean(stats["hit"]) if stats["hit"] else 0, 3
                    ),
                    "avg_mrr": round(
                        statistics.mean(stats["mrr"]) if stats["mrr"] else 0, 3
                    ),
                    "avg_response_time_ms": round(
                        statistics.mean(stats["time"]) if stats["time"] else 0, 1
                    ),
                    "avg_recall_at_5": round(
                        statistics.mean(stats["recall"]) if stats["recall"] else 0, 3
                    ),
                    "avg_f1_at_5": round(
                        statistics.mean(stats["f1"]) if stats["f1"] else 0, 3
                    ),
                    "avg_ndcg_at_5": round(
                        statistics.mean(stats["ndcg"]) if stats["ndcg"] else 0, 3
                    ),
                    # LLM 性能指标
                    "avg_llm_input_tokens": round(
                        statistics.mean(stats["llm_input_tokens"]) if stats["llm_input_tokens"] else 0, 1
                    ),
                    "avg_llm_output_tokens": round(
                        statistics.mean(stats["llm_output_tokens"]) if stats["llm_output_tokens"] else 0, 1
                    ),
                    "avg_llm_tokens_per_second": round(
                        statistics.mean(stats["llm_tokens_per_second"]) if stats["llm_tokens_per_second"] else 0, 2
                    ),
                    "avg_llm_ttft_ms": round(
                        statistics.mean(stats["llm_ttft"]) if stats["llm_ttft"] else 0, 1
                    ),
                }
                for diff, stats in by_difficulty.items()
            },
            "by_category": {
                cat: {
                    "count": len(stats["p1"]),
                    "avg_precision_at_1": round(
                        statistics.mean(stats["p1"]) if stats["p1"] else 0, 3
                    ),
                    "avg_keyword_hit_rate": round(
                        statistics.mean(stats["hit"]) if stats["hit"] else 0, 3
                    ),
                    "avg_mrr": round(
                        statistics.mean(stats["mrr"]) if stats["mrr"] else 0, 3
                    ),
                    "avg_recall_at_5": round(
                        statistics.mean(stats["recall"]) if stats["recall"] else 0, 3
                    ),
                    "avg_f1_at_5": round(
                        statistics.mean(stats["f1"]) if stats["f1"] else 0, 3
                    ),
                    "avg_ndcg_at_5": round(
                        statistics.mean(stats["ndcg"]) if stats["ndcg"] else 0, 3
                    ),
                    # LLM 性能指标
                    "avg_llm_input_tokens": round(
                        statistics.mean(stats["llm_input_tokens"]) if stats["llm_input_tokens"] else 0, 1
                    ),
                    "avg_llm_output_tokens": round(
                        statistics.mean(stats["llm_output_tokens"]) if stats["llm_output_tokens"] else 0, 1
                    ),
                    "avg_llm_tokens_per_second": round(
                        statistics.mean(stats["llm_tokens_per_second"]) if stats["llm_tokens_per_second"] else 0, 2
                    ),
                    "avg_llm_ttft_ms": round(
                        statistics.mean(stats["llm_ttft"]) if stats["llm_ttft"] else 0, 1
                    ),
                }
                for cat, stats in by_category.items()
            },
            "problem_cases": [
                {
                    "id": r["case_info"]["id"],
                    "query": r["query"],
                    "hit_rate": r["keyword_analysis"]["hit_rate"],
                    "missed_keywords": r["keyword_analysis"]["missed"],
                    "precision_at_1": r["metrics"]["precision_at_1"],
                    "ndcg_at_5": r["metrics"]["ndcg_at_5"],
                    "topic_coverage": r["topic_coverage"]["coverage_rate"],
                }
                for r in poor_cases
            ],
            "good_cases": [
                {
                    "id": r["case_info"]["id"],
                    "query": r["query"],
                    "hit_rate": r["keyword_analysis"]["hit_rate"],
                    "precision_at_1": r["metrics"]["precision_at_1"],
                    "ndcg_at_5": r["metrics"]["ndcg_at_5"],
                    "topic_coverage": r["topic_coverage"]["coverage_rate"],
                }
                for r in good_cases
            ],
            "failed_cases": [
                {"id": r["case_info"]["id"], "query": r["query"], "error": r["error"]}
                for r in failed_cases
            ],
        }

    def calculate_score(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """计算综合评分"""
        stats = analysis.get("statistics", {})

        score = 0
        grade_descriptions = []

        avg_p1 = stats.get("avg_precision_at_1", 0)
        if avg_p1 >= 0.7:
            score += 20
            grade_descriptions.append("🟢 P@1 精确率优秀 (+20)")
        elif avg_p1 >= 0.5:
            score += 15
            grade_descriptions.append("🟡 P@1 精确率良好 (+15)")
        elif avg_p1 >= 0.3:
            score += 8
            grade_descriptions.append("🟠 P@1 精确率一般 (+8)")

        avg_ndcg = stats.get("avg_ndcg_at_5", 0)
        if avg_ndcg >= 0.7:
            score += 20
            grade_descriptions.append("🟢 NDCG@5 排序质量优秀 (+20)")
        elif avg_ndcg >= 0.5:
            score += 15
            grade_descriptions.append("🟡 NDCG@5 排序质量良好 (+15)")
        elif avg_ndcg >= 0.3:
            score += 8
            grade_descriptions.append("🟠 NDCG@5 排序质量一般 (+8)")

        avg_f1 = stats.get("avg_f1_at_5", 0)
        if avg_f1 >= 0.7:
            score += 20
            grade_descriptions.append("🟢 F1@5 平衡指标优秀 (+20)")
        elif avg_f1 >= 0.5:
            score += 15
            grade_descriptions.append("🟡 F1@5 平衡指标良好 (+15)")
        elif avg_f1 >= 0.3:
            score += 8
            grade_descriptions.append("🟠 F1@5 平衡指标一般 (+8)")

        avg_hit = stats.get("avg_keyword_hit_rate", 0)
        if avg_hit >= 0.8:
            score += 15
            grade_descriptions.append("🟢 关键词命中率优秀 (+15)")
        elif avg_hit >= 0.6:
            score += 10
            grade_descriptions.append("🟡 关键词命中率良好 (+10)")
        elif avg_hit >= 0.4:
            score += 5
            grade_descriptions.append("🟠 关键词命中率一般 (+5)")

        avg_topic = stats.get("avg_topic_coverage", 0)
        if avg_topic >= 0.8:
            score += 10
            grade_descriptions.append("🟢 主题覆盖率优秀 (+10)")
        elif avg_topic >= 0.6:
            score += 7
            grade_descriptions.append("🟡 主题覆盖率良好 (+7)")
        elif avg_topic >= 0.4:
            score += 3
            grade_descriptions.append("🟠 主题覆盖率一般 (+3)")

        avg_semantic = stats.get("avg_semantic_similarity", 0)
        if avg_semantic >= 0.8:
            score += 5
            grade_descriptions.append("🟢 语义相似度优秀 (+5)")
        elif avg_semantic >= 0.6:
            score += 3
            grade_descriptions.append("🟡 语义相似度良好 (+3)")

        avg_mrr = stats.get("avg_mrr", 0)
        if avg_mrr >= 0.5:
            score += 5
            grade_descriptions.append("🟢 MRR 优秀 (+5)")

        avg_time = stats.get("avg_response_time_ms", 0)
        if avg_time <= 100:
            score += 5
            grade_descriptions.append("🟢 响应速度优秀 (+5)")
        elif avg_time <= 500:
            score += 3
            grade_descriptions.append("🟡 响应速度良好 (+3)")

        if score >= 85:
            grade = "🟢 优秀"
        elif score >= 70:
            grade = "🟡 良好"
        elif score >= 55:
            grade = "🟠 一般"
        else:
            grade = "🔴 需改进"

        return {
            "total_score": score,
            "max_score": 100,
            "grade": grade,
            "grade_descriptions": grade_descriptions,
        }

    def save_report(
        self,
        results: List[Dict],
        analysis: Dict[str, Any],
        score: Dict[str, Any],
        test_data_info: Dict[str, Any],
        test_file: str,
        baseline_results: Optional[List[Dict]] = None,
        baseline_analysis: Optional[Dict[str, Any]] = None,
        baseline_score: Optional[Dict[str, Any]] = None,
    ) -> str:
        """保存测试报告"""
        timestamp = self.evaluation_time.strftime("%Y%m%d_%H%M%S")

        reranker_enabled = False
        if results and results[0]["model_info"].get("reranker_enabled"):
            reranker_enabled = True

        report_data = {
            "evaluation_info": {
                "timestamp": self.evaluation_time.isoformat(),
                "test_file": str(test_file),
                "evaluator": "enhanced_eval.py",
                "version": "2.1-autodl",
                "reranker_enabled": reranker_enabled,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
            "dataset_info": test_data_info,
            "score_info": score,
            "analysis": analysis,
            "detailed_results": results,
        }

        if baseline_results:
            report_data["baseline_results"] = baseline_results
            report_data["baseline_analysis"] = baseline_analysis
            report_data["baseline_score"] = baseline_score

        json_file = self.output_dir / f"rag_evaluation_report_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        md_file = self.output_dir / f"rag_evaluation_summary_{timestamp}.md"
        md_content = self.generate_markdown_report(
            results,
            analysis,
            score,
            test_data_info,
            timestamp,
            baseline_results=baseline_results,
            baseline_analysis=baseline_analysis,
            baseline_score=baseline_score,
        )
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        return str(json_file)

    def generate_markdown_report(
        self,
        results: List[Dict],
        analysis: Dict[str, Any],
        score: Dict[str, Any],
        test_data_info: Dict[str, Any],
        timestamp: str,
        baseline_results: Optional[List[Dict]] = None,
        baseline_analysis: Optional[Dict[str, Any]] = None,
        baseline_score: Optional[Dict[str, Any]] = None,
    ) -> str:
        """生成Markdown格式报告"""
        stats = analysis["statistics"]

        reranker_enabled = False
        reranker_info = ""
        if results and results[0]["model_info"].get("reranker_enabled"):
            reranker_enabled = True
            reranker_info = (
                f" ({results[0]['model_info'].get('reranker_type', 'unknown')} 重排序)"
            )

        device_info = "GPU" if torch.cuda.is_available() else "CPU"

        md = f"""# RAG系统测评报告{reranker_info}

## 📊 测评概览

- **测评时间**: {self.evaluation_time.strftime("%Y-%m-%d %H:%M:%S")}
- **运行设备**: {device_info}
- **数据集版本**: {test_data_info.get("version", "unknown")}
- **测试用例总数**: {analysis["total_tests"]}
- **有效测试**: {analysis["valid_tests"]}
- **失败测试**: {analysis["failed_tests"]}
- **重排序**: {"启用" if reranker_enabled else "禁用"}

## 🏆 综合评分

**{score["grade"]} - {score["total_score"]}/100**

"""

        for desc in score["grade_descriptions"]:
            md += f"- {desc}\n"

        if baseline_analysis and baseline_score:
            baseline_stats = baseline_analysis["statistics"]
            md += f"""
### 🔄 重排序效果对比

| 指标 | 基础检索 | 重排序 | 改进 |
|------|----------|--------|------|
| **综合评分** | {baseline_score["total_score"]}/100 | {score["total_score"]}/100 | **{score["total_score"] - baseline_score["total_score"]:+d}** |
| **评级** | {baseline_score["grade"]} | {score["grade"]} | {"⬆️ 提升" if score["total_score"] > baseline_score["total_score"] else "⬇️ 下降" if score["total_score"] < baseline_score["total_score"] else "➡️ 持平"} |
| P@1 精确率 | {baseline_stats["avg_precision_at_1"]:.3f} | {stats["avg_precision_at_1"]:.3f} | {stats["avg_precision_at_1"] - baseline_stats["avg_precision_at_1"]:+.3f} |
| NDCG@5 | {baseline_stats.get("avg_ndcg_at_5", 0):.3f} | {stats.get("avg_ndcg_at_5", 0):.3f} | {stats.get("avg_ndcg_at_5", 0) - baseline_stats.get("avg_ndcg_at_5", 0):+.3f} |
| F1@5 | {baseline_stats.get("avg_f1_at_5", 0):.3f} | {stats.get("avg_f1_at_5", 0):.3f} | {stats.get("avg_f1_at_5", 0) - baseline_stats.get("avg_f1_at_5", 0):+.3f} |
| 关键词命中率 | {baseline_stats["avg_keyword_hit_rate"]:.1%} | {stats["avg_keyword_hit_rate"]:.1%} | {stats["avg_keyword_hit_rate"] - baseline_stats["avg_keyword_hit_rate"]:+.1%} |
| 响应时间 | {baseline_stats["avg_response_time_ms"]:.1f}ms | {stats["avg_response_time_ms"]:.1f}ms | {stats["avg_response_time_ms"] - baseline_stats["avg_response_time_ms"]:+.1f}ms |

"""

        md += f"""
## 📈 关键指标

| 指标 | 数值 | 评价 |
|------|------|------|
| 平均响应时间 | {stats["avg_response_time_ms"]:.1f}ms | {"优秀" if stats["avg_response_time_ms"] <= 100 else "良好" if stats["avg_response_time_ms"] <= 500 else "一般"} |
| P@1 精确率 | {stats["avg_precision_at_1"]:.3f} | {"优秀" if stats["avg_precision_at_1"] >= 0.7 else "良好" if stats["avg_precision_at_1"] >= 0.5 else "一般"} |
| P@3 精确率 | {stats["avg_precision_at_3"]:.3f} | {"优秀" if stats["avg_precision_at_3"] >= 0.7 else "良好" if stats["avg_precision_at_3"] >= 0.5 else "一般"} |
| P@5 精确率 | {stats["avg_precision_at_5"]:.3f} | {"优秀" if stats["avg_precision_at_5"] >= 0.8 else "良好" if stats["avg_precision_at_5"] >= 0.6 else "一般"} |
| **Recall@5** | **{stats.get("avg_recall_at_5", 0):.3f}** | {"优秀" if stats.get("avg_recall_at_5", 0) >= 0.7 else "良好" if stats.get("avg_recall_at_5", 0) >= 0.5 else "一般"} |
| **F1@5** | **{stats.get("avg_f1_at_5", 0):.3f}** | {"优秀" if stats.get("avg_f1_at_5", 0) >= 0.7 else "良好" if stats.get("avg_f1_at_5", 0) >= 0.5 else "一般"} |
| **NDCG@5** | **{stats.get("avg_ndcg_at_5", 0):.3f}** | {"优秀" if stats.get("avg_ndcg_at_5", 0) >= 0.7 else "良好" if stats.get("avg_ndcg_at_5", 0) >= 0.5 else "一般"} |
| MRR | {stats["avg_mrr"]:.3f} | {"优秀" if stats["avg_mrr"] >= 0.5 else "良好" if stats["avg_mrr"] >= 0.3 else "一般"} |
| 关键词命中率 | {stats["avg_keyword_hit_rate"]:.1%} | {"优秀" if stats["avg_keyword_hit_rate"] >= 0.8 else "良好" if stats["avg_keyword_hit_rate"] >= 0.6 else "一般"} |
| **主题覆盖率** | **{stats.get("avg_topic_coverage", 0):.1%}** | {"优秀" if stats.get("avg_topic_coverage", 0) >= 0.8 else "良好" if stats.get("avg_topic_coverage", 0) >= 0.6 else "一般"} |
| **语义相似度** | **{stats.get("avg_semantic_similarity", 0):.3f}** | {"优秀" if stats.get("avg_semantic_similarity", 0) >= 0.8 else "良好" if stats.get("avg_semantic_similarity", 0) >= 0.6 else "一般"} |

### 🆕 新增指标说明

- **Recall@5**: 召回率@5，衡量检索结果的完整性
- **F1@5**: F1分数@5，精确率和召回率的调和平均
- **NDCG@5**: 归一化折损累积增益@5，考虑排序质量
- **主题覆盖率**: expected_topics的覆盖情况分析
- **语义相似度**: 使用embedding模型计算答案相似度

## ⚡ LLM 生成性能

| 指标 | 数值 | 评价 |
|------|------|------|
| 平均输入 Token | {stats.get('avg_llm_input_tokens', 0):.1f} | prompt 长度 |
| 平均输出 Token | {stats.get('avg_llm_output_tokens', 0):.1f} | 生成长度 |
| 首 Token 时延 | {stats.get('avg_time_to_first_token_ms', 0):.1f}ms | {"优秀" if stats.get('avg_time_to_first_token_ms', 0) < 1000 else "良好" if stats.get('avg_time_to_first_token_ms', 0) < 2000 else "一般"} |
| 平均生成时间 | {stats.get('avg_generation_time_ms', 0):.1f}ms | 纯生成阶段耗时 |
| **生成速度** | **{stats.get('avg_tokens_per_second', 0):.2f} tokens/s** | {"优秀" if stats.get('avg_tokens_per_second', 0) > 25 else "良好" if stats.get('avg_tokens_per_second', 0) > 15 else "一般"} |

### 📝 性能指标说明

- **输入 Token**: 送入模型的 prompt token 数量
- **输出 Token**: 模型生成的答案 token 数量
- **首 Token 时延 (TTFT)**: 从请求发送到首个 token 生成的时间（反映模型加载和预热速度）
- **生成时间**: 从首个 token 到生成结束的纯生成阶段时间
- **生成速度**: output_tokens / generation_time，模型解码效率的核心指标

### 🎯 性能评价标准

| 指标 | 优秀 | 良好 | 一般 |
|------|------|------|------|
| 生成速度 | > 25 tokens/s | 15-25 tokens/s | < 15 tokens/s |
| 首 Token 时延 | < 1000ms | 1000-2000ms | > 2000ms |
| 输出 Token 数 | 100-300 | 50-100 或 300-500 | < 50 或 > 500 |

## 📊 按难度分析

"""

        for diff, diff_stats in analysis["by_difficulty"].items():
            md += f"### {diff.upper()}\n"
            md += f"- 测试数量: {diff_stats['count']}\n"
            md += f"- P@1 精确率: {diff_stats['avg_precision_at_1']:.3f}\n"
            md += f"- **NDCG@5**: {diff_stats['avg_ndcg_at_5']:.3f}\n"
            md += f"- **F1@5**: {diff_stats['avg_f1_at_5']:.3f}\n"
            md += f"- **Recall@5**: {diff_stats['avg_recall_at_5']:.3f}\n"
            md += f"- 关键词命中率: {diff_stats['avg_keyword_hit_rate']:.1%}\n"
            md += f"- 平均响应时间: {diff_stats['avg_response_time_ms']:.1f}ms\n"
            # 添加LLM性能指标（如果有数据）
            if diff_stats.get('avg_llm_tokens_per_second', 0) > 0:
                md += f"- **生成速度**: {diff_stats['avg_llm_tokens_per_second']:.2f} tokens/s\n"
                md += f"- 首Token时延: {diff_stats.get('avg_llm_ttft_ms', 0):.1f}ms\n"
                md += f"- 平均输出Token: {diff_stats.get('avg_llm_output_tokens', 0):.1f}\n"
            md += "\n"

        if analysis["problem_cases"]:
            md += "## ⚠️ 问题用例分析\n\n"
            for case in analysis["problem_cases"][:5]:
                md += f"### {case['id']}\n"
                md += f"**查询**: {case['query']}\n"
                md += f"**关键词命中率**: {case['hit_rate']:.1%}\n"
                md += f"**NDCG@5**: {case['ndcg_at_5']:.3f}\n"
                md += f"**主题覆盖率**: {case['topic_coverage']:.1%}\n"
                md += f"**未命中关键词**: {', '.join(case['missed_keywords'])}\n"
                md += f"**P@1**: {case['precision_at_1']:.3f}\n\n"

        md += """
## 📁 文件说明

- `rag_evaluation_report_{timestamp}.json`: 完整测试数据（JSON格式）
- `rag_evaluation_summary_{timestamp}.md`: 本摘要报告（Markdown格式）

## 💡 优化建议

"""

        if stats["avg_precision_at_1"] < 0.6:
            md += "- 🔴 检索精确率偏低，建议优化嵌入模型或重排序策略\n"
        if stats.get("avg_ndcg_at_5", 0) < 0.5:
            md += "- 🔴 排序质量不佳(NDCG@5)，建议引入重排序模型或优化相似度计算\n"
        if stats.get("avg_f1_at_5", 0) < 0.5:
            md += "- 🟡 F1分数偏低，需要平衡精确率和召回率，调整检索参数\n"
        if stats.get("avg_recall_at_5", 0) < 0.5:
            md += "- 🟡 召回率不足，建议增加top_k或扩展文档库\n"
        if stats["avg_keyword_hit_rate"] < 0.7:
            md += "- 🟡 关键词覆盖率不足，建议扩展文档内容或优化查询理解\n"
        if stats.get("avg_topic_coverage", 0) < 0.6:
            md += "- 🟡 主题覆盖率偏低，建议丰富各主题相关文档\n"
        if (
            stats.get("avg_semantic_similarity", 0) < 0.6
            and stats.get("avg_semantic_similarity", 0) > 0
        ):
            md += "- 🟡 语义相似度不足，建议优化embedding模型或答案生成质量\n"
        if stats["avg_mrr"] < 0.3:
            md += "- 🟡 MRR偏低，建议优化排序算法确保最相关结果排在首位\n"

        return md

    def run_evaluation(
        self,
        test_file: str,
        limit: Optional[int] = None,
        enable_rerank: bool = True,
        reranker_type: str = "bge",
        compare_with_baseline: bool = True,
    ) -> str:
        """运行完整测评"""
        print("\n" + "=" * 80)
        print("🚀 RAG系统增强测评 (AutoDL版)")
        print(f"💻 设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        if enable_rerank:
            print(f"🔄 重排序: {reranker_type.upper()}")
        else:
            print("📊 基础检索模式")
        print("=" * 80)
        print(f"测评时间: {self.evaluation_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if not self.init_services(
            enable_rerank=enable_rerank, reranker_type=reranker_type
        ):
            raise RuntimeError("服务初始化失败")

        test_data = self.load_test_data(test_file)

        metadata = test_data.get("metadata", {})
        retrieval_cases = test_data.get("retrieval_test_cases", []) + test_data.get(
            "retrieval_test_cases_part2", []
        )

        print(f"\n📊 数据集信息:")
        print(f"   版本: {metadata.get('version', 'unknown')}")
        print(f"   描述: {metadata.get('description', 'N/A')}")
        print(f"   检索测试: {len(retrieval_cases)} 条")

        print(f"\n🎯 主要测试: {'启用' if enable_rerank else '禁用'}重排序")
        results = self.evaluate_retrieval_cases(
            retrieval_cases,
            limit,
            enable_rerank=enable_rerank,
            reranker_type=reranker_type,
        )

        baseline_results = None
        if compare_with_baseline and enable_rerank:
            print(f"\n📊 对比测试: 禁用重排序")
            baseline_results = self.evaluate_retrieval_cases(
                retrieval_cases, limit, enable_rerank=False
            )

        print("\n📊 分析测试结果...")
        analysis = self.analyze_results(results)

        baseline_analysis = None
        if baseline_results:
            print("📊 分析基线测试结果...")
            baseline_analysis = self.analyze_results(baseline_results)

        print("🏆 计算综合评分...")
        score = self.calculate_score(analysis)
        baseline_score = None
        if baseline_analysis:
            baseline_score = self.calculate_score(baseline_analysis)

        print("💾 保存测试报告...")
        report_file = self.save_report(
            results,
            analysis,
            score,
            metadata,
            test_file,
            baseline_results=baseline_results,
            baseline_analysis=baseline_analysis,
            baseline_score=baseline_score,
        )

        self.print_summary(analysis, score, baseline_analysis, baseline_score)
        
        # 测试完成后，如果LLM模型常驻显存，则卸载释放资源
        if self.keep_llm_loaded and self._llm_client is not None:
            print("\n🧹 清理资源：卸载LLM模型...")
            self.unload_llm_model()

        return report_file

    def print_summary(
        self,
        analysis: Dict[str, Any],
        score: Dict[str, Any],
        baseline_analysis: Optional[Dict[str, Any]] = None,
        baseline_score: Optional[Dict[str, Any]] = None,
    ):
        """打印测评摘要"""
        print("\n" + "=" * 80)
        print("📊 测评摘要")
        if baseline_analysis:
            print("🔄 包含重排序效果对比")
        print("=" * 80)

        stats = analysis["statistics"]
        device = "GPU" if torch.cuda.is_available() else "CPU"

        print(f"\n💻 运行设备: {device}")
        print(f"🔍 整体性能:")
        print(
            f"   测试数量: {analysis['total_tests']} (有效:{analysis['valid_tests']}, 失败:{analysis['failed_tests']})"
        )
        print(f"   平均响应时间: {stats['avg_response_time_ms']:.1f}ms")
        print(f"   P@1 精确率: {stats['avg_precision_at_1']:.3f}")
        print(f"   P@3 精确率: {stats['avg_precision_at_3']:.3f}")
        print(f"   P@5 精确率: {stats['avg_precision_at_5']:.3f}")
        print(f"   🆕 Recall@5: {stats.get('avg_recall_at_5', 0):.3f}")
        print(f"   🆕 F1@5: {stats.get('avg_f1_at_5', 0):.3f}")
        print(f"   🆕 NDCG@5: {stats.get('avg_ndcg_at_5', 0):.3f}")
        print(f"   MRR: {stats['avg_mrr']:.3f}")
        print(f"   关键词命中率: {stats['avg_keyword_hit_rate']:.1%}")
        print(f"   🆕 主题覆盖率: {stats.get('avg_topic_coverage', 0):.1%}")
        print(f"   🆕 语义相似度: {stats.get('avg_semantic_similarity', 0):.3f}")
        
        # LLM 性能指标
        if stats.get('avg_llm_tokens_per_second', 0) > 0:
            print(f"\n   ⚡ LLM生成性能:")
            print(f"      平均输入Token: {stats.get('avg_llm_input_tokens', 0):.1f}")
            print(f"      平均输出Token: {stats.get('avg_llm_output_tokens', 0):.1f}")
            print(f"      首Token时延: {stats.get('avg_time_to_first_token_ms', 0):.1f}ms")
            print(f"      平均生成时间: {stats.get('avg_generation_time_ms', 0):.1f}ms")
            print(f"      ⚡ 生成速度: {stats.get('avg_llm_tokens_per_second', 0):.2f} tokens/s")
        
        # 打印 LLM 性能指标
        if stats.get('avg_tokens_per_second', 0) > 0:
            print(f"\n⚡ LLM生成性能:")
            print(f"   平均输入Token: {stats.get('avg_llm_input_tokens', 0):.1f}")
            print(f"   平均输出Token: {stats.get('avg_llm_output_tokens', 0):.1f}")
            print(f"   首Token时延: {stats.get('avg_time_to_first_token_ms', 0):.1f}ms")
            print(f"   平均生成时间: {stats.get('avg_generation_time_ms', 0):.1f}ms")
            print(f"   ⚡ 生成速度: {stats.get('avg_tokens_per_second', 0):.2f} tokens/s")

        if baseline_analysis:
            baseline_stats = baseline_analysis["statistics"]
            print(f"\n🔄 重排序效果对比:")
            score_improvement = score["total_score"] - baseline_score["total_score"]
            print(
                f"   综合评分: {baseline_score['total_score']} → {score['total_score']} ({score_improvement:+d}分)"
            )
            print(
                f"   P@1: {baseline_stats['avg_precision_at_1']:.3f} → {stats['avg_precision_at_1']:.3f} ({stats['avg_precision_at_1'] - baseline_stats['avg_precision_at_1']:+.3f})"
            )
            print(
                f"   NDCG@5: {baseline_stats.get('avg_ndcg_at_5', 0):.3f} → {stats.get('avg_ndcg_at_5', 0):.3f} ({stats.get('avg_ndcg_at_5', 0) - baseline_stats.get('avg_ndcg_at_5', 0):+.3f})"
            )
            print(
                f"   F1@5: {baseline_stats.get('avg_f1_at_5', 0):.3f} → {stats.get('avg_f1_at_5', 0):.3f} ({stats.get('avg_f1_at_5', 0) - baseline_stats.get('avg_f1_at_5', 0):+.3f})"
            )
            print(
                f"   响应时间: {baseline_stats['avg_response_time_ms']:.1f}ms → {stats['avg_response_time_ms']:.1f}ms ({stats['avg_response_time_ms'] - baseline_stats['avg_response_time_ms']:+.1f}ms)"
            )

        print(f"\n📈 按难度分析:")
        for diff, diff_stats in analysis["by_difficulty"].items():
            print(
                f"   {diff:6s}: P@1={diff_stats['avg_precision_at_1']:.2f}, "
                f"NDCG5={diff_stats['avg_ndcg_at_5']:.2f}, "
                f"F1@5={diff_stats['avg_f1_at_5']:.2f}, "
                f"关键词={diff_stats['avg_keyword_hit_rate']:.0%} ({diff_stats['count']}条)"
            )

        if analysis["problem_cases"]:
            print(f"\n⚠️  问题用例: {len(analysis['problem_cases'])} 条")
            for case in analysis["problem_cases"][:2]:
                print(f"   • {case['query'][:40]}... (命中:{case['hit_rate']:.0%})")

        print(f"\n🏆 综合评分: {score['grade']} - {score['total_score']}/100")

        for desc in score["grade_descriptions"]:
            print(f"   {desc}")

        print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG系统增强测评脚本 (AutoDL版)")
    parser.add_argument(
        "--test-file",
        type=str,
        default=str(TEST_DATASET_PATH),
        help=f"测试数据文件 (默认: {TEST_DATASET_PATH})",
    )
    parser.add_argument("--limit", type=int, help="限制测试数量")
    parser.add_argument(
        "--output-dir", type=str, default="test_reports", help="输出目录"
    )
    parser.add_argument(
        "--enable-rerank",
        action="store_true",
        default=True,
        help="启用重排序（默认启用）",
    )
    parser.add_argument("--disable-rerank", action="store_true", help="禁用重排序")
    parser.add_argument(
        "--reranker-type",
        type=str,
        default="bge",
        choices=["bge", "cross-encoder", "none"],
        help="重排序器类型",
    )
    parser.add_argument(
        "--compare", action="store_true", default=True, help="与基线对比（默认启用）"
    )
    parser.add_argument("--no-compare", action="store_true", help="禁用基线对比")
    parser.add_argument(
        "--vector-db-dir",
        type=str,
        default=str(VECTOR_DB_DIR),
        help=f"向量数据库目录 (默认: {VECTOR_DB_DIR})",
    )

    args = parser.parse_args()

    enable_rerank = args.enable_rerank and not args.disable_rerank
    compare_with_baseline = args.compare and not args.no_compare

    evaluator = RAGEvaluator(
        output_dir=args.output_dir,
        vector_db_path=args.vector_db_dir,
    )

    try:
        report_file = evaluator.run_evaluation(
            args.test_file,
            args.limit,
            enable_rerank=enable_rerank,
            reranker_type=args.reranker_type,
            compare_with_baseline=compare_with_baseline,
        )
        print(f"\n✅ 测评完成，报告已保存: {report_file}")
    except Exception as e:
        print(f"\n❌ 测评失败: {e}")
        logger.error(f"测评失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
