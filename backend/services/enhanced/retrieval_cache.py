"""
检索缓存与质量门控
实现功能：检索结果缓存、多维度质量评估、自动修复策略
"""

import json
import hashlib
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from utils.logger import logger


class QualityTier(str, Enum):
    """质量等级"""

    HIGH = "high"  # 高质量 (k=5)
    MEDIUM = "medium"  # 中质量 (k=10)
    LOW = "low"  # 低质量 (需要修复)


@dataclass
class RetrievalResult:
    """检索结果"""

    doc_id: str
    content: str
    score: float
    source: str  # vector, keyword, hyde
    metadata: Dict = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """质量评估指标"""

    relevance_coverage: float  # 相关性覆盖率
    diversity_entropy: float  # 结果熵值 (多样性)
    confidence_distribution: float  # 置信度分布
    recency_score: float  # 时效性分数
    consistency_score: float  # 一致性分数
    overall_score: float  # 综合质量分


@dataclass
class QualityGateResult:
    """质量门控结果"""

    passed: bool
    tier: QualityTier
    candidates: List[RetrievalResult]
    metrics: QualityMetrics
    feedback: Optional[Dict] = None  # 修复建议


class RetrievalCache:
    """检索缓存管理器"""

    def __init__(self, redis_client=None, default_ttl: int = 1800):
        """
        Args:
            redis_client: Redis客户端
            default_ttl: 默认缓存时间(秒)，默认30分钟
        """
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.memory_cache = {}  # 内存缓存作为降级
        self._cache_hits = 0
        self._cache_misses = 0

    def _generate_cache_key(self, query: str, filters: Dict = None) -> str:
        """生成缓存键"""
        # 规范化查询
        normalized = query.lower().strip()
        # 组合查询和过滤条件
        cache_data = {"query": normalized, "filters": filters or {}}
        # 生成哈希
        key_str = json.dumps(cache_data, sort_keys=True)
        hash_val = hashlib.sha256(key_str.encode()).hexdigest()[:16]
        return f"retrieval:{hash_val}"

    async def get(
        self, query: str, filters: Dict = None
    ) -> Optional[List[RetrievalResult]]:
        """
        获取缓存的检索结果

        Returns:
            缓存的结果列表，如果不存在返回None
        """
        key = self._generate_cache_key(query, filters)

        try:
            if self.redis:
                data = self.redis.get(key)
                if data:
                    self._cache_hits += 1
                    results = self._deserialize_results(data)
                    logger.debug(f"Cache hit for query: {query[:50]}...")
                    return results
            else:
                # 使用内存缓存
                if key in self.memory_cache:
                    self._cache_hits += 1
                    cached = self.memory_cache[key]
                    # 检查是否过期
                    if cached["expire_time"] > time.time():
                        return cached["results"]
                    else:
                        del self.memory_cache[key]
        except Exception as e:
            logger.warning(f"Cache get error: {e}")

        self._cache_misses += 1
        return None

    async def set(
        self,
        query: str,
        results: List[RetrievalResult],
        filters: Dict = None,
        ttl: int = None,
    ):
        """
        设置检索结果缓存

        Args:
            query: 查询
            results: 检索结果
            filters: 过滤条件
            ttl: 过期时间(秒)
        """
        key = self._generate_cache_key(query, filters)
        ttl = ttl or self.default_ttl

        try:
            data = self._serialize_results(results)

            if self.redis:
                self.redis.setex(key, ttl, data)
            else:
                # 使用内存缓存
                self.memory_cache[key] = {
                    "results": results,
                    "expire_time": time.time() + ttl,
                }
                # 限制内存缓存大小
                if len(self.memory_cache) > 1000:
                    # 移除最早的条目
                    oldest_key = next(iter(self.memory_cache))
                    del self.memory_cache[oldest_key]

            logger.debug(f"Cached {len(results)} results for query: {query[:50]}...")

        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    def _serialize_results(self, results: List[RetrievalResult]) -> str:
        """序列化结果"""
        data = [
            {
                "doc_id": r.doc_id,
                "content": r.content,
                "score": r.score,
                "source": r.source,
                "metadata": r.metadata,
            }
            for r in results
        ]
        return json.dumps(data)

    def _deserialize_results(self, data: str) -> List[RetrievalResult]:
        """反序列化结果"""
        items = json.loads(data)
        return [
            RetrievalResult(
                doc_id=item["doc_id"],
                content=item["content"],
                score=item["score"],
                source=item["source"],
                metadata=item.get("metadata", {}),
            )
            for item in items
        ]

    def get_stats(self) -> Dict:
        """获取缓存统计"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "memory_size": len(self.memory_cache),
        }


class QualityGate:
    """检索质量门控"""

    def __init__(self, threshold: float = 0.6):
        """
        Args:
            threshold: 质量阈值 (0-1)
        """
        self.threshold = threshold

    def evaluate(
        self,
        candidates: List[RetrievalResult],
        query: str,
        scene_tags: List[str] = None,
    ) -> QualityGateResult:
        """
        评估检索结果质量

        Args:
            candidates: 检索候选结果
            query: 原始查询
            scene_tags: 场景标签

        Returns:
            QualityGateResult: 质量评估结果
        """
        scene_tags = scene_tags or []

        # 1. 计算相关性覆盖率
        relevance_coverage = self._calculate_relevance_coverage(candidates, query)

        # 2. 计算多样性熵
        diversity_entropy = self._calculate_diversity_entropy(candidates)

        # 3. 计算置信度分布
        confidence_dist = self._calculate_confidence_distribution(candidates)

        # 4. 计算时效性分数
        recency_score = self._calculate_recency_score(candidates, scene_tags)

        # 5. 计算一致性分数
        consistency_score = self._calculate_consistency_score(candidates)

        # 6. 计算综合质量分 (加权平均)
        weights = {
            "relevance": 0.35,
            "diversity": 0.20,
            "confidence": 0.25,
            "recency": 0.10,
            "consistency": 0.10,
        }

        overall_score = (
            relevance_coverage * weights["relevance"]
            + diversity_entropy * weights["diversity"]
            + confidence_dist * weights["confidence"]
            + recency_score * weights["recency"]
            + consistency_score * weights["consistency"]
        )

        metrics = QualityMetrics(
            relevance_coverage=relevance_coverage,
            diversity_entropy=diversity_entropy,
            confidence_distribution=confidence_dist,
            recency_score=recency_score,
            consistency_score=consistency_score,
            overall_score=overall_score,
        )

        # 判断质量等级
        if overall_score >= 0.8:
            tier = QualityTier.HIGH
            passed = True
        elif overall_score >= self.threshold:
            tier = QualityTier.MEDIUM
            passed = True
        else:
            tier = QualityTier.LOW
            passed = False

        # 构建反馈
        feedback = None
        if not passed:
            feedback = self._generate_repair_feedback(metrics, candidates, query)

        result = QualityGateResult(
            passed=passed,
            tier=tier,
            candidates=candidates,
            metrics=metrics,
            feedback=feedback,
        )

        logger.info(
            f"Quality evaluation: score={overall_score:.2f}, "
            f"tier={tier.value}, passed={passed}, "
            f"candidates={len(candidates)}"
        )

        return result

    def _calculate_relevance_coverage(
        self, candidates: List[RetrievalResult], query: str
    ) -> float:
        """计算相关性覆盖率"""
        if not candidates:
            return 0.0

        # 提取查询关键词
        query_keywords = set(self._extract_keywords(query))

        if not query_keywords:
            return 0.5  # 默认中等分数

        # 计算每个候选覆盖的关键词
        coverage_scores = []
        for candidate in candidates[:10]:  # 只看top10
            content_keywords = set(self._extract_keywords(candidate.content))
            if content_keywords:
                overlap = query_keywords & content_keywords
                coverage = len(overlap) / len(query_keywords)
                coverage_scores.append(coverage)

        if not coverage_scores:
            return 0.0

        # 取平均覆盖率
        avg_coverage = sum(coverage_scores) / len(coverage_scores)
        return min(avg_coverage * 2, 1.0)  # 放大信号，但不超过1

    def _calculate_diversity_entropy(self, candidates: List[RetrievalResult]) -> float:
        """计算结果多样性 (使用熵)"""
        if len(candidates) < 2:
            return 0.5

        # 基于来源分布计算熵
        source_counts = {}
        for c in candidates:
            source_counts[c.source] = source_counts.get(c.source, 0) + 1

        total = len(candidates)
        entropy = 0.0

        for count in source_counts.values():
            p = count / total
            if p > 0:
                import math

                entropy -= p * math.log2(p)  # 正确的熵计算

        # 归一化到0-1
        max_entropy = len(source_counts)
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0.5

        return max(0, min(1, normalized_entropy + 0.5))

    def _calculate_confidence_distribution(
        self, candidates: List[RetrievalResult]
    ) -> float:
        """计算置信度分布"""
        if not candidates:
            return 0.0

        scores = [c.score for c in candidates[:10]]

        if not scores:
            return 0.0

        # 计算高分候选的比例
        high_confidence_threshold = 0.5
        high_conf_count = sum(1 for s in scores if s > high_confidence_threshold)

        ratio = high_conf_count / len(scores)

        # 也考虑平均分数
        avg_score = sum(scores) / len(scores)

        # 综合分数
        return ratio * 0.6 + avg_score * 0.4

    def _calculate_recency_score(
        self, candidates: List[RetrievalResult], scene_tags: List[str]
    ) -> float:
        """计算时效性分数"""
        if not candidates:
            return 0.0

        # 检查是否有Temporal标签
        is_temporal = "Temporal" in scene_tags

        recency_scores = []
        for c in candidates:
            # 从metadata中提取时间信息
            metadata = c.metadata
            doc_date = metadata.get("upload_time") or metadata.get("created_at")

            if doc_date:
                try:
                    # 简化的时效性计算
                    # 实际应该解析日期并计算与当前时间的差距
                    recency_scores.append(0.8)  # 假设较新
                except:
                    recency_scores.append(0.5)
            else:
                recency_scores.append(0.5)

        if not recency_scores:
            return 0.5

        avg_recency = sum(recency_scores) / len(recency_scores)

        # 如果查询是时间敏感的，提高时效性权重
        if is_temporal:
            return min(1.0, avg_recency * 1.2)

        return avg_recency

    def _calculate_consistency_score(self, candidates: List[RetrievalResult]) -> float:
        """计算结果一致性分数"""
        if len(candidates) < 2:
            return 0.8  # 单个结果默认较高一致性

        # 简化的冲突检测：检查top结果之间是否有明显矛盾
        # 实际应该使用更复杂的NLP技术检测事实冲突

        top_scores = [c.score for c in candidates[:5]]
        score_variance = self._calculate_variance(top_scores)

        # 分数方差小表示一致性高
        if score_variance < 0.01:
            return 0.9
        elif score_variance < 0.05:
            return 0.7
        else:
            return 0.5

    def _calculate_variance(self, values: List[float]) -> float:
        """计算方差"""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        import re

        words = re.findall(r"[\w\u4e00-\u9fa5]+", text.lower())
        stop_words = {
            "的",
            "了",
            "在",
            "是",
            "我",
            "有",
            "和",
            "就",
            "不",
            "人",
            "都",
            "一",
            "上",
            "也",
            "很",
            "到",
            "说",
            "要",
            "去",
            "你",
            "会",
            "着",
            "没有",
            "看",
            "好",
            "自己",
            "这",
        }
        return [w for w in words if len(w) > 1 and w not in stop_words]

    def _generate_repair_feedback(
        self, metrics: QualityMetrics, candidates: List[RetrievalResult], query: str
    ) -> Dict:
        """生成修复建议"""
        issues = []
        suggestions = []
        action = "retry"

        # 诊断问题
        if metrics.relevance_coverage < 0.3:
            issues.append("relevance_coverage_low")
            suggestions.append("扩展查询关键词")
            suggestions.append("使用同义词")

        if metrics.diversity_entropy < 0.3:
            issues.append("diversity_low")
            suggestions.append("放宽过滤条件")
            suggestions.append("增加检索源多样性")

        if metrics.recency_score < 0.3:
            issues.append("recency_low")
            suggestions.append("放宽时间过滤")

        if not candidates:
            issues.append("no_results")
            action = "clarify"
            suggestions.append("请更具体地描述您的问题")

        # 如果歧义度高，建议澄清
        if len(issues) >= 3:
            action = "clarify"

        return {
            "action": action,
            "issues": issues,
            "suggestions": suggestions,
            "strategy": self._determine_repair_strategy(issues),
        }

    def _determine_repair_strategy(self, issues: List[str]) -> str:
        """确定修复策略"""
        if "no_results" in issues:
            return "expand_query"
        elif "relevance_coverage_low" in issues:
            return "synonym_expansion"
        elif "diversity_low" in issues:
            return "diversity_rerank"
        else:
            return "broaden_filters"


# 单例
retrieval_cache = RetrievalCache()
quality_gate = QualityGate()

__all__ = [
    "RetrievalCache",
    "QualityGate",
    "RetrievalResult",
    "QualityMetrics",
    "QualityGateResult",
    "QualityTier",
    "retrieval_cache",
    "quality_gate",
]
