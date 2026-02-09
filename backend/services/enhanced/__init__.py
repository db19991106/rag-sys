"""
增强版RAG服务模块
"""
from .scene_tagger import SceneTagger, SceneTag, SceneTagResult, scene_tagger
from .query_enhancer import QueryEnhancer, EnhancedQuery, SubQuery, HypothesisDoc, query_enhancer
from .retrieval_cache import RetrievalCache, QualityGate, RetrievalResult, QualityMetrics, QualityGateResult, QualityTier, retrieval_cache, quality_gate
from .generation_cache import GenerationCache, GenerationCacheEntry, generation_cache
from .rag_pipeline import EnhancedRAGPipeline, RAGResponse, enhanced_rag_pipeline

__all__ = [
    # 场景标签
    "SceneTagger",
    "SceneTag", 
    "SceneTagResult",
    "scene_tagger",
    # 查询增强
    "QueryEnhancer",
    "EnhancedQuery",
    "SubQuery",
    "HypothesisDoc",
    "query_enhancer",
    # 检索缓存与质量门控
    "RetrievalCache",
    "QualityGate",
    "RetrievalResult",
    "QualityMetrics",
    "QualityGateResult",
    "QualityTier",
    "retrieval_cache",
    "quality_gate",
    # 生成缓存
    "GenerationCache",
    "GenerationCacheEntry",
    "generation_cache",
    # RAG流水线
    "EnhancedRAGPipeline",
    "RAGResponse",
    "enhanced_rag_pipeline",
]
