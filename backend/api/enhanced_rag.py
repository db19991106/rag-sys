"""
增强版RAG API路由
整合所有增强功能：场景标签、HyDE、问题分解、质量门控、缓存
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from services.enhanced.rag_pipeline import enhanced_rag_pipeline, RAGResponse
from services.enhanced.scene_tagger import scene_tagger, SceneTagResult
from services.enhanced.query_enhancer import query_enhancer, EnhancedQuery
from services.enhanced.retrieval_cache import retrieval_cache, quality_gate
from services.enhanced.generation_cache import generation_cache
from services.intent_recognizer import intent_recognizer
from utils.logger import logger


router = APIRouter(prefix="/rag-enhanced", tags=["增强版RAG"])


class RAGGenerateRequest(BaseModel):
    """RAG生成请求"""

    query: str = Field(..., min_length=1, max_length=2000, description="用户查询")
    session_id: Optional[str] = Field(None, description="会话ID")
    config: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="配置参数"
    )
    enable_hyde: bool = Field(True, description="启用HyDE")
    enable_decomposition: bool = Field(True, description="启用问题分解")
    quality_threshold: float = Field(0.6, ge=0, le=1, description="质量阈值")


class Citation(BaseModel):
    """引用信息"""

    id: int
    doc_id: str
    source: str
    preview: Optional[str] = None


class RAGGenerateResponse(BaseModel):
    """RAG生成响应"""

    answer: str
    citations: List[Citation]
    session_id: str
    trace_id: str
    scene_tags: List[str]
    metadata: Dict[str, Any]
    pending_action: Optional[str] = None


class SceneTagRequest(BaseModel):
    """场景标签识别请求"""

    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None


class SceneTagResponse(BaseModel):
    """场景标签识别响应"""

    query: str
    tags: List[str]
    confidence: Dict[str, float]
    details: Dict[str, Any]
    routing_strategy: str


class QueryEnhanceRequest(BaseModel):
    """查询增强请求"""

    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    scene_tags: List[str] = Field(default_factory=list)


class SubQueryInfo(BaseModel):
    """子查询信息"""

    id: str
    text: str
    dependencies: List[str]
    query_type: str


class QueryEnhanceResponse(BaseModel):
    """查询增强响应"""

    original_query: str
    main_query: str
    sub_queries: List[SubQueryInfo]
    hypotheses_count: int
    enhancements_applied: List[str]
    filters: Dict[str, Any]


class CacheStatsResponse(BaseModel):
    """缓存统计响应"""

    retrieval_cache: Dict[str, Any]
    generation_cache: Dict[str, Any]


@router.post("/generate", response_model=RAGGenerateResponse)
async def generate(request: RAGGenerateRequest, http_request: Request):
    """
    增强版RAG生成

    完整流程：
    1. 意图识别 + 缓存
    2. 场景标签识别
    3. 查询增强 (指代消解/HyDE/问题分解)
    4. 检索缓存检查
    5. 多路检索
    6. 质量门控
    7. 重排序
    8. 生成缓存检查
    9. RAG生成
    10. 会话更新
    """
    try:
        # 获取追踪ID
        trace_id = getattr(http_request.state, "trace_id", str(uuid.uuid4()))
        user_id = getattr(http_request.state, "user_id", None)

        logger.info(
            f"Enhanced RAG generate: query='{request.query[:50]}...', trace_id={trace_id}"
        )

        # 调用增强版RAG流水线
        result = await enhanced_rag_pipeline.generate(
            query=request.query,
            session_id=request.session_id,
            user_id=user_id,
            trace_id=trace_id,
            config=request.config,
        )

        # 转换为响应格式
        citations = [Citation(**c) for c in result.citations]

        return RAGGenerateResponse(
            answer=result.answer,
            citations=citations,
            session_id=result.session_id,
            trace_id=result.trace_id,
            scene_tags=result.scene_tags,
            metadata=result.metadata,
            pending_action=result.pending_action,
        )

    except Exception as e:
        logger.error(f"Enhanced RAG generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@router.post("/scene-tags", response_model=SceneTagResponse)
async def analyze_scene_tags(request: SceneTagRequest, http_request: Request):
    """分析查询的场景标签"""
    try:
        # 加载会话上下文
        session_context = None
        if request.session_id:
            from services.conversation_manager import conversation_manager

            session = conversation_manager.get_conversation(request.session_id)
            if session:
                session_context = {
                    "has_history": len(session.messages) > 0,
                    "entities": [],  # 简化处理
                    "history_summary": "",
                }

        # 场景标签识别
        result = scene_tagger.tag(request.query, session_context)

        return SceneTagResponse(
            query=request.query,
            tags=[t.value for t in result.tags],
            confidence=result.confidence,
            details=result.details,
            routing_strategy=result.routing_strategy,
        )

    except Exception as e:
        logger.error(f"Scene tag analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@router.post("/enhance-query", response_model=QueryEnhanceResponse)
async def enhance_query(request: QueryEnhanceRequest, http_request: Request):
    """增强查询 (指代消解/HyDE/问题分解)"""
    try:
        # 加载会话上下文
        session_context = {"has_history": False, "entities": []}
        if request.session_id:
            from services.conversation_manager import conversation_manager

            session = conversation_manager.get_conversation(request.session_id)
            if session:
                session_context = {
                    "has_history": len(session.messages) > 0,
                    "entities": [],
                }

        # 查询增强
        enhanced = query_enhancer.enhance(
            query=request.query,
            session_context=session_context,
            scene_tags=request.scene_tags,
        )

        # 转换为响应格式
        sub_queries = [
            SubQueryInfo(
                id=sq.id,
                text=sq.text,
                dependencies=sq.dependencies,
                query_type=sq.query_type,
            )
            for sq in enhanced.sub_queries
        ]

        return QueryEnhanceResponse(
            original_query=enhanced.original_query,
            main_query=enhanced.main_query,
            sub_queries=sub_queries,
            hypotheses_count=len(enhanced.hypotheses),
            enhancements_applied=enhanced.enhancements_applied,
            filters=enhanced.filters,
        )

    except Exception as e:
        logger.error(f"Query enhancement error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"增强失败: {str(e)}")


@router.get("/cache-stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """获取缓存统计"""
    try:
        retrieval_stats = retrieval_cache.get_stats()
        generation_stats = generation_cache.get_stats()

        return CacheStatsResponse(
            retrieval_cache=retrieval_stats, generation_cache=generation_stats
        )

    except Exception as e:
        logger.error(f"Cache stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")


@router.post("/clear-cache")
async def clear_cache(cache_type: str = "all"):
    """清除缓存"""
    try:
        import asyncio

        if cache_type in ["all", "retrieval"]:
            await retrieval_cache.invalidate()

        if cache_type in ["all", "generation"]:
            await generation_cache.invalidate()

        return {"message": f"已清除 {cache_type} 缓存", "status": "success"}

    except Exception as e:
        logger.error(f"Clear cache error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"清除缓存失败: {str(e)}")


@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "enhanced_rag": "available",
        "timestamp": datetime.utcnow().isoformat(),
    }
