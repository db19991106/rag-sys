from typing import List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import numpy as np
from models import RetrievalConfig, RetrievalResponse
from services.retriever import retriever
from services.vector_db import vector_db_manager
from services.embedding import embedding_service
from services.reranker import reranker_manager
from utils.logger import logger


class SearchRequest(BaseModel):
    query: str
    config: RetrievalConfig


class SimilarChunksRequest(BaseModel):
    chunk_id: str
    content: str
    similarity_threshold: float = 0.7
    top_k: int = 5


class SimilarChunkResult(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    chunk_num: int
    content: str
    similarity: float


class SimilarChunksResponse(BaseModel):
    chunk_id: str
    similar_chunks: List[SimilarChunkResult]
    total: int


router = APIRouter(prefix="/retrieval", tags=["检索"])


@router.post("/search", response_model=RetrievalResponse)
async def search(request: SearchRequest):
    """
    执行检索

    Args:
        request: 检索请求，包含查询文本和配置
    """
    try:
        response = retriever.retrieve(request.query, request.config)
        logger.info(f"检索完成: 查询='{request.query}', 结果数={response.total}")
        return response
    except Exception as e:
        logger.error(f"检索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


@router.post("/similar-chunks", response_model=SimilarChunksResponse)
async def find_similar_chunks(request: SimilarChunksRequest):
    """
    查找与指定chunk相似的其他chunks
    
    Args:
        request: 包含chunk_id、内容和相似度阈值的请求
    """
    try:
        # 检查模型是否已加载
        if not embedding_service.is_loaded():
            raise HTTPException(
                status_code=400,
                detail="嵌入模型未加载，请先调用 /embedding/load"
            )
        
        # 检查向量数据库是否已初始化
        if not vector_db_manager.db:
            raise HTTPException(
                status_code=400,
                detail="向量数据库未初始化"
            )
        
        # 将chunk内容转换为向量
        # 注意：encode返回的是二维数组，取第一个元素
        vectors = embedding_service.encode([request.content])
        if vectors.ndim == 2:
            query_vector = vectors[0]  # 取第一个向量
        else:
            query_vector = vectors
        
        # 确保查询向量是二维数组 (1, dimension)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        logger.info(f"查询向量维度: {query_vector.shape}")
        
        # 搜索向量数据库（多获取一些结果，因为要排除自己）
        search_k = min(request.top_k * 2, 20)  # 最多搜索20个
        distances, metadata_list = vector_db_manager.search(query_vector, search_k)
        
        if len(distances) == 0 or len(metadata_list) == 0:
            return SimilarChunksResponse(
                chunk_id=request.chunk_id,
                similar_chunks=[],
                total=0
            )
        
        # 处理结果
        similar_chunks = []
        distances_row = distances[0]
        metadata_row = metadata_list[0]
        
        for distance, meta in zip(distances_row, metadata_row):
            # 跳过自己的chunk
            result_chunk_id = meta.get('chunk_id', '')
            if result_chunk_id == request.chunk_id:
                continue
            
            # 计算相似度（使用余弦相似度）
            similarity = 1 - (distance ** 2) / 2
            
            # 过滤低于阈值的
            if similarity < request.similarity_threshold:
                continue
            
            # 获取文档信息
            document_id = meta.get('document_id', '')
            document_name = meta.get('document_name', 'Unknown')
            
            similar_chunk = SimilarChunkResult(
                chunk_id=result_chunk_id,
                document_id=document_id,
                document_name=document_name,
                chunk_num=meta.get('chunk_num', 0),
                content=meta.get('content', ''),
                similarity=float(similarity)
            )
            
            similar_chunks.append(similar_chunk)
            
            # 只返回top_k个结果
            if len(similar_chunks) >= request.top_k:
                break
        
        logger.info(f"相似chunk检索完成: chunk_id='{request.chunk_id}', 找到{len(similar_chunks)}个相似片段")
        
        return SimilarChunksResponse(
            chunk_id=request.chunk_id,
            similar_chunks=similar_chunks,
            total=len(similar_chunks)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"相似chunk检索失败: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"相似chunk检索失败: {str(e)}")


@router.post("/reranker/initialize")
async def initialize_reranker(
    reranker_type: str = Query(..., description="重排序器类型: cross_encoder, colbert, mmr"),
    model_name: str = Query(..., description="模型名称"),
    top_k: int = Query(10, ge=1, le=100, description="重排序保留的top_k数量"),
    threshold: float = Query(0.0, ge=0.0, le=1.0, description="重排序分数阈值")
):
    """
    初始化重排序器
    
    Args:
        reranker_type: 重排序器类型
        model_name: 模型名称
        top_k: 保留的top_k数量
        threshold: 分数阈值
    """
    try:
        success = reranker_manager.initialize(
            reranker_type=reranker_type,
            model_name=model_name,
            device="cuda",
            top_k=top_k,
            threshold=threshold
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="重排序器初始化失败")
        
        logger.info(f"重排序器初始化成功: type={reranker_type}, model={model_name}")
        
        return {
            "success": True,
            "message": f"重排序器初始化成功: {reranker_type}",
            "type": reranker_type,
            "model": model_name
        }
    except Exception as e:
        logger.error(f"重排序器初始化失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重排序器初始化失败: {str(e)}")


@router.get("/reranker/status")
async def get_reranker_status():
    """
    获取重排序器状态
    """
    try:
        status = reranker_manager.get_status()
        return status
    except Exception as e:
        logger.error(f"获取重排序器状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取重排序器状态失败: {str(e)}")