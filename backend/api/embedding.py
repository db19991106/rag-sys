from fastapi import APIRouter, HTTPException
from models import EmbeddingConfig, EmbeddingResponse, ApiResponse
from services.embedding import embedding_service
from utils.logger import logger


router = APIRouter(prefix="/embedding", tags=["向量嵌入"])


@router.post("/load", response_model=EmbeddingResponse)
async def load_embedding_model(config: EmbeddingConfig):
    """
    加载嵌入模型

    Args:
        config: 嵌入配置
    """
    try:
        response = embedding_service.load_model(config)
        logger.info(f"嵌入模型加载: {response.message}")
        
        # 初始化意图识别器
        from services.intent_recognizer import intent_recognizer
        intent_recognizer.initialize(embedding_service)
        logger.info("意图识别器已初始化")
        
        return response
    except Exception as e:
        logger.error(f"加载嵌入模型失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"加载嵌入模型失败: {str(e)}")


@router.get("/status", response_model=ApiResponse)
async def get_embedding_status():
    """
    获取嵌入模型状态
    """
    try:
        is_loaded = embedding_service.is_loaded()
        dimension = embedding_service.get_dimension()

        return ApiResponse(
            success=True,
            message="获取状态成功",
            data={
                "is_loaded": is_loaded,
                "dimension": dimension,
                "model_name": embedding_service.config.model_name if embedding_service.config else None
            }
        )
    except Exception as e:
        logger.error(f"获取嵌入模型状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取嵌入模型状态失败: {str(e)}")


@router.post("/text", response_model=ApiResponse)
async def embed_text(text: str):
    """
    文本向量化API

    Args:
        text: 要向量化的文本

    Returns:
        向量化结果
    """
    try:
        if not text:
            return ApiResponse(
                success=False,
                message="文本不能为空",
                data={}
            )

        if not embedding_service.is_loaded():
            return ApiResponse(
                success=False,
                message="模型未加载，请先加载嵌入模型",
                data={}
            )

        # 编码文本为向量
        vectors = embedding_service.encode([text])
        if len(vectors) == 0:
            return ApiResponse(
                success=False,
                message="向量化失败",
                data={}
            )

        vector = vectors[0].tolist()

        return ApiResponse(
            success=True,
            message="向量化成功",
            data={
                "vector": vector,
                "dimension": len(vector),
                "model_name": embedding_service.config.model_name if embedding_service.config else None
            }
        )
    except Exception as e:
        logger.error(f"文本向量化失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文本向量化失败: {str(e)}")


@router.post("/text/batch", response_model=ApiResponse)
async def embed_text_batch(texts: list[str]):
    """
    批量文本向量化API

    Args:
        texts: 要向量化的文本列表

    Returns:
        批量向量化结果
    """
    try:
        if not texts or len(texts) == 0:
            return ApiResponse(
                success=False,
                message="文本列表不能为空",
                data={}
            )

        if not embedding_service.is_loaded():
            return ApiResponse(
                success=False,
                message="模型未加载，请先加载嵌入模型",
                data={}
            )

        # 限制批量大小，避免资源消耗过大
        max_batch_size = 100
        if len(texts) > max_batch_size:
            return ApiResponse(
                success=False,
                message=f"批量大小不能超过 {max_batch_size}",
                data={}
            )

        # 编码文本为向量
        vectors = embedding_service.encode(texts)
        if len(vectors) == 0:
            return ApiResponse(
                success=False,
                message="向量化失败",
                data={}
            )

        # 转换为列表格式
        vector_list = vectors.tolist()

        return ApiResponse(
            success=True,
            message="批量向量化成功",
            data={
                "vectors": vector_list,
                "dimension": len(vector_list[0]) if vector_list else 0,
                "model_name": embedding_service.config.model_name if embedding_service.config else None,
                "count": len(vector_list)
            }
        )
    except Exception as e:
        logger.error(f"批量文本向量化失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量文本向量化失败: {str(e)}")