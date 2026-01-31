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