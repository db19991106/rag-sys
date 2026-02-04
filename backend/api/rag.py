from fastapi import APIRouter, HTTPException
from models import RAGRequest, RAGResponse, IntentResult
from services.rag_generator import rag_generator
from services.embedding import embedding_service
from services.intent_recognizer import intent_recognizer, IntentType, IntentConfig
from utils.logger import logger


router = APIRouter(prefix="/rag", tags=["RAG生成"])


@router.post("/generate", response_model=RAGResponse)
async def generate_answer(request: RAGRequest):
    """
    RAG 生成回答

    Args:
        request: RAG 请求
    """
    logger.info("=" * 80)
    logger.info("收到智能对话请求")
    logger.info(f"用户查询: {request.query}")
    
    # 意图识别
    intent, confidence, intent_details = intent_recognizer.recognize(request.query)
    
    logger.info(f"识别意图: {intent.value} (置信度: {confidence:.2f})")
    logger.info(f"识别方法: {intent_details.get('method', 'unknown')}")
    
    # 根据意图动态调整检索配置
    intent_config = IntentConfig.get_config(intent)
    adjusted_config = request.retrieval_config.model_copy(
        update={
            "top_k": intent_config["top_k"],
            "similarity_threshold": intent_config["similarity_threshold"]
        }
    )
    
    logger.info(f"原始检索配置: top_k={request.retrieval_config.top_k}, "
                f"阈值={request.retrieval_config.similarity_threshold}")
    logger.info(f"调整后检索配置: top_k={adjusted_config.top_k}, "
                f"阈值={adjusted_config.similarity_threshold}")
    logger.info(f"意图描述: {intent_config['description']}")
    
    logger.info(f"生成配置: provider={request.generation_config.llm_provider}, "
                f"model={request.generation_config.llm_model}, "
                f"temperature={request.generation_config.temperature}")

    try:
        # 使用调整后的配置执行检索和生成
        response = rag_generator.generate(
            request.query,
            adjusted_config,
            request.generation_config,
            request.conversation_id
        )

        logger.info(f"RAG 生成完成")
        logger.info(f"  - 查询: {request.query}")
        logger.info(f"  - 识别意图: {intent.value} (置信度: {confidence:.2f})")
        logger.info(f"  - 检索耗时: {response.retrieval_time_ms:.2f}ms")
        logger.info(f"  - 生成耗时: {response.generation_time_ms:.2f}ms")
        logger.info(f"  - 总耗时: {response.total_time_ms:.2f}ms")
        logger.info(f"  - 检索到上下文片段数: {len(response.context_chunks)}")

        if response.context_chunks:
            logger.info(f"检索结果详情:")
            for i, chunk in enumerate(response.context_chunks, 1):
                logger.info(f"  片段{i}: 相似度={chunk.similarity:.4f}, "
                           f"文档ID={chunk.document_id}, "
                           f"内容长度={len(chunk.content)}字符")

        logger.info(f"  - 生成回答: {response.answer[:200]}{'...' if len(response.answer) > 200 else ''}")
        logger.info("=" * 80)

        return response
    except Exception as e:
        logger.error(f"RAG 生成失败: {str(e)}")
        logger.error(f"错误详情: {type(e).__name__}")
        logger.error("=" * 80)
        raise HTTPException(status_code=500, detail=f"RAG 生成失败: {str(e)}")


@router.post("/recognize-intent", response_model=IntentResult)
async def recognize_intent(request: dict):
    """
    识别用户查询的意图
    
    Args:
        request: 包含query字段的请求体
    """
    try:
        query = request.get("query", "")
        if not query:
            raise HTTPException(status_code=400, detail="查询内容不能为空")
        
        # 确保意图识别器已初始化
        if not intent_recognizer._initialized:
            # 使用默认生成配置初始化意图识别器
            from models import GenerationConfig
            default_config = GenerationConfig(
                llm_provider="local",
                llm_model="Qwen2.5-0.5B-Instruct",
                temperature=0.1,
                max_tokens=100,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            intent_recognizer.initialize_with_config(default_config)
        
        intent, confidence, details = intent_recognizer.recognize(query)
        
        logger.info(f"意图识别: 查询='{query}', 意图={intent.value}, 置信度={confidence:.2f}")
        
        return IntentResult(
            intent=intent,
            confidence=confidence,
            details=details
        )
    except Exception as e:
        logger.error(f"意图识别失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"意图识别失败: {str(e)}")