from fastapi import APIRouter, HTTPException
from models import RAGRequest, RAGResponse, IntentResult
from services.rag_generator import rag_generator
from services.embedding import embedding_service
from services.intent_recognizer import intent_recognizer, IntentType, IntentConfig
from services.query_cache import query_cache
from utils.logger import logger


router = APIRouter(prefix="/rag", tags=["RAG生成"])


@router.post("/generate", response_model=RAGResponse)
async def generate_answer(request: RAGRequest):
    """
    RAG 生成回答

    根据意图类型选择不同的回答策略：
    - 业务意图（薪酬福利、考勤休假等）：检索知识库回答
    - 闲聊意图：直接使用LLM回答，不检索知识库
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
    
    # 判断是否使用知识库
    use_knowledge_base = IntentConfig.should_use_knowledge_base(intent)
    
    if use_knowledge_base:
        # 业务意图：调整检索配置
        adjusted_config = request.retrieval_config.model_copy(
            update={
                "top_k": intent_config["top_k"],
                "similarity_threshold": intent_config["similarity_threshold"]
            }
        )
        
        logger.info(f"业务意图 - 使用知识库检索")
        logger.info(f"原始检索配置: top_k={request.retrieval_config.top_k}, "
                    f"阈值={request.retrieval_config.similarity_threshold}")
        logger.info(f"调整后检索配置: top_k={adjusted_config.top_k}, "
                    f"阈值={adjusted_config.similarity_threshold}")
        
        # 获取目标文档列表
        target_docs = IntentConfig.get_target_documents(intent)
        if target_docs:
            logger.info(f"目标文档: {target_docs[:3]}{'...' if len(target_docs) > 3 else ''}")
    else:
        # 闲聊意图：不检索知识库
        adjusted_config = request.retrieval_config.model_copy(
            update={
                "top_k": 0,  # 不检索
                "similarity_threshold": 1.0  # 不过滤
            }
        )
        logger.info(f"闲聊意图 - 直接使用LLM回答，不检索知识库")
    
    logger.info(f"意图描述: {intent_config['description']}")
    logger.info(f"生成配置: provider={request.generation_config.llm_provider}, "
                f"model={request.generation_config.llm_model}, "
                f"temperature={request.generation_config.temperature}")

    # ========== 缓存检查 ==========
    # 只有业务意图且使用知识库时才使用缓存
    if use_knowledge_base:
        retrieval_config_dict = {
            "top_k": adjusted_config.top_k,
            "similarity_threshold": adjusted_config.similarity_threshold,
            "enable_rerank": adjusted_config.enable_rerank,
        }
        
        cached_response = query_cache.get(
            query=request.query,
            retrieval_config=retrieval_config_dict,
            conversation_id=request.conversation_id,
        )
        
        if cached_response:
            logger.info(f"✅ 缓存命中！跳过检索和生成")
            cached_data = cached_response.get("data", cached_response)
            
            # 构建响应对象
            response = RAGResponse(
                query=cached_data.get("query", request.query),
                answer=cached_data.get("answer", ""),
                context_chunks=[],  # 缓存响应不返回上下文
                generation_time_ms=0,
                retrieval_time_ms=0,
                total_time_ms=1,  # 几乎为0
            )
            
            logger.info(f"  - 缓存回答: {response.answer[:100]}...")
            logger.info(f"  - 总耗时: 1ms (缓存)")
            logger.info("=" * 80)
            return response
        
        logger.info(f"缓存未命中，继续执行检索和生成")
        logger.info(f"缓存统计: {query_cache.get_stats()}")

    try:
        if use_knowledge_base:
            # 业务意图：执行检索和生成
            response = rag_generator.generate(
                request.query,
                adjusted_config,
                request.generation_config,
                request.conversation_id,
                intent.value  # 传入意图类型
            )
            
            # ========== 存入缓存 ==========
            cache_data = {
                "query": request.query,
                "answer": response.answer,
                "intent": intent.value,
                "confidence": confidence,
                "retrieval_time_ms": response.retrieval_time_ms,
                "generation_time_ms": response.generation_time_ms,
                "total_time_ms": response.total_time_ms,
            }
            query_cache.set(
                query=request.query,
                retrieval_config=retrieval_config_dict,
                response_data=cache_data,
                conversation_id=request.conversation_id,
            )
            logger.info(f"✅ 已存入缓存")
            
        else:
            # 闲聊意图：直接使用LLM生成，不检索
            response = rag_generator.generate_without_retrieval(
                request.query,
                request.generation_config,
                request.conversation_id,
                prompt_prefix=intent_config.get("prompt_template", "")
            )

        logger.info(f"RAG 生成完成")
        logger.info(f"  - 查询: {request.query}")
        logger.info(f"  - 识别意图: {intent.value} (置信度: {confidence:.2f})")
        logger.info(f"  - 使用知识库: {use_knowledge_base}")
        logger.info(f"  - 检索耗时: {response.retrieval_time_ms:.2f}ms")
        logger.info(f"  - 生成耗时: {response.generation_time_ms:.2f}ms")
        logger.info(f"  - 总耗时: {response.total_time_ms:.2f}ms")
        
        if use_knowledge_base and response.context_chunks:
            logger.info(f"  - 检索到上下文片段数: {len(response.context_chunks)}")
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
            # 使用配置中的LLM提供商
            from models import GenerationConfig
            from config import settings
            
            # 使用配置中的llm_provider，如果是local则尝试使用vllm
            provider = settings.llm_provider
            if provider == "local" and settings.vllm_enabled:
                provider = "vllm"
            
            default_config = GenerationConfig(
                llm_provider=provider,
                llm_model=settings.llm_model,
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
