"""
RAG 流式生成 API
支持 Server-Sent Events (SSE) 格式的流式输出
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
import json
import asyncio
import time

from models import RAGRequest, RAGResponse, RetrievalResult
from services.rag_generator import rag_generator
from services.retriever import retriever
from services.embedding import embedding_service
from services.intent_recognizer import intent_recognizer, IntentConfig
from utils.logger import logger


router = APIRouter(prefix="/rag", tags=["RAG流式生成"])


class StreamRAGRequest(BaseModel):
    """流式RAG请求"""
    query: str
    conversation_id: Optional[str] = None
    top_k: int = 5
    similarity_threshold: float = 0.2
    temperature: float = 0.1  # 降低随机性，使输出更确定性
    max_tokens: int = 512


async def stream_rag_response(
    query: str,
    retrieval_config,
    generation_config,
    conversation_id: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    流式生成RAG响应
    
    Yields:
        SSE格式的数据块
    """
    start_time = time.time()
    
    try:
        # 1. 意图识别
        intent, confidence, intent_details = intent_recognizer.recognize(query)
        intent_config = IntentConfig.get_config(intent)
        use_knowledge_base = IntentConfig.should_use_knowledge_base(intent)
        
        # 发送意图识别结果
        intent_data = {
            "type": "intent",
            "intent": intent.value,
            "confidence": confidence,
            "use_knowledge_base": use_knowledge_base
        }
        yield f"data: {json.dumps(intent_data, ensure_ascii=False)}\n\n"
        
        # 2. 检索（如果需要）
        retrieval_time_ms = 0
        context_chunks = []
        
        if use_knowledge_base:
            retrieval_start = time.time()
            retrieval_response = retriever.retrieve(
                query, retrieval_config
            )
            retrieval_time_ms = (time.time() - retrieval_start) * 1000
            context_chunks = retrieval_response.results
            
            # 发送检索结果
            retrieval_data = {
                "type": "retrieval",
                "total": len(context_chunks),
                "latency_ms": retrieval_time_ms,
                "chunks": [
                    {
                        "document_name": c.document_name,
                        "similarity": c.similarity,
                        "content_preview": c.content[:100] + "..." if len(c.content) > 100 else c.content
                    }
                    for c in context_chunks[:3]  # 只发送前3个预览
                ]
            }
            yield f"data: {json.dumps(retrieval_data, ensure_ascii=False)}\n\n"
        
        # 3. 流式生成答案
        generation_start = time.time()
        
        # 调用流式生成方法（传入意图类型）
        async for chunk in rag_generator.generate_stream(
            query=query,
            retrieval_config=retrieval_config,
            generation_config=generation_config,
            conversation_id=conversation_id,
            context_chunks=context_chunks,
            use_knowledge_base=use_knowledge_base,
            intent_type=intent.value  # 传入意图类型
        ):
            if chunk:
                text_data = {
                    "type": "token",  # 改为 token，与前端期望一致
                    "content": chunk
                }
                yield f"data: {json.dumps(text_data, ensure_ascii=False)}\n\n"
        
        generation_time_ms = (time.time() - generation_start) * 1000
        total_time_ms = (time.time() - start_time) * 1000
        
        # 4. 发送完成信号
        complete_data = {
            "type": "done",  # 改为 done，与前端期望一致
            "retrieval_time_ms": retrieval_time_ms,
            "generation_time_ms": generation_time_ms,
            "total_time_ms": total_time_ms
        }
        yield f"data: {json.dumps(complete_data, ensure_ascii=False)}\n\n"
        
        logger.info(f"流式RAG完成: 查询='{query}', 总耗时={total_time_ms:.2f}ms")
        
    except Exception as e:
        logger.error(f"流式RAG生成失败: {str(e)}")
        error_data = {
            "type": "error",
            "message": str(e)
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"


@router.post("/generate/stream")
async def generate_stream(request: StreamRAGRequest):
    """
    流式生成回答 (SSE格式)
    
    返回格式:
    - type: intent - 意图识别结果
    - type: retrieval - 检索结果
    - type: text - 生成的文本片段
    - type: complete - 生成完成
    - type: error - 错误信息
    """
    logger.info(f"收到流式RAG请求: {request.query}")
    
    # 构建配置
    from models import RetrievalConfig, GenerationConfig
    
    retrieval_config = RetrievalConfig(
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
    )
    
    generation_config = GenerationConfig(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )
    
    return StreamingResponse(
        stream_rag_response(
            query=request.query,
            retrieval_config=retrieval_config,
            generation_config=generation_config,
            conversation_id=request.conversation_id
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用nginx缓冲
        }
    )


@router.get("/generate/stream/test")
async def test_stream():
    """测试流式输出"""
    async def test_generator():
        test_text = "这是一个流式输出测试，每个字符将逐个返回。"
        for char in test_text:
            data = {"type": "text", "content": char}
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.05)  # 模拟生成延迟
        
        complete_data = {"type": "complete", "total_time_ms": 1000}
        yield f"data: {json.dumps(complete_data, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        test_generator(),
        media_type="text/event-stream"
    )
