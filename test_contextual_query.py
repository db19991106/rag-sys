#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试上下文相关查询处理
"""

import sys
import os
from typing import Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.conversation_manager import conversation_manager
from backend.services.rag_generator import rag_generator
from backend.models import RetrievalConfig, GenerationConfig


def test_contextual_query():
    """
    测试上下文相关查询处理
    """
    print("=== 测试上下文相关查询处理 ===")
    print()
    
    # 1. 创建对话
    conversation = conversation_manager.create_conversation("test_user", "测试用户")
    conversation_id = conversation.id
    print(f"创建对话成功，对话ID: {conversation_id}")
    print()
    
    # 2. 初始问题
    initial_query = "我是一名主管，我现在要申请通讯费报销，报销标准是什么？"
    print(f"初始问题: {initial_query}")
    print()
    
    # 3. 配置
    retrieval_config = RetrievalConfig(
        top_k=3,
        algorithm="cosine",
        enable_rerank=True,
        reranker_type="cross_encoder",
        reranker_model="BAAI/bge-reranker-large",
        device="cuda",
        enable_query_expansion=True,
        similarity_threshold=0.5
    )
    
    generation_config = GenerationConfig(
        llm_provider="local",
        llm_model="Qwen2.5-0.5B-Instruct",
        temperature=0.3,
        max_tokens=500,
        top_p=0.8,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    
    # 4. 处理初始问题
    print("处理初始问题...")
    print("=" * 50)
    
    response = rag_generator.generate(
        initial_query,
        retrieval_config,
        generation_config,
        conversation_id
    )
    
    print(f"初始问题回答: {response.answer}")
    print(f"检索耗时: {response.retrieval_time_ms:.2f}ms")
    print(f"生成耗时: {response.generation_time_ms:.2f}ms")
    print(f"总耗时: {response.total_time_ms:.2f}ms")
    print()
    
    # 5. 添加回答到对话历史
    conversation_manager.add_message(conversation_id, "assistant", response.answer)
    
    # 6. 上下文相关问题
    contextual_query = "那经理呢？"
    print(f"上下文相关问题: {contextual_query}")
    print()
    
    # 7. 处理上下文相关问题
    print("处理上下文相关问题...")
    print("=" * 50)
    
    contextual_response = rag_generator.generate(
        contextual_query,
        retrieval_config,
        generation_config,
        conversation_id
    )
    
    print(f"上下文相关问题回答: {contextual_response.answer}")
    print(f"检索耗时: {contextual_response.retrieval_time_ms:.2f}ms")
    print(f"生成耗时: {contextual_response.generation_time_ms:.2f}ms")
    print(f"总耗时: {contextual_response.total_time_ms:.2f}ms")
    print()
    
    # 8. 验证结果
    print("=== 验证结果 ===")
    if "经理" in contextual_response.answer:
        print("✅ 成功：回答中包含了'经理'相关信息")
    else:
        print("❌ 失败：回答中没有包含'经理'相关信息")
    
    if "标准" in contextual_response.answer or "报销" in contextual_response.answer:
        print("✅ 成功：回答中包含了'标准'或'报销'相关信息")
    else:
        print("❌ 失败：回答中没有包含'标准'或'报销'相关信息")
    
    print()
    print("=== 测试完成 ===")


if __name__ == "__main__":
    try:
        test_contextual_query()
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
