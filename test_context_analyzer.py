#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试上下文分析器
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.conversation_manager import conversation_manager
from backend.services.context_analyzer import context_analyzer


def test_context_analyzer():
    """
    测试上下文分析器
    """
    print("=== 测试上下文分析器 ===")
    print()
    
    # 1. 创建对话
    conversation = conversation_manager.create_conversation("test_user", "测试用户")
    conversation_id = conversation.id
    print(f"创建对话成功，对话ID: {conversation_id}")
    print()
    
    # 2. 添加初始问题和回答
    initial_query = "我是一名主管，我现在要申请通讯费报销，报销标准是什么？"
    initial_answer = "您作为主管，通讯费报销的标准是150元/月。需要提供运营商发票，无论是个人名头还是公司名头的发票都可以。"
    
    conversation_manager.add_message(conversation_id, "user", initial_query)
    conversation_manager.add_message(conversation_id, "assistant", initial_answer)
    
    print(f"初始问题: {initial_query}")
    print(f"初始回答: {initial_answer}")
    print()
    
    # 3. 获取对话历史
    conversation = conversation_manager.get_conversation(conversation_id)
    conversation_history = conversation.messages
    print(f"对话历史消息数: {len(conversation_history)}")
    print()
    
    # 4. 测试上下文相关查询
    test_queries = [
        "那经理呢？",
        "那总监呢？",
        "那员工的标准呢？",
        "那高级主管的政策呢？",
        "那其他人呢？"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"测试查询 {i+1}: {query}")
        print("-" * 40)
        
        # 分析上下文
        analysis_result = context_analyzer.analyze_context(conversation_history, query)
        
        print(f"是否上下文相关: {analysis_result['is_contextual']}")
        print(f"主要话题: {analysis_result['main_topic']}")
        print(f"实体: {analysis_result['entities']}")
        print(f"重写查询: {analysis_result['rewritten_query']}")
        print(f"上下文摘要: {analysis_result['context_summary'][:100]}...")
        print()
    
    # 5. 测试指代消解
    print("=== 测试指代消解 ===")
    print()
    
    # 添加一个包含实体的问题
    entity_query = "我想了解一下张三的报销情况"
    entity_answer = "张三的通讯费报销标准是150元/月，与主管级别相同。"
    
    conversation_manager.add_message(conversation_id, "user", entity_query)
    conversation_manager.add_message(conversation_id, "assistant", entity_answer)
    
    # 更新对话历史
    conversation = conversation_manager.get_conversation(conversation_id)
    conversation_history = conversation.messages
    
    # 测试指代查询
    coreference_queries = [
        "他的标准是多少？",
        "这个人的政策呢？",
        "他的要求是什么？"
    ]
    
    for i, query in enumerate(coreference_queries):
        print(f"测试指代查询 {i+1}: {query}")
        print("-" * 40)
        
        # 分析上下文
        analysis_result = context_analyzer.analyze_context(conversation_history, query)
        
        print(f"是否上下文相关: {analysis_result['is_contextual']}")
        print(f"主要话题: {analysis_result['main_topic']}")
        print(f"实体: {analysis_result['entities']}")
        print(f"重写查询: {analysis_result['rewritten_query']}")
        print()
    
    print("=== 测试完成 ===")


if __name__ == "__main__":
    try:
        test_context_analyzer()
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
