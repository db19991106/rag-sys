#!/usr/bin/env python3
"""
测试上下文分析器
"""

import sys

sys.path.insert(0, "/root/autodl-tmp/rag/backend")

from datetime import datetime
from services.context_analyzer import context_analyzer
from models import Message

# 模拟对话历史
history = [
    Message(
        id="msg_1",
        role="user",
        content="我是一名主管，我现在要申请通讯费报销，报销标准是什么？",
        conversation_id="conv_test",
        timestamp=datetime.now(),
    ),
    Message(
        id="msg_2",
        role="assistant",
        content="作为主管，您的通讯费报销标准是150元/月。需要提供运营商发票。",
        conversation_id="conv_test",
        timestamp=datetime.now(),
    ),
]

# 当前查询
current_query = "那部门总监呢？"

print("=" * 60)
print("🧪 测试上下文分析器")
print("=" * 60)
print(f"\n对话历史:")
for i, msg in enumerate(history, 1):
    print(f"  {i}. {msg.role}: {msg.content[:50]}...")

print(f"\n当前查询: {current_query}")

# 调用分析器
try:
    result = context_analyzer.analyze_context(history, current_query)
    print(f"\n✅ 分析结果:")
    print(f"  is_contextual: {result['is_contextual']}")
    print(f"  main_topic: {result['main_topic']}")
    print(f"  entities: {result['entities']}")
    print(f"  rewritten_query: {result['rewritten_query']}")
    print(
        f"  context_summary: {result['context_summary'][:100]}..."
        if result["context_summary"]
        else "  context_summary: (空)"
    )
except Exception as e:
    print(f"\n❌ 分析失败: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
