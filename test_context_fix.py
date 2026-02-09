"""
测试上下文理解修复效果
"""
from services.context_analyzer import context_analyzer
from models import Message
from datetime import datetime

def test_entity_passing():
    """测试实体信息是否正确传递到改写阶段"""
    
    # 模拟对话历史
    conversation_history = [
        Message(
            id="msg1",
            conversation_id="test_conv",
            role="user",
            content="我是一名高管，我的报销标准是多少？",
            timestamp=datetime.now()
        ),
        Message(
            id="msg2",
            conversation_id="test_conv",
            role="assistant",
            content="高管的报销标准如下：飞机商务舱、住宿800元/晚...",
            timestamp=datetime.now()
        )
    ]
    
    # 第二轮查询
    current_query = "我如果想报销交通呢？"
    
    # 分析上下文
    result = context_analyzer.analyze_context(conversation_history, current_query)
    
    print("=" * 60)
    print("上下文分析结果")
    print("=" * 60)
    print(f"原始查询: {current_query}")
    print(f"是否依赖上下文: {result['is_contextual']}")
    print(f"核心主题: {result['main_topic']}")
    print(f"关键实体: {result['entities']}")
    print(f"上下文摘要: {result['context_summary']}")
    print(f"改写后查询: {result['rewritten_query']}")
    print("=" * 60)
    
    # 验证
    rewritten = result['rewritten_query']
    if "高管" in rewritten:
        print("✅ 测试通过：改写后的查询保留了'高管'身份信息")
    else:
        print("❌ 测试失败：改写后的查询丢失了'高管'身份信息")
        print(f"   期望包含'高管'，实际得到: {rewritten}")
    
    return "高管" in rewritten

if __name__ == "__main__":
    success = test_entity_passing()
    exit(0 if success else 1)
