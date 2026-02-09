"""
诊断第二轮对话问题的根本原因
"""
import sys
sys.path.insert(0, '/root/autodl-tmp/rag/backend')

from services.context_analyzer import context_analyzer
from services.rag_generator import rag_generator
from models import Message
from datetime import datetime

def analyze_second_round():
    """分析第二轮对话的处理流程"""
    
    print("=" * 70)
    print("第二轮对话问题诊断")
    print("=" * 70)
    
    # 模拟两轮对话历史
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
            content="高管报销标准：飞机商务舱、住宿800元/晚...",
            timestamp=datetime.now()
        ),
        Message(
            id="msg3",
            conversation_id="test_conv",
            role="user",
            content="我如果想报销差旅费，我能报销多少？",
            timestamp=datetime.now()
        )
    ]
    
    current_query = "我如果想报销差旅费，我能报销多少？"
    
    print(f"\n当前查询: {current_query}")
    print("-" * 70)
    
    # 1. 测试上下文分析
    print("\n【步骤1】上下文分析:")
    result = context_analyzer.analyze_context(conversation_history, current_query)
    
    print(f"  是否依赖上下文: {result['is_contextual']}")
    print(f"  核心主题: {result['main_topic']}")
    print(f"  关键实体: {result['entities']}")
    print(f"  上下文摘要: {result['context_summary']}")
    print(f"  改写后查询: {result['rewritten_query']}")
    
    # 2. 检查Prompt构建
    print("\n【步骤2】Prompt构建分析:")
    
    # 模拟构建的Prompt
    document_context = """根据文档中的信息，作为高管（总监及以上），您的报销标准如下：

1. **交通费用**：
   - 飞机：商务舱或头等舱（高铁商务座）
   - 火车：高铁商务座
   - 市内交通：实报实销（含出租车）

2. **住宿费用**：
   - 一线城市（北上广深）：800元/晚
   - 新一线城市/省会：600元/晚
   - 其他城市：500元/晚

3. **差旅补贴**：
   - 一线城市：180元/天
   - 其他城市：150元/天"""
    
    context_summary = result['context_summary']
    full_context = f"对话历史摘要: {context_summary}\n\n{document_context}" if context_summary else document_context
    
    # 当前使用的Prompt
    current_prompt = f"""请根据以下参考文档回答问题。如果文档中没有相关信息，请明确说明。

参考文档:
{full_context}

问题: {result['rewritten_query']}

回答:"""
    
    print("  当前Prompt（实际使用）:")
    print(f"  {current_prompt[:300]}...")
    print("\n  ❌ 问题: Prompt中没有提到用户的'高管'身份！")
    
    # 3. 建议的Prompt
    print("\n【步骤3】建议的改进Prompt:")
    entities_str = ", ".join(result['entities']) if result['entities'] else ""
    improved_prompt = f"""请根据以下参考文档回答问题。如果文档中没有相关信息，请明确说明。

用户身份信息: 高管

参考文档:
{full_context}

问题: {result['rewritten_query']}

回答要求: 请根据用户的高管身份，只提供与高管相关的报销标准信息。"""
    
    print(f"  {improved_prompt[:350]}...")
    print("\n  ✅ 改进: 明确告诉LLM用户是高管，要求只提供相关信息")
    
    # 4. 问题总结
    print("\n" + "=" * 70)
    print("问题总结")
    print("=" * 70)
    print("""
问题1: 查询改写阶段 ✓ 已修复
  - 传递了entities参数
  - 改写后的查询应该包含"高管"

问题2: Prompt构建阶段 ✗ 存在问题  
  - _build_prompt()方法没有接收entities参数
  - Prompt中没有体现用户身份信息
  - LLM不知道应该只回答高管相关内容

问题3: 检索阶段 ? 可能存在问题
  - 检索结果包含所有职级的文档
  - 需要在检索时过滤或加权高管相关内容

根本原因:
  虽然查询被正确改写，但在生成回答的Prompt中没有
  明确指示LLM只关注"高管"相关信息，导致LLM生成了
  涵盖所有职级的完整回答。
""")

if __name__ == "__main__":
    analyze_second_round()
