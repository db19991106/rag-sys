#!/usr/bin/env python3
"""
全面测试增强型切分功能
测试各种内容类型的切分效果，包括表格、列表、流程步骤和短段落
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.services.chunker import Chunker
from backend.models import ChunkType, ChunkConfig


def test_enhanced_chunking():
    """
    测试增强型切分功能
    """
    # 创建切分器实例
    chunker = Chunker()
    
    # 测试内容：包含表格、列表、流程步骤和短段落
    test_content = """
# 测试文档

## 1. 管理原则

以下是公司的管理原则：

1. 客户至上：始终以客户需求为中心
2. 创新驱动：不断探索新的解决方案
3. 团队协作：发挥集体智慧
4. 诚信经营：坚持诚实守信的原则

## 2. 报销流程

### 2.1 报销步骤

Step 1: 收集报销凭证
- 发票
- 收据
- 其他证明材料

Step 2: 填写报销单
- 详细填写费用明细
- 注明报销原因
- 部门主管签字

Step 3: 提交财务审核
- 财务部门会在3个工作日内完成审核
- 审核通过后会通知申请人

Step 4: 报销款到账
- 审核通过后，报销款会在5个工作日内到账

## 3. 薪资标准

| 职级 | 基本工资 |
|------|----------|
| 助理级 | 4000 |
| 专员级 | 5000 |
| 主管级 | 6000 |
| 经理级 | 8000 |
| 总监级 | 12000 |

## 4. 公司福利

公司提供以下福利：

- 五险一金
- 带薪年假
- 节日福利
- 定期体检
- 员工培训

## 5. 简短段落

这是一个简短的段落，用于测试小片段过滤功能。

这是另一个简短段落。

## 6. 项目管理流程

### 6.1 项目启动

1. 确定项目目标
2. 组建项目团队
3. 制定项目计划

### 6.2 项目执行

1. 定期召开项目例会
2. 及时解决项目问题
3. 监控项目进度

### 6.3 项目收尾

1. 完成项目交付
2. 进行项目验收
3. 总结项目经验

    """
    
    # 配置切分参数
    config = ChunkConfig(
        type=ChunkType.ENHANCED,
        chunk_token_size=512,
        overlapped_percent=0.1,
        delimiters=["\n"],
        enable_children=False,
        children_delimiters=[]
    )
    
    print("\n=== 测试增强型切分功能 ===\n")
    print("测试内容包含：")
    print("- 表格（薪资标准）")
    print("- 列表（管理原则、公司福利）")
    print("- 流程步骤（报销流程）")
    print("- 短段落（简短段落部分）")
    print("- 多级标题结构")
    print()
    
    # 执行切分
    chunks = chunker.chunk(test_content, "test_doc", config)
    
    print(f"切分结果：共生成 {len(chunks)} 个片段\n")
    
    # 打印每个片段
    for i, chunk in enumerate(chunks):
        print(f"=== 片段 {i+1} ===")
        print(f"长度：{len(chunk.content)} 字符")
        print("内容：")
        print(chunk.content)
        print()
    
    # 验证结果
    print("=== 验证结果 ===")
    
    # 检查表格完整性
    table_found = False
    for chunk in chunks:
        if "| 职级 | 基本工资 |" in chunk.content and "| 经理级 | 8000 |" in chunk.content:
            table_found = True
            print("✓ 表格完整性：表格未被水平拆分")
            break
    if not table_found:
        print("✗ 表格完整性：表格可能被拆分")
    
    # 检查列表聚合
    list_found = False
    for chunk in chunks:
        if "1. 客户至上" in chunk.content and "4. 诚信经营" in chunk.content:
            list_found = True
            print("✓ 列表聚合：同一子节的列表项被保持在一起")
            break
    if not list_found:
        print("✗ 列表聚合：同一子节的列表项可能被拆分")
    
    # 检查流程步骤连续性
    steps_found = False
    for chunk in chunks:
        if "Step 1: 收集报销凭证" in chunk.content and "Step 4: 报销款到账" in chunk.content:
            steps_found = True
            print("✓ 流程步骤连续性：完整流程被保持在一起")
            break
    if not steps_found:
        print("✗ 流程步骤连续性：流程步骤可能被拆分")
    
    # 检查小片段过滤
    small_fragments = [chunk for chunk in chunks if len(chunk.content) < 50 and not chunk.content.strip().startswith("#")]
    if len(small_fragments) == 0:
        print("✓ 小片段过滤：没有发现小于50字符的非标题片段")
    else:
        print(f"✗ 小片段过滤：发现 {len(small_fragments)} 个小于50字符的非标题片段")
    
    print()
    print("=== 测试完成 ===")


if __name__ == "__main__":
    test_enhanced_chunking()
