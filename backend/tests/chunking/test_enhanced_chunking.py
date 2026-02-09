import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.chunker import Chunker
from models import ChunkConfig, ChunkType

# 创建包含各种结构的测试文档
test_content = '''# 测试文档：增强型切分策略

## 第一章 引言
这是测试文档的引言部分，简短介绍增强型切分策略的目的和重要性。

## 第二章 表格完整性测试

### 2.1 简单表格

| 职级 | 标准 | 备注 |
|------|------|------|
| 总监级 | 800 | 一线城市 |
| 经理级 | 600 | 一线城市 |
| 主管级 | 400 | 一线城市 |

### 2.2 复杂表格

| 城市类别 | 一线城市（北上广深） | 新一线/省会 | 其他城市 |
|---------|------------------|------------|----------|
| 总监级 | 800 | 600 | 500 |
| 经理级 | 600 | 450 | 350 |
| 主管级 | 400 | 300 | 250 |

## 第三章 列表项聚合测试

### 3.1 数字编号列表

1. 管理原则一：责任明确，分工合理
2. 管理原则二：目标导向，结果优先
3. 管理原则三：沟通顺畅，协作高效
4. 管理原则四：持续改进，创新发展

### 3.2 中文数字列表

一、项目启动阶段
二、需求分析阶段
三、设计开发阶段
四、测试验收阶段
五、上线运维阶段

### 3.3 破折号列表

- 核心价值观：诚信、创新、协作、共赢
- 企业使命：为客户创造价值，为员工创造机会
- 企业愿景：成为行业领先的解决方案提供商

## 第四章 流程步骤连续性测试

### 4.1 英文流程步骤

Step 1: 提交申请 - 填写并提交电子版申请表
Step 2: 初步审核 - 人力资源部门进行初步资格审核
Step 3: 面试评估 - 相关部门负责人进行面试
Step 4: 背景调查 - 对拟录用人员进行背景调查
Step 5: 录用决策 - 综合评估后做出录用决策

### 4.2 中文流程步骤

步骤 1：准备材料 - 收集必要的申请材料
步骤 2：在线申请 - 通过公司官网提交申请
步骤 3：资格审查 - 招聘团队进行资格审查
步骤 4：笔试环节 - 相关岗位需要参加笔试
步骤 5：面试环节 - 通过资格审查的候选人进入面试
步骤 6：最终评估 - 综合各环节表现进行评估

## 第五章 小片段过滤测试

### 5.1 短段落

这是一个短段落，用于测试小片段过滤功能。

### 5.2 独立标题

## 第六章 综合测试

### 6.1 混合内容

| 类型 | 数量 | 状态 |
|------|------|------|
| A类 | 10 | 正常 |
| B类 | 20 | 正常 |
| C类 | 15 | 异常 |

1. 第一步：识别问题
2. 第二步：分析原因
3. 第三步：制定方案
4. 第四步：实施方案
5. 第五步：验证结果

Step 1: 准备工作 - 收集相关资料
Step 2: 分析现状 - 了解当前情况
Step 3: 制定计划 - 确定改进方向
Step 4: 执行计划 - 实施改进措施
Step 5: 评估效果 - 验证改进成果

短段落测试。

另一个短段落。
'''

# 创建切分器实例
chunker = Chunker()

# 创建增强型切分配置
config = ChunkConfig(type=ChunkType.ENHANCED)

# 执行切分
print("开始测试增强型切分策略...")
print(f"原始内容字符数: {len(test_content)}")

chunks = chunker.chunk(test_content, 'test_enhanced', config)

# 输出结果
print(f"\n增强型切分完成，生成了 {len(chunks)} 个片段")
print("\n片段详情：")
for i, chunk in enumerate(chunks):
    print(f"\n片段 {i+1}:")
    print(f"字符数: {len(chunk.content)}")
    print(f"内容前150字符: {chunk.content[:150]}...")

# 验证表格完整性
print("\n\n表格完整性验证：")
table_chunks = [chunk for chunk in chunks if '|' in chunk.content and '---' in chunk.content]
print(f"识别到 {len(table_chunks)} 个表格片段")

# 检查是否有表格被水平拆分
for i, table_chunk in enumerate(table_chunks):
    lines = table_chunk.content.strip().split('\n')
    # 检查表格结构
    has_header = any('---' in line for line in lines)
    has_data_rows = any('|' in line and '---' not in line for line in lines)
    print(f"表格 {i+1}: 包含标题={has_header}, 包含数据行={has_data_rows}")

# 验证列表项聚合
print("\n\n列表项聚合验证：")
list_chunks = [chunk for chunk in chunks if ('1.' in chunk.content and '2.' in chunk.content) or 
                                      ('一、' in chunk.content and '二、' in chunk.content) or 
                                      ('-' in chunk.content and chunk.content.count('-') >= 2)]
print(f"识别到 {len(list_chunks)} 个列表聚合片段")

# 验证流程步骤连续性
print("\n\n流程步骤连续性验证：")
process_chunks = [chunk for chunk in chunks if ('Step 1' in chunk.content and ('Step 2' in chunk.content or 'Step 3' in chunk.content)) or 
                                         ('步骤 1' in chunk.content and ('步骤 2' in chunk.content or '步骤 3' in chunk.content))]
print(f"识别到 {len(process_chunks)} 个流程步骤片段")

# 验证小片段过滤
print("\n\n小片段过滤验证：")
small_chunks = [chunk for chunk in chunks if len(chunk.content) < 50]
print(f"剩余小片段数量: {len(small_chunks)}")
for i, small_chunk in enumerate(small_chunks):
    print(f"小片段 {i+1}: {len(small_chunk.content)} 字符 - 内容: {small_chunk.content.strip()}")

# 验证内容完整性
total_characters = sum(len(chunk.content) for chunk in chunks)
original_characters = len(test_content.strip())
print(f"\n\n内容完整性验证：")
print(f"原始内容字符数: {original_characters}")
print(f"拆分后总字符数: {total_characters}")
print(f"内容完整性: {'完整' if abs(original_characters - total_characters) < 100 else '可能存在丢失'}")
print(f"字符数差异: {abs(original_characters - total_characters)}")

# 验证结构维持
print("\n\n结构维持验证：")
# 检查是否包含所有标题
all_headings = ['# 测试文档', '## 第一章', '## 第二章', '### 2.1', '### 2.2', '## 第三章', '### 3.1', '### 3.2', '### 3.3', '## 第四章', '### 4.1', '### 4.2', '## 第五章', '### 5.1', '### 5.2', '## 第六章', '### 6.1']

missing_headings = []
for heading in all_headings:
    found = any(heading in chunk.content for chunk in chunks)
    if not found:
        missing_headings.append(heading)

if missing_headings:
    print(f"缺失的标题: {missing_headings}")
else:
    print("所有标题均已保留")

print("\n增强型切分策略测试完成！")
