"""
测试智能切分功能
验证智能切分选项是否与PDF切分使用相同的逻辑
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.chunker import Chunker
from models import ChunkConfig, ChunkType

# 测试文档内容
TEST_CONTENT = """
# 财务报销管理制度

## 第一章 总则

### 1.1 目的
为规范公司财务报销管理，加强内部控制，提高工作效率，特制定本制度。

### 1.2 适用范围
本制度适用于公司全体员工的日常费用报销、差旅费报销、业务招待费报销等所有财务报销事项。

## 第二章 报销流程

### 2.1 报销申请
员工应在费用发生后5个工作日内，通过OA系统提交报销申请，并附上相关原始凭证。

### 2.2 审批流程
报销申请需经过以下审批环节：
1. 部门负责人审核
2. 财务部门审核
3. 分管领导审批
4. 总经理审批（金额超过5000元）

## 第三章 差旅费报销

### 3.1 住宿标准
| 职级 | 住宿标准（元/晚） | 城市类别 |
|------|------------------|----------|
| 高管 | 800              | 一类城市 |
| 中层 | 600              | 一类城市 |
| 普通员工 | 400              | 一类城市 |
| 高管 | 600              | 二类城市 |
| 中层 | 400              | 二类城市 |
| 普通员工 | 300              | 二类城市 |

### 3.2 交通标准
1. 高管：可乘坐飞机经济舱、高铁一等座
2. 中层：可乘坐飞机经济舱、高铁二等座
3. 普通员工：可乘坐高铁二等座、硬卧

## 第四章 业务招待费

### 4.1 招待标准
业务招待费应遵循合理、必要、节约的原则，单次招待费用不得超过2000元。

### 4.2 审批权限
| 招待金额 | 审批权限 |
|----------|----------|
| ≤1000元 | 部门负责人 |
| 1001-2000元 | 分管领导 |
| ＞2000元 | 总经理 |

## 第五章 附则

### 5.1 解释权
本制度由财务部负责解释和修订。

### 5.2 生效日期
本制度自2024年1月1日起生效。
"""

def test_intelligent_chunking():
    """测试智能切分"""
    chunker = Chunker()
    
    # 测试智能切分
    config_intelligent = ChunkConfig(type=ChunkType.INTELLIGENT)
    chunks_intelligent = chunker.chunk(TEST_CONTENT, "test_doc_intelligent", config_intelligent)
    
    print("=== 智能切分结果 ===")
    print(f"生成片段数: {len(chunks_intelligent)}")
    for i, chunk in enumerate(chunks_intelligent, 1):
        print(f"\n片段 {i} (长度: {chunk.length}):")
        print(chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content)
    
    # 测试PDF切分
    config_pdf = ChunkConfig(type=ChunkType.PDF)
    chunks_pdf = chunker.chunk(TEST_CONTENT, "test_doc_pdf", config_pdf)
    
    print("\n=== PDF切分结果 ===")
    print(f"生成片段数: {len(chunks_pdf)}")
    for i, chunk in enumerate(chunks_pdf, 1):
        print(f"\n片段 {i} (长度: {chunk.length}):")
        print(chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content)
    
    # 验证两者是否使用相同的逻辑
    print("\n=== 验证结果 ===")
    print(f"智能切分片段数: {len(chunks_intelligent)}")
    print(f"PDF切分片段数: {len(chunks_pdf)}")
    print(f"片段数是否相同: {len(chunks_intelligent) == len(chunks_pdf)}")
    
    # 验证内容是否相似
    if len(chunks_intelligent) == len(chunks_pdf):
        for i in range(len(chunks_intelligent)):
            len_intel = len(chunks_intelligent[i].content)
            len_pdf = len(chunks_pdf[i].content)
            print(f"片段 {i+1} 长度差异: {abs(len_intel - len_pdf)}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_intelligent_chunking()
