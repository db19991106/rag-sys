#!/usr/bin/env python3

import sys
import os
import json
from pathlib import Path

# 配置路径
DOCS_DIR = Path("/root/autodl-tmp/rag/backend/tests/evaluation/data/docs")
VECTOR_DB_DIR = Path("/root/autodl-tmp/rag/backend/tests/evaluation/vector_db")

def main():
    print("🚀 简单构建测试")
    print("=" * 50)
    
    # 1. 检查目录
    print(f"📁 文档目录: {DOCS_DIR}")
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        # 创建测试文档
        test_content = """# 测试财务报销文档

## 1. 总则

### 1.1 适用范围
全体员工因公发生的费用支出，包括差旅费、业务招待费等。

## 2. 报销标准

### 2.1 差旅费
8-9级普通员工适用本制度对应职级报销标准。

### 2.2 住宿标准
| 职级 | 一线城市 | 其他城市 |
|------|----------|---------|
| 8-9级 | 500元 | 350元 |
| 10-11级 | 600元 | 450元 |

## 3. 审批流程

### 3.1 基本流程
提交申请 → 部门审批 → 财务审核 → 付款
"""
        with open(DOCS_DIR / "test_finance.md", "w", encoding="utf-8") as f:
            f.write(test_content)
    
        print("   创建测试文档: test_finance.md")
    
    doc_count = len(list(DOCS_DIR.glob('*.md')))
    print(f"📄 找到文档: {doc_count} 个")
    
    # 2. 简单切分
    print("✂️ 执行简单切分...")
    chunks = []
    
    sections = test_content.split("##")
    for i, section in enumerate(sections[1:], 1):  # 跳过第一行
        if section.strip():
            chunks.append({
                "id": f"chunk_{i}",
                "content": f"## {section.strip()}",
                "metadata": {"section": f"第{i}节", "doc_id": "test_doc"},
                "chunk_type": "text"
            })
    
    print(f"   生成 {len(chunks)} 个片段")
    
    # 3. 保存切分结果
    print("💾 保存切分结果...")
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    chunk_file = VECTOR_DB_DIR / "chunks.json"
    with open(chunk_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    print(f"   保存到: {chunk_file}")
    
    # 4. 统计信息
    print("=" * 50)
    print("✅ 构建完成统计:")
    print(f"   📄 文档数量: {len(list(DOCS_DIR.glob('*.md'))}")
    print(f"   ✂️ 片段数量: {len(chunks)}")
    print(f"   💾 数据目录: {VECTOR_DB_DIR}")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = main()
    print("构建成功！" if success else "构建失败！")
    sys.exit(0 if success else 1)