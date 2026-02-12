#!/usr/bin/env python3
"""
简单测试脚本，避免被系统日志干扰
"""

import sys
import os
from pathlib import Path

# 添加backend路径
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# 禁用系统日志
os.environ["NO_SYSTEM_LOG"] = "1"
os.environ["DISABLE_SYSTEM_LOG"] = "true"

from eval_config import get_config
from services.document_parser import DocumentParser
from services.financial_chunker_v2 import FinancialDocumentChunker


def main():
    print("=" * 60)
    print("🚀 简单构建测试")
    print("=" * 60)

    # 1. 测试配置加载
    print("\n📋 测试配置加载...")
    config = get_config("financial")
    print(f"✅ 配置加载成功")
    print(f"   文档目录: {config['docs_dir']}")
    print(f"   向量库目录: {config['vector_db_dir']}")
    print(f"   切分方法: {config['chunking_method']}")

    # 2. 测试文档扫描
    print("\n📁 测试文档扫描...")
    docs_dir = config["docs_dir"]
    if not docs_dir.exists():
        print(f"❌ 文档目录不存在: {docs_dir}")
        return

    doc_files = list(docs_dir.glob("**/*.md"))
    print(f"✅ 找到 {len(doc_files)} 个文档")
    for doc in doc_files:
        print(f"   - {doc.name}")

    # 3. 测试文档解析
    print("\n📄 测试文档解析...")
    for doc_file in doc_files:
        try:
            content = DocumentParser.parse(str(doc_file))
            print(f"✅ {doc_file.name}: 解析成功 ({len(content)} 字符)")
            return content  # 只测试第一个
        except Exception as e:
            print(f"❌ {doc_file.name}: 解析失败 - {e}")
            return None

    print("❌ 没有可用的文档")

    # 4. 测试切分
    print("\n✂️ 测试财务切分...")
    if content:
        try:
            chunker = FinancialDocumentChunker(max_chunk_size=1000)
            chunks = chunker.chunk_document(content, doc_id="test")
            print(f"✅ 切分成功: {len(chunks)} 个片段")

            # 显示前3个片段
            for i, chunk in enumerate(chunks[:3], 1):
                print(f"   片段 {i}: {chunk.content[:100]}...")

        except Exception as e:
            print(f"❌ 切分失败: {e}")

    print("\n✅ 测试完成！")
    return True


if __name__ == "__main__":
    main()
