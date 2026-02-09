#!/usr/bin/env python3
"""
测试PDF智能切分策略
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
from services.chunker import RAGFlowChunker
from models import ChunkConfig, ChunkType


def test_pdf_chunking():
    """
    测试PDF智能切分策略
    """
    # 读取测试文档
    test_file = os.path.join(os.path.dirname(__file__), 'data', 'docs', 'baoxiao.pdf')
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 创建切分器
    chunker = RAGFlowChunker()
    
    # 测试1: 使用PDF切分类型
    print("测试1: 使用PDF切分类型")
    config = ChunkConfig(
        type=ChunkType.PDF,
        chunk_token_size=512,
        overlapped_percent=0.1
    )
    
    chunks = chunker.chunk(content, "test_pdf", config)
    print(f"PDF切分结果: {len(chunks)} 个片段")
    for i, chunk in enumerate(chunks):
        print(f"\n片段 {i+1} (长度: {chunk.length}):")
        print(chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content)
    
    # 测试2: 使用智能切分（自动检测PDF类型）
    print("\n\n测试2: 使用智能切分（自动检测PDF类型）")
    config = ChunkConfig(
        type=ChunkType.INTELLIGENT,
        chunk_token_size=512,
        overlapped_percent=0.1
    )
    
    chunks = chunker.chunk(content, "test_intelligent", config)
    print(f"智能切分结果: {len(chunks)} 个片段")
    for i, chunk in enumerate(chunks):
        print(f"\n片段 {i+1} (长度: {chunk.length}):")
        print(chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content)
    
    # 测试3: 验证切分质量
    print("\n\n测试3: 验证切分质量")
    # 检查是否有小片段
    small_chunks = [chunk for chunk in chunks if len(chunk.content) < 80]
    print(f"小片段数量: {len(small_chunks)}")
    for chunk in small_chunks:
        print(f"小片段: {chunk.content[:100]}...")
    
    # 检查是否有超长片段
    long_chunks = [chunk for chunk in chunks if len(chunk.content) > 1500]
    print(f"超长片段数量: {len(long_chunks)}")
    for chunk in long_chunks:
        print(f"超长片段长度: {len(chunk.content)}")
    
    # 检查表格是否被正确处理
    table_chunks = [chunk for chunk in chunks if '|' in chunk.content and '---' in chunk.content]
    print(f"表格片段数量: {len(table_chunks)}")
    for i, chunk in enumerate(table_chunks):
        print(f"\n表格片段 {i+1}:")
        print(chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content)


if __name__ == "__main__":
    test_pdf_chunking()
