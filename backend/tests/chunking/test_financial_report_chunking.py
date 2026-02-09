#!/usr/bin/env python3
"""
测试财务报告文档的智能分块功能
验证分块结果是否与 baoxiao_chunking_final.md 中的最终方案完全匹配
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.chunker import RAGFlowChunker
from models import ChunkConfig, ChunkType


def test_financial_report_chunking():
    """
    测试财务报告文档的智能分块功能
    """
    print("=== 测试财务报告文档智能分块 ===")
    
    # 读取测试文档
    test_file_path = os.path.join(os.path.dirname(__file__), "data", "docs", "baoxiao.md")
    
    if not os.path.exists(test_file_path):
        print(f"错误：测试文件不存在 - {test_file_path}")
        return False
    
    with open(test_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    print(f"读取测试文档成功，总长度：{len(content)} 字符")
    
    # 初始化分块器
    chunker = RAGFlowChunker()
    
    # 配置分块参数
    config = ChunkConfig(
        type=ChunkType.INTELLIGENT,
        chunk_token_size=512,
        overlapped_percent=0.1,
        delimiters=["\n"],
        enable_children=False,
        children_delimiters=[]
    )
    
    # 执行分块
    print("执行智能分块...")
    chunk_infos = chunker.chunk(content, "test_doc", config)
    
    print(f"分块完成，共生成 {len(chunk_infos)} 个片段")
    
    # 验证分块结果
    print("\n=== 验证分块结果 ===")
    
    # 检查分块数量是否正确
    expected_chunk_count = 12
    if len(chunk_infos) == expected_chunk_count:
        print(f"✓ 分块数量正确：{len(chunk_infos)} 个片段")
    else:
        print(f"✗ 分块数量错误：期望 {expected_chunk_count} 个，实际 {len(chunk_infos)} 个")
    
    # 检查每个片段的内容
    for i, chunk_info in enumerate(chunk_infos):
        chunk_num = i + 1
        chunk_content = chunk_info.content
        
        print(f"\n=== 片段 {chunk_num} ===")
        print(f"长度：{len(chunk_content)} 字符")
        print(f"完整内容：")
        print("-" * 80)
        print(chunk_content)
        print("-" * 80)
        
        # 检查片段是否为空
        if not chunk_content or not chunk_content.strip():
            print(f"✗ 片段 {chunk_num} 内容为空")
        else:
            print(f"✓ 片段 {chunk_num} 内容正常")
    
    # 检查关键片段的内容
    print("\n=== 检查关键片段 ===")
    
    # 检查片段1：文档标题 + 第一章 总则
    if len(chunk_infos) >= 1:
        chunk1 = chunk_infos[0].content
        if "财务报销管理制度" in chunk1 and "第一章 总则" in chunk1:
            print("✓ 片段1 包含文档标题和第一章总则")
        else:
            print("✗ 片段1 缺少文档标题或第一章总则")
    
    # 检查片段2：第二章 2.1 差旅费标准
    if len(chunk_infos) >= 2:
        chunk2 = chunk_infos[1].content
        if "2.1 差旅费标准" in chunk2 and "交通工具" in chunk2 and "住宿标准" in chunk2:
            print("✓ 片段2 包含完整的2.1差旅费标准")
        else:
            print("✗ 片段2 缺少2.1差旅费标准的关键内容")
    
    # 检查片段7：第三章 3.1 报销流程
    if len(chunk_infos) >= 7:
        chunk7 = chunk_infos[6].content
        if "第三章 报销流程与审批权限" in chunk7 and "报销流程" in chunk7:
            print("✓ 片段7 包含第三章标题和报销流程")
        else:
            print("✗ 片段7 缺少第三章标题或报销流程")
    
    print("\n=== 测试完成 ===")
    return len(chunk_infos) == expected_chunk_count


if __name__ == "__main__":
    success = test_financial_report_chunking()
    if success:
        print("\n🎉 测试通过：财务报告文档智能分块功能正常")
        sys.exit(0)
    else:
        print("\n❌ 测试失败：财务报告文档智能分块功能存在问题")
        sys.exit(1)
