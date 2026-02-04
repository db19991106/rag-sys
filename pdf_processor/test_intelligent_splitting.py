#!/usr/bin/env python3
# 智能切分功能测试脚本

import os
import sys
import json
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.preprocessing.pdf_preprocessor import pdf_preprocessor
from app.layout_analysis.layout_analyzer import layout_analyzer
from app.content_extraction.content_extractor import content_extractor
from app.content_organization.content_organizer import content_organizer
from utils.utils import logger


def test_pdf_processing(pdf_path: str) -> Dict[str, Any]:
    """
    测试PDF处理流程
    
    Args:
        pdf_path: PDF文件路径
        
    Returns:
        处理结果
    """
    try:
        if not os.path.exists(pdf_path):
            logger.error(f"PDF文件不存在: {pdf_path}")
            return {
                "status": "error",
                "message": f"PDF文件不存在: {pdf_path}"
            }
        
        logger.info(f"开始处理PDF文件: {pdf_path}")
        
        # 1. 预处理
        logger.info("1. 开始预处理...")
        preprocess_result = pdf_preprocessor.preprocess_pdf(pdf_path)
        
        if preprocess_result.get("status") != "success":
            logger.error(f"预处理失败: {preprocess_result.get('message', '未知错误')}")
            return preprocess_result
        
        # 2. 版面分析
        logger.info("2. 开始版面分析...")
        layout_result = layout_analyzer.analyze_layout(preprocess_result)
        
        if layout_result.get("status") != "success":
            logger.error(f"版面分析失败: {layout_result.get('message', '未知错误')}")
            return layout_result
        
        # 3. 内容提取
        logger.info("3. 开始内容提取...")
        extract_result = content_extractor.extract_content(layout_result)
        
        if extract_result.get("status") != "success":
            logger.error(f"内容提取失败: {extract_result.get('message', '未知错误')}")
            return extract_result
        
        # 添加PDF路径到提取结果，用于文件类型识别
        extract_result["pdf_path"] = pdf_path
        
        # 4. 内容组织（包含智能切分）
        logger.info("4. 开始内容组织和智能切分...")
        organize_result = content_organizer.organize_content(extract_result)
        
        if organize_result.get("status") != "success":
            logger.error(f"内容组织失败: {organize_result.get('message', '未知错误')}")
            return organize_result
        
        logger.info(f"PDF处理完成: 生成 {organize_result.get('total_chunks', 0)} 个chunk")
        
        # 保存结果
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON结果
        output_json = os.path.join(output_dir, f"result_{os.path.basename(pdf_path).replace('.pdf', '.json')}")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(organize_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"结果保存到: {output_json}")
        
        # 保存Markdown结果
        if 'markdown_content' in organize_result:
            output_md = os.path.join(output_dir, f"result_{os.path.basename(pdf_path).replace('.pdf', '.md')}")
            with open(output_md, 'w', encoding='utf-8') as f:
                f.write(organize_result['markdown_content'])
            
            logger.info(f"Markdown结果保存到: {output_md}")
        
        # 打印chunk信息
        chunks = organize_result.get('chunks', [])
        logger.info("\n=== Chunk信息 ===")
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get('chunk_id', f'chunk_{i}')
            chunk_type = chunk.get('type', 'unknown')
            chunk_content = chunk.get('content', '').strip()[:100] + '...' if len(chunk.get('content', '')) > 100 else chunk.get('content', '')
            logger.info(f"Chunk {i+1} [{chunk_id}] ({chunk_type}): {chunk_content}")
        
        return organize_result
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }


def main():
    """
    主函数
    """
    # 测试文件路径
    test_files = [
        "/root/autodl-tmp/rag/backend/data/docs/baoxiao.pdf",  # 报销单
        # 可以添加其他测试文件
    ]
    
    for pdf_path in test_files:
        logger.info(f"\n====================================")
        logger.info(f"测试文件: {pdf_path}")
        logger.info(f"====================================")
        
        result = test_pdf_processing(pdf_path)
        
        if result.get("status") == "success":
            logger.info("测试成功!")
        else:
            logger.error(f"测试失败: {result.get('message', '未知错误')}")


if __name__ == "__main__":
    main()
