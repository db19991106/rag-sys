# PDF文档智能处理系统主入口文件

import os
import argparse
from typing import Dict, Any

from app.preprocessing.pdf_preprocessor import pdf_preprocessor
from app.layout_analysis.layout_analyzer import layout_analyzer
from app.content_extraction.content_extractor import content_extractor
from app.content_organization.content_organizer import content_organizer
from app.output_formatting.output_formatter import output_formatter
from utils.utils import logger, ensure_directory_exists


class PDFProcessor:
    """PDF处理器"""
    
    def __init__(self):
        """初始化PDF处理器"""
        logger.info("初始化PDF文档智能处理系统...")
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        处理PDF文件
        
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
            
            # 1. PDF类型检测与预处理
            logger.info("=== 步骤1: PDF类型检测与预处理 ===")
            preprocessing_result = pdf_preprocessor.preprocess_pdf(pdf_path)
            if preprocessing_result.get("status") != "success":
                logger.error(f"PDF预处理失败: {preprocessing_result.get('message', '未知错误')}")
                return preprocessing_result
            
            # 2. 版面分析
            logger.info("=== 步骤2: 版面分析 ===")
            layout_analysis_result = layout_analyzer.analyze_layout(preprocessing_result)
            if layout_analysis_result.get("status") != "success":
                logger.error(f"版面分析失败: {layout_analysis_result.get('message', '未知错误')}")
                return layout_analysis_result
            
            # 3. 内容提取
            logger.info("=== 步骤3: 内容提取 ===")
            extraction_result = content_extractor.extract_content(layout_analysis_result)
            if extraction_result.get("status") != "success":
                logger.error(f"内容提取失败: {extraction_result.get('message', '未知错误')}")
                return extraction_result
            
            # 4. 内容组织
            logger.info("=== 步骤4: 内容组织 ===")
            organization_result = content_organizer.organize_content(extraction_result)
            if organization_result.get("status") != "success":
                logger.error(f"内容组织失败: {organization_result.get('message', '未知错误')}")
                return organization_result
            
            # 5. 输出格式处理
            logger.info("=== 步骤5: 输出格式处理 ===")
            output_result = output_formatter.format_output(organization_result)
            if output_result.get("status") != "success":
                logger.error(f"输出格式处理失败: {output_result.get('message', '未知错误')}")
                return output_result
            
            logger.info("PDF文档处理完成！")
            return output_result
            
        except Exception as e:
            logger.error(f"PDF处理失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PDF文档智能处理系统")
    parser.add_argument("pdf_path", help="PDF文件路径")
    args = parser.parse_args()
    
    # 处理PDF文件
    processor = PDFProcessor()
    result = processor.process_pdf(args.pdf_path)
    
    # 输出处理结果
    if result.get("status") == "success":
        logger.info(f"处理成功！生成 {result.get('total_chunks', 0)} 个chunk")
        logger.info(f"输出文件:")
        for file_type, file_path in result.get('output_files', {}).items():
            logger.info(f"{file_type}: {file_path}")
    else:
        logger.error(f"处理失败: {result.get('message', '未知错误')}")


if __name__ == "__main__":
    main()
