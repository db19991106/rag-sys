# PDF文档智能处理系统测试脚本

import os
import sys
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import PDFProcessor
from utils.utils import logger


def test_pdf_processor():
    """测试PDF处理器"""
    # 测试文件路径
    test_pdf_path = "test_sample.pdf"
    
    # 检查测试文件是否存在
    if not os.path.exists(test_pdf_path):
        logger.error(f"测试PDF文件不存在: {test_pdf_path}")
        logger.info("请将测试PDF文件放在当前目录下，命名为test_sample.pdf")
        return False
    
    logger.info("=== 开始测试PDF文档智能处理系统 ===")
    
    # 创建PDF处理器
    processor = PDFProcessor()
    
    # 处理PDF文件
    result = processor.process_pdf(test_pdf_path)
    
    # 检查处理结果
    if result.get("status") == "success":
        logger.info("=== 测试成功！===")
        logger.info(f"PDF类型: {result.get('pdf_type', '未知')}")
        logger.info(f"生成chunk数量: {result.get('total_chunks', 0)}")
        logger.info(f"输出文件:")
        for file_type, file_path in result.get('output_files', {}).items():
            logger.info(f"{file_type}: {file_path}")
        
        # 验证输出文件
        for file_type, file_path in result.get('output_files', {}).items():
            if os.path.exists(file_path):
                logger.info(f"{file_type}文件生成成功: {file_path}")
                
                # 验证JSON文件格式
                if file_type == "json":
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            chunks = json.load(f)
                            logger.info(f"JSON文件包含 {len(chunks)} 个chunk")
                            
                            # 验证每个chunk的格式
                            for i, chunk in enumerate(chunks):
                                required_fields = ["chunk_id", "type", "content", "metadata"]
                                for field in required_fields:
                                    if field not in chunk:
                                        logger.error(f"Chunk {i} 缺少字段: {field}")
                                        return False
                            logger.info("所有chunk格式验证通过！")
                    except Exception as e:
                        logger.error(f"验证JSON文件失败: {str(e)}")
                        return False
            else:
                logger.error(f"{file_type}文件生成失败: {file_path}")
                return False
        
        logger.info("=== 所有测试通过！===")
        return True
    else:
        logger.error(f"测试失败: {result.get('message', '未知错误')}")
        return False


if __name__ == "__main__":
    test_pdf_processor()
