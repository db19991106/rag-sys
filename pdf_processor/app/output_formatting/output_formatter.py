# PDF文档智能处理系统输出格式处理模块

import os
import json
import markdown
from typing import List, Dict, Any
from datetime import datetime

from config.config import settings
from utils.utils import logger, ensure_directory_exists, dict_to_json


class OutputFormatter:
    """输出格式处理器"""
    
    def __init__(self):
        """初始化输出格式处理器"""
        # 确保输出目录存在
        ensure_directory_exists(settings.output_format.output_dir)
    
    def format_output(self, organization_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化输出
        
        Args:
            organization_result: 内容组织结果
        
        Returns:
            格式化输出结果
        """
        try:
            if organization_result.get("status") != "success":
                logger.error(f"输出格式化失败: 内容组织结果无效")
                return {
                    "status": "error",
                    "message": "内容组织结果无效"
                }
            
            chunks = organization_result.get("chunks", [])
            pdf_type = organization_result.get("pdf_type", "text")
            
            # 生成JSON格式的输出
            logger.info("生成JSON格式输出...")
            json_output = self._generate_json_output(chunks)
            
            # 生成Markdown格式的输出
            logger.info("生成Markdown格式输出...")
            markdown_output = self._generate_markdown_output(chunks)
            
            # 保存输出文件
            output_files = self._save_output_files(json_output, markdown_output)
            
            logger.info(f"输出格式化完成: 生成 {len(chunks)} 个chunk")
            
            return {
                "status": "success",
                "pdf_type": pdf_type,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "output_files": output_files
            }
            
        except Exception as e:
            logger.error(f"输出格式化失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _generate_json_output(self, chunks: List[Dict[str, Any]]) -> str:
        """
        生成JSON格式输出
        
        Args:
            chunks: chunk列表
        
        Returns:
            JSON字符串
        """
        # 确保每个chunk都符合指定格式
        formatted_chunks = []
        
        for chunk in chunks:
            formatted_chunk = {
                "chunk_id": chunk.get("chunk_id", ""),
                "type": chunk.get("type", "text"),
                "content": chunk.get("content", ""),
                "metadata": chunk.get("metadata", {})
            }
            formatted_chunks.append(formatted_chunk)
        
        # 生成JSON字符串
        return json.dumps(formatted_chunks, ensure_ascii=False, indent=2)
    
    def _generate_markdown_output(self, chunks: List[Dict[str, Any]]) -> str:
        """
        生成Markdown格式输出
        
        Args:
            chunks: chunk列表
        
        Returns:
            Markdown字符串
        """
        markdown_content = "# PDF文档内容\n\n"
        
        for chunk in chunks:
            chunk_type = chunk.get("type", "text")
            chunk_content = chunk.get("content", "")
            
            if chunk_type == "title":
                markdown_content += f"## {chunk_content}\n\n"
            elif chunk_type == "text":
                markdown_content += f"{chunk_content}\n\n"
            elif chunk_type == "image":
                markdown_content += f"{chunk_content}\n\n"
            elif chunk_type == "table":
                markdown_content += f"{chunk_content}\n\n"
        
        return markdown_content
    
    def _save_output_files(self, json_output: str, markdown_output: str) -> Dict[str, str]:
        """
        保存输出文件
        
        Args:
            json_output: JSON输出
            markdown_output: Markdown输出
        
        Returns:
            输出文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_files = {}
        
        # 保存JSON文件
        json_file_path = os.path.join(
            settings.output_format.output_dir,
            f"pdf_output_{timestamp}.json"
        )
        with open(json_file_path, "w", encoding="utf-8") as f:
            f.write(json_output)
        output_files["json"] = json_file_path
        
        # 保存Markdown文件
        markdown_file_path = os.path.join(
            settings.output_format.output_dir,
            f"pdf_output_{timestamp}.md"
        )
        with open(markdown_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_output)
        output_files["markdown"] = markdown_file_path
        
        logger.info(f"输出文件保存成功: {json_file_path}")
        logger.info(f"输出文件保存成功: {markdown_file_path}")
        
        return output_files


# 创建输出格式处理器实例
output_formatter = OutputFormatter()
