# PDF文档智能处理系统内容组织模块

import os
import re
from typing import List, Dict, Any, Tuple

from config.config import settings
from utils.utils import logger, sort_elements_by_reading_order, split_text, ensure_directory_exists
from .file_type_identifier import file_type_identifier
from .splitting_rules import splitting_rules


class ContentOrganizer:
    """内容组织器"""
    
    def __init__(self):
        """初始化内容组织器"""
        # 确保输出目录存在
        ensure_directory_exists(settings.output_format.output_dir)
    
    def organize_content(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        组织PDF内容
        
        Args:
            extraction_result: 内容提取结果
        
        Returns:
            内容组织结果
        """
        try:
            if extraction_result.get("status") != "success":
                logger.error(f"内容组织失败: 内容提取结果无效")
                return {
                    "status": "error",
                    "message": "内容提取结果无效"
                }
            
            pdf_type = extraction_result.get("pdf_type", "text")
            pages = extraction_result.get("pages", [])
            pdf_path = extraction_result.get("pdf_path", "")
            
            organized_pages = []
            all_elements = []
            
            # 收集所有元素
            for page_info in pages:
                page_num = page_info.get("page_num", 0)
                elements = page_info.get("elements", [])
                all_elements.extend(elements)
            
            # 恢复阅读顺序
            logger.info("恢复阅读顺序...")
            sorted_elements = self._restore_reading_order(all_elements)
            
            # 智能切分
            logger.info("智能切分内容...")
            chunks = self._intelligent_split(sorted_elements, pdf_path)
            
            # 标准化输出
            logger.info("标准化输出...")
            output_result = self._standardize_output(chunks)
            
            logger.info(f"内容组织完成: 生成 {len(output_result['chunks'])} 个chunk")
            
            return {
                "status": "success",
                "pdf_type": pdf_type,
                "chunks": output_result["chunks"],
                "total_chunks": len(output_result["chunks"]),
                "markdown_content": output_result.get("markdown_content", "")
            }
            
        except Exception as e:
            logger.error(f"内容组织失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _restore_reading_order(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        恢复阅读顺序
        
        Args:
            elements: 元素列表
        
        Returns:
            排序后的元素列表
        """
        # 确保每个元素都有bbox字段
        elements_with_bbox = []
        for element in elements:
            if "bbox" in element:
                elements_with_bbox.append(element)
        
        # 按照阅读顺序排序
        sorted_elements = sort_elements_by_reading_order(elements_with_bbox)
        return sorted_elements
    
    def _intelligent_split(self, elements: List[Dict[str, Any]], pdf_path: str = "") -> List[Dict[str, Any]]:
        """
        智能切分内容
        
        Args:
            elements: 排序后的元素列表
            pdf_path: PDF文件路径
        
        Returns:
            切分后的chunk列表
        """
        # 根据配置选择切分策略
        splitting_strategy = settings.content_organization.splitting_strategy
        
        if splitting_strategy == "intelligent" and settings.content_organization.intelligent_splitting_enabled:
            # 智能切分策略
            logger.info("使用智能切分策略...")
            # 识别文件类型
            file_type_info = file_type_identifier.identify_file_type(pdf_path, elements)
            file_type = file_type_info.get("type", "general")
            
            # 应用基于文件类型的切分规则
            chunks = splitting_rules.apply_splitting_strategy(file_type, elements)
        elif splitting_strategy == "size_based":
            # 基于大小的切分策略
            logger.info("使用基于大小的切分策略...")
            chunks = self._size_based_split(elements)
        elif splitting_strategy == "section_based":
            # 基于章节的切分策略
            logger.info("使用基于章节的切分策略...")
            chunks = self._section_based_split(elements)
        elif splitting_strategy == "page_based":
            # 基于页面的切分策略
            logger.info("使用基于页面的切分策略...")
            chunks = self._page_based_split(elements)
        else:
            # 默认使用智能切分
            logger.info("使用默认切分策略...")
            chunks = self._size_based_split(elements)
        
        return chunks
    
    def _size_based_split(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        基于大小的切分策略
        
        Args:
            elements: 排序后的元素列表
        
        Returns:
            切分后的chunk列表
        """
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        chunk_size = settings.content_organization.chunk_size
        
        for element in elements:
            element_content = element.get("content", "")
            element_size = len(element_content)
            
            if current_chunk_size + element_size > chunk_size:
                if current_chunk:
                    chunks.append({
                        "elements": current_chunk,
                        "size": current_chunk_size
                    })
                    current_chunk = [element]
                    current_chunk_size = element_size
            else:
                current_chunk.append(element)
                current_chunk_size += element_size
        
        if current_chunk:
            chunks.append({
                "elements": current_chunk,
                "size": current_chunk_size
            })
        
        return chunks
    
    def _section_based_split(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        基于章节的切分策略
        
        Args:
            elements: 排序后的元素列表
        
        Returns:
            切分后的chunk列表
        """
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        
        for element in elements:
            element_type = element.get("type", "text")
            element_content = element.get("content", "")
            element_size = len(element_content)
            
            # 如果是标题元素，且当前chunk不为空，则保存当前chunk并开始新chunk
            if element_type == "title" and current_chunk:
                chunks.append({
                    "elements": current_chunk,
                    "size": current_chunk_size
                })
                current_chunk = [element]
                current_chunk_size = element_size
            else:
                current_chunk.append(element)
                current_chunk_size += element_size
        
        if current_chunk:
            chunks.append({
                "elements": current_chunk,
                "size": current_chunk_size
            })
        
        return chunks
    
    def _page_based_split(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        基于页面的切分策略
        
        Args:
            elements: 排序后的元素列表
        
        Returns:
            切分后的chunk列表
        """
        chunks = []
        current_chunk = []
        current_chunk_size = 0
        current_page = None
        
        for element in elements:
            element_page = element.get("page_num", 0)
            element_content = element.get("content", "")
            element_size = len(element_content)
            
            if current_page is None:
                current_page = element_page
            elif element_page != current_page:
                # 页面变化，保存当前chunk并开始新chunk
                if current_chunk:
                    chunks.append({
                        "elements": current_chunk,
                        "size": current_chunk_size
                    })
                    current_chunk = [element]
                    current_chunk_size = element_size
                    current_page = element_page
            else:
                current_chunk.append(element)
                current_chunk_size += element_size
        
        if current_chunk:
            chunks.append({
                "elements": current_chunk,
                "size": current_chunk_size
            })
        
        return chunks
    
    def _standardize_output(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        标准化输出
        
        Args:
            chunks: 切分后的chunk列表
        
        Returns:
            标准化输出结果
        """
        standardized_chunks = []
        markdown_content = ""
        
        for i, chunk in enumerate(chunks):
            chunk_elements = chunk.get("elements", [])
            chunk_id = f"{chunk_elements[0].get('type', 'text')}_{i}"
            
            # 构建chunk内容
            chunk_content = ""
            chunk_metadata = {
                "page": chunk_elements[0].get("page_num", 0),
                "element_type": chunk_elements[0].get("type", "text")
            }
            
            # 处理不同类型的元素
            for element in chunk_elements:
                element_type = element.get("type", "text")
                element_content = element.get("content", "")
                
                if element_type == "title":
                    # 标题内容
                    chunk_content += f"## {element_content}\n\n"
                    chunk_metadata["level"] = "h2"
                elif element_type == "text":
                    # 正文内容
                    chunk_content += f"{element_content}\n\n"
                elif element_type == "image":
                    # 图片内容
                    image_path = element.get("path", "")
                    ocr_text = element.get("ocr_text", "")
                    chunk_content += f"![图片]({image_path})\n"
                    if ocr_text:
                        chunk_content += f"图片OCR文本: {ocr_text}\n\n"
                    chunk_metadata["image_path"] = image_path
                    chunk_metadata["ocr_text"] = ocr_text
                elif element_type == "table":
                    # 表格内容
                    chunk_content += f"{element_content}\n\n"
                    chunk_metadata["row_count"] = 3  # 示例值
                    chunk_metadata["is_partial"] = False
            
            # 创建标准化chunk
            standardized_chunk = {
                "chunk_id": chunk_id,
                "type": chunk_elements[0].get("type", "text"),
                "content": chunk_content.strip(),
                "metadata": chunk_metadata
            }
            
            standardized_chunks.append(standardized_chunk)
            markdown_content += chunk_content
        
        return {
            "chunks": standardized_chunks,
            "markdown_content": markdown_content.strip()
        }


# 创建内容组织器实例
content_organizer = ContentOrganizer()
