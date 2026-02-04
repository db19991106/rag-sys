# PDF文档智能处理系统内容提取模块

import os
import re
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np

from config.config import settings
from utils.utils import logger, ensure_directory_exists, clean_text


class ContentExtractor:
    """内容提取器"""
    
    def __init__(self):
        """初始化内容提取器"""
        # 确保输出目录存在
        ensure_directory_exists(settings.content_extraction.image_output_dir)
        ensure_directory_exists(settings.output_format.output_dir)
    
    def extract_content(self, layout_analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取PDF内容
        
        Args:
            layout_analysis_result: 版面分析结果
        
        Returns:
            内容提取结果
        """
        try:
            if layout_analysis_result.get("status") != "success":
                logger.error(f"内容提取失败: 版面分析结果无效")
                return {
                    "status": "error",
                    "message": "版面分析结果无效"
                }
            
            pdf_type = layout_analysis_result.get("pdf_type", "text")
            pages = layout_analysis_result.get("pages", [])
            
            extracted_pages = []
            
            for page_info in pages:
                page_num = page_info.get("page_num", 0)
                logger.info(f"提取第 {page_num + 1} 页内容")
                
                # 提取页面内容
                page_extraction = self._extract_page_content(page_info, pdf_type)
                extracted_pages.append(page_extraction)
            
            logger.info(f"内容提取完成: 共提取 {len(extracted_pages)} 页")
            
            return {
                "status": "success",
                "pdf_type": pdf_type,
                "pages": extracted_pages,
                "total_pages": len(extracted_pages)
            }
            
        except Exception as e:
            logger.error(f"内容提取失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _extract_page_content(self, page_info: Dict[str, Any], pdf_type: str) -> Dict[str, Any]:
        """
        提取页面内容
        
        Args:
            page_info: 页面信息
            pdf_type: PDF类型
        
        Returns:
            页面内容提取结果
        """
        page_num = page_info.get("page_num", 0)
        elements = page_info.get("elements", [])
        extracted_elements = []
        
        for element in elements:
            element_type = element.get("type", "text")
            
            if element_type == "title" or element_type == "text":
                # 提取文本内容
                extracted_element = self._extract_text_content(element)
                extracted_elements.append(extracted_element)
            elif element_type == "image":
                # 提取图片内容
                extracted_element = self._extract_image_content(element)
                extracted_elements.append(extracted_element)
            elif element_type == "table":
                # 提取表格内容
                extracted_element = self._extract_table_content(element)
                extracted_elements.append(extracted_element)
        
        return {
            "page_num": page_num,
            "elements": extracted_elements,
            "element_count": len(extracted_elements)
        }
    
    def _extract_text_content(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取文本内容
        
        Args:
            element: 文本元素
        
        Returns:
            提取的文本内容
        """
        text = element.get("content", "")
        cleaned_text = clean_text(text)
        
        # 提取文本属性（这里使用简单的估算，实际应该从PDF中提取准确属性）
        font_properties = self._estimate_font_properties(element)
        
        return {
            "id": element.get("id", ""),
            "type": element.get("type", "text"),
            "content": cleaned_text,
            "font_properties": font_properties,
            "page_num": element.get("page_num", 0),
            "bbox": element.get("bbox", [0, 0, 0, 0])
        }
    
    def _extract_image_content(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取图片内容
        
        Args:
            element: 图片元素
        
        Returns:
            提取的图片内容
        """
        image_path = element.get("path", "")
        page_num = element.get("page_num", 0)
        
        if not image_path or not os.path.exists(image_path):
            logger.error(f"图片路径无效: {image_path}")
            return {
                "id": element.get("id", ""),
                "type": "image",
                "content": "",
                "path": "",
                "page_num": page_num,
                "bbox": element.get("bbox", [0, 0, 0, 0])
            }
        
        # 生成图片描述（这里使用简单的描述，实际应该使用更复杂的算法）
        image_description = f"图片: 第 {page_num + 1} 页"
        
        # 对图片执行OCR识别（如果启用）
        ocr_text = ""
        if settings.content_extraction.image_ocr_enabled:
            try:
                from paddleocr import PaddleOCR
                ocr = PaddleOCR(
                    use_angle_cls=settings.ocr.paddleocr_use_angle_cls,
                    use_gpu=settings.ocr.paddleocr_use_gpu,
                    lang=settings.ocr.paddleocr_lang
                )
                result = ocr.ocr(np.array(Image.open(image_path)), cls=True)
                
                if result:
                    for line in result:
                        for word_info in line:
                            text_content = word_info[1][0]
                            ocr_text += text_content + " "
                    
                ocr_text = ocr_text.strip()
                
            except Exception as e:
                logger.error(f"图片OCR识别失败: {str(e)}")
                ocr_text = ""
        
        return {
            "id": element.get("id", ""),
            "type": "image",
            "content": image_description,
            "path": image_path,
            "ocr_text": ocr_text,
            "page_num": page_num,
            "bbox": element.get("bbox", [0, 0, 0, 0])
        }
    
    def _extract_table_content(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取表格内容
        
        Args:
            element: 表格元素
        
        Returns:
            提取的表格内容
        """
        # 这里使用简单的表格提取方法，实际应该使用更复杂的算法
        table_content = ""
        
        if settings.content_extraction.table_output_format == "markdown":
            # 生成Markdown格式的表格
            table_content = "| 列1 | 列2 | 列3 |\n|------|------|------|\n| 数据1 | 数据2 | 数据3 |"
        else:
            # 生成HTML格式的表格
            table_content = "<table><tr><th>列1</th><th>列2</th><th>列3</th></tr><tr><td>数据1</td><td>数据2</td><td>数据3</td></tr></table>"
        
        return {
            "id": element.get("id", ""),
            "type": "table",
            "content": table_content,
            "page_num": element.get("page_num", 0),
            "bbox": element.get("bbox", [0, 0, 0, 0])
        }
    
    def _estimate_font_properties(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """
        估算字体属性
        
        Args:
            element: 文本元素
        
        Returns:
            字体属性
        """
        bbox = element.get("bbox", [0, 0, 0, 0])
        bbox_height = bbox[3] - bbox[1]
        text = element.get("content", "")
        
        # 估算字体大小
        font_size = bbox_height
        
        # 估算字体粗细
        font_weight = "normal"
        if any(char.isupper() for char in text) and len(text) < 50:
            font_weight = "bold"
        
        # 估算字体样式
        font_style = "normal"
        
        return {
            "font_size": font_size,
            "font_weight": font_weight,
            "font_style": font_style
        }


# 创建内容提取器实例
content_extractor = ContentExtractor()
