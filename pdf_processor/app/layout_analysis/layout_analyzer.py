# PDF文档智能处理系统版面分析模块

import os
import re
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np

from config.config import settings
from utils.utils import logger, get_bbox_area


class LayoutAnalyzer:
    """版面分析器"""
    
    def __init__(self):
        """初始化版面分析器"""
        pass
    
    def analyze_layout(self, preprocessing_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析PDF版面
        
        Args:
            preprocessing_result: 预处理结果
        
        Returns:
            版面分析结果
        """
        try:
            if preprocessing_result.get("status") != "success":
                logger.error(f"版面分析失败: 预处理结果无效")
                return {
                    "status": "error",
                    "message": "预处理结果无效"
                }
            
            pdf_type = preprocessing_result.get("pdf_type", "text")
            pages = preprocessing_result.get("pages", [])
            
            analyzed_pages = []
            
            for page_info in pages:
                page_num = page_info.get("page_num", 0)
                logger.info(f"分析第 {page_num + 1} 页版面")
                
                # 分析页面版面
                page_analysis = self._analyze_page_layout(page_info, pdf_type)
                analyzed_pages.append(page_analysis)
            
            logger.info(f"版面分析完成: 共分析 {len(analyzed_pages)} 页")
            
            return {
                "status": "success",
                "pdf_type": pdf_type,
                "pages": analyzed_pages,
                "total_pages": len(analyzed_pages)
            }
            
        except Exception as e:
            logger.error(f"版面分析失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _analyze_page_layout(self, page_info: Dict[str, Any], pdf_type: str) -> Dict[str, Any]:
        """
        分析页面版面
        
        Args:
            page_info: 页面信息
            pdf_type: PDF类型
        
        Returns:
            页面版面分析结果
        """
        page_num = page_info.get("page_num", 0)
        elements = []
        
        # 处理文本块
        if "text_blocks" in page_info:
            text_blocks = page_info.get("text_blocks", [])
            for i, block in enumerate(text_blocks):
                bbox = block.get("bbox", [0, 0, 0, 0])
                text = block.get("text", "")
                
                # 过滤小面积文本块
                if get_bbox_area(bbox) < settings.layout_analysis.min_element_area:
                    continue
                
                # 过滤空文本
                if not text.strip():
                    continue
                
                # 文本块分类
                element_type = self._classify_text_block(text, bbox)
                
                elements.append({
                    "id": f"text_{page_num}_{i}",
                    "type": element_type,
                    "bbox": bbox,
                    "content": text,
                    "page_num": page_num
                })
        
        # 处理图片
        if "images" in page_info:
            images = page_info.get("images", [])
            for i, img_info in enumerate(images):
                image_path = img_info.get("path", "")
                if image_path and os.path.exists(image_path):
                    # 获取图片大小
                    try:
                        with Image.open(image_path) as img:
                            width, height = img.size
                            
                            # 估算图片在PDF中的位置（这里使用简单的估算，实际应该从PDF中提取准确位置）
                            bbox = [100, 100, 100 + width, 100 + height]
                            
                            elements.append({
                                "id": f"image_{page_num}_{i}",
                                "type": "image",
                                "bbox": bbox,
                                "path": image_path,
                                "page_num": page_num
                            })
                    except Exception as e:
                        logger.error(f"处理图片失败: {str(e)}")
        
        # 处理扫描型PDF的图像
        if pdf_type == "scanned" and "image_path" in page_info:
            image_path = page_info.get("image_path", "")
            if image_path and os.path.exists(image_path):
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        bbox = [0, 0, width, height]
                        
                        elements.append({
                            "id": f"scanned_image_{page_num}",
                            "type": "image",
                            "bbox": bbox,
                            "path": image_path,
                            "page_num": page_num
                        })
                except Exception as e:
                    logger.error(f"处理扫描型PDF图像失败: {str(e)}")
        
        # 检测表格
        if pdf_type == "text":
            # 这里使用简单的表格检测方法，实际应该使用更复杂的算法
            table_elements = self._detect_tables(page_info)
            elements.extend(table_elements)
        
        # 过滤小元素
        elements = [elem for elem in elements if get_bbox_area(elem.get("bbox", [0, 0, 0, 0])) >= settings.layout_analysis.min_element_area]
        
        return {
            "page_num": page_num,
            "elements": elements,
            "element_count": len(elements)
        }
    
    def _classify_text_block(self, text: str, bbox: List[float]) -> str:
        """
        分类文本块
        
        Args:
            text: 文本内容
            bbox: 边界框
        
        Returns:
            文本块类型: "title" 或 "text"
        """
        # 简单的文本块分类逻辑
        # 1. 基于文本长度和字体大小（通过bbox高度估算）
        bbox_height = bbox[3] - bbox[1]
        text_length = len(text.strip())
        
        # 标题通常较短且字体较大
        if bbox_height > 20 and text_length < 100:
            # 检查是否包含标题特征词
            title_patterns = [r"^第.*章", r"^.*节", r"^.*条", r"^.*款", r"^.*项", r"^.*目"]
            for pattern in title_patterns:
                if re.search(pattern, text.strip()):
                    return "title"
            
            # 检查是否全部为大写或加粗
            if text.isupper() or any(char.isupper() for char in text) and text_length < 50:
                return "title"
        
        return "text"
    
    def _detect_tables(self, page_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        检测表格
        
        Args:
            page_info: 页面信息
        
        Returns:
            表格元素列表
        """
        tables = []
        page_num = page_info.get("page_num", 0)
        
        # 这里使用简单的表格检测方法，实际应该使用更复杂的算法
        # 例如，检测文本块的排列是否形成表格结构
        text_blocks = page_info.get("text_blocks", [])
        
        if len(text_blocks) > 10:
            # 简单估算：如果页面中有很多文本块，可能包含表格
            # 实际应用中应该使用更精确的表格检测算法
            pass
        
        return tables


# 创建版面分析器实例
layout_analyzer = LayoutAnalyzer()
