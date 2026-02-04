# PDF类型检测与预处理模块

import os
import re
import tempfile
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import numpy as np

from config.config import settings
from utils.utils import ensure_directory_exists, logger


class PDFPreprocessor:
    """PDF预处理器"""
    
    def __init__(self):
        """初始化PDF预处理器"""
        # 初始化OCR引擎
        if settings.pdf_preprocessing.ocr_enabled:
            # 检查PaddleOCR版本，使用兼容的参数
            try:
                # 尝试使用最新版本的参数
                self.ocr = PaddleOCR(
                    use_angle_cls=settings.ocr.paddleocr_use_angle_cls,
                    lang=settings.ocr.paddleocr_lang
                )
            except Exception as e:
                logger.warning(f"PaddleOCR初始化失败: {str(e)}")
                logger.warning("尝试使用兼容模式初始化PaddleOCR")
                # 兼容模式
                self.ocr = PaddleOCR(
                    use_angle_cls=settings.ocr.paddleocr_use_angle_cls,
                    lang=settings.ocr.paddleocr_lang
                )
        else:
            self.ocr = None
        
        # 确保输出目录存在
        ensure_directory_exists(settings.content_extraction.image_output_dir)
    
    def detect_pdf_type(self, pdf_path: str) -> str:
        """
        检测PDF类型
        
        Args:
            pdf_path: PDF文件路径
        
        Returns:
            PDF类型: "text" (文本型PDF) 或 "scanned" (扫描型PDF)
        """
        try:
            # 打开PDF文件
            doc = fitz.open(pdf_path)
            
            # 检测是否为文本型PDF
            is_text_pdf = False
            text_count = 0
            
            # 遍历前几页进行检测
            for page_num in range(min(5, len(doc))):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    text_count += 1
                
                # 如果找到足够的文本，判断为文本型PDF
                if text_count >= 2:
                    is_text_pdf = True
                    break
            
            doc.close()
            
            if is_text_pdf:
                logger.info(f"PDF类型检测结果: 文本型PDF")
                return "text"
            else:
                logger.info(f"PDF类型检测结果: 扫描型PDF")
                return "scanned"
                
        except Exception as e:
            logger.error(f"PDF类型检测失败: {str(e)}")
            # 默认视为扫描型PDF
            return "scanned"
    
    def preprocess_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        预处理PDF文件
        
        Args:
            pdf_path: PDF文件路径
        
        Returns:
            预处理结果
        """
        try:
            # 检测PDF类型
            pdf_type = self.detect_pdf_type(pdf_path)
            
            if pdf_type == "text":
                # 文本型PDF预处理
                return self._preprocess_text_pdf(pdf_path)
            else:
                # 扫描型PDF预处理
                return self._preprocess_scanned_pdf(pdf_path)
                
        except Exception as e:
            logger.error(f"PDF预处理失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "pdf_type": "unknown"
            }
    
    def _preprocess_text_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        预处理文本型PDF
        
        Args:
            pdf_path: PDF文件路径
        
        Returns:
            预处理结果
        """
        try:
            doc = fitz.open(pdf_path)
            pages = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # 提取文本内容
                text = page.get_text()
                
                # 提取文本块信息
                text_blocks = []
                for block in page.get_text("blocks"):
                    bbox = block[:4]
                    text = block[4]
                    if text.strip():
                        text_blocks.append({
                            "bbox": bbox,
                            "text": text,
                            "type": "text"
                        })
                
                # 提取图片信息
                images = []
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_data = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # 保存图片
                    image_path = os.path.join(
                        settings.content_extraction.image_output_dir,
                        f"page_{page_num}_img_{img_index}.{image_ext}"
                    )
                    
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                    
                    images.append({
                        "index": img_index,
                        "path": image_path,
                        "ext": image_ext
                    })
                
                pages.append({
                    "page_num": page_num,
                    "text": text,
                    "text_blocks": text_blocks,
                    "images": images,
                    "type": "text"
                })
            
            doc.close()
            
            logger.info(f"文本型PDF预处理完成: {len(pages)}页")
            
            return {
                "status": "success",
                "pdf_type": "text",
                "pages": pages,
                "total_pages": len(pages)
            }
            
        except Exception as e:
            logger.error(f"文本型PDF预处理失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "pdf_type": "text"
            }
    
    def _preprocess_scanned_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        预处理扫描型PDF
        
        Args:
            pdf_path: PDF文件路径
        
        Returns:
            预处理结果
        """
        try:
            # 将PDF转换为图像
            images = convert_from_path(
                pdf_path,
                dpi=settings.pdf_preprocessing.pdf2image_dpi,
                fmt="png"
            )
            
            pages = []
            
            for page_num, image in enumerate(images):
                # 对图像进行OCR识别
                if self.ocr:
                    result = self.ocr.ocr(np.array(image), cls=True)
                    
                    # 提取文本内容
                    text = ""
                    text_blocks = []
                    
                    if result:
                        for line in result:
                            for word_info in line:
                                bbox = word_info[0]
                                text_content = word_info[1][0]
                                confidence = word_info[1][1]
                                
                                # 转换bbox格式
                                bbox = [
                                    bbox[0][0],
                                    bbox[0][1],
                                    bbox[2][0],
                                    bbox[2][1]
                                ]
                                
                                text += text_content + " "
                                text_blocks.append({
                                    "bbox": bbox,
                                    "text": text_content,
                                    "confidence": confidence,
                                    "type": "text"
                                })
                    
                    # 保存图像
                    image_path = os.path.join(
                        settings.content_extraction.image_output_dir,
                        f"page_{page_num}.png"
                    )
                    image.save(image_path)
                    
                    pages.append({
                        "page_num": page_num,
                        "text": text.strip(),
                        "text_blocks": text_blocks,
                        "image_path": image_path,
                        "type": "scanned"
                    })
                else:
                    # 没有OCR引擎，只保存图像
                    image_path = os.path.join(
                        settings.content_extraction.image_output_dir,
                        f"page_{page_num}.png"
                    )
                    image.save(image_path)
                    
                    pages.append({
                        "page_num": page_num,
                        "text": "",
                        "text_blocks": [],
                        "image_path": image_path,
                        "type": "scanned"
                    })
            
            logger.info(f"扫描型PDF预处理完成: {len(pages)}页")
            
            return {
                "status": "success",
                "pdf_type": "scanned",
                "pages": pages,
                "total_pages": len(pages)
            }
            
        except Exception as e:
            logger.error(f"扫描型PDF预处理失败: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "pdf_type": "scanned"
            }


# 创建预处理器实例
pdf_preprocessor = PDFPreprocessor()
