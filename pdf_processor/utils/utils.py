# PDF文档智能处理系统工具函数

import os
import re
import uuid
import json
import markdown
from typing import List, Dict, Any, Tuple
from datetime import datetime


# ========== 文件操作相关 ==========
def ensure_directory_exists(directory: str) -> None:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def generate_unique_id(prefix: str = "") -> str:
    """
    生成唯一标识符
    
    Args:
        prefix: 前缀
    
    Returns:
        唯一标识符
    """
    unique_id = str(uuid.uuid4()).replace("-", "")[:16]
    if prefix:
        return f"{prefix}_{unique_id}"
    return unique_id


# ========== 文本处理相关 ==========
def clean_text(text: str) -> str:
    """
    清理文本
    
    Args:
        text: 原始文本
    
    Returns:
        清理后的文本
    """
    # 移除多余的空白字符
    text = re.sub(r"\s+|\n+", " ", text)
    # 移除首尾空白字符
    text = text.strip()
    return text


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    切分文本
    
    Args:
        text: 原始文本
        chunk_size: 切分大小
        chunk_overlap: 重叠大小
    
    Returns:
        切分后的文本列表
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        if end > text_length:
            end = text_length
        chunks.append(text[start:end])
        start = end - chunk_overlap
    
    return chunks


# ========== 格式转换相关 ==========
def markdown_to_html(markdown_text: str) -> str:
    """
    将Markdown文本转换为HTML
    
    Args:
        markdown_text: Markdown文本
    
    Returns:
        HTML文本
    """
    return markdown.markdown(markdown_text)


def dict_to_json(data: Dict[str, Any]) -> str:
    """
    将字典转换为JSON字符串
    
    Args:
        data: 字典数据
    
    Returns:
        JSON字符串
    """
    return json.dumps(data, ensure_ascii=False, indent=2)


# ========== 坐标处理相关 ==========
def get_bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """
    获取边界框的中心点坐标
    
    Args:
        bbox: 边界框 (x1, y1, x2, y2)
    
    Returns:
        中心点坐标 (x, y)
    """
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def get_bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    """
    获取边界框的面积
    
    Args:
        bbox: 边界框 (x1, y1, x2, y2)
    
    Returns:
        面积
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


# ========== 排序相关 ==========
def sort_elements_by_reading_order(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按照阅读顺序排序元素
    
    Args:
        elements: 元素列表，每个元素必须包含bbox字段
    
    Returns:
        排序后的元素列表
    """
    # 按照从上到下、从左到右的顺序排序
    elements.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return elements


# ========== 日志相关 ==========
def setup_logger(name: str = "pdf_processor") -> Any:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
    
    Returns:
        日志记录器
    """
    import logging
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 文件处理器
        log_file = f"pdf_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        
        # 日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger


# 创建全局日志记录器
logger = setup_logger()
