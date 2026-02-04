# PDF文档智能处理系统配置文件

from pydantic_settings import BaseSettings
from typing import List, Dict, Any
from pydantic import BaseModel

# 导入logger
from utils.utils import logger


class PDFPreprocessingConfig(BaseModel):
    """PDF预处理配置"""
    # 扫描型PDF处理配置
    ocr_enabled: bool = True
    ocr_language: str = "ch"
    pdf2image_dpi: int = 300


class LayoutAnalysisConfig(BaseModel):
    """版面分析配置"""
    # 元素检测配置
    element_detection_threshold: float = 0.5
    min_element_area: int = 100


class ContentExtractionConfig(BaseModel):
    """内容提取配置"""
    # 文本提取配置
    text_extraction_enabled: bool = True
    font_property_extraction: bool = True
    
    # 表格提取配置
    table_extraction_enabled: bool = True
    table_output_format: str = "markdown"  # markdown or html
    
    # 图片提取配置
    image_extraction_enabled: bool = True
    image_output_dir: str = "./extracted_images"
    image_ocr_enabled: bool = True


class ContentOrganizationConfig(BaseModel):
    """内容组织配置"""
    # 阅读顺序恢复配置
    reading_order_strategy: str = "top_to_bottom_left_to_right"
    
    # 内容切分配置
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # 智能切分配置
    intelligent_splitting_enabled: bool = True
    splitting_strategy: str = "intelligent"  # intelligent, size_based, section_based, page_based
    
    # 智能切分参数
    intelligent_splitting_sensitivity: float = 0.6
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    special_rules_enabled: bool = True
    special_rules: Dict[str, Any] = {}



class OutputFormatConfig(BaseModel):
    """输出格式配置"""
    # 输出格式配置
    output_formats: List[str] = ["json", "markdown"]
    output_dir: str = "./output"


class OCRConfig(BaseModel):
    """OCR配置"""
    # PaddleOCR配置
    paddleocr_use_angle_cls: bool = True
    paddleocr_use_gpu: bool = False
    paddleocr_lang: str = "ch"


class Settings(BaseSettings):
    """系统配置类"""
    # 基础配置
    project_name: str = "PDF文档智能处理系统"
    version: str = "1.0.0"
    debug: bool = True
    
    # PDF处理配置
    pdf_preprocessing: PDFPreprocessingConfig = PDFPreprocessingConfig()
    
    # 版面分析配置
    layout_analysis: LayoutAnalysisConfig = LayoutAnalysisConfig()
    
    # 内容提取配置
    content_extraction: ContentExtractionConfig = ContentExtractionConfig()
    
    # 内容组织配置
    content_organization: ContentOrganizationConfig = ContentOrganizationConfig()
    
    # 输出格式配置
    output_format: OutputFormatConfig = OutputFormatConfig()
    
    # OCR配置
    ocr: OCRConfig = OCRConfig()
    
    # 系统设置文件路径
    system_settings_file: str = ""        
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# 创建全局配置实例
settings = Settings()


def load_system_settings(system_settings_file: str = "") -> None:
    """
    从系统设置文件加载配置
    
    Args:
        system_settings_file: 系统设置文件路径
    """
    import json
    from pathlib import Path
    
    try:
        if not system_settings_file:
            # 默认路径
            system_settings_file = Path("../backend/data/vector_db") / "system_settings.json"
        
        settings_path = Path(system_settings_file)
        if settings_path.exists():
            with open(settings_path, 'r', encoding='utf-8') as f:
                system_settings = json.load(f)
            
            # 更新智能切分配置
            if "intelligent_splitting_enabled" in system_settings:
                settings.content_organization.intelligent_splitting_enabled = system_settings["intelligent_splitting_enabled"]
            if "splitting_strategy" in system_settings:
                settings.content_organization.splitting_strategy = system_settings["splitting_strategy"]
            if "intelligent_splitting_sensitivity" in system_settings:
                settings.content_organization.intelligent_splitting_sensitivity = system_settings["intelligent_splitting_sensitivity"]
            if "min_chunk_size" in system_settings:
                settings.content_organization.min_chunk_size = system_settings["min_chunk_size"]
            if "max_chunk_size" in system_settings:
                settings.content_organization.max_chunk_size = system_settings["max_chunk_size"]
            if "special_rules_enabled" in system_settings:
                settings.content_organization.special_rules_enabled = system_settings["special_rules_enabled"]
            
            logger.info(f"已从系统设置文件加载配置: {system_settings_file}")
        else:
            logger.info(f"系统设置文件不存在: {system_settings_file}")
    except Exception as e:
        logger.error(f"加载系统设置失败: {str(e)}")


# 尝试加载系统设置
load_system_settings(settings.system_settings_file)
