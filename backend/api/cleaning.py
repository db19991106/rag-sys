from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models import ApiResponse
from services.document_cleaner import document_cleaner
from utils.logger import logger


router = APIRouter(prefix="/cleaning", tags=["数据清洗"])


class CleaningConfig(BaseModel):
    """数据清洗配置"""
    remove_whitespace: bool = True
    remove_special_chars: bool = False
    normalize_quotes: bool = True
    remove_html_tags: bool = False
    remove_urls: bool = False
    remove_emails: bool = False
    remove_numbers: bool = False
    remove_chinese_punctuation: bool = False
    remove_english_punctuation: bool = False
    normalize_whitespace: bool = True
    remove_duplicate_lines: bool = False
    trim_lines: bool = True
    custom_regex: Optional[list] = None
    remove_empty_lines: bool = True
    convert_full_to_half: bool = True


class CleaningRequest(BaseModel):
    """数据清洗请求"""
    content: str
    config: Optional[CleaningConfig] = None
    preset: Optional[str] = None  # 预设配置: minimal, aggressive, text_only, html_clean


class CleaningResponse(BaseModel):
    """数据清洗响应"""
    original_length: int
    cleaned_length: int
    content: str
    config_used: Dict[str, Any]


@router.post("/clean", response_model=CleaningResponse)
async def clean_document(request: CleaningRequest):
    """
    清洗文档内容

    Args:
        request: 清洗请求，包含内容和配置

    Returns:
        清洗后的内容
    """
    try:
        # 获取配置
        if request.preset:
            # 使用预设配置
            preset_configs = document_cleaner.get_preset_configs()
            if request.preset not in preset_configs:
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的预设配置: {request.preset}，可选: {list(preset_configs.keys())}"
                )
            config = preset_configs[request.preset]
        elif request.config:
            # 使用自定义配置
            config = request.config.dict()
        else:
            # 使用默认配置
            config = document_cleaner.get_default_config()

        # 执行清洗
        cleaned_content = document_cleaner.clean(request.content, config)

        return CleaningResponse(
            original_length=len(request.content),
            cleaned_length=len(cleaned_content),
            content=cleaned_content,
            config_used=config
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档清洗失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文档清洗失败: {str(e)}")


@router.get("/presets")
async def get_presets():
    """
    获取所有预设配置

    Returns:
        预设配置列表
    """
    try:
        presets = document_cleaner.get_preset_configs()
        return {
            "presets": list(presets.keys()),
            "configs": presets
        }
    except Exception as e:
        logger.error(f"获取预设配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取预设配置失败: {str(e)}")


@router.get("/default-config")
async def get_default_config():
    """
    获取默认配置

    Returns:
        默认清洗配置
    """
    try:
        config = document_cleaner.get_default_config()
        return config
    except Exception as e:
        logger.error(f"获取默认配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取默认配置失败: {str(e)}")


@router.post("/preview")
async def preview_cleaning(request: CleaningRequest):
    """
    预览清洗效果（只返回前后对比，不保存）

    Args:
        request: 清洗请求

    Returns:
        清洗前后对比
    """
    try:
        # 获取配置
        if request.preset:
            preset_configs = document_cleaner.get_preset_configs()
            if request.preset not in preset_configs:
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的预设配置: {request.preset}"
                )
            config = preset_configs[request.preset]
        elif request.config:
            config = request.config.dict()
        else:
            config = document_cleaner.get_default_config()

        # 执行清洗
        cleaned_content = document_cleaner.clean(request.content, config)

        return {
            "original": request.content,
            "cleaned": cleaned_content,
            "original_length": len(request.content),
            "cleaned_length": len(cleaned_content),
            "reduction_rate": f"{(1 - len(cleaned_content) / len(request.content)) * 100:.2f}%",
            "config_used": config
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预览清洗失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预览清洗失败: {str(e)}")