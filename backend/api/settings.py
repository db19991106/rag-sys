from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
import json
import sys
import os

# 添加父目录到路径以导入config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import logger

# 动态导入settings以避免命名冲突
config_module = __import__('config')
settings_config = config_module.settings

router = APIRouter(prefix="/settings", tags=["系统设置"])


class SystemSettings(BaseModel):
    """系统设置模型"""
    # 嵌入模型配置
    embedding_model_type: str = "bge"
    embedding_model_name: str = "BAAI/bge-base-zh-v1.5"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 32
    
    # 重排序配置
    enable_rerank: bool = False
    reranker_type: str = "cross_encoder"
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_top_k: int = 10
    reranker_threshold: float = 0.0
    
    # 向量数据库配置
    vector_db_type: str = "faiss"
    vector_db_dimension: int = 768
    vector_db_index_type: str = "HNSW"
    vector_db_host: Optional[str] = None
    vector_db_port: Optional[int] = None
    vector_db_collection_name: Optional[str] = None
    
    # 智能切分配置
    intelligent_splitting_enabled: bool = True
    splitting_strategy: str = "intelligent"
    intelligent_splitting_sensitivity: float = 0.6
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    special_rules_enabled: bool = True


# 设置文件路径
SETTINGS_FILE = Path(settings_config.vector_db_dir) / "system_settings.json"


def _load_settings() -> Dict[str, Any]:
    """从文件加载设置"""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载设置文件失败: {str(e)}")
    return {}


def _save_settings(settings_dict: Dict[str, Any]):
    """保存设置到文件"""
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings_dict, f, ensure_ascii=False, indent=2)
        logger.info("系统设置已保存")
    except Exception as e:
        logger.error(f"保存设置文件失败: {str(e)}")
        raise


@router.get("/", response_model=SystemSettings)
async def get_settings():
    """
    获取系统设置
    """
    try:
        saved_settings = _load_settings()
        # 合并默认设置和保存的设置
        default_settings = SystemSettings().dict()
        merged_settings = {**default_settings, **saved_settings}
        return SystemSettings(**merged_settings)
    except Exception as e:
        logger.error(f"获取系统设置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取系统设置失败: {str(e)}")


@router.post("/", response_model=SystemSettings)
async def update_settings(settings: SystemSettings):
    """
    更新系统设置
    """
    try:
        settings_dict = settings.dict()
        _save_settings(settings_dict)
        logger.info(f"系统设置已更新: {settings_dict}")
        return settings
    except Exception as e:
        logger.error(f"更新系统设置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新系统设置失败: {str(e)}")


@router.post("/reset")
async def reset_settings():
    """
    重置系统设置为默认值
    """
    try:
        default_settings = SystemSettings()
        settings_dict = default_settings.dict()
        _save_settings(settings_dict)
        logger.info("系统设置已重置为默认值")
        return {"success": True, "message": "系统设置已重置为默认值"}
    except Exception as e:
        logger.error(f"重置系统设置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重置系统设置失败: {str(e)}")
