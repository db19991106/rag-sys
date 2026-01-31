import logging
import sys
from pathlib import Path
from config import settings


def setup_logger(name: str = "rag_backend") -> logging.Logger:
    """配置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level))

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level))

    # 文件处理器
    file_handler = logging.FileHandler(settings.log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()