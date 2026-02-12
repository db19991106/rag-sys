import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from config import settings
from .rotating_logger import SmartRotatingFileHandler


class StreamToLogger:
    """
    将标准输出和标准错误重定向到日志记录器
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''
        # 保存原始的stdout/stderr，用于调用其方法
        self.original_stream = sys.stdout if log_level == logging.INFO else sys.stderr

    def flush(self):
        """
        刷新缓冲区
        """
        if self.linebuf:
            try:
                self.logger.log(self.log_level, self.linebuf.rstrip())
                self.linebuf = ''
            except Exception:
                # 避免递归错误
                pass

    def close(self):
        """
        关闭流
        """
        self.flush()

    def isatty(self):
        """
        检查是否是终端
        """
        try:
            return self.original_stream.isatty()
        except:
            return False

    def fileno(self):
        """
        获取文件描述符
        """
        try:
            return self.original_stream.fileno()
        except:
            return -1

    def readable(self):
        """
        检查是否可读
        """
        return False

    def writable(self):
        """
        检查是否可写
        """
        return True

    def seekable(self):
        """
        检查是否可查找
        """
        return False

    def detach(self):
        """
        分离流
        """
        return self

    def read(self, size=-1):
        """
        读取数据
        """
        return ''

    def readline(self, size=-1):
        """
        读取一行
        """
        return ''

    def readlines(self, hint=-1):
        """
        读取所有行
        """
        return []

    def seek(self, offset, whence=0):
        """
        查找位置
        """
        pass

    def tell(self):
        """
        获取当前位置
        """
        return 0

    def truncate(self, size=None):
        """
        截断流
        """
        pass

    def write(self, s):
        """
        写入数据
        """
        # 如果流已关闭，直接忽略
        if not s:
            return
        try:
            for line in s.rstrip().splitlines():
                if line:
                    self.logger.log(self.log_level, line.rstrip())
        except Exception:
            # 忽略所有错误，避免递归
            pass

    def writelines(self, lines):
        """
        写入多行
        """
        for line in lines:
            self.write(line)


def setup_logger(name: str = "rag_backend") -> logging.Logger:
    """配置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level))

    # 清空现有的处理器，确保使用新的配置
    if logger.handlers:
        logger.handlers.clear()

    # 添加敏感信息过滤器
    from .sensitive_filter import SensitiveDataFilter
    sensitive_filter = SensitiveDataFilter()
    logger.addFilter(sensitive_filter)

    # 控制台处理器 - 使用原始的sys.stdout避免递归
    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setLevel(getattr(logging, settings.log_level))

    # 文件处理器 - 使用智能轮转处理器
    log_file_path = "/root/autodl-tmp/rag/logs/app.log"
    try:
        # 使用智能轮转文件处理器，支持文件大小和时间间隔两种轮转策略
        file_handler = SmartRotatingFileHandler(
            log_file_path,
            max_bytes=5*1024*1024,  # 5MB
            backup_count=7,         # 保留7个备份
            when="midnight",        # 每天午夜轮转
            interval=1,
            encoding='utf-8',
            delay=False
        )
        file_handler.setLevel(logging.DEBUG)
    except Exception:
        # 创建一个简单的文件处理器作为备用
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8', delay=False)
        file_handler.setLevel(logging.DEBUG)

    # 日志格式 - 包含毫秒级时间戳
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S.%f'
    )
    
    # 自定义格式化器，只保留毫秒
    class CustomFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            if datefmt:
                # 格式化时间，只保留前三位毫秒
                return datetime.fromtimestamp(record.created).strftime(datefmt)[:-3]
            return super().formatTime(record, datefmt)
    
    # 使用自定义格式化器
    custom_formatter = CustomFormatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S.%f'
    )
    
    console_handler.setFormatter(custom_formatter)
    file_handler.setFormatter(custom_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # 配置第三方库的日志级别
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)

    return logger


def redirect_stdout_stderr(logger):
    """
    重定向标准输出和标准错误到日志记录器
    """
    # 保存原始的stdout和stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # 创建流重定向器
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    return original_stdout, original_stderr


def restore_stdout_stderr(original_stdout, original_stderr):
    """
    恢复原始的标准输出和标准错误
    """
    sys.stdout = original_stdout
    sys.stderr = original_stderr


def validate_logger_config(logger):
    """
    验证日志配置有效性

    Args:
        logger: 日志记录器实例

    Returns:
        bool: 配置是否有效
    """
    is_valid = True
    
    # 检查是否有处理器
    if not logger.handlers:
        is_valid = False
    
    # 测试日志输出
    try:
        logger.debug("日志系统自检")
    except Exception:
        is_valid = False
    
    return is_valid


logger = setup_logger()

# 重定向标准输出和标准错误到日志记录器
# 检查环境变量，允许在脚本执行时禁用重定向以避免递归
import os
if os.environ.get('RAG_DISABLE_STDOUT_REDIRECT', '').lower() != 'true':
    original_stdout, original_stderr = redirect_stdout_stderr(logger)
    # 验证日志配置
    validate_logger_config(logger)
else:
    # 不重定向，保持原始stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    # 不重定向时不输出日志，避免与脚本自身的日志配置冲突
    pass