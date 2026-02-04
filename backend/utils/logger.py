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
            self.logger.log(self.log_level, self.linebuf.rstrip())
            self.linebuf = ''

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
        try:
            for line in s.rstrip().splitlines():
                self.logger.log(self.log_level, line.rstrip())
        except Exception as e:
            # 写入失败时的错误处理
            print(f"Error writing to log stream: {str(e)}")

    def writelines(self, lines):
        """
        写入多行
        """
        for line in lines:
            self.write(line)


def setup_logger(name: str = "rag_backend") -> logging.Logger:
    """配置日志记录器"""
    # 打印调试信息
    print(f"DEBUG: log_file = {settings.log_file}")
    print(f"DEBUG: log_level = {settings.log_level}")
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level))

    # 清空现有的处理器，确保使用新的配置
    if logger.handlers:
        print(f"DEBUG: Clearing existing handlers: {len(logger.handlers)}")
        logger.handlers.clear()

    # 添加敏感信息过滤器
    from .sensitive_filter import SensitiveDataFilter
    sensitive_filter = SensitiveDataFilter()
    logger.addFilter(sensitive_filter)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level))

    # 文件处理器 - 使用智能轮转处理器
    log_file_path = "/root/autodl-tmp/rag/logs/app.log"
    print(f"DEBUG: Using hardcoded log file path: {log_file_path}")
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
        print(f"DEBUG: Smart rotating file handler created successfully")
    except Exception as e:
        print(f"DEBUG: Error creating smart rotating file handler: {str(e)}")
        # 创建一个简单的文件处理器作为备用
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8', delay=False)
        file_handler.setLevel(logging.DEBUG)
        print(f"DEBUG: Fallback to simple file handler")

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

    print(f"DEBUG: Adding console handler")
    logger.addHandler(console_handler)
    print(f"DEBUG: Adding file handler")
    logger.addHandler(file_handler)
    print(f"DEBUG: Total handlers: {len(logger.handlers)}")

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
    验证日志配置有效性并输出诊断信息

    Args:
        logger: 日志记录器实例

    Returns:
        bool: 配置是否有效
    """
    print("\n=== 日志系统自检 ===")
    is_valid = True

    # 检查处理器配置
    print(f"处理器数量: {len(logger.handlers)}")
    for i, handler in enumerate(logger.handlers):
        print(f"处理器 {i}: {type(handler).__name__}")
        if hasattr(handler, 'baseFilename'):
            log_file = getattr(handler, 'baseFilename')
            print(f"  日志文件: {log_file}")
            
            # 检查文件路径是否存在
            log_dir = os.path.dirname(log_file)
            if log_dir:
                if os.path.exists(log_dir):
                    print(f"  目录存在: {log_dir}")
                else:
                    print(f"  警告: 目录不存在，将自动创建: {log_dir}")
            
            # 检查文件权限
            try:
                if os.path.exists(log_file):
                    if os.access(log_file, os.W_OK):
                        print(f"  文件可写: {log_file}")
                    else:
                        print(f"  错误: 文件不可写: {log_file}")
                        is_valid = False
                else:
                    # 检查目录权限
                    if os.access(log_dir, os.W_OK):
                        print(f"  目录可写: {log_dir}")
                    else:
                        print(f"  错误: 目录不可写: {log_dir}")
                        is_valid = False
            except Exception as e:
                print(f"  权限检查失败: {str(e)}")
                is_valid = False

    # 检查日志级别
    print(f"日志级别: {logging.getLevelName(logger.level)}")

    # 测试日志输出
    print("测试日志输出...")
    test_message = "日志系统自检测试消息"
    try:
        logger.info(test_message)
        print("  测试日志输出成功")
    except Exception as e:
        print(f"  错误: 测试日志输出失败: {str(e)}")
        is_valid = False

    # 检查标准输出重定向
    print("检查标准输出重定向...")
    try:
        print("  标准输出重定向测试")
        print("  标准错误重定向测试", file=sys.stderr)
        print("  标准输出重定向成功")
    except Exception as e:
        print(f"  错误: 标准输出重定向失败: {str(e)}")
        is_valid = False

    print(f"\n=== 自检结果: {'有效' if is_valid else '无效'} ===\n")
    return is_valid


logger = setup_logger()

# 重定向标准输出和标准错误到日志记录器
original_stdout, original_stderr = redirect_stdout_stderr(logger)

# 验证日志配置
validate_logger_config(logger)