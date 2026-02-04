import logging
import os
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor


class SmartRotatingFileHandler(logging.FileHandler):
    """
    智能日志轮转处理器，支持文件大小和时间间隔两种轮转策略
    """

    def __init__(
        self,
        filename,
        max_bytes=5*1024*1024,  # 默认5MB
        backup_count=7,         # 默认保留7个备份
        when="midnight",        # 默认每天午夜轮转
        interval=1,             # 默认间隔1个时间单位
        encoding="utf-8",
        delay=False
    ):
        """
        初始化智能日志轮转处理器

        Args:
            filename: 日志文件路径
            max_bytes: 文件大小阈值，达到此大小后轮转
            backup_count: 保留的备份文件数量
            when: 时间轮转间隔单位 (S, M, H, D, midnight, W0-W6)
            interval: 时间轮转间隔数量
            encoding: 日志文件编码
            delay: 是否延迟创建日志文件
        """
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.when = when
        self.interval = interval
        self.encoding = encoding
        self.delay = delay
        
        # 计算时间轮转的秒数
        self.interval_seconds = self._calculate_interval_seconds()
        
        # 计算下一次轮转的时间
        self.rollover_at = self._calculate_next_rollover_time()
        
        # 创建线程池用于异步处理文件操作
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # 确保日志目录存在
        log_dir = os.path.dirname(filename)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        super().__init__(filename, mode='a', encoding=encoding, delay=delay)

    def _calculate_interval_seconds(self):
        """
        计算时间轮转的间隔秒数
        """
        if self.when == 'S':
            return 1
        elif self.when == 'M':
            return 60
        elif self.when == 'H':
            return 3600
        elif self.when == 'D' or self.when == 'midnight':
            return 86400
        elif self.when.startswith('W'):
            return 604800  # 一周
        else:
            return 86400  # 默认一天

    def _calculate_next_rollover_time(self):
        """
        计算下一次轮转的时间
        """
        current_time = time.time()
        if self.when == 'midnight':
            # 计算到下一个午夜的时间
            today = datetime.fromtimestamp(current_time)
            tomorrow = today.replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow = tomorrow.replace(day=tomorrow.day + 1)
            return tomorrow.timestamp()
        elif self.when.startswith('W'):
            # 计算到下一个指定星期几的时间
            today = datetime.fromtimestamp(current_time)
            weekday = int(self.when[1])
            days_ahead = weekday - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            next_weekday = today.replace(hour=0, minute=0, second=0, microsecond=0)
            next_weekday = next_weekday.replace(day=next_weekday.day + days_ahead)
            return next_weekday.timestamp()
        else:
            # 简单地在当前时间基础上添加间隔
            return current_time + self.interval_seconds * self.interval

    def shouldRollover(self, record):
        """
        检查是否应该进行日志轮转

        Args:
            record: 日志记录对象

        Returns:
            bool: 是否应该轮转
        """
        # 检查文件大小
        if self.max_bytes > 0:
            try:
                if os.path.exists(self.baseFilename):
                    file_size = os.path.getsize(self.baseFilename)
                    if file_size >= self.max_bytes:
                        return True
            except Exception:
                pass

        # 检查时间
        current_time = time.time()
        if current_time >= self.rollover_at:
            return True

        return False

    def doRollover(self):
        """
        执行日志轮转
        """
        try:
            # 异步执行轮转操作，避免阻塞主线程
            self.executor.submit(self._perform_rollover)
        except Exception as e:
            # 轮转失败时的错误处理
            print(f"Error during log rollover: {str(e)}")

    def _perform_rollover(self):
        """
        执行实际的轮转操作
        """
        max_retries = 3
        retry_delay = 1  # 1秒

        for attempt in range(max_retries):
            try:
                # 关闭当前文件
                if self.stream:
                    self.stream.close()
                    self.stream = None

                # 生成备份文件名 - 使用要求的格式
                timestamp = datetime.now().strftime('%Y%m%d.%H%M%S')
                base_name = os.path.basename(self.baseFilename)
                name_parts = base_name.split('.')
                if len(name_parts) > 1:
                    name = '.'.join(name_parts[:-1])
                    ext = name_parts[-1]
                    backup_name = f"{name}.{timestamp}"
                else:
                    backup_name = f"{base_name}.{timestamp}"

                backup_path = os.path.join(os.path.dirname(self.baseFilename), backup_name)

                # 如果备份文件已存在，添加序号
                counter = 1
                while os.path.exists(backup_path):
                    backup_name = f"{name}.{timestamp}_{counter}"
                    backup_path = os.path.join(os.path.dirname(self.baseFilename), backup_name)
                    counter += 1

                # 重命名当前文件为备份文件
                if os.path.exists(self.baseFilename):
                    os.rename(self.baseFilename, backup_path)
                    print(f"Log file rolled over to: {backup_path}")

                # 清理旧的备份文件
                self._cleanup_old_backups()

                # 重新打开日志文件
                if not self.delay:
                    self.stream = self._open()

                # 更新下一次轮转时间
                self.rollover_at = self._calculate_next_rollover_time()

                return  # 成功完成，退出重试循环

            except Exception as e:
                # 轮转失败时的错误处理
                print(f"Error performing log rollover (attempt {attempt + 1}/{max_retries}): {str(e)}")
                
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    # 最后一次尝试失败
                    print(f"Max retries reached. Log rollover failed.")
                    # 尝试重新打开日志文件
                    try:
                        if not self.delay:
                            self.stream = self._open()
                    except Exception:
                        pass

    def _cleanup_old_backups(self):
        """
        清理旧的备份文件，只保留指定数量的备份
        """
        try:
            log_dir = os.path.dirname(self.baseFilename)
            base_name = os.path.basename(self.baseFilename)
            name_parts = base_name.split('.')
            if len(name_parts) > 1:
                name_pattern = '.'.join(name_parts[:-1])
            else:
                name_pattern = base_name

            # 收集所有备份文件
            backup_files = []
            for file in os.listdir(log_dir):
                # 匹配备份文件格式: app.log.YYYYMMDD.HHMMSS 或 app.log.YYYYMMDD.HHMMSS_1
                file_path = os.path.join(log_dir, file)
                if os.path.isfile(file_path) and file != base_name:
                    # 检查文件名是否符合备份格式
                    if file.startswith(name_pattern + '.'):
                        timestamp_part = file[len(name_pattern) + 1:]
                        # 检查时间戳部分是否符合格式
                        if len(timestamp_part) >= 15:  # YYYYMMDD.HHMMSS 最少15个字符
                            backup_files.append((os.path.getmtime(file_path), file_path))

            # 按修改时间排序，保留最新的备份文件
            backup_files.sort(reverse=True)
            if len(backup_files) > self.backup_count:
                for _, file_path in backup_files[self.backup_count:]:
                    try:
                        os.remove(file_path)
                        print(f"Cleaned up old backup file: {file_path}")
                    except Exception as e:
                        print(f"Error cleaning up old backup file {file_path}: {str(e)}")

        except Exception as e:
            print(f"Error cleaning up old backups: {str(e)}")

    def emit(self, record):
        """
        发送日志记录

        Args:
            record: 日志记录对象
        """
        max_retries = 3
        retry_delay = 0.1  # 100毫秒

        for attempt in range(max_retries):
            try:
                # 检查是否需要轮转
                if self.shouldRollover(record):
                    self.doRollover()

                # 调用父类的emit方法
                super().emit(record)

                # 刷新流，确保日志实时写入
                self.flush()

                return  # 成功完成，退出重试循环

            except Exception as e:
                # 写入失败时的错误处理
                print(f"Error emitting log record (attempt {attempt + 1}/{max_retries}): {str(e)}")
                
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                    
                    # 尝试重新打开流
                    try:
                        if self.stream:
                            self.stream.close()
                            self.stream = None
                        if not self.delay:
                            self.stream = self._open()
                    except Exception as reopen_error:
                        print(f"Error reopening log stream: {str(reopen_error)}")
                else:
                    # 最后一次尝试失败
                    print(f"Max retries reached. Log record emission failed.")
                    # 尝试重新打开流作为最后的努力
                    try:
                        if self.stream:
                            self.stream.close()
                            self.stream = None
                        if not self.delay:
                            self.stream = self._open()
                    except Exception:
                        pass
                    # 即使失败，也不抛出异常，确保应用程序主流程不受影响

    def close(self):
        """
        关闭日志处理器
        """
        try:
            # 关闭流
            if self.stream:
                self.stream.close()
                self.stream = None

            # 关闭线程池
            self.executor.shutdown(wait=False)

        except Exception as e:
            print(f"Error closing log handler: {str(e)}")

        super().close()


def setup_rotating_logger(name="rag_backend", max_bytes=5*1024*1024, backup_count=7):
    """
    配置带轮转功能的日志记录器

    Args:
        name: 日志记录器名称
        max_bytes: 文件大小阈值，达到此大小后轮转
        backup_count: 保留的备份文件数量

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 清空现有的处理器，确保使用新的配置
    if logger.handlers:
        logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 智能轮转文件处理器
    log_file_path = "/root/autodl-tmp/rag/logs/app.log"
    
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file_path)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    try:
        file_handler = SmartRotatingFileHandler(
            log_file_path,
            max_bytes=max_bytes,
            backup_count=backup_count,
            when="midnight",
            interval=1,
            encoding="utf-8",
            delay=False
        )
        file_handler.setLevel(logging.DEBUG)
        print(f"Smart rotating file handler created successfully for: {log_file_path}")
    except Exception as e:
        print(f"Error creating smart rotating file handler: {str(e)}")
        # 创建一个简单的文件处理器作为备用
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8", delay=False)
        file_handler.setLevel(logging.DEBUG)

    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    print(f"Rotating logger setup complete with {len(logger.handlers)} handlers")
    return logger


# 全局日志记录器实例
rotating_logger = setup_rotating_logger()
