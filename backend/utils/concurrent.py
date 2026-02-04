from typing import Any, Callable, List, Optional, Dict
import asyncio
import concurrent.futures
import threading
from functools import wraps


class ConcurrentManager:
    """并发管理器"""

    def __init__(self):
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, (os.cpu_count() or 4) * 4)
        )
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=min(8, os.cpu_count() or 4)
        )
        self.semaphore = asyncio.Semaphore(100)  # 限制并发数为100
        self.task_queue = asyncio.Queue(maxsize=1000)  # 任务队列
        self._start_worker()

    def _start_worker(self):
        """启动工作线程"""
        asyncio.create_task(self._worker_task())

    async def _worker_task(self):
        """工作任务"""
        while True:
            try:
                task, callback = await self.task_queue.get()
                try:
                    result = await task
                    if callback:
                        await callback(result)
                except Exception as e:
                    from utils.logger import logger
                    logger.error(f"执行任务失败: {str(e)}")
                finally:
                    self.task_queue.task_done()
            except Exception as e:
                from utils.logger import logger
                logger.error(f"工作线程出错: {str(e)}")
                await asyncio.sleep(1)

    async def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """
        在线程池中执行同步函数

        Args:
            func: 同步函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            函数返回值
        """
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.thread_pool,
                lambda: func(*args, **kwargs)
            )

    async def run_in_process(self, func: Callable, *args, **kwargs) -> Any:
        """
        在进程池中执行CPU密集型任务

        Args:
            func: 同步函数
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            函数返回值
        """
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.process_pool,
                lambda: func(*args, **kwargs)
            )

    def submit_task(self, task: asyncio.Task, callback: Optional[Callable] = None):
        """
        提交任务到队列

        Args:
            task: 异步任务
            callback: 回调函数
        """
        try:
            self.task_queue.put_nowait((task, callback))
            return True
        except asyncio.QueueFull:
            from utils.logger import logger
            logger.warning("任务队列已满")
            return False

    async def gather_with_concurrency(self, limit: int, *tasks):
        """
        限制并发数的gather

        Args:
            limit: 并发限制
            *tasks: 任务列表

        Returns:
            任务结果列表
        """
        semaphore = asyncio.Semaphore(limit)

        async def sem_task(task):
            async with semaphore:
                return await task

        return await asyncio.gather(*(sem_task(task) for task in tasks))

    def shutdown(self):
        """
        关闭线程池和进程池
        """
        self.thread_pool.shutdown(wait=False)
        self.process_pool.shutdown(wait=False)


# 修复导入
import os

# 全局并发管理器实例
concurrent_manager = ConcurrentManager()


# 异步执行装饰器
def async_executor(func):
    """
    异步执行装饰器

    Args:
        func: 同步函数

    Returns:
        异步函数
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await concurrent_manager.run_in_thread(func, *args, **kwargs)
    return wrapper


# 进程执行装饰器（用于CPU密集型任务）
def process_executor(func):
    """
    进程执行装饰器

    Args:
        func: 同步函数

    Returns:
        异步函数
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await concurrent_manager.run_in_process(func, *args, **kwargs)
    return wrapper


# 限流装饰器
def rate_limit(max_calls: int, period: int):
    """
    限流装饰器

    Args:
        max_calls: 最大调用次数
        period: 时间窗口（秒）

    Returns:
        装饰器函数
    """
    calls = []
    lock = threading.RLock()

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal calls
            import time

            current_time = time.time()

            with lock:
                # 清理过期的调用记录
                calls = [t for t in calls if current_time - t < period]
                
                # 检查是否超过限制
                if len(calls) >= max_calls:
                    wait_time = period - (current_time - calls[0])
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    # 再次清理
                    current_time = time.time()
                    calls = [t for t in calls if current_time - t < period]

                # 记录本次调用
                calls.append(current_time)

            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            nonlocal calls
            import time

            current_time = time.time()

            with lock:
                # 清理过期的调用记录
                calls = [t for t in calls if current_time - t < period]
                
                # 检查是否超过限制
                if len(calls) >= max_calls:
                    wait_time = period - (current_time - calls[0])
                    if wait_time > 0:
                        time.sleep(wait_time)
                    # 再次清理
                    current_time = time.time()
                    calls = [t for t in calls if current_time - t < period]

                # 记录本次调用
                calls.append(current_time)

            return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# 重试装饰器
def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    重试装饰器

    Args:
        max_attempts: 最大尝试次数
        delay: 初始延迟（秒）
        backoff: 退避因子

    Returns:
        装饰器函数
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay

            while attempts < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise
                    from utils.logger import logger
                    logger.warning(f"尝试 {attempts}/{max_attempts} 失败: {str(e)}")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay

            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise
                    from utils.logger import logger
                    logger.warning(f"尝试 {attempts}/{max_attempts} 失败: {str(e)}")
                    import time
                    time.sleep(current_delay)
                    current_delay *= backoff

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class AsyncTask:
    """异步任务类"""

    def __init__(self):
        self.tasks: Dict[str, asyncio.Task] = {}

    async def create_task(self, coro, task_id: Optional[str] = None) -> str:
        """
        创建异步任务

        Args:
            coro: 协程
            task_id: 任务ID

        Returns:
            任务ID
        """
        import uuid
        task_id = task_id or str(uuid.uuid4())
        task = asyncio.create_task(coro)
        self.tasks[task_id] = task
        
        # 任务完成后清理
        def cleanup(task):
            if task_id in self.tasks:
                del self.tasks[task_id]

        task.add_done_callback(cleanup)
        return task_id

    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务

        Args:
            task_id: 任务ID

        Returns:
            是否取消成功
        """
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.cancel()
            return True
        return False

    def get_task_status(self, task_id: str) -> Optional[str]:
        """
        获取任务状态

        Args:
            task_id: 任务ID

        Returns:
            任务状态
        """
        if task_id in self.tasks:
            task = self.tasks[task_id]
            return task._state
        return None


# 全局异步任务管理器实例
async_task_manager = AsyncTask()
