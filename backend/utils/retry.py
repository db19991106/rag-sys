"""
重试机制工具
为失败的操作提供自动重试功能，使用指数退避策略
"""
import time
import functools
from typing import Callable, Type, Tuple, Optional, Any
from utils.logger import logger


class RetryException(Exception):
    """重试失败异常"""
    pass


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        base_delay: 基础延迟时间（秒）
        backoff_factor: 退避因子（每次重试延迟乘以这个因子）
        max_delay: 最大延迟时间（秒）
        exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # 如果是最后一次尝试，不再重试
                    if attempt == max_attempts:
                        logger.error(
                            f"函数 {func.__name__} 重试 {max_attempts} 次后仍然失败: {str(e)}"
                        )
                        raise RetryException(
                            f"重试 {max_attempts} 次后失败: {str(e)}"
                        ) from e
                    
                    # 计算延迟时间（指数退避）
                    delay = min(
                        base_delay * (backoff_factor ** (attempt - 1)),
                        max_delay
                    )
                    
                    logger.warning(
                        f"函数 {func.__name__} 第 {attempt} 次尝试失败: {str(e)}"
                        f"，将在 {delay:.2f} 秒后重试"
                    )
                    
                    # 调用回调函数
                    if on_retry:
                        on_retry(attempt, e)
                    
                    # 等待后重试
                    time.sleep(delay)
            
            # 理论上不会执行到这里
            raise last_exception
        
        return wrapper
    return decorator


class RetryWithFallback:
    """带降级的重试器"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        fallback_value: Any = None,
        fallback_exception: bool = False
    ):
        """
        初始化重试器
        
        Args:
            max_attempts: 最大重试次数
            base_delay: 基础延迟时间
            fallback_value: 降级返回值
            fallback_exception: 是否在降级时抛出异常
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.fallback_value = fallback_value
        self.fallback_exception = fallback_exception
    
    def execute(
        self,
        func: Callable,
        *args,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        **kwargs
    ) -> Any:
        """
        执行函数，失败时降级
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            exceptions: 需要重试的异常类型
            **kwargs: 函数关键字参数
            
        Returns:
            函数返回值或降级值
        """
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                
                if attempt == self.max_attempts:
                    logger.warning(
                        f"函数 {func.__name__} 重试 {self.max_attempts} 次后失败，"
                        f"使用降级方案"
                    )
                    break
                
                delay = self.base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"函数 {func.__name__} 第 {attempt} 次尝试失败: {str(e)}"
                    f"，将在 {delay:.2f} 秒后重试"
                )
                time.sleep(delay)
        
        # 降级处理
        if self.fallback_exception:
            raise RetryException(
                f"重试失败，降级处理抛出异常: {str(last_exception)}"
            ) from last_exception
        else:
            logger.info(f"函数 {func.__name__} 降级返回默认值")
            return self.fallback_value


# 默认重试配置
DEFAULT_RETRY_CONFIG = {
    "max_attempts": 3,
    "base_delay": 1.0,
    "backoff_factor": 2.0,
    "max_delay": 60.0
}

# LLM调用重试配置
LLM_RETRY_CONFIG = {
    "max_attempts": 2,  # LLM调用重试次数不宜过多
    "base_delay": 2.0,
    "backoff_factor": 2.0,
    "max_delay": 10.0
}

# 文档解析重试配置
DOCUMENT_PARSER_RETRY_CONFIG = {
    "max_attempts": 3,
    "base_delay": 0.5,
    "backoff_factor": 2.0,
    "max_delay": 5.0
}

# API调用重试配置
API_RETRY_CONFIG = {
    "max_attempts": 3,
    "base_delay": 1.0,
    "backoff_factor": 2.0,
    "max_delay": 30.0
}