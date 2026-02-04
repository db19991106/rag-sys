from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
import hashlib
import asyncio


class CacheManager:
    """缓存管理器"""

    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = 3600  # 默认缓存1小时
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """启动定期清理任务"""
        asyncio.create_task(self._cleanup_expired_task())

    async def _cleanup_expired_task(self):
        """
        定期清理过期的缓存
        """
        while True:
            try:
                self._cleanup_expired()
                await asyncio.sleep(3600)  # 每小时清理一次
            except Exception as e:
                from utils.logger import logger
                logger.error(f"清理缓存失败: {str(e)}")
                await asyncio.sleep(7200)  # 出错后暂停2小时

    def _cleanup_expired(self):
        """
        清理过期的缓存
        """
        current_time = datetime.now()
        expired_keys = []

        for key, item in self.cache.items():
            if item.get('expires_at') and current_time > item['expires_at']:
                expired_keys.append(key)

        for key in expired_keys:
            del self.cache[key]

    def _generate_key(self, prefix: str, data: Any) -> str:
        """
        生成缓存键

        Args:
            prefix: 前缀
            data: 数据

        Returns:
            缓存键
        """
        import json
        try:
            data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        except:
            data_str = str(data)
        hash_str = hashlib.md5(data_str.encode()).hexdigest()
        return f"{prefix}:{hash_str}"

    def set(self, 
            key: str, 
            value: Any, 
            ttl: Optional[int] = None):
        """
        设置缓存

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
        """
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        self.cache[key] = {
            "value": value,
            "created_at": datetime.now(),
            "expires_at": expires_at,
            "ttl": ttl
        }

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存

        Args:
            key: 缓存键

        Returns:
            缓存值，如果不存在或已过期返回 None
        """
        item = self.cache.get(key)
        if not item:
            return None

        # 检查是否过期
        if item.get('expires_at') and datetime.now() > item['expires_at']:
            del self.cache[key]
            return None

        return item['value']

    def delete(self, key: str):
        """
        删除缓存

        Args:
            key: 缓存键
        """
        if key in self.cache:
            del self.cache[key]

    def clear(self, prefix: Optional[str] = None):
        """
        清空缓存

        Args:
            prefix: 前缀，只清空指定前缀的缓存
        """
        if prefix:
            keys_to_delete = [key for key in self.cache if key.startswith(prefix)]
            for key in keys_to_delete:
                del self.cache[key]
        else:
            self.cache.clear()

    def get_size(self) -> int:
        """
        获取缓存大小

        Returns:
            缓存项数量
        """
        return len(self.cache)

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            统计信息
        """
        current_time = datetime.now()
        active_items = 0
        expired_items = 0

        for item in self.cache.values():
            if item.get('expires_at') and current_time > item['expires_at']:
                expired_items += 1
            else:
                active_items += 1

        return {
            "total_items": len(self.cache),
            "active_items": active_items,
            "expired_items": expired_items,
            "default_ttl": self.default_ttl
        }


# 全局缓存管理器实例
cache_manager = CacheManager()


class AsyncCacheManager:
    """异步缓存管理器"""

    def __init__(self):
        self.cache = cache_manager

    async def set(self, 
                  key: str, 
                  value: Any, 
                  ttl: Optional[int] = None):
        """
        异步设置缓存

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
        """
        self.cache.set(key, value, ttl)

    async def get(self, key: str) -> Optional[Any]:
        """
        异步获取缓存

        Args:
            key: 缓存键

        Returns:
            缓存值
        """
        return self.cache.get(key)

    async def delete(self, key: str):
        """
        异步删除缓存

        Args:
            key: 缓存键
        """
        self.cache.delete(key)

    async def clear(self, prefix: Optional[str] = None):
        """
        异步清空缓存

        Args:
            prefix: 前缀
        """
        self.cache.clear(prefix)


# 全局异步缓存管理器实例
async_cache_manager = AsyncCacheManager()


# 缓存装饰器
def cached(ttl: int = 3600, key_prefix: Optional[str] = None):
    """
    缓存装饰器

    Args:
        ttl: 过期时间（秒）
        key_prefix: 键前缀

    Returns:
        装饰器函数
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # 生成缓存键
            prefix = key_prefix or func.__name__
            key = cache_manager._generate_key(prefix, {"args": args, "kwargs": kwargs})

            # 尝试从缓存获取
            cached_value = cache_manager.get(key)
            if cached_value is not None:
                return cached_value

            # 执行函数
            result = await func(*args, **kwargs)

            # 缓存结果
            cache_manager.set(key, result, ttl)
            return result

        def sync_wrapper(*args, **kwargs):
            # 生成缓存键
            prefix = key_prefix or func.__name__
            key = cache_manager._generate_key(prefix, {"args": args, "kwargs": kwargs})

            # 尝试从缓存获取
            cached_value = cache_manager.get(key)
            if cached_value is not None:
                return cached_value

            # 执行函数
            result = func(*args, **kwargs)

            # 缓存结果
            cache_manager.set(key, result, ttl)
            return result

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
