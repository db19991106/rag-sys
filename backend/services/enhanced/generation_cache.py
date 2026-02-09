"""
生成结果缓存
实现功能：生成结果缓存、Prompt哈希去重
"""

import json
import hashlib
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from utils.logger import logger


@dataclass
class GenerationCacheEntry:
    """生成缓存条目"""

    answer: str
    citations: list
    tokens_used: int
    created_at: float
    access_count: int = 0


class GenerationCache:
    """生成结果缓存管理器"""

    def __init__(self, redis_client=None, default_ttl: int = 7200):
        """
        Args:
            redis_client: Redis客户端
            default_ttl: 默认缓存时间(秒)，默认2小时
        """
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.memory_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _generate_cache_key(self, prompt: str, model_config: Dict = None) -> str:
        """
        生成缓存键

        基于Prompt内容生成哈希作为缓存键
        """
        # 规范化Prompt
        normalized = prompt.strip()

        # 组合Prompt和模型配置
        cache_data = {"prompt": normalized, "model": model_config or {}}

        key_str = json.dumps(cache_data, sort_keys=True)
        hash_val = hashlib.sha256(key_str.encode()).hexdigest()[:20]
        return f"generation:{hash_val}"

    async def get(
        self, prompt: str, model_config: Dict = None
    ) -> Optional[GenerationCacheEntry]:
        """
        获取缓存的生成结果

        Returns:
            缓存的生成结果，如果不存在返回None
        """
        key = self._generate_cache_key(prompt, model_config)

        try:
            if self.redis:
                data = self.redis.get(key)
                if data:
                    self._cache_hits += 1
                    entry_data = json.loads(data)
                    entry = GenerationCacheEntry(**entry_data)
                    entry.access_count += 1

                    # 更新访问计数
                    self.redis.setex(key, self.default_ttl, json.dumps(entry.__dict__))

                    logger.debug(
                        f"Generation cache hit [access_count={entry.access_count}]"
                    )
                    return entry
            else:
                # 使用内存缓存
                if key in self.memory_cache:
                    self._cache_hits += 1
                    cached = self.memory_cache[key]

                    # 检查是否过期
                    if cached["expire_time"] > time.time():
                        entry = cached["entry"]
                        entry.access_count += 1
                        return entry
                    else:
                        del self.memory_cache[key]
        except Exception as e:
            logger.warning(f"Generation cache get error: {e}")

        self._cache_misses += 1
        return None

    async def set(
        self,
        prompt: str,
        answer: str,
        citations: list = None,
        tokens_used: int = 0,
        model_config: Dict = None,
        ttl: int = None,
    ):
        """
        设置生成结果缓存

        Args:
            prompt: 生成Prompt
            answer: 生成的答案
            citations: 引用列表
            tokens_used: 使用的Token数
            model_config: 模型配置
            ttl: 过期时间(秒)
        """
        key = self._generate_cache_key(prompt, model_config)
        ttl = ttl or self.default_ttl

        try:
            entry = GenerationCacheEntry(
                answer=answer,
                citations=citations or [],
                tokens_used=tokens_used,
                created_at=time.time(),
                access_count=1,
            )

            data = json.dumps(entry.__dict__)

            if self.redis:
                self.redis.setex(key, ttl, data)
            else:
                # 使用内存缓存
                self.memory_cache[key] = {
                    "entry": entry,
                    "expire_time": time.time() + ttl,
                }

                # 限制缓存大小
                if len(self.memory_cache) > 500:
                    # 移除最早的条目
                    oldest_key = next(iter(self.memory_cache))
                    del self.memory_cache[oldest_key]

            logger.debug(f"Cached generation result [key={key[:20]}...]")

        except Exception as e:
            logger.warning(f"Generation cache set error: {e}")

    def get_stats(self) -> Dict:
        """获取缓存统计"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0

        # 计算节省的Token
        total_tokens_saved = 0
        for cached in self.memory_cache.values():
            entry = cached["entry"]
            # 假设每次命中节省一次完整的生成调用
            total_tokens_saved += entry.tokens_used * (entry.access_count - 1)

        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "memory_size": len(self.memory_cache),
            "estimated_tokens_saved": total_tokens_saved,
        }

    async def invalidate(self, pattern: str = None):
        """
        使缓存失效

        Args:
            pattern: 匹配模式，如果为None则清空所有
        """
        try:
            if self.redis and pattern:
                # 使用Redis的SCAN和DEL
                for key in self.redis.scan_iter(match=pattern):
                    self.redis.delete(key)
            elif self.redis:
                # 清空所有生成缓存
                for key in self.redis.scan_iter(match="generation:*"):
                    self.redis.delete(key)
            else:
                # 清空内存缓存
                if pattern:
                    keys_to_delete = [
                        k for k in self.memory_cache.keys() if pattern in k
                    ]
                    for k in keys_to_delete:
                        del self.memory_cache[k]
                else:
                    self.memory_cache.clear()

            logger.info(f"Invalidated generation cache [pattern={pattern}]")

        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")


# 单例
generation_cache = GenerationCache()

__all__ = ["GenerationCache", "GenerationCacheEntry", "generation_cache"]
