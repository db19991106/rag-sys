"""
查询缓存服务

支持内存缓存（LRU）和Redis缓存，用于加速重复查询的响应速度。
"""

import time
import hashlib
import json
from typing import Optional, Dict, Any, Tuple
from collections import OrderedDict
from threading import Lock
from utils.logger import logger


class QueryCache:
    """
    查询缓存服务
    
    功能：
    - 内存LRU缓存（默认）
    - 可选Redis缓存
    - TTL过期机制
    - 缓存命中率统计
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 1800,  # 默认30分钟
        enable_redis: bool = False,
        redis_url: Optional[str] = None,
    ):
        """
        初始化缓存
        
        Args:
            max_size: 最大缓存条目数
            ttl_seconds: 缓存过期时间（秒）
            enable_redis: 是否启用Redis
            redis_url: Redis连接URL
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_redis = enable_redis
        self.redis_url = redis_url
        
        # 内存缓存（使用OrderedDict实现LRU）
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = Lock()
        
        # 统计信息
        self._hits = 0
        self._misses = 0
        
        # Redis客户端（延迟初始化）
        self._redis_client = None
        
        if enable_redis:
            self._init_redis()
        
        logger.info(
            f"查询缓存初始化完成: max_size={max_size}, "
            f"ttl={ttl_seconds}s, redis={enable_redis}"
        )
    
    def _init_redis(self):
        """初始化Redis连接"""
        try:
            import redis
            self._redis_client = redis.from_url(
                self.redis_url or "redis://localhost:6379/0",
                decode_responses=True
            )
            # 测试连接
            self._redis_client.ping()
            logger.info("Redis缓存连接成功")
        except ImportError:
            logger.warning("redis库未安装，回退到内存缓存")
            self.enable_redis = False
        except Exception as e:
            logger.warning(f"Redis连接失败: {e}，回退到内存缓存")
            self.enable_redis = False
    
    def _generate_cache_key(
        self,
        query: str,
        retrieval_config: Dict[str, Any],
        conversation_id: Optional[str] = None,
    ) -> str:
        """
        生成缓存键
        
        使用查询文本和配置参数生成唯一键
        """
        # 标准化查询（去除多余空格、转小写）
        normalized_query = " ".join(query.lower().split())
        
        # 构建键的组成部分
        key_parts = {
            "query": normalized_query,
            "top_k": retrieval_config.get("top_k", 5),
            "threshold": retrieval_config.get("similarity_threshold", 0.3),
            "rerank": retrieval_config.get("enable_rerank", False),
        }
        
        # 如果有对话ID，加入键中（保持上下文一致性）
        if conversation_id:
            key_parts["conv_id"] = conversation_id[:8]  # 只取前8位避免键过长
        
        # 生成哈希键
        key_str = json.dumps(key_parts, sort_keys=True)
        return f"rag:cache:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    def get(
        self,
        query: str,
        retrieval_config: Dict[str, Any],
        conversation_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        从缓存获取结果
        
        Args:
            query: 查询文本
            retrieval_config: 检索配置
            conversation_id: 对话ID
            
        Returns:
            缓存的响应数据，如果不存在则返回None
        """
        cache_key = self._generate_cache_key(query, retrieval_config, conversation_id)
        
        # 尝试从Redis获取
        if self.enable_redis and self._redis_client:
            try:
                cached = self._redis_client.get(cache_key)
                if cached:
                    self._hits += 1
                    logger.debug(f"Redis缓存命中: {cache_key[:20]}...")
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis读取失败: {e}")
        
        # 从内存缓存获取
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                
                # 检查是否过期
                if time.time() - entry["timestamp"] > self.ttl_seconds:
                    del self._cache[cache_key]
                    self._misses += 1
                    logger.debug(f"缓存过期: {cache_key[:20]}...")
                    return None
                
                # LRU: 移到末尾（最近使用）
                self._cache.move_to_end(cache_key)
                self._hits += 1
                logger.debug(f"内存缓存命中: {cache_key[:20]}...")
                return entry["data"]
        
        self._misses += 1
        return None
    
    def set(
        self,
        query: str,
        retrieval_config: Dict[str, Any],
        response_data: Dict[str, Any],
        conversation_id: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        将结果存入缓存
        
        Args:
            query: 查询文本
            retrieval_config: 检索配置
            response_data: 响应数据
            conversation_id: 对话ID
            ttl: 自定义过期时间（秒）
            
        Returns:
            是否成功存入
        """
        cache_key = self._generate_cache_key(query, retrieval_config, conversation_id)
        ttl = ttl or self.ttl_seconds
        entry = {
            "data": response_data,
            "timestamp": time.time(),
            "query": query[:100],  # 保存查询前100字符用于调试
        }
        
        # 存入Redis
        if self.enable_redis and self._redis_client:
            try:
                self._redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(entry, ensure_ascii=False)
                )
                logger.debug(f"存入Redis缓存: {cache_key[:20]}...")
            except Exception as e:
                logger.warning(f"Redis写入失败: {e}")
        
        # 存入内存缓存
        with self._lock:
            # LRU: 如果已满，删除最旧的条目
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            self._cache[cache_key] = entry
            logger.debug(f"存入内存缓存: {cache_key[:20]}...")
        
        return True
    
    def invalidate(
        self,
        query: str,
        retrieval_config: Dict[str, Any],
        conversation_id: Optional[str] = None,
    ) -> bool:
        """
        使指定缓存失效
        
        Args:
            query: 查询文本
            retrieval_config: 检索配置
            conversation_id: 对话ID
            
        Returns:
            是否成功删除
        """
        cache_key = self._generate_cache_key(query, retrieval_config, conversation_id)
        
        # 从Redis删除
        if self.enable_redis and self._redis_client:
            try:
                self._redis_client.delete(cache_key)
            except Exception as e:
                logger.warning(f"Redis删除失败: {e}")
        
        # 从内存删除
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                return True
        
        return False
    
    def clear(self) -> int:
        """
        清空所有缓存
        
        Returns:
            清除的条目数
        """
        count = 0
        
        # 清空Redis
        if self.enable_redis and self._redis_client:
            try:
                keys = self._redis_client.keys("rag:cache:*")
                if keys:
                    count += self._redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis清空失败: {e}")
        
        # 清空内存
        with self._lock:
            count += len(self._cache)
            self._cache.clear()
        
        logger.info(f"已清空 {count} 条缓存")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.2%}",
            "cache_size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "redis_enabled": self.enable_redis,
        }
    
    def get_hot_queries(self, top_n: int = 10) -> list:
        """
        获取热门查询（基于缓存访问频率）
        
        注意：这是一个简化实现，仅返回当前缓存中的查询
        
        Args:
            top_n: 返回条目数
            
        Returns:
            热门查询列表
        """
        with self._lock:
            # 从最近使用的条目中提取查询
            queries = []
            for key, entry in reversed(list(self._cache.items())):
                if "query" in entry:
                    queries.append(entry["query"])
                if len(queries) >= top_n:
                    break
            return queries


# 全局缓存实例（使用默认配置）
query_cache = QueryCache(
    max_size=1000,
    ttl_seconds=1800,  # 30分钟
    enable_redis=False,
)
