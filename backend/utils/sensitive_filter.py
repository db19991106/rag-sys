"""
日志敏感信息过滤器
自动过滤日志中的敏感信息
"""
import logging
import re
from typing import Any, Dict
from utils.security import secure_config


class SensitiveDataFilter(logging.Filter):
    """敏感数据过滤器 - 自动过滤日志中的敏感信息"""
    
    # 敏感字段模式
    SENSITIVE_PATTERNS = [
        (r'password["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)', 'password'),
        (r'secret["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)', 'secret'),
        (r'token["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)', 'token'),
        (r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)', 'api_key'),
        (r'jwt["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)', 'jwt'),
        (r'authorization["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)', 'authorization'),
        (r'Bearer\s+([^\s]+)', 'bearer_token'),
    ]
    
    def __init__(self):
        super().__init__()
        # 编译正则表达式
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), name)
            for pattern, name in self.SENSITIVE_PATTERNS
        ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """过滤日志记录"""
        # 过滤消息
        record.msg = self._sanitize(str(record.msg))
        
        # 过滤args
        if record.args:
            record.args = tuple(
                self._sanitize(str(arg)) if isinstance(arg, str) else arg
                for arg in record.args
            )
        
        return True
    
    def _sanitize(self, text: str) -> str:
        """清理文本中的敏感信息"""
        sanitized = text
        
        # 应用所有模式
        for pattern, name in self.compiled_patterns:
            # 替换敏感值为掩码
            sanitized = pattern.sub(
                lambda m: m.group(0).replace(m.group(1), secure_config.mask_sensitive(m.group(1))),
                sanitized
            )
        
        return sanitized


def sanitize_dict(data: Dict[str, Any], sensitive_keys: set = None) -> Dict[str, Any]:
    """
    清理字典中的敏感信息
    
    Args:
        data: 要清理的字典
        sensitive_keys: 敏感字段集合
        
    Returns:
        清理后的字典
    """
    if sensitive_keys is None:
        sensitive_keys = {
            'password', 'secret', 'token', 'api_key', 'jwt',
            'authorization', 'access_token', 'refresh_token',
            'private_key', 'credential'
        }
    
    sanitized = {}
    
    for key, value in data.items():
        key_lower = key.lower()
        
        # 检查是否为敏感字段
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            # 脱敏处理
            if isinstance(value, str):
                sanitized[key] = secure_config.mask_sensitive(value)
            elif isinstance(value, dict):
                sanitized[key] = sanitize_dict(value, sensitive_keys)
            else:
                sanitized[key] = "******"
        elif isinstance(value, dict):
            # 递归处理嵌套字典
            sanitized[key] = sanitize_dict(value, sensitive_keys)
        elif isinstance(value, list):
            # 处理列表
            sanitized[key] = [
                sanitize_dict(item, sensitive_keys) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value
    
    return sanitized


def sanitize_log_extra(extra: Dict[str, Any]) -> Dict[str, Any]:
    """
    清理日志extra参数中的敏感信息
    
    Args:
        extra: 日志extra参数
        
    Returns:
        清理后的extra参数
    """
    return sanitize_dict(extra)