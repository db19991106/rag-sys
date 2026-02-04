"""
敏感信息加密/解密工具
用于保护配置文件中的敏感信息（API密钥、密码等）
"""
import base64
import hashlib
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Optional


class SecureConfig:
    """安全配置管理器 - 用于加密/解密敏感配置"""
    
    def __init__(self, master_key: Optional[str] = None):
        """
        初始化安全配置管理器
        
        Args:
            master_key: 主密钥，如果不提供则从环境变量SECRET_KEY获取
        """
        # 获取主密钥
        self.master_key = master_key or os.environ.get(
            "SECRET_KEY", 
            "default-secret-key-please-change-in-production"
        )
        
        # 从主密钥派生加密密钥
        self._cipher = self._create_cipher()
    
    def _create_cipher(self) -> Fernet:
        """从主密钥创建加密器"""
        # 使用PBKDF2从主密钥派生加密密钥
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'rag_system_salt',  # 在生产环境中应该使用随机salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        return Fernet(key)
    
    def encrypt(self, plaintext: str) -> str:
        """
        加密明文
        
        Args:
            plaintext: 要加密的明文
            
        Returns:
            加密后的字符串（Base64编码）
        """
        if not plaintext:
            return ""
        
        encrypted = self._cipher.encrypt(plaintext.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, encrypted: str) -> str:
        """
        解密密文
        
        Args:
            encrypted: 加密后的字符串（Base64编码）
            
        Returns:
            解密后的明文
        """
        if not encrypted:
            return ""
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted.encode())
            decrypted = self._cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception:
            # 如果解密失败，返回原字符串（向后兼容）
            return encrypted
    
    def encrypt_api_key(self, api_key: str) -> str:
        """
        加密API密钥
        
        Args:
            api_key: API密钥
            
        Returns:
            加密后的API密钥
        """
        # 对API密钥进行哈希，然后加密
        hashed = hashlib.sha256(api_key.encode()).hexdigest()
        return self.encrypt(hashed)
    
    def mask_sensitive(self, value: str, visible_chars: int = 4) -> str:
        """
        掩码敏感信息（用于日志输出）
        
        Args:
            value: 敏感值
            visible_chars: 保留可见的字符数（开头和结尾）
            
        Returns:
            掩码后的值
        """
        if not value or len(value) <= visible_chars * 2:
            return "*" * len(value)
        
        return value[:visible_chars] + "*" * (len(value) - visible_chars * 2) + value[-visible_chars:]


# 全局实例
secure_config = SecureConfig()


def encrypt_sensitive_config(config_dict: dict, sensitive_keys: list) -> dict:
    """
    加密配置字典中的敏感字段
    
    Args:
        config_dict: 配置字典
        sensitive_keys: 敏感字段列表
        
    Returns:
        加密后的配置字典
    """
    encrypted = config_dict.copy()
    for key in sensitive_keys:
        if key in encrypted and encrypted[key]:
            encrypted[key] = secure_config.encrypt(str(encrypted[key]))
    return encrypted


def decrypt_sensitive_config(config_dict: dict, sensitive_keys: list) -> dict:
    """
    解密配置字典中的敏感字段
    
    Args:
        config_dict: 配置字典
        sensitive_keys: 敏感字段列表
        
    Returns:
        解密后的配置字典
    """
    decrypted = config_dict.copy()
    for key in sensitive_keys:
        if key in decrypted and decrypted[key]:
            decrypted[key] = secure_config.decrypt(str(decrypted[key]))
    return decrypted