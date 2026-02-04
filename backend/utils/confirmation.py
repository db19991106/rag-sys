"""
敏感操作确认令牌管理
为危险操作提供二次确认机制
"""
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional
from utils.logger import logger


class ConfirmationManager:
    """确认令牌管理器"""
    
    def __init__(self, token_expiry_minutes: int = 10):
        """
        初始化确认令牌管理器
        
        Args:
            token_expiry_minutes: 令牌过期时间（分钟）
        """
        self.token_expiry_minutes = token_expiry_minutes
        # 存储确认令牌（实际应用中应该使用Redis或数据库）
        self.tokens: Dict[str, Dict] = {}
    
    def generate_token(
        self,
        operation: str,
        resource_id: str,
        user_id: Optional[str] = None
    ) -> str:
        """
        生成确认令牌
        
        Args:
            operation: 操作类型（delete, batch_delete等）
            resource_id: 资源ID
            user_id: 用户ID
            
        Returns:
            确认令牌
        """
        # 生成唯一的令牌
        token = secrets.token_urlsafe(32)
        
        # 计算令牌hash（用于验证）
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        # 存储令牌信息
        self.tokens[token_hash] = {
            "operation": operation,
            "resource_id": resource_id,
            "user_id": user_id,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(minutes=self.token_expiry_minutes),
            "used": False
        }
        
        # 清理过期令牌
        self._cleanup_expired_tokens()
        
        logger.info(
            f"生成确认令牌: operation={operation}, resource_id={resource_id}, "
            f"user_id={user_id}, token_hash={token_hash[:8]}..."
        )
        
        return token
    
    def verify_token(
        self,
        token: str,
        operation: str,
        resource_id: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        验证确认令牌
        
        Args:
            token: 确认令牌
            operation: 操作类型
            resource_id: 资源ID
            user_id: 用户ID
            
        Returns:
            是否验证成功
        """
        # 计算令牌hash
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        # 检查令牌是否存在
        if token_hash not in self.tokens:
            logger.warning(f"确认令牌不存在: token_hash={token_hash[:8]}...")
            return False
        
        token_info = self.tokens[token_hash]
        
        # 检查令牌是否已使用
        if token_info["used"]:
            logger.warning(f"确认令牌已使用: token_hash={token_hash[:8]}...")
            return False
        
        # 检查令牌是否过期
        if datetime.now() > token_info["expires_at"]:
            logger.warning(f"确认令牌已过期: token_hash={token_hash[:8]}...")
            return False
        
        # 验证操作类型和资源ID
        if token_info["operation"] != operation:
            logger.warning(
                f"确认令牌操作类型不匹配: expected={operation}, "
                f"actual={token_info['operation']}"
            )
            return False
        
        if token_info["resource_id"] != resource_id:
            logger.warning(
                f"确认令牌资源ID不匹配: expected={resource_id}, "
                f"actual={token_info['resource_id']}"
            )
            return False
        
        # 如果指定了用户ID，验证用户ID
        if user_id and token_info["user_id"] != user_id:
            logger.warning(
                f"确认令牌用户ID不匹配: expected={user_id}, "
                f"actual={token_info['user_id']}"
            )
            return False
        
        # 标记令牌为已使用
        token_info["used"] = True
        
        logger.info(
            f"确认令牌验证成功: operation={operation}, resource_id={resource_id}, "
            f"user_id={user_id}, token_hash={token_hash[:8]}..."
        )
        
        return True
    
    def _cleanup_expired_tokens(self):
        """清理过期令牌"""
        now = datetime.now()
        expired_tokens = [
            token_hash
            for token_hash, token_info in self.tokens.items()
            if now > token_info["expires_at"] or token_info["used"]
        ]
        
        for token_hash in expired_tokens:
            del self.tokens[token_hash]
        
        if expired_tokens:
            logger.debug(f"清理了 {len(expired_tokens)} 个过期或已使用的令牌")
    
    def revoke_token(self, token: str) -> bool:
        """
        撤销确认令牌
        
        Args:
            token: 确认令牌
            
        Returns:
            是否成功撤销
        """
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        if token_hash in self.tokens:
            del self.tokens[token_hash]
            logger.info(f"撤销确认令牌: token_hash={token_hash[:8]}...")
            return True
        
        return False


# 全局确认令牌管理器实例
confirmation_manager = ConfirmationManager()