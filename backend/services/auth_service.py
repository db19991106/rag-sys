from typing import Optional, Dict, List
from datetime import datetime, timedelta
import jwt
import hashlib
import secrets
import re
from config import settings
from utils.logger import logger


class PasswordValidator:
    """密码复杂度验证器"""
    
    @staticmethod
    def validate(password: str) -> tuple[bool, str]:
        """
        验证密码复杂度
        
        要求：
        - 最少8个字符
        - 至少1个数字
        - 至少1个字母
        - 可选：至少1个特殊字符（推荐）
        
        Returns:
            (是否有效, 错误消息)
        """
        if len(password) < 8:
            return False, "密码长度至少8个字符"
        
        if not re.search(r'[A-Za-z]', password):
            return False, "密码必须包含至少1个字母"
        
        if not re.search(r'\d', password):
            return False, "密码必须包含至少1个数字"
        
        # 推荐包含特殊字符，但不强制
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            logger.warning("密码不包含特殊字符，建议添加以提高安全性")
        
        return True, ""


class AuthService:
    """认证服务"""

    def __init__(self):
        self.secret_key = settings.jwt_secret_key
        self.algorithm = settings.jwt_algorithm
        self.access_token_expire_minutes = settings.jwt_access_token_expire_minutes
        self.refresh_token_expire_days = 7  # 刷新令牌有效期7天
        
        # 模拟用户数据库（使用更强的密码哈希）
        self.users = {
            "admin": {
                "id": "1",
                "username": "admin",
                "password_hash": self._hash_password("Admin@123"),  # 更强的密码
                "email": "admin@example.com",
                "role": "admin",
                "created_at": datetime.now().isoformat(),
                "failed_attempts": 0,
                "locked_until": None
            },
            "editor": {
                "id": "2",
                "username": "editor",
                "password_hash": self._hash_password("Editor@123"),
                "email": "editor@example.com",
                "role": "editor",
                "created_at": datetime.now().isoformat(),
                "failed_attempts": 0,
                "locked_until": None
            },
            "viewer": {
                "id": "3",
                "username": "viewer",
                "password_hash": self._hash_password("Viewer@123"),
                "email": "viewer@example.com",
                "role": "viewer",
                "created_at": datetime.now().isoformat(),
                "failed_attempts": 0,
                "locked_until": None
            }
        }
        
        # 刷新令牌存储（实际应用中应该使用Redis或数据库）
        self.refresh_tokens = {}
        
        # 角色权限映射
        self.role_permissions = {
            "admin": [
                "manage_users",
                "manage_documents",
                "manage_settings",
                "view_audit_logs",
                "manage_vector_db",
                "manage_embedding_models",
                "run_evaluations",
                "export_data"
            ],
            "editor": [
                "manage_documents",
                "manage_vector_db",
                "run_evaluations"
            ],
            "viewer": [
                "view_documents",
                "run_retrieval",
                "run_generation"
            ],
            "user": [
                "run_retrieval",
                "run_generation"
            ]
        }
        
        # 最大登录失败次数
        self.max_failed_attempts = 5
        # 账户锁定时间（分钟）
        self.lockout_duration = 30

    def _hash_password(self, password: str) -> str:
        """
        使用更强的密码哈希算法（PBKDF2 + SHA256）
        """
        salt = secrets.token_hex(16)  # 随机salt
        # 使用PBKDF2进行密码哈希
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 迭代次数
        )
        return f"{salt}${key.hex()}"

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """
        验证密码
        """
        try:
            salt, key_hex = password_hash.split('$')
            key = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            return key.hex() == key_hex
        except Exception:
            return False

    def _check_account_locked(self, username: str) -> bool:
        """检查账户是否被锁定"""
        if username not in self.users:
            return False
        
        user = self.users[username]
        if user.get("locked_until"):
            locked_until = datetime.fromisoformat(user["locked_until"])
            if datetime.now() < locked_until:
                return True
            else:
                # 锁定期已过，重置
                user["failed_attempts"] = 0
                user["locked_until"] = None
        
        return False

    def _record_failed_attempt(self, username: str):
        """记录登录失败"""
        if username not in self.users:
            return
        
        user = self.users[username]
        user["failed_attempts"] = user.get("failed_attempts", 0) + 1
        
        # 如果超过最大失败次数，锁定账户
        if user["failed_attempts"] >= self.max_failed_attempts:
            lock_until = datetime.now() + timedelta(minutes=self.lockout_duration)
            user["locked_until"] = lock_until.isoformat()
            logger.warning(f"账户 {username} 因多次登录失败被锁定到 {lock_until}")

    def _reset_failed_attempts(self, username: str):
        """重置登录失败次数"""
        if username in self.users:
            self.users[username]["failed_attempts"] = 0
            self.users[username]["locked_until"] = None

    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """
        验证用户（带账户锁定机制）

        Args:
            username: 用户名
            password: 密码

        Returns:
            用户信息，如果验证失败返回 None
        """
        # 检查账户是否存在
        if username not in self.users:
            self._record_failed_attempt(username)
            logger.warning(f"登录失败: 用户不存在 - {username}")
            return None
        
        # 检查账户是否被锁定
        if self._check_account_locked(username):
            logger.warning(f"登录失败: 账户已锁定 - {username}")
            return None
        
        user = self.users[username]
        
        # 验证密码
        if not self._verify_password(password, user["password_hash"]):
            self._record_failed_attempt(username)
            logger.warning(f"登录失败: 密码错误 - {username}")
            return None
        
        # 登录成功，重置失败次数
        self._reset_failed_attempts(username)
        
        # 返回用户信息（不包含密码哈希）
        return {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "role": user["role"]
        }

    def create_access_token(self, data: Dict) -> str:
        """
        创建访问令牌

        Args:
            data: 令牌数据

        Returns:
            JWT 令牌
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def create_refresh_token(self, data: Dict) -> str:
        """
        创建刷新令牌

        Args:
            data: 令牌数据

        Returns:
            JWT 刷新令牌
        """
        to_encode = data.copy()
        # 生成唯一的token ID
        token_id = secrets.token_urlsafe(32)
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh", "jti": token_id})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        # 存储刷新令牌
        self.refresh_tokens[token_id] = {
            "user_id": data.get("sub"),
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expire.isoformat()
        }
        
        return encoded_jwt

    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """
        使用刷新令牌获取新的访问令牌

        Args:
            refresh_token: 刷新令牌

        Returns:
            新的访问令牌，如果验证失败返回 None
        """
        try:
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])
            
            # 验证令牌类型
            if payload.get("type") != "refresh":
                return None
            
            # 验证令牌是否在存储中
            token_id = payload.get("jti")
            if token_id not in self.refresh_tokens:
                return None
            
            # 创建新的访问令牌
            access_data = {
                "sub": payload.get("sub"),
                "username": payload.get("username"),
                "role": payload.get("role")
            }
            return self.create_access_token(access_data)
        except jwt.PyJWTError:
            return None

    def revoke_refresh_token(self, refresh_token: str) -> bool:
        """
        撤销刷新令牌

        Args:
            refresh_token: 刷新令牌

        Returns:
            是否成功
        """
        try:
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])
            token_id = payload.get("jti")
            
            if token_id in self.refresh_tokens:
                del self.refresh_tokens[token_id]
                return True
            
            return False
        except jwt.PyJWTError:
            return False

    def verify_token(self, token: str) -> Optional[Dict]:
        """
        验证令牌

        Args:
            token: JWT 令牌

        Returns:
            令牌数据，如果验证失败返回 None
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("令牌已过期")
            return None
        except jwt.InvalidTokenError:
            logger.warning("无效的令牌")
            return None

    def get_user_permissions(self, role: str) -> List[str]:
        """
        获取用户权限

        Args:
            role: 用户角色

        Returns:
            权限列表
        """
        return self.role_permissions.get(role, [])

    def check_permission(self, role: str, permission: str) -> bool:
        """
        检查权限

        Args:
            role: 用户角色
            permission: 权限名称

        Returns:
            是否有权限
        """
        permissions = self.get_user_permissions(role)
        return permission in permissions


# 全局认证服务实例
auth_service = AuthService()
password_validator = PasswordValidator()
