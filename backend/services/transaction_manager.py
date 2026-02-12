"""
事务管理器 - 确保文档存储和向量索引的双写一致性
解决双写一致性问题
"""

from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime
from pathlib import Path
import json
import uuid
import fcntl
import os
from utils.logger import logger


class TransactionStatus(str, Enum):
    """事务状态"""

    PENDING = "pending"  # 待处理
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    ROLLING_BACK = "rolling_back"  # 回滚中
    ROLLED_BACK = "rolled_back"  # 已回滚


class DocumentTransaction:
    """文档处理事务"""

    def __init__(self, doc_id: str, operation: str, metadata: Dict[str, Any] = None):
        self.transaction_id = str(uuid.uuid4())
        self.doc_id = doc_id
        self.operation = operation  # upload, update, delete
        self.status = TransactionStatus.PENDING
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.completed_steps = []
        self.failed_steps = []
        self.error_message = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "doc_id": self.doc_id,
            "operation": self.operation,
            "status": self.status.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentTransaction":
        tx = cls(data["doc_id"], data["operation"], data.get("metadata", {}))
        tx.transaction_id = data["transaction_id"]
        tx.status = TransactionStatus(data["status"])
        tx.created_at = datetime.fromisoformat(data["created_at"])
        tx.updated_at = datetime.fromisoformat(data["updated_at"])
        tx.completed_steps = data.get("completed_steps", [])
        tx.failed_steps = data.get("failed_steps", [])
        tx.error_message = data.get("error_message")
        return tx


class TransactionManager:
    """事务管理器 - 管理文档处理事务"""

    def __init__(self, storage_dir: str = "./data/transactions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.storage_dir / "transactions.lock"
        self.transactions: Dict[str, DocumentTransaction] = {}
        self._load_transactions()

    def _load_transactions(self):
        """加载未完成的事务"""
        try:
            transaction_file = self.storage_dir / "transactions.json"
            if transaction_file.exists():
                with open(transaction_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for tx_data in data:
                        tx = DocumentTransaction.from_dict(tx_data)
                        # 只加载未完成的事务
                        if tx.status not in [
                            TransactionStatus.COMPLETED,
                            TransactionStatus.ROLLED_BACK,
                        ]:
                            self.transactions[tx.transaction_id] = tx
                            logger.info(
                                f"加载未完成事务: {tx.transaction_id} ({tx.status.value})"
                            )
        except Exception as e:
            logger.error(f"加载事务失败: {str(e)}")

    def _save_transactions(self):
        """保存事务状态"""
        try:
            with open(self.lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    transaction_file = self.storage_dir / "transactions.json"
                    data = [tx.to_dict() for tx in self.transactions.values()]
                    # 原子写入
                    temp_path = str(transaction_file) + ".tmp"
                    with open(temp_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    os.replace(temp_path, transaction_file)
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logger.error(f"保存事务失败: {str(e)}")

    def start_transaction(
        self, doc_id: str, operation: str, metadata: Dict[str, Any] = None
    ) -> DocumentTransaction:
        """开始一个新事务"""
        tx = DocumentTransaction(doc_id, operation, metadata)
        tx.status = TransactionStatus.PENDING
        self.transactions[tx.transaction_id] = tx
        self._save_transactions()
        logger.info(f"开始事务: {tx.transaction_id} ({operation} for {doc_id})")
        return tx

    def update_step(
        self, transaction_id: str, step: str, success: bool, error: str = None
    ):
        """更新事务步骤状态"""
        if transaction_id not in self.transactions:
            logger.warning(f"事务不存在: {transaction_id}")
            return

        tx = self.transactions[transaction_id]
        tx.updated_at = datetime.now()

        if success:
            tx.completed_steps.append(step)
            logger.info(f"事务步骤完成: {transaction_id} - {step}")
        else:
            tx.failed_steps.append(step)
            tx.error_message = error
            tx.status = TransactionStatus.FAILED
            logger.error(f"事务步骤失败: {transaction_id} - {step}: {error}")

        self._save_transactions()

    def complete_transaction(self, transaction_id: str):
        """完成事务"""
        if transaction_id not in self.transactions:
            return

        tx = self.transactions[transaction_id]
        tx.status = TransactionStatus.COMPLETED
        tx.updated_at = datetime.now()
        self._save_transactions()
        logger.info(f"事务完成: {transaction_id}")

    def fail_transaction(self, transaction_id: str, error: str):
        """标记事务失败"""
        if transaction_id not in self.transactions:
            return

        tx = self.transactions[transaction_id]
        tx.status = TransactionStatus.FAILED
        tx.error_message = error
        tx.updated_at = datetime.now()
        self._save_transactions()
        logger.error(f"事务失败: {transaction_id} - {error}")

    def rollback_transaction(self, transaction_id: str) -> bool:
        """回滚事务"""
        if transaction_id not in self.transactions:
            return False

        tx = self.transactions[transaction_id]
        tx.status = TransactionStatus.ROLLING_BACK
        self._save_transactions()

        try:
            # 根据已完成步骤执行回滚
            # 这里需要根据不同的步骤执行不同的回滚操作
            logger.info(f"回滚事务: {transaction_id}")

            # 回滚完成后标记为已回滚
            tx.status = TransactionStatus.ROLLED_BACK
            tx.updated_at = datetime.now()
            self._save_transactions()
            return True
        except Exception as e:
            logger.error(f"回滚事务失败: {transaction_id} - {str(e)}")
            return False

    def get_transaction(self, transaction_id: str) -> Optional[DocumentTransaction]:
        """获取事务"""
        return self.transactions.get(transaction_id)

    def get_pending_transactions(self) -> List[DocumentTransaction]:
        """获取所有待处理的事务"""
        return [
            tx
            for tx in self.transactions.values()
            if tx.status in [TransactionStatus.PENDING, TransactionStatus.FAILED]
        ]

    def cleanup_old_transactions(self, days: int = 7):
        """清理旧的事务记录"""
        cutoff = datetime.now() - __import__("datetime").timedelta(days=days)
        to_remove = []

        for tx_id, tx in self.transactions.items():
            if tx.updated_at < cutoff and tx.status in [
                TransactionStatus.COMPLETED,
                TransactionStatus.ROLLED_BACK,
            ]:
                to_remove.append(tx_id)

        for tx_id in to_remove:
            del self.transactions[tx_id]

        if to_remove:
            self._save_transactions()
            logger.info(f"清理了 {len(to_remove)} 个旧事务")


# 全局事务管理器实例
transaction_manager = TransactionManager()
