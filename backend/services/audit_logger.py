from typing import Dict, Any, List
from datetime import datetime
import json
from pathlib import Path
from utils.logger import logger
from config import settings


class AuditLogger:
    """审计日志记录器"""

    def __init__(self):
        self.log_file = Path(settings.audit_log_file)
        self.logs: List[Dict[str, Any]] = []
        self._load_logs()

    def _load_logs(self):
        """从文件加载日志"""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for log in data:
                        if 'timestamp' in log:
                            log['timestamp'] = datetime.fromisoformat(log['timestamp'])
                        self.logs.append(log)
                logger.info(f"加载了 {len(self.logs)} 条审计日志")
            except Exception as e:
                logger.error(f"加载审计日志失败: {str(e)}")

    def _save_logs(self):
        """保存日志到文件"""
        try:
            # 只保留最近10000条日志
            logs_to_save = self.logs[-10000:]
            data = []
            for log in logs_to_save:
                log_copy = log.copy()
                if isinstance(log_copy.get('timestamp'), datetime):
                    log_copy['timestamp'] = log_copy['timestamp'].isoformat()
                data.append(log_copy)

            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存审计日志失败: {str(e)}")

    def log(self, 
            user_id: str, 
            username: str, 
            action: str, 
            module: str, 
            details: Dict[str, Any] = None,
            ip_address: str = None,
            user_agent: str = None):
        """
        记录审计日志

        Args:
            user_id: 用户ID
            username: 用户名
            action: 操作类型
            module: 模块名称
            details: 详细信息
            ip_address: IP地址
            user_agent: 用户代理
        """
        log_entry = {
            "log_id": self._generate_log_id(),
            "user_id": user_id,
            "username": username,
            "action": action,
            "module": module,
            "details": details or {},
            "ip_address": ip_address,
            "user_agent": user_agent,
            "timestamp": datetime.now()
        }

        self.logs.append(log_entry)
        self._save_logs()
        logger.info(f"审计日志: {username} - {action} - {module}")

    def log_system_event(self, 
                        event_type: str, 
                        message: str, 
                        severity: str = "info",
                        details: Dict[str, Any] = None):
        """
        记录系统事件

        Args:
            event_type: 事件类型
            message: 事件消息
            severity: 严重程度
            details: 详细信息
        """
        log_entry = {
            "log_id": self._generate_log_id(),
            "user_id": "system",
            "username": "system",
            "action": event_type,
            "module": "system",
            "details": {
                "message": message,
                "severity": severity,
                **(details or {})
            },
            "timestamp": datetime.now()
        }

        self.logs.append(log_entry)
        self._save_logs()
        logger.info(f"系统事件: {event_type} - {message}")

    def get_logs(self, 
                 limit: int = 100, 
                 offset: int = 0, 
                 user_id: str = None,
                 action: str = None,
                 module: str = None,
                 start_time: datetime = None,
                 end_time: datetime = None) -> List[Dict[str, Any]]:
        """
        获取审计日志

        Args:
            limit: 限制数量
            offset: 偏移量
            user_id: 用户ID过滤
            action: 操作类型过滤
            module: 模块过滤
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            日志列表
        """
        filtered_logs = self.logs.copy()

        # 应用过滤条件
        if user_id:
            filtered_logs = [log for log in filtered_logs if log.get('user_id') == user_id]
        if action:
            filtered_logs = [log for log in filtered_logs if log.get('action') == action]
        if module:
            filtered_logs = [log for log in filtered_logs if log.get('module') == module]
        if start_time:
            filtered_logs = [log for log in filtered_logs if log.get('timestamp') >= start_time]
        if end_time:
            filtered_logs = [log for log in filtered_logs if log.get('timestamp') <= end_time]

        # 按时间倒序排序
        filtered_logs.sort(key=lambda x: x.get('timestamp'), reverse=True)

        # 应用分页
        return filtered_logs[offset:offset + limit]

    def get_log_count(self, 
                     user_id: str = None,
                     action: str = None,
                     module: str = None,
                     start_time: datetime = None,
                     end_time: datetime = None) -> int:
        """
        获取日志数量

        Args:
            user_id: 用户ID过滤
            action: 操作类型过滤
            module: 模块过滤
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            日志数量
        """
        filtered_logs = self.logs.copy()

        # 应用过滤条件
        if user_id:
            filtered_logs = [log for log in filtered_logs if log.get('user_id') == user_id]
        if action:
            filtered_logs = [log for log in filtered_logs if log.get('action') == action]
        if module:
            filtered_logs = [log for log in filtered_logs if log.get('module') == module]
        if start_time:
            filtered_logs = [log for log in filtered_logs if log.get('timestamp') >= start_time]
        if end_time:
            filtered_logs = [log for log in filtered_logs if log.get('timestamp') <= end_time]

        return len(filtered_logs)

    def _generate_log_id(self) -> str:
        """生成日志ID"""
        import uuid
        return str(uuid.uuid4())

    def export_logs(self, 
                   format: str = "json",
                   start_time: datetime = None,
                   end_time: datetime = None) -> str:
        """
        导出审计日志

        Args:
            format: 导出格式
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            导出文件路径
        """
        filtered_logs = self.logs.copy()

        if start_time:
            filtered_logs = [log for log in filtered_logs if log.get('timestamp') >= start_time]
        if end_time:
            filtered_logs = [log for log in filtered_logs if log.get('timestamp') <= end_time]

        export_file = Path(settings.log_dir) / f"audit_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"

        try:
            if format == "json":
                data = []
                for log in filtered_logs:
                    log_copy = log.copy()
                    if isinstance(log_copy.get('timestamp'), datetime):
                        log_copy['timestamp'] = log_copy['timestamp'].isoformat()
                    data.append(log_copy)

                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            elif format == "csv":
                import csv
                with open(export_file, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['log_id', 'user_id', 'username', 'action', 'module', 'timestamp', 'details'])
                    for log in filtered_logs:
                        writer.writerow([
                            log.get('log_id'),
                            log.get('user_id'),
                            log.get('username'),
                            log.get('action'),
                            log.get('module'),
                            log.get('timestamp').isoformat() if isinstance(log.get('timestamp'), datetime) else '',
                            json.dumps(log.get('details'), ensure_ascii=False)
                        ])
        except Exception as e:
            logger.error(f"导出审计日志失败: {str(e)}")
            raise

        return str(export_file)


# 全局审计日志实例
audit_logger = AuditLogger()
