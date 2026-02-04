from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from pathlib import Path
from utils.logger import logger
from config import settings
import inspect
import asyncio
from concurrent.futures import ThreadPoolExecutor


class StateLogger:
    """状态变化日志记录器"""

    def __init__(self):
        self.log_file = Path(settings.state_log_file)
        self.logs: List[Dict[str, Any]] = []
        self._load_logs()
        self.executor = ThreadPoolExecutor(max_workers=2)

    def _load_logs(self):
        """从文件加载日志"""
        if self.log_file.exists():
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        logger.info("状态日志文件为空，初始化为空列表")
                        self.logs = []
                        return
                    data = json.loads(content)
                    for log in data:
                        if "timestamp" in log:
                            log["timestamp"] = datetime.fromisoformat(log["timestamp"])
                        self.logs.append(log)
                logger.info(f"加载了 {len(self.logs)} 条状态变化日志")
            except Exception as e:
                logger.error(f"加载状态变化日志失败: {str(e)}")

    async def _save_logs_async(self):
        """异步保存日志到文件"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                self.executor, self._save_logs_sync
            )
        except Exception as e:
            logger.error(f"异步保存状态变化日志失败: {str(e)}")

    def _save_logs_sync(self):
        """同步保存日志到文件"""
        try:
            # 只保留最近50000条日志
            logs_to_save = self.logs[-50000:]
            data = []
            for log in logs_to_save:
                log_copy = log.copy()
                if isinstance(log_copy.get("timestamp"), datetime):
                    log_copy["timestamp"] = log_copy["timestamp"].isoformat()
                data.append(log_copy)

            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存状态变化日志失败: {str(e)}")

    def log_state_change(
        self,
        state_type: str,
        entity_id: str,
        previous_state: Optional[Dict[str, Any]],
        current_state: Optional[Dict[str, Any]],
        action: str,
        source: str,
        details: Optional[Dict[str, Any]] = None,
        level: str = "info",
    ):
        """
        记录状态变化

        Args:
            state_type: 状态类型
            entity_id: 实体ID
            previous_state: 变化前的状态
            current_state: 变化后的状态
            action: 触发变化的操作
            source: 变化的来源
            details: 详细信息
            level: 日志级别
        """
        # 获取调用者信息
        caller_frame = inspect.currentframe().f_back
        caller_info = inspect.getframeinfo(caller_frame)
        caller_file = caller_info.filename.split("/")[-1]
        caller_line = caller_info.lineno

        log_entry = {
            "log_id": self._generate_log_id(),
            "timestamp": datetime.now(),
            "level": level,
            "state_type": state_type,
            "entity_id": entity_id,
            "previous_state": previous_state,
            "current_state": current_state,
            "action": action,
            "source": source,
            "caller": f"{caller_file}:{caller_line}",
            "details": details or {},
        }

        self.logs.append(log_entry)

        # 异步保存，避免阻塞主线程
        loop = (
            asyncio.get_event_loop()
            if asyncio.get_event_loop().is_running()
            else asyncio.new_event_loop()
        )
        loop.create_task(self._save_logs_async())

        # 记录到控制台
        if level == "error":
            logger.error(f"状态变化: {state_type} - {entity_id} - {action} - {source}")
        elif level == "warning":
            logger.warning(
                f"状态变化: {state_type} - {entity_id} - {action} - {source}"
            )
        else:
            logger.info(f"状态变化: {state_type} - {entity_id} - {action} - {source}")

    def log_system_state(
        self,
        component: str,
        state: str,
        details: Optional[Dict[str, Any]] = None,
        level: str = "info",
    ):
        """
        记录系统组件状态

        Args:
            component: 系统组件
            state: 组件状态
            details: 详细信息
            level: 日志级别
        """
        log_entry = {
            "log_id": self._generate_log_id(),
            "timestamp": datetime.now(),
            "level": level,
            "state_type": "system",
            "entity_id": component,
            "previous_state": None,
            "current_state": {"state": state},
            "action": "state_change",
            "source": "system",
            "details": details or {},
        }

        self.logs.append(log_entry)

        # 异步保存
        loop = (
            asyncio.get_event_loop()
            if asyncio.get_event_loop().is_running()
            else asyncio.new_event_loop()
        )
        loop.create_task(self._save_logs_async())

        # 记录到控制台
        if level == "error":
            logger.error(f"系统状态: {component} - {state}")
        elif level == "warning":
            logger.warning(f"系统状态: {component} - {state}")
        else:
            logger.info(f"系统状态: {component} - {state}")

    def get_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        state_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取状态变化日志

        Args:
            limit: 限制数量
            offset: 偏移量
            state_type: 状态类型过滤
            entity_id: 实体ID过滤
            level: 日志级别过滤
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            日志列表
        """
        filtered_logs = self.logs.copy()

        # 应用过滤条件
        if state_type:
            filtered_logs = [
                log for log in filtered_logs if log.get("state_type") == state_type
            ]
        if entity_id:
            filtered_logs = [
                log for log in filtered_logs if log.get("entity_id") == entity_id
            ]
        if level:
            filtered_logs = [log for log in filtered_logs if log.get("level") == level]
        if start_time:
            filtered_logs = [
                log for log in filtered_logs if log.get("timestamp") >= start_time
            ]
        if end_time:
            filtered_logs = [
                log for log in filtered_logs if log.get("timestamp") <= end_time
            ]

        # 按时间倒序排序
        filtered_logs.sort(key=lambda x: x.get("timestamp"), reverse=True)

        # 应用分页
        return filtered_logs[offset : offset + limit]

    def get_log_count(
        self,
        state_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        """
        获取日志数量

        Args:
            state_type: 状态类型过滤
            entity_id: 实体ID过滤
            level: 日志级别过滤
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            日志数量
        """
        filtered_logs = self.logs.copy()

        # 应用过滤条件
        if state_type:
            filtered_logs = [
                log for log in filtered_logs if log.get("state_type") == state_type
            ]
        if entity_id:
            filtered_logs = [
                log for log in filtered_logs if log.get("entity_id") == entity_id
            ]
        if level:
            filtered_logs = [log for log in filtered_logs if log.get("level") == level]
        if start_time:
            filtered_logs = [
                log for log in filtered_logs if log.get("timestamp") >= start_time
            ]
        if end_time:
            filtered_logs = [
                log for log in filtered_logs if log.get("timestamp") <= end_time
            ]

        return len(filtered_logs)

    def _generate_log_id(self) -> str:
        """生成日志ID"""
        import uuid

        return str(uuid.uuid4())

    def export_logs(
        self,
        format: str = "json",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> str:
        """
        导出状态变化日志

        Args:
            format: 导出格式
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            导出文件路径
        """
        filtered_logs = self.logs.copy()

        if start_time:
            filtered_logs = [
                log for log in filtered_logs if log.get("timestamp") >= start_time
            ]
        if end_time:
            filtered_logs = [
                log for log in filtered_logs if log.get("timestamp") <= end_time
            ]

        export_file = (
            Path(settings.log_dir)
            / f"state_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        )

        try:
            if format == "json":
                data = []
                for log in filtered_logs:
                    log_copy = log.copy()
                    if isinstance(log_copy.get("timestamp"), datetime):
                        log_copy["timestamp"] = log_copy["timestamp"].isoformat()
                    data.append(log_copy)

                with open(export_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            elif format == "csv":
                import csv

                with open(export_file, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "log_id",
                            "timestamp",
                            "level",
                            "state_type",
                            "entity_id",
                            "action",
                            "source",
                            "details",
                        ]
                    )
                    for log in filtered_logs:
                        writer.writerow(
                            [
                                log.get("log_id"),
                                log.get("timestamp").isoformat()
                                if isinstance(log.get("timestamp"), datetime)
                                else "",
                                log.get("level"),
                                log.get("state_type"),
                                log.get("entity_id"),
                                log.get("action"),
                                log.get("source"),
                                json.dumps(log.get("details"), ensure_ascii=False),
                            ]
                        )
        except Exception as e:
            logger.error(f"导出状态变化日志失败: {str(e)}")
            raise

        return str(export_file)

    def clear_old_logs(self, days: int = 7):
        """
        清理旧日志

        Args:
            days: 保留最近多少天的日志
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        self.logs = [log for log in self.logs if log.get("timestamp") >= cutoff_time]

        loop = (
            asyncio.get_event_loop()
            if asyncio.get_event_loop().is_running()
            else asyncio.new_event_loop()
        )
        loop.create_task(self._save_logs_async())

        logger.info(f"清理了 {days} 天前的状态变化日志")


# 全局状态变化日志实例
state_logger = StateLogger()
