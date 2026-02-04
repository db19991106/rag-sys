from typing import Dict, Any, List
from datetime import datetime
import psutil
import time
import asyncio
from utils.logger import logger
from services.audit_logger import audit_logger


class HealthMonitor:
    """健康监控类"""

    def __init__(self):
        self.metrics: Dict[str, List[Dict[str, Any]]] = {
            "cpu": [],
            "memory": [],
            "disk": [],
            "network": [],
            "response_time": [],
            "error_rate": []
        }
        self.alert_thresholds = {
            "cpu": 80.0,  # 80%
            "memory": 85.0,  # 85%
            "disk": 90.0,  # 90%
            "response_time": 5.0,  # 5秒
            "error_rate": 5.0  # 5%
        }
        self._start_monitoring()

    def _start_monitoring(self):
        """启动监控任务"""
        asyncio.create_task(self._collect_metrics_task())

    async def _collect_metrics_task(self):
        """定期收集指标的任务"""
        while True:
            try:
                self.collect_system_metrics()
                await asyncio.sleep(30)  # 每30秒收集一次
            except Exception as e:
                logger.error(f"收集指标失败: {str(e)}")
                await asyncio.sleep(60)  # 出错后暂停1分钟

    def collect_system_metrics(self):
        """
        收集系统指标
        """
        timestamp = datetime.now()

        # 收集CPU指标
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics["cpu"].append({
            "timestamp": timestamp,
            "value": cpu_percent
        })

        # 收集内存指标
        memory = psutil.virtual_memory()
        self.metrics["memory"].append({
            "timestamp": timestamp,
            "value": memory.percent,
            "total": memory.total,
            "available": memory.available
        })

        # 收集磁盘指标
        disk = psutil.disk_usage('/')
        self.metrics["disk"].append({
            "timestamp": timestamp,
            "value": disk.percent,
            "total": disk.total,
            "used": disk.used,
            "free": disk.free
        })

        # 收集网络指标
        net_io = psutil.net_io_counters()
        self.metrics["network"].append({
            "timestamp": timestamp,
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        })

        # 限制每个指标的历史数据量
        for metric_name in self.metrics:
            if len(self.metrics[metric_name]) > 2880:  # 48小时 * 12次/小时
                self.metrics[metric_name] = self.metrics[metric_name][-2880:]

        # 检查是否需要告警
        self._check_thresholds()

    def record_response_time(self, endpoint: str, response_time: float):
        """
        记录API响应时间

        Args:
            endpoint: API端点
            response_time: 响应时间（秒）
        """
        timestamp = datetime.now()
        self.metrics["response_time"].append({
            "timestamp": timestamp,
            "endpoint": endpoint,
            "value": response_time
        })

        # 检查响应时间是否超过阈值
        if response_time > self.alert_thresholds["response_time"]:
            self._trigger_alert(
                "response_time",
                f"API响应时间过长: {endpoint} - {response_time:.2f}秒",
                {"endpoint": endpoint, "response_time": response_time}
            )

    def record_error_rate(self, error_rate: float):
        """
        记录错误率

        Args:
            error_rate: 错误率（百分比）
        """
        timestamp = datetime.now()
        self.metrics["error_rate"].append({
            "timestamp": timestamp,
            "value": error_rate
        })

    def _check_thresholds(self):
        """
        检查指标是否超过阈值
        """
        # 检查CPU
        if self.metrics["cpu"]:
            latest_cpu = self.metrics["cpu"][-1]["value"]
            if latest_cpu > self.alert_thresholds["cpu"]:
                self._trigger_alert(
                    "cpu",
                    f"CPU使用率过高: {latest_cpu:.1f}%",
                    {"value": latest_cpu}
                )

        # 检查内存
        if self.metrics["memory"]:
            latest_memory = self.metrics["memory"][-1]["value"]
            if latest_memory > self.alert_thresholds["memory"]:
                self._trigger_alert(
                    "memory",
                    f"内存使用率过高: {latest_memory:.1f}%",
                    {"value": latest_memory}
                )

        # 检查磁盘
        if self.metrics["disk"]:
            latest_disk = self.metrics["disk"][-1]["value"]
            if latest_disk > self.alert_thresholds["disk"]:
                self._trigger_alert(
                    "disk",
                    f"磁盘使用率过高: {latest_disk:.1f}%",
                    {"value": latest_disk}
                )

    def _trigger_alert(self, metric: str, message: str, details: Dict[str, Any]):
        """
        触发告警

        Args:
            metric: 指标名称
            message: 告警消息
            details: 详细信息
        """
        logger.warning(f"[告警] {message}")
        audit_logger.log_system_event(
            "alert",
            message,
            "warning",
            {
                "metric": metric,
                "threshold": self.alert_thresholds.get(metric),
                **details
            }
        )

    def get_health_status(self) -> Dict[str, Any]:
        """
        获取系统健康状态

        Returns:
            健康状态信息
        """
        status = "healthy"
        issues = []

        # 检查CPU
        if self.metrics["cpu"]:
            latest_cpu = self.metrics["cpu"][-1]["value"]
            if latest_cpu > self.alert_thresholds["cpu"]:
                status = "degraded"
                issues.append(f"CPU使用率过高: {latest_cpu:.1f}%")

        # 检查内存
        if self.metrics["memory"]:
            latest_memory = self.metrics["memory"][-1]["value"]
            if latest_memory > self.alert_thresholds["memory"]:
                status = "degraded"
                issues.append(f"内存使用率过高: {latest_memory:.1f}%")

        # 检查磁盘
        if self.metrics["disk"]:
            latest_disk = self.metrics["disk"][-1]["value"]
            if latest_disk > self.alert_thresholds["disk"]:
                status = "critical"
                issues.append(f"磁盘使用率过高: {latest_disk:.1f}%")

        return {
            "status": status,
            "timestamp": datetime.now(),
            "issues": issues,
            "metrics": {
                "cpu": self.metrics["cpu"][-1] if self.metrics["cpu"] else None,
                "memory": self.metrics["memory"][-1] if self.metrics["memory"] else None,
                "disk": self.metrics["disk"][-1] if self.metrics["disk"] else None,
                "network": self.metrics["network"][-1] if self.metrics["network"] else None
            }
        }

    def get_metrics_history(self, metric: str, minutes: int = 60) -> List[Dict[str, Any]]:
        """
        获取指标历史数据

        Args:
            metric: 指标名称
            minutes: 历史分钟数

        Returns:
            指标历史数据
        """
        if metric not in self.metrics:
            return []

        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            item for item in self.metrics[metric]
            if item["timestamp"] >= cutoff_time
        ]

    def reset_metrics(self):
        """
        重置所有指标
        """
        self.metrics = {
            "cpu": [],
            "memory": [],
            "disk": [],
            "network": [],
            "response_time": [],
            "error_rate": []
        }


# 修复导入
from datetime import timedelta

# 全局健康监控实例
health_monitor = HealthMonitor()
