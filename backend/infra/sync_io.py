"""端边云同步 IO 实现。

domain/sync.py 只持有端口契约与编排；JSONL 落盘（JsonlQueue）与 HTTP 推送
（UrllibJsonPoster）的 IO 均落在 infra，经依赖注入装配——domain 运行期
不直接触碰文件系统/网络。
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from backend.domain.interfaces import HttpPushPort, QueuePort
from backend.infra.metrics import get_metrics

_LOG = logging.getLogger("scandetection.sync.io")


class JsonlQueue(QueuePort):
    """JSONL 持久化队列（线程安全追加 + 行数观测，原 LocalAdapter 内嵌 IO 外移）。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def append(self, record: Any) -> None:
        """追加一行 JSON（record 需可 JSON 序列化；default=str 兜底非标准类型）。"""
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def count(self) -> int:
        """队列现有行数（观测用）；文件不存在视为 0。"""
        if not self._path.exists():
            return 0
        with self._lock:
            return sum(1 for _ in self._path.open("r", encoding="utf-8"))


class UrllibJsonPoster(HttpPushPort):
    """urllib JSON POST（尽力而为：失败仅告警，不阻断主流程）。

    传输加密由调用方负责（建议 https + 端点鉴权），本实现不内置 TLS 终止逻辑。
    超时经配置注入。
    """

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout

    def post(self, endpoint: str, token: str | None, record: Any) -> None:
        try:
            import urllib.request

            payload = json.dumps(record, ensure_ascii=False, default=str).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                _ = resp.status
            get_metrics().inc("sync_push_total", labels={"adapter": "http", "result": "success"})
        except Exception as exc:  # noqa: BLE001 - 同步尽力而为，失败不阻断主流程
            get_metrics().inc("sync_push_total", labels={"adapter": "http", "result": "failure"})
            _LOG.warning("HttpSync push 失败（已本地留档）: %s", exc)
