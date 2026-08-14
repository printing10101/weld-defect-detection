"""端边云同步 v1 LocalAdapter（§7.6，M6 实现）。

规格书：v1 = LocalAdapter（空操作/本地）；v3 = CloudAdapter（联邦平均）。
v1 语义——数据**不出本机**（§7.5 本地优先）：
- push：把待同步记录追加到本地待同步队列（JSONL 落盘），不发起任何网络调用；
- pull：本地无远端，恒返回空列表；
- federate：无远端可联邦，空操作（v3 才实现联邦平均）。

这样 v1 在"数据不出本机 + 接口契约完整"下可演进到 v3 云端适配器，
替换实现即切换，不影响上层调用方。
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

_LOG = logging.getLogger("scandetection.sync")


class LocalAdapter:
    """本地同步适配器（§7.6 v1：空操作/本地，数据不出本机）。"""

    name = "local"

    def __init__(self, queue_path: str | Path | None = None) -> None:
        """queue_path 缺省时不做持久化（纯空操作）；提供时记录 push 到 JSONL。"""
        self._path = Path(queue_path) if queue_path else None
        self._lock = threading.Lock()

    def push(self, record) -> None:
        """本地记录待同步项（不发起网络）。record 需可 JSON 序列化。"""
        if self._path is None:
            return
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def pull(self) -> list[dict[str, Any]]:
        """本地无远端：恒返回空列表。"""
        return []

    def federate(self, weights) -> None:
        """v3 才实现联邦平均；v1 空操作（无远端可联邦）。"""
        return

    @property
    def pending_count(self) -> int:
        """本地待同步队列条数（观测用）。"""
        if self._path is None or not self._path.exists():
            return 0
        with self._lock:
            return sum(1 for _ in self._path.open("r", encoding="utf-8"))


class HttpSyncAdapter:
    """HTTP 同步适配器（§7.6 v3 前哨：推送到可配置端点，数据不出本机由调用方保证）。

    与 LocalAdapter 同契约：
    - push：本地落 JSONL（可观测）+ POST 到 http_endpoint（Bearer token 可选）；
      网络失败仅告警、不抛（同步是尽力而为，不阻断主流程）；
    - pull：返回空列表（云侧拉取留待 v3 联邦）；
    - federate：空操作（v3 才实现联邦平均）。

    传输加密由调用方负责（建议 https + 端点鉴权），本适配器不内置 TLS 终止逻辑。
    """

    name = "http"

    def __init__(
        self,
        endpoint: str | None,
        token: str | None = None,
        queue_path: str | Path | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError("HttpSyncAdapter 需要配置 sync.http_endpoint")
        self._endpoint = endpoint
        self._token = token
        self._path = Path(queue_path) if queue_path else None
        self._lock = threading.Lock()

    def push(self, record) -> None:
        """本地记录 + 推送远端（尽力而为，失败仅告警）。"""
        if self._path is not None:
            with self._lock:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        try:
            import urllib.request

            payload = json.dumps(record, ensure_ascii=False, default=str).encode("utf-8")
            req = urllib.request.Request(
                self._endpoint,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            if self._token:
                req.add_header("Authorization", f"Bearer {self._token}")
            with urllib.request.urlopen(req, timeout=10) as resp:
                _ = resp.status
        except Exception as exc:  # noqa: BLE001 - 同步尽力而为，失败不阻断主流程
            _LOG.warning("HttpSync push 失败（已本地留档）: %s", exc)

    def pull(self) -> list[dict[str, Any]]:
        """云侧拉取留待 v3；当前返回空列表。"""
        return []

    def federate(self, weights) -> None:
        """v3 才实现联邦平均；当前空操作。"""
        return

    @property
    def pending_count(self) -> int:
        """本地待同步队列条数（观测用）。"""
        if self._path is None or not self._path.exists():
            return 0
        with self._lock:
            return sum(1 for _ in self._path.open("r", encoding="utf-8"))
