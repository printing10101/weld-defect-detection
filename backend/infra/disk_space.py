"""磁盘水位告警（S-20）：data 分区剩余空间后台监测，低水位落安全告警+审计。

后台守护线程周期统计 ``data_dir`` 所在文件系统的可用空间（free/total），
当剩余空间低于阈值时触发 security alert（kind=disk_space_low）并主审计链留痕，
避免磁盘写满导致 SQLite 写失败 / WAL 膨胀 / PDF 报告无法落盘的隐性故障
（长跑 100h 稳定的磁盘层兜底）。

判定（任一触发即告警）：
- ``free_ratio_pct``（剩余比例 %）< ``warn_ratio_pct``；
- ``free_bytes``（剩余绝对字节）< ``warn_min_bytes``（默认 1 GiB）。

告警去重：同一次连续低水位只告警一次（防刷屏，与 MemoryWatchdog 语义一致）；
回落到阈值之上后再超限，才允许再次告警。

RSS 采样的数据分区路径：Windows/POSIX 均可经 ``shutil.disk_usage`` 取路径所在
文件系统的用量。测试可注入 ``sampler`` 返回确定性的 (free, total)。

配置节 disk_space（默认 enabled=true）：见 configs/default.yaml。

线程模式复刻 MemoryWatchdog（S-09）：start/stop + daemon 线程 + check_once，
Registry.lifespan 装配，健康检查经 ``/health` 的 ``disk_space`` 字段暴露。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from backend.infra.timeutil import fmt_naive_utc

_LOG = logging.getLogger("scandetection.disk")

_GB = 1024 * 1024 * 1024


def sample_disk_usage(path: str | Path) -> tuple[int, int]:
    """取路径所在文件系统的磁盘用量，返回 (free_bytes, total_bytes)。失败抛 OSError。"""
    import shutil

    usage = shutil.disk_usage(str(Path(path)))
    return usage.free, usage.total


class DiskWatchdog:
    """磁盘水位看门狗（后台线程；start/stop 生命周期，snapshot 可观测）。

    复用 MemoryWatchdog 的线程/去重范式；``sampler`` 可注入（测试确定性），
    缺省用 sample_disk_usage 取 data_dir 所在分区用量。
    """

    def __init__(
        self,
        *,
        interval_sec: float,
        warn_ratio_pct: float,
        warn_min_bytes: int,
        data_dir: str | Path,
        raise_alert: Callable[..., Any] | None = None,
        append_audit: Callable[..., Any] | None = None,
        sampler: Callable[[], tuple[int, int]] | None = None,
    ) -> None:
        """raise_alert/append_audit 由 Registry 注入（security_store / repository）。"""
        self._interval = max(1.0, float(interval_sec))
        self._warn_ratio = max(0.0, float(warn_ratio_pct))
        self._warn_bytes = max(0, int(warn_min_bytes))
        self._data_dir = Path(data_dir)
        self._raise_alert = raise_alert
        self._append_audit = append_audit
        self._sampler = sampler or (lambda: sample_disk_usage(self._data_dir))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._alerted = False
        self.breach_count = 0
        self.last_free_bytes: int | None = None
        self.last_total_bytes: int | None = None
        self.last_ratio_pct: float | None = None
        self.last_check_at: str | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="disk-watchdog", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.check_once()
            except Exception as exc:  # noqa: BLE001 - 看门狗自身故障不拖垮进程
                _LOG.warning("disk watchdog check failed: %s", exc)

    # ---- 检查 ----------------------------------------------------------------
    def check_once(self) -> dict[str, Any]:
        """单次采样+判定（公开便于测试/外部触发）。返回本次快照。"""
        try:
            free, total = self._sampler()
        except OSError as exc:  # 采样失败：保守不告警（避免误报），如实记录
            _LOG.warning("disk usage sample failed: %s", exc)
            return self.snapshot()
        self.last_free_bytes = free
        self.last_total_bytes = total
        self.last_ratio_pct = (free / total * 100.0) if total > 0 else 0.0
        self.last_check_at = fmt_naive_utc()
        low = free < self._warn_bytes or self.last_ratio_pct < self._warn_ratio
        if low:
            self.breach_count += 1
            if not self._alerted:
                self._alerted = True
                self._notify()
        else:
            self._alerted = False  # 回落：允许下次超限再次告警
        return self.snapshot()

    def _notify(self) -> None:
        """告警（security alert）+ 审计留痕（均 best-effort）。"""
        free_gb = (self.last_free_bytes or 0) / _GB
        total_gb = (self.last_total_bytes or 0) / _GB
        msg = (
            f"磁盘水位告警：data 分区剩余 {self.last_ratio_pct:.2f}%"
            f"（{free_gb:.2f} GiB / {total_gb:.2f} GiB），"
            f"低于阈值（比例 {self._warn_ratio}% 或 绝对 {self._warn_bytes / _GB:.2f} GiB）"
        )
        if self._raise_alert is not None:
            try:
                self._raise_alert(
                    kind="disk_space_low",
                    level="warn",
                    message=msg,
                    detail={
                        "free_bytes": self.last_free_bytes,
                        "total_bytes": self.last_total_bytes,
                        "free_ratio_pct": self.last_ratio_pct,
                        "warn_ratio_pct": self._warn_ratio,
                        "warn_min_bytes": self._warn_bytes,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("disk_space_low 告警落库失败: %s", exc)
        if self._append_audit is not None:
            try:
                self._append_audit(
                    actor="system",
                    action="disk_space",
                    object_type="filesystem",
                    object_id="data",
                    before=None,
                    after={
                        "free_bytes": self.last_free_bytes,
                        "total_bytes": self.last_total_bytes,
                        "free_ratio_pct": self.last_ratio_pct,
                    },
                    note=msg,
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("disk watchdog 审计落库失败: %s", exc)

    # ---- 可观测 ----------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self._thread is not None and self._thread.is_alive(),
            "interval_sec": self._interval,
            "warn_ratio_pct": self._warn_ratio,
            "warn_min_bytes": self._warn_bytes,
            "free_bytes": self.last_free_bytes,
            "total_bytes": self.last_total_bytes,
            "free_ratio_pct": self.last_ratio_pct,
            "last_check_at": self.last_check_at,
            "breach_count": self.breach_count,
        }
