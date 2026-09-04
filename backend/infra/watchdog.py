"""内存看门狗（S-09）：进程 RSS 周期采样，超阈值告警 + 可选优雅重启标记。

后台守护线程每隔 ``interval_sec`` 采样一次进程常驻内存（RSS）：
- 超过 ``rss_warn_mb`` → security alert（kind=memory_pressure，跨阈值只告警一次，
  回落后再超限才再次告警，防刷屏）+ 主审计链留痕；
- 超过 ``rss_restart_mb`` 且 ``graceful_restart``=true → 写重启标记文件
  ``<data_dir>/restart_required``（含原因与时间戳）。Tauri 壳（main.rs 的
  supervisor）周期检测到该标记即重启后端，实现内存超限的优雅自愈。
  标记文件本身不触发本进程强制终止——进程继续运行，由告警+审计+壳侧重启兜底。

RSS 采样优先 psutil；不可用时尽力回退：
- Windows：ctypes GetProcessMemoryInfo（WorkingSetSize）；
- POSIX：resource.getrusage(RUSAGE_SELF).ru_maxrss（峰值语义，非当前值，
  报告时如实标注 source）。

配置节 watchdog（默认 enabled=false，不影响既有测试/部署），状态经 /health
的 ``watchdog`` 字段暴露（Registry.health）。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

_LOG = logging.getLogger("scandetection.watchdog")

_RESTART_MARKER = "restart_required"


def sample_rss_mb() -> tuple[float, str]:
    """采样当前进程 RSS（MB）。返回 (rss_mb, source)。采样失败返回 (-1.0, "unavailable")。"""
    try:
        import psutil  # type: ignore[import-not-found]

        return float(psutil.Process().memory_info().rss) / (1024 * 1024), "psutil"
    except ImportError:
        pass
    except Exception:  # noqa: BLE001 - psutil 异常走回退
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = _PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
            # 必须声明 restype/argtypes：否则 GetCurrentProcess 的 64 位伪句柄会被
            # ctypes 截断成 32 位，GetProcessMemoryInfo 因句柄无效而返回 0（RSS==0，
            # 采样失效）。实测补齐签名后返回 1、正常读到工作集。
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p  # type: ignore[attr-defined]
            psapi.GetProcessMemoryInfo.restype = ctypes.c_int  # type: ignore[attr-defined]
            psapi.GetProcessMemoryInfo.argtypes = [  # type: ignore[attr-defined]
                ctypes.c_void_p,
                ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
                wintypes.DWORD,
            ]
            handle = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return float(pmc.WorkingSetSize) / (1024 * 1024), "ctypes-psapi"
        except Exception:  # noqa: BLE001 - 尽力采样，失败不致命
            pass
    else:
        try:
            import resource

            # POSIX：ru_maxrss 为峰值（KB on Linux, bytes on macOS），如实标注。
            peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if sys_platform() == "darwin":
                return peak / (1024 * 1024), "resource-peak"
            return peak / 1024.0, "resource-peak"
        except Exception:  # noqa: BLE001
            pass
    return -1.0, "unavailable"


def sys_platform() -> str:
    import sys

    return sys.platform


class MemoryWatchdog:
    """RSS 看门狗（后台线程；start/stop 生命周期，snapshot 可观测）。"""

    def __init__(
        self,
        *,
        interval_sec: float,
        rss_warn_mb: float,
        rss_restart_mb: float,
        graceful_restart: bool,
        data_dir: str | Path,
        raise_alert: Callable[..., None] | None = None,
        append_audit: Callable[..., None] | None = None,
        sampler: Callable[[], tuple[float, str]] | None = None,
    ) -> None:
        """raise_alert/append_audit 由 Registry 注入（security_store / repository）。

        sampler 可注入（测试确定性）；缺省用 sample_rss_mb。
        """
        self._interval = max(1.0, float(interval_sec))
        self._warn = float(rss_warn_mb)
        self._restart = float(rss_restart_mb)
        self._graceful = bool(graceful_restart)
        self._data_dir = Path(data_dir)
        self._raise_alert = raise_alert
        self._append_audit = append_audit
        self._sampler = sampler or sample_rss_mb
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._alerted = False  # 跨阈值只告警一次，回落后再超限才再次告警
        self.last_rss_mb: float | None = None
        self.last_sample_source: str | None = None
        self.last_sample_at: str | None = None
        self.breach_count = 0

    # ---- 生命周期 ------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="memory-watchdog", daemon=True)
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
                _LOG.warning("watchdog check failed: %s", exc)

    # ---- 检查 ----------------------------------------------------------------
    def check_once(self) -> dict[str, Any]:
        """单次采样+判定（公开便于测试/外部触发）。返回本次快照。"""
        rss, source = self._sampler()
        self.last_rss_mb = rss
        self.last_sample_source = source
        self.last_sample_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        if rss < 0:
            return self.snapshot()
        if rss >= self._warn:
            self.breach_count += 1
            if not self._alerted:
                self._alerted = True
                self._notify(rss)
            if rss >= self._restart and self._graceful:
                self._write_restart_marker(rss)
        else:
            self._alerted = False  # 回落：允许下次超限再次告警
        return self.snapshot()

    def _notify(self, rss: float) -> None:
        """告警（security alert）+ 审计留痕（均 best-effort）。"""
        msg = f"内存看门狗：RSS {rss:.0f} MB 超过告警阈值 {self._warn:.0f} MB"
        if self._raise_alert is not None:
            try:
                self._raise_alert(
                    kind="memory_pressure",
                    level="warn",
                    message=msg,
                    detail={"rss_mb": rss, "threshold_mb": self._warn},
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("memory_pressure 告警落库失败: %s", exc)
        if self._append_audit is not None:
            try:
                self._append_audit(
                    actor="system",
                    action="watchdog_alert",
                    object_type="process",
                    object_id="memory",
                    before=None,
                    after={"rss_mb": rss, "threshold_mb": self._warn},
                    note=msg,
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("watchdog 审计落库失败: %s", exc)

    def _write_restart_marker(self, rss: float) -> None:
        """写优雅重启标记文件（诚实边界：当前壳侧未消费，仅落盘+日志）。"""
        marker = self._data_dir / _RESTART_MARKER
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                json_dumps(
                    {
                        "reason": "rss_over_threshold",
                        "rss_mb": rss,
                        "threshold_mb": self._restart,
                        "at": self.last_sample_at,
                        "note": "Tauri 壳检测到本文件后将重启后端（优雅自愈）",
                    }
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            _LOG.warning("重启标记文件写入失败: %s", exc)

    # ---- 可观测 ----------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self._thread is not None and self._thread.is_alive(),
            "interval_sec": self._interval,
            "rss_warn_mb": self._warn,
            "rss_restart_mb": self._restart,
            "graceful_restart": self._graceful,
            "last_rss_mb": self.last_rss_mb,
            "sample_source": self.last_sample_source,
            "last_sample_at": self.last_sample_at,
            "breach_count": self.breach_count,
            "restart_marker_pending": (self._data_dir / _RESTART_MARKER).is_file(),
        }


def json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)
