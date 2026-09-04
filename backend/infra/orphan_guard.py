"""父进程孤儿兜底（S-* 可靠性）。

当后端由 Tauri 壳拉起时，Tauri 通过环境变量 ``SCANDETECTION_PARENT_PID`` 传入
壳自身 PID。本模块启动一个后台守护线程，周期探测父进程是否存活：

- 父进程仍在 → 继续；
- 父进程已退出（壳被强杀 / 崩溃 / 任务管理器结束等，窗口 Destroyed 事件不会
  触发，壳内 supervisor 也随之消失）→ 日志留痕后 ``os._exit(0)`` 自杀退出，
  释放端口与资源，避免遗留孤儿后端长期占用 18773 / 占内存。

边界：进程探测失败时**保守判定"父进程存活"**（不自杀），避免误杀在线后端。
该机制仅在由壳启动（env 存在且为正整数 PID）时武装；直接命令行跑 uvicorn 时不生效。
"""

from __future__ import annotations

import logging
import os
import threading
import time

_LOG = logging.getLogger("scandetection.orphan_guard")

_PARENT_PID_ENV = "SCANDETECTION_PARENT_PID"


# Windows 侧进程存活探测所需的原生 API，一次性初始化（放模块顶部避免每轮探测重复加载）。
# 关键点：HANDLE 是 64 位指针，必须把 restype 声明为 c_void_p，否则 ctypes 默认的 c_int
# (32 位) 会把高位非零的句柄截断成 0，造成"进程明明存在却判定不存在"的误杀。
# use_last_error=True 让 ctypes 可靠保存/读取线程 last-error，避免 GetLastError 时序被改写。


def _window_probe(pid: int) -> bool:
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        open_proc = getattr(k32, "OpenProcess", None)
        if open_proc is not None:
            open_proc.restype = ctypes.c_void_p  # type: ignore[attr-defined]
            open_proc.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]  # type: ignore[attr-defined]
            # PROCESS_QUERY_LIMITED_INFORMATION 级别句柄即可判存在，通常跨权限可用。
            handle = open_proc(0x1000, False, wintypes.DWORD(pid))
            if not handle:
                # 句柄为空：区分"进程不存在"与"存在但句柄受限"。
                return int(ctypes.get_last_error() or 0) == 5  # 5=ACCESS_DENIED
            try:
                # 光有句柄还不够——被 TerminateProcess 的进程对象仍可被打开，
                # 会造成"死人判活"的误判。必须再看退出码：STILL_ACTIVE=259 才算存活。
                get_exit = getattr(k32, "GetExitCodeProcess", None)
                if get_exit is not None:
                    get_exit.restype = wintypes.BOOL  # type: ignore[attr-defined]
                    get_exit.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]  # type: ignore[attr-defined]
                    code = wintypes.DWORD(0)
                    if get_exit(handle, ctypes.byref(code)):
                        return int(code.value) == 259
                    return True  # 读取退出码失败，保守判存活
                return True  # 极老环境无 GetExitCodeProcess，退回仅凭句柄
            finally:
                close = getattr(k32, "CloseHandle", None)
                if close is not None:
                    close(handle)
        # 极旧/精简环境无 OpenProcess：回退到枚举快照（尽力而为）。
        snap = getattr(k32, "CreateToolhelp32Snapshot", None)
        if snap is None:
            return True
    except (OSError, ValueError, AttributeError, ImportError):
        # 任一环节异常都保守判存活，避免误杀在线的后端。
        return True
    return _window_probe_snap(pid)


def _window_probe_snap(pid: int) -> bool:
    """回退方案：用 Toolhelp32 进程快照判断 pid 是否存在（无句柄权限依赖）。"""
    import ctypes
    from ctypes import wintypes

    try:
        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        TH32CS_SNAPPROCESS = 0x00000002
        snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if not snap or snap == -1:
            return True  # 快照失败保守判存活
        try:

            class _PROCESSENTRY32W(ctypes.Structure):
                _fields_ = [
                    ("dwSize", wintypes.DWORD),
                    ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.POINTER(wintypes.ULONG)),
                    ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", wintypes.LONG),
                    ("dwFlags", wintypes.DWORD),
                    ("szExeFile", wintypes.WCHAR * 260),
                ]

            k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
            k32.Process32FirstW.restype = wintypes.BOOL
            k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
            k32.Process32NextW.restype = wintypes.BOOL
            entry = _PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
            found = bool(k32.Process32FirstW(snap, ctypes.byref(entry)))
            while found:
                if int(entry.th32ProcessID) == pid:
                    return True
                found = bool(k32.Process32NextW(snap, ctypes.byref(entry)))
            return False
        finally:
            k32.CloseHandle(snap)
    except Exception:  # noqa: BLE001 - 异常保守判存活
        return True


def parent_alive(pid: int) -> bool:
    """探测 pid 进程是否存活。失败时保守返回 True（不误杀）。"""
    if pid <= 0:
        return True
    if os.name == "nt":
        return _window_probe(pid)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True


def _guard_loop(parent_pid: int, interval_sec: float) -> None:
    while True:
        if not parent_alive(parent_pid):
            _LOG.error(
                "父进程(pid=%s)已退出，孤儿后端按兜底策略自杀退出",
                parent_pid,
            )
            # 日志已由 handler 逐条 flush；此处显式退出，即时释放端口/内存。
            os._exit(0)
        time.sleep(interval_sec)


def start_orphan_guard_if_spawned(interval_sec: float = 3.0) -> threading.Thread | None:
    """若由 Tauri 壳启动（env 提供了正整数父 PID），武装孤儿兜底线程。

    直接命令行运行（无 env）返回 None，不产生任何副作用。
    """
    raw = os.environ.get(_PARENT_PID_ENV)
    if not raw:
        return None
    try:
        pid = int(raw)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    thread = threading.Thread(
        target=_guard_loop,
        args=(pid, max(1.0, float(interval_sec))),
        name="orphan-guard",
        daemon=True,
    )
    thread.start()
    _LOG.info("orphan-guard armed (parent pid=%s)", pid)
    return thread
