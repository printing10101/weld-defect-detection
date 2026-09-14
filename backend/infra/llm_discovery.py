"""本地 LLM 资源发现：扫描本机 GGUF + 探测已运行的 llama 兼容服务 + GPU 显存探测。

三层能力（**全部只读**）
------------------------
1. :func:`scan_gguf_files` —— 遍历目录树收集 ``.gguf``（跳过系统目录、不跟随
   symlink/junction、权限错误计数跳过），逐个经 :mod:`backend.infra.gguf_meta`
   解析头部元数据；
2. :func:`probe_service` —— 探测本机 OpenAI 兼容端点（llama.cpp 的 ``/v1/models``
   与 ``/health``），列出服务端已加载模型、识别是否需要鉴权；
3. :func:`read_gpu_vram` —— 取 NVIDIA 显存总量/可用量，供显存可行性判定。

:class:`LlmDiscovery` 把三者编排为「后台扫描 + 进度上报 + 可取消 + 结果缓存」，
供 ``app.routers.llm`` 以同步/异步两种方式消费。

安全边界（与项目硬化口径一致）
------------------------------
- **只读**：不修改、不移动、不删除任何文件；
- 跳过系统与易变目录（``Windows``/``$Recycle.Bin``/``System Volume Information``/
  ``WinSxS``/``temp`` …）：那里不可能存放用户模型，只会拖慢扫描并撞权限错误；
- **不跟随** symlink / junction（Windows 的兼容性 junction 会造成目录循环与越界）；
- 权限错误**计数跳过**而非中断，单个坏目录不影响整体结果；
- 每个候选文件只读头部（``gguf_meta.HEADER_READ_BYTES``），绝不整文件载入；
- 服务探测**仅限回环地址**（复用 :func:`llm_server.is_loopback_host`），
  ``egress_guard`` 对回环本就放行，不产生任何外发；
  API Key 只经环境变量读取，**绝不落盘、绝不写日志**。

缓存
----
结果落 ``data/llm_scan_cache.json``（含扫描根、时间戳、逐文件 size/mtime）。
``cached_results`` 在缓存未过期且覆盖当前扫描根时直接复用，避免每次启动全盘重扫
（本机实测全盘 63.7 s / 30 万目录）。
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.infra.gguf_meta import GgufMeta, read_gguf_meta
from backend.infra.llm_server import is_loopback_host

_LOG = logging.getLogger("scandetection.llm_discovery")

_MIB = 1024 * 1024
_GIB = 1024 * 1024 * 1024

# 跳过：系统目录 / 回收站 / 卷影 / 临时 / 版本控制 / 依赖目录
SKIP_DIR_NAMES = frozenset(
    {
        "$recycle.bin",
        "system volume information",
        "$windows.~bt",
        "$windows.~ws",
        "recovery",
        "perflogs",
        "msocache",
        "config.msi",
        "winsxs",
        "temp",
        "node_modules",
        ".git",
        ".cargo",
        ".rustup",
    }
)

HEADER_PROBE_TIMEOUT_SEC = 1.5
_NVIDIA_SMI_TIMEOUT_SEC = 5.0
_PROGRESS_EVERY_DIRS = 2000
_CACHE_VERSION = 1
DEFAULT_CACHE_TTL_SEC = 24 * 3600


# --------------------------------------------------------------------------- #
# 目录遍历
# --------------------------------------------------------------------------- #


def _is_reparse_point(entry: os.DirEntry) -> bool:
    """junction / symlink 检测：命中即不跟随，避免循环与越界。"""
    try:
        if entry.is_symlink():
            return True
        attrs = entry.stat(follow_symlinks=False).st_file_attributes  # type: ignore[attr-defined]
        return bool(attrs & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT
    except (OSError, AttributeError):
        return False


def fixed_drive_roots() -> list[str]:
    """本机存在的固定盘根，用于默认全盘扫描范围。

    Windows：``C:\\`` ``D:\\`` …；POSIX（麒麟/UOS）无盘符概念，返回 ``/``
    （此前恒返回空列表会让 Linux 部署下"默认扫描范围"为空，扫不到任何模型）。
    """
    if os.name != "nt":
        return ["/"]
    return [f"{d}:\\" for d in "CDEFGHIJ" if Path(f"{d}:\\").exists()]


def scan_gguf_files(
    roots: list[str] | list[Path],
    *,
    parse_meta: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[list[GgufMeta], dict[str, Any]]:
    """遍历 ``roots`` 收集 ``.gguf`` 并解析元数据。

    ``progress(scanned_dirs, found, current_dir)`` 每遍历若干目录回调一次；
    ``should_cancel()`` 返回 True 时**优雅停止**并在 stats 里置 ``cancelled``
    （返回已收集的部分结果，而非清空——取消不等于丢数据）。

    返回 ``(metas, stats)``；任何单个文件的失败都不会中断整体扫描。
    """
    roots_path = [Path(r) for r in roots]
    found: list[GgufMeta] = []
    stats: dict[str, Any] = {
        "dirs": 0,
        "skipped_dirs": 0,
        "permission_errors": 0,
        "cancelled": False,
        "roots": [str(r) for r in roots_path],
    }
    stack = [r for r in roots_path if r.exists()]
    started = time.perf_counter()

    while stack:
        if should_cancel is not None and should_cancel():
            stats["cancelled"] = True
            break
        current = stack.pop()
        stats["dirs"] += 1  # 计数语义 = **访问过的目录数**（含根本身），供进度与统计
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_symlink():
                            # 符号链接（目录或文件）一律不跟随、不重复计入，但必须留痕：
                            # 否则目录 symlink 会静默消失，skipped_dirs 统计与实际行为脱节。
                            stats["skipped_dirs"] += 1
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name.lower() in SKIP_DIR_NAMES or _is_reparse_point(entry):
                                stats["skipped_dirs"] += 1
                                continue
                            stack.append(Path(entry.path))
                        elif entry.name.lower().endswith(".gguf"):
                            meta = (
                                read_gguf_meta(entry.path)
                                if parse_meta
                                else GgufMeta(
                                    path=entry.path,
                                    size_bytes=entry.stat(follow_symlinks=False).st_size,
                                    ok=True,
                                )
                            )
                            found.append(meta)
                    except (OSError, PermissionError):
                        stats["permission_errors"] += 1
        except (PermissionError, OSError):
            stats["permission_errors"] += 1
        if progress is not None and stats["dirs"] % _PROGRESS_EVERY_DIRS == 0:
            progress(stats["dirs"], len(found), str(current))

    stats["elapsed_sec"] = round(time.perf_counter() - started, 2)
    stats["found"] = len(found)
    return found, stats


def model_key(meta: GgufMeta) -> str:
    """同模型多副本的归并键：名称 + 架构 + 量化 + 字节数（同尺寸即视为同一权重）。"""
    return f"{meta.name or Path(meta.path).stem}|{meta.architecture}|{meta.quant}|{meta.size_bytes}"


def dedupe_models(metas: list[GgufMeta]) -> tuple[list[GgufMeta], list[GgufMeta]]:
    """归并重复权重：返回 ``(唯一项, 副本)``。

    同一模型常被复制到多个目录（本机 Qwen3-4B 存在 3 份），列表里全列会淹没
    有效信息。保留**路径最短**的那份作为主条目（通常是最「正规」的安装位置），
    其余作为副本由上层折叠展示。
    """
    best: dict[str, GgufMeta] = {}
    dupes: list[GgufMeta] = []
    for meta in metas:
        key = model_key(meta)
        cur = best.get(key)
        if cur is None:
            best[key] = meta
        elif (len(meta.path), meta.path) < (len(cur.path), cur.path):
            dupes.append(cur)
            best[key] = meta
        else:
            dupes.append(meta)
    return list(best.values()), dupes


# --------------------------------------------------------------------------- #
# GPU 显存探测
# --------------------------------------------------------------------------- #


@dataclass
class GpuInfo:
    """NVIDIA 显存快照（无 NVIDIA 卡 / 无 nvidia-smi 时上层拿到 ``None``）。"""

    name: str = ""
    total_bytes: int = 0
    free_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_bytes": self.total_bytes,
            "free_bytes": self.free_bytes,
        }


def read_gpu_vram() -> GpuInfo | None:
    """经 ``nvidia-smi`` 取显存总量/可用量；不可用时返回 ``None``（不猜、不兜造）。"""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    line = proc.stdout.strip().splitlines()[0]
    try:
        name, total_mib, free_mib = [s.strip() for s in line.rsplit(",", 2)]
        return GpuInfo(
            name=name,
            total_bytes=int(total_mib) * _MIB,
            free_bytes=int(free_mib) * _MIB,
        )
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# 已运行服务探测
# --------------------------------------------------------------------------- #


@dataclass
class ServiceProbe:
    """单个 llama 兼容端点的探测结果。"""

    host: str
    port: int
    reachable: bool = False
    needs_auth: bool = False
    health_ok: bool = False
    models: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def openai_base_url(self) -> str:
        """OpenAI 兼容基址（供 ``openai`` 客户端 / llama-server 外部模式使用）。"""
        return f"{self.base_url}/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "base_url": self.base_url,
            "openai_base_url": self.openai_base_url,
            "reachable": self.reachable,
            "needs_auth": self.needs_auth,
            "health_ok": self.health_ok,
            "models": list(self.models),
            "error": self.error,
        }


def _http_get_json(
    url: str, *, timeout: float, api_key: str | None = None
) -> tuple[int, Any | None]:
    """GET JSON，返回 ``(status_code, payload)``；网络层失败统一返回 ``(0, None)``。

    HTTP 错误码（401/403/404…）原样返回——它们本身就是有效探测信息。
    """
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        # 目标地址已由 probe_service 的回环门禁强制约束，不存在外发风险
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(64 * 1024)
            return int(resp.status), (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        return int(exc.code), None
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return 0, None


def probe_service(
    host: str,
    port: int,
    *,
    timeout: float = HEADER_PROBE_TIMEOUT_SEC,
    api_key: str | None = None,
    api_key_env: str = "",
) -> ServiceProbe:
    """探测 ``host:port`` 上的 OpenAI 兼容服务（llama.cpp / Ollama / LM Studio）。

    **仅允许回环地址**——非回环直接返回 ``error="non_loopback"``，不做任何外发。
    ``api_key_env`` 指向环境变量名，用于需要鉴权的实例（如带 ``--api-key`` 启动的
    llama-server）；Key 只在此处读取用于请求头，不落盘、不写日志。
    """
    probe = ServiceProbe(host=host, port=port)
    if not is_loopback_host(host):
        probe.error = "non_loopback"
        _LOG.warning("拒绝探测非回环端点: %s:%s", host, port)
        return probe

    key = api_key
    if not key and api_key_env:
        key = os.environ.get(api_key_env) or None

    health_status, _ = _http_get_json(f"{probe.base_url}/health", timeout=timeout, api_key=key)
    probe.health_ok = health_status == 200

    status, payload = _http_get_json(f"{probe.base_url}/v1/models", timeout=timeout, api_key=key)
    if status == 200 and isinstance(payload, dict):
        probe.reachable = True
        probe.models = [
            str(item["id"])
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        ]
    elif status in (401, 403):
        # 服务活着但要凭据：这是有效信息，不能当成「不可达」
        probe.reachable = True
        probe.needs_auth = True
        probe.error = "auth_required"
    elif status:
        probe.reachable = True
        probe.error = f"http_{status}"
    else:
        probe.error = "unreachable"
    return probe


def probe_services(
    endpoints: list[tuple[str, int]],
    *,
    timeout: float = HEADER_PROBE_TIMEOUT_SEC,
    api_key_env: str = "",
) -> list[ServiceProbe]:
    """批量探测端点，跳过不可达项不抛异常（供 ``GET /llm/services`` 直接消费）。"""
    return [
        probe_service(host, port, timeout=timeout, api_key_env=api_key_env)
        for host, port in endpoints
    ]


# --------------------------------------------------------------------------- #
# 缓存
# --------------------------------------------------------------------------- #


def _cache_payload(
    metas: list[GgufMeta], gpu: GpuInfo | None, stats: dict[str, Any]
) -> dict[str, Any]:
    return {
        "version": _CACHE_VERSION,
        "scanned_at": time.time(),
        "roots": stats.get("roots", []),
        "stats": {k: v for k, v in stats.items() if k != "roots"},
        "gpu": gpu.to_dict() if gpu else None,
        "files": [m.to_dict() for m in metas],
    }


def save_cache(path: str | Path, payload: dict[str, Any]) -> bool:
    """原子写缓存（临时文件 + ``os.replace``）；失败返回 False，不影响主流程。"""
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, target)
        return True
    except OSError:
        _LOG.warning("写入扫描缓存失败: %s", target, exc_info=True)
        return False


def load_cache(
    path: str | Path, *, max_age_sec: float = DEFAULT_CACHE_TTL_SEC
) -> dict[str, Any] | None:
    """读缓存；文件缺失/损坏/版本不符/过期一律返回 ``None``（调用方转去重扫）。"""
    target = Path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("version") != _CACHE_VERSION:
        return None
    if max_age_sec > 0 and time.time() - float(data.get("scanned_at") or 0) > max_age_sec:
        return None
    return data


def cache_to_metas(data: dict[str, Any]) -> list[GgufMeta]:
    """把缓存里的 ``files`` 还原成 :class:`GgufMeta`（字段缺失按默认值兜底）。"""
    metas: list[GgufMeta] = []
    for raw in data.get("files", []):
        if not isinstance(raw, dict) or not raw.get("path"):
            continue
        kwargs = {k: v for k, v in raw.items() if k in GgufMeta.__dataclass_fields__}
        metas.append(GgufMeta(**kwargs))
    return metas


# --------------------------------------------------------------------------- #
# 编排
# --------------------------------------------------------------------------- #


class LlmDiscovery:
    """本地 LLM 资源发现编排：后台扫描 + 进度 + 取消 + 缓存。

    状态快照经 :meth:`status` 读取（线程安全）；:meth:`start` 幂等——已在扫描中
    返回 False，不重复起线程。取消后**保留已扫到的部分结果**（取消不等于丢数据）。
    """

    def __init__(
        self,
        *,
        roots: list[str] | None = None,
        cache_file: str | Path,
        parse_meta: bool = True,
        cache_ttl_sec: float = DEFAULT_CACHE_TTL_SEC,
    ) -> None:
        self._roots = list(roots) if roots else fixed_drive_roots()
        self._cache_file = Path(cache_file)
        self._parse_meta = parse_meta
        self._cache_ttl = cache_ttl_sec
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._metas: list[GgufMeta] = []
        self._gpu: GpuInfo | None = None
        self._stats: dict[str, Any] = {}
        self._state = "idle"  # idle / scanning / done / cancelled / error
        self._progress: dict[str, Any] = {"dirs": 0, "found": 0, "current": ""}
        self._error: str | None = None

    # -- 只读属性 ----------------------------------------------------------- #

    @property
    def roots(self) -> list[str]:
        return list(self._roots)

    @property
    def cache_file(self) -> Path:
        return self._cache_file

    def status(self) -> dict[str, Any]:
        """当前状态快照（含进度；``scanning`` 时 ``progress`` 有值）。"""
        with self._lock:
            return {
                "state": self._state,
                "roots": list(self._roots),
                "progress": dict(self._progress),
                "found": len(self._metas),
                "elapsed_sec": self._stats.get("elapsed_sec"),
                "error": self._error,
            }

    def results(self) -> tuple[list[GgufMeta], GpuInfo | None, dict[str, Any]]:
        """最近一次扫描结果 ``(metas, gpu, stats)``（未扫过时为空）。"""
        with self._lock:
            return list(self._metas), self._gpu, dict(self._stats)

    # -- 缓存 --------------------------------------------------------------- #

    def cached(self, *, force: bool = False) -> dict[str, Any] | None:
        """读缓存；``force=True`` 时忽略 TTL（仍校验版本/格式）。"""
        data = load_cache(self._cache_file, max_age_sec=0 if force else self._cache_ttl)
        if data is None:
            return None
        # 缓存必须覆盖当前扫描根，否则视为失效（换了范围就该重扫）
        cached_roots = {str(r).lower().rstrip("\\/") for r in data.get("roots", [])}
        want_roots = {r.lower().rstrip("\\/") for r in self._roots}
        if not want_roots.issubset(cached_roots):
            return None
        return data

    # -- 扫描 --------------------------------------------------------------- #

    def scan_now(self) -> dict[str, Any]:
        """同步扫描（阻塞至结束）；供测试与「立刻要结果」的调用方使用。"""
        self._cancel.clear()
        with self._lock:
            self._state = "scanning"
            self._progress = {"dirs": 0, "found": 0, "current": ""}
            self._error = None
        try:
            metas, stats = scan_gguf_files(
                self._roots,
                parse_meta=self._parse_meta,
                progress=self._on_progress,
                should_cancel=self._cancel.is_set,
            )
            gpu = read_gpu_vram()
        except Exception as exc:  # noqa: BLE001 - 发现失败不能拖垮服务启动
            with self._lock:
                self._state = "error"
                self._error = f"{exc.__class__.__name__}: {exc}"
            _LOG.exception("本地模型扫描失败")
            return self.status()
        with self._lock:
            self._metas = metas
            self._gpu = gpu
            self._stats = stats
            self._state = "cancelled" if stats.get("cancelled") else "done"
        save_cache(self._cache_file, _cache_payload(metas, gpu, stats))
        return self.status()

    def start(self) -> bool:
        """起后台扫描线程；已在扫描中返回 ``False``（幂等，不重复起线程）。"""
        with self._lock:
            if self._state == "scanning":
                return False
            self._state = "scanning"
            self._cancel.clear()
            self._progress = {"dirs": 0, "found": 0, "current": ""}
            self._error = None
        self._thread = threading.Thread(
            target=self.scan_now, name="llm-discovery-scan", daemon=True
        )
        self._thread.start()
        return True

    def cancel(self) -> bool:
        """请求取消扫描；已在扫描中返回 True（结果为已扫到的部分）。"""
        with self._lock:
            if self._state != "scanning":
                return False
        self._cancel.set()
        return True

    def join(self, timeout: float | None = None) -> None:
        """等待后台扫描结束（测试与关停路径用）。"""
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    def _on_progress(self, dirs: int, found: int, current: str) -> None:
        with self._lock:
            self._progress = {"dirs": dirs, "found": found, "current": current}


__all__ = [
    "DEFAULT_CACHE_TTL_SEC",
    "HEADER_PROBE_TIMEOUT_SEC",
    "SKIP_DIR_NAMES",
    "GpuInfo",
    "LlmDiscovery",
    "ServiceProbe",
    "cache_to_metas",
    "dedupe_models",
    "fixed_drive_roots",
    "load_cache",
    "model_key",
    "probe_service",
    "probe_services",
    "read_gpu_vram",
    "save_cache",
    "scan_gguf_files",
]
