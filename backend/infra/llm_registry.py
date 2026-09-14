"""本地大模型注册表：发现清单 + 选中指针持久化 + 生效配置推导。

职责（对齐 ``infra.model_registry`` 的既有模式）
------------------------------------------------
- **发现**：编排 :mod:`backend.infra.llm_discovery`（全盘扫描 GGUF + 探测 llama
  兼容端点 + GPU 显存探测），把「本机有哪些模型」汇总成统一清单；
- **清单**：本地 GGUF 与「已有服务已加载的模型」合并为同一结构
  （:class:`LlmModelEntry`），**全部条目都返回**——跑不动的模型也照列，只是带
  ``vram.verdict`` 标注（用户明确要求不做折叠过滤）；
- **选中**：把「用哪个模型」持久化到 ``data/llm_registry.json``，并把选中项推导
  成一份生效的 :class:`~backend.infra.config.LlmCfg`（managed=自拉起该 GGUF /
  external=连那个端点），供 ``LlamaServerManager`` 直接消费——管理器保持"哑"，
  不需要认识注册表；
- **目录**：用户在界面添加的额外模型目录持久化，与配置文件里的内置目录取并集。

id 语义
-------
- 本地：``gguf:<sha1(规范化路径)[:12]>``——同一路径稳定，同一模型的不同副本
  各自独立（副本关系由 ``primary``/``duplicate_of`` 表达，而非合并掉）；
- 服务：``svc:<sha1(端点|模型名)[:12]>``。
不采用「文件内容哈希」：本机最大权重单文件 16.5 GB，选中一次全量哈希不可接受。

安全边界
--------
- 只读：不移动/删除任何模型文件；添加目录仅写注册表状态文件；
- 目录路径经 ``resolve_config_path`` 解析后**必须存在且是目录**才接受，拒绝空串；
- 服务探测**仅限回环**（由 ``llm_discovery.probe_service`` 强制），API Key 只从
  环境变量读、不落盘不写日志；
- 状态文件原子写（临时文件 + ``os.replace``），损坏时回落默认值而非崩溃。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.infra.config import LlmCfg, resolve_config_path
from backend.infra.gguf_meta import GgufMeta, estimate_vram
from backend.infra.llm_discovery import (
    DEFAULT_CACHE_TTL_SEC,
    GpuInfo,
    LlmDiscovery,
    ServiceProbe,
    cache_to_metas,
    fixed_drive_roots,
    load_cache,
    model_key,
    probe_service,
    read_gpu_vram,
    scan_gguf_files,
)
from backend.infra.paths import resolve_data_path

_LOG = logging.getLogger("scandetection.llm_registry")

_STATE_VERSION = 1
SOURCE_LOCAL = "local"
SOURCE_SERVICE = "service"
MODE_MANAGED = "managed"
MODE_EXTERNAL = "external"

# 端点探测并发度：4 个预设端点串行最坏 4×timeout，并发后最坏 1×timeout
_PROBE_WORKERS = 4

# 盘根判定：显式目录若是盘根，交给全盘扫描（避免"同步直扫 C:\"这类病态开销）
_DRIVE_ROOT_RE = re.compile(r"^[A-Za-z]:[\\/]?$")


def _is_drive_root(p: str) -> bool:
    s = str(p).strip()
    return bool(_DRIVE_ROOT_RE.match(s)) or s in ("/", "\\")


def _local_id(path: str) -> str:
    """本地模型 id：规范化路径的 sha1 前 12 位（路径稳定则 id 稳定）。"""
    norm = str(Path(path)).replace("/", "\\").lower()
    return "gguf:" + hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def _service_id(base_url: str, name: str) -> str:
    return "svc:" + hashlib.sha1(f"{base_url.lower()}|{name}".encode()).hexdigest()[:12]


def _norm_dir(p: str) -> str:
    """目录比较键：去除尾部分隔符并小写（Windows 路径大小写不敏感）。"""
    return str(Path(p)).rstrip("\\/").lower()


@dataclass
class LlmModelEntry:
    """注册表中的单个可选模型（本地 GGUF 或已有服务上的模型）。"""

    id: str
    name: str
    source: str = SOURCE_LOCAL  # local | service
    display_name: str = ""
    path: str = ""  # local：GGUF 绝对路径
    endpoint: str = ""  # service：OpenAI 兼容 base_url
    size_bytes: int = 0
    architecture: str = ""
    quant: str = ""
    max_ctx: int | None = None  # 模型**支持上限**，非服务实际 -c
    is_embedding: bool = False
    available: bool = True  # false = 头部不可解析 / 文件不可读
    error: str | None = None
    active: bool = False
    primary: bool = True  # false = 同模型的另一份副本
    duplicate_of: str | None = None  # 副本指向的主条目 id
    duplicate_count: int = 0  # 主条目：共有几份副本
    vram: dict[str, Any] | None = None  # estimate_vram(...).to_dict()（service 无）
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name or self.name,
            "source": self.source,
            "path": self.path,
            "endpoint": self.endpoint,
            "size_bytes": self.size_bytes,
            "architecture": self.architecture,
            "quant": self.quant,
            "max_ctx": self.max_ctx,
            "is_embedding": self.is_embedding,
            "available": self.available,
            "error": self.error,
            "active": self.active,
            "primary": self.primary,
            "duplicate_of": self.duplicate_of,
            "duplicate_count": self.duplicate_count,
            "vram": self.vram,
            "notes": list(self.notes),
        }


class LlmRegistry:
    """本地大模型发现 + 选中管理。

    ``state_file`` / ``cache_file`` 默认取配置中的 ``data/`` 相对路径，经
    :func:`backend.infra.paths.resolve_data_path` 解析（随用户数据目录重定向，
    卸载不留残留）。
    """

    def __init__(
        self,
        cfg: LlmCfg,
        *,
        state_file: str | Path | None = None,
        cache_file: str | Path | None = None,
    ) -> None:
        self._cfg = cfg
        self._state_file = (
            Path(state_file)
            if state_file is not None
            else resolve_data_path(cfg.registry_state_file)
        )
        self._cache_file = (
            Path(cache_file) if cache_file is not None else resolve_data_path(cfg.scan_cache_file)
        )
        self._lock = threading.Lock()
        self._state: dict[str, Any] = self._load_state()
        self._discovery: LlmDiscovery | None = None
        # 显式声明目录的扫描结果（进程内缓存一次）：让"添加目录"立刻能看到模型，
        # 而不必等一次 60s 级的全盘扫描。目录增删时失效重扫。
        self._dir_metas_cache: list[GgufMeta] | None = None
        # 最近一次清单快照（id → 条目）：``select`` 据此解析 id，避免"端点抖动
        # 一下，用户刚在界面上看到的模型就选不中了"（探测结果瞬时不可复用，
        # 但用户点的是列表里那一项，语义上必须可选）。
        self._seen: dict[str, LlmModelEntry] = {}

    # ---------- 持久化 ----------

    def _load_state(self) -> dict[str, Any]:
        """读状态文件；缺失/损坏/版本不符一律回落默认（不抛，不阻断装配）。"""
        default: dict[str, Any] = {
            "version": _STATE_VERSION,
            "active_id": None,
            "mode": None,
            "endpoint": "",
            "model_path": "",
            "model_dirs": [],
        }
        try:
            raw = json.loads(Path(self._state_file).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default
        if not isinstance(raw, dict) or raw.get("version") != _STATE_VERSION:
            _LOG.warning("LLM 注册表状态文件版本不符或格式非法，回落到默认: %s", self._state_file)
            return default
        dirs = [str(d) for d in raw.get("model_dirs", []) if isinstance(d, str) and d.strip()]
        return {
            "version": _STATE_VERSION,
            "active_id": raw.get("active_id") or None,
            "mode": raw.get("mode") or None,
            "endpoint": str(raw.get("endpoint") or ""),
            "model_path": str(raw.get("model_path") or ""),
            "model_dirs": dirs,
        }

    def _save_state(self) -> None:
        """原子写状态文件；失败仅告警（选中丢失可重新选，不阻断运行）。"""
        with self._lock:
            payload = dict(self._state)
        payload["version"] = _STATE_VERSION
        payload["updated_at"] = time.time()
        try:
            target = Path(self._state_file)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f"{target.name}.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, target)
        except OSError as exc:
            _LOG.warning("LLM 注册表状态写入失败（不影响运行）: %s", exc)

    # ---------- 目录 ----------

    @property
    def state_file(self) -> Path:
        return self._state_file

    @property
    def cache_file(self) -> Path:
        return self._cache_file

    def config_model_dirs(self) -> list[str]:
        """配置文件声明的模型目录（内置，不可从界面删除）。"""
        return list(self._cfg.model_dirs)

    def extra_model_dirs(self) -> list[str]:
        """用户在界面添加的模型目录（持久化）。"""
        with self._lock:
            return list(self._state.get("model_dirs", []))

    def all_model_dirs(self) -> list[str]:
        """配置内置目录 + 用户添加目录的并集（保留顺序、忽略大小写重复）。"""
        out: list[str] = []
        seen: set[str] = set()
        for d in [*self.config_model_dirs(), *self.extra_model_dirs()]:
            key = _norm_dir(d)
            if key and key not in seen:
                seen.add(key)
                out.append(d)
        return out

    def add_dir(self, path: str) -> str:
        """添加模型目录。路径须解析后真实存在且为目录，否则 ``ValueError``。"""
        raw = (path or "").strip()
        if not raw:
            raise ValueError("目录路径不能为空")
        resolved = resolve_config_path(raw)
        if not resolved.is_dir():
            raise ValueError(f"目录不存在或不是目录: {resolved}")
        norm = _norm_dir(str(resolved))
        if norm in {_norm_dir(d) for d in self.config_model_dirs()}:
            return str(resolved)  # 已是内置目录，幂等
        with self._lock:
            existing = {_norm_dir(d) for d in self._state.get("model_dirs", [])}
            if norm not in existing:
                self._state.setdefault("model_dirs", []).append(str(resolved))
        self._save_state()
        self._invalidate_dirs()
        return str(resolved)

    def remove_dir(self, path: str) -> bool:
        """移除用户添加的目录；返回是否真的移除了（不存在返回 False）。"""
        norm = _norm_dir(path)
        removed = False
        with self._lock:
            dirs = list(self._state.get("model_dirs", []))
            kept = [d for d in dirs if _norm_dir(d) != norm]
            if len(kept) != len(dirs):
                self._state["model_dirs"] = kept
                removed = True
        if removed:
            self._save_state()
            self._invalidate_dirs()
        return removed

    # ---------- 扫描 ----------

    def declared_dirs(self) -> list[str]:
        """显式声明的模型目录（已解析为绝对路径），供"目录直扫"使用。"""
        return [str(resolve_config_path(d)) for d in self.all_model_dirs()]

    def scan_roots(self) -> list[str]:
        """**全盘扫描**范围：配置扫描根；未配置则本机全部固定盘。"""
        return [str(r) for r in (self._cfg.scan_roots or fixed_drive_roots())]

    def _ensure_discovery(self) -> LlmDiscovery:
        """按当前全盘扫描范围取（必要时重建）发现器。"""
        desired = self.scan_roots()
        with self._lock:
            cur = self._discovery
            if cur is not None and cur.roots == desired:
                return cur
            if cur is not None:
                cur.cancel()
            fresh = LlmDiscovery(roots=desired, cache_file=self._cache_file)
            self._discovery = fresh
            return fresh

    def _invalidate_dirs(self) -> None:
        """声明目录变化时只失效目录直扫缓存——不取消正在跑的全盘扫描。"""
        with self._lock:
            self._dir_metas_cache = None

    def start_scan(self) -> bool:
        """起后台全盘扫描；已在扫描中返回 False（幂等）。"""
        return self._ensure_discovery().start()

    def cancel_scan(self) -> bool:
        with self._lock:
            disc = self._discovery
        return disc.cancel() if disc is not None else False

    def scan_now(self) -> dict[str, Any]:
        """同步全盘扫描（阻塞；供测试与「立刻要结果」用）。"""
        return self._ensure_discovery().scan_now()

    def scan_status(self) -> dict[str, Any]:
        """扫描状态快照；从未扫描过时返回 ``state=idle`` 的空壳。"""
        roots = self.scan_roots()
        declared = self.declared_dirs()
        with self._lock:
            disc = self._discovery
        if disc is None:
            base = {
                "state": "idle",
                "roots": roots,
                "declared_dirs": declared,
                "progress": {"dirs": 0, "found": 0, "current": ""},
                "found": 0,
                "elapsed_sec": None,
                "error": None,
            }
        else:
            base = disc.status()
            base["roots"] = roots
            base["declared_dirs"] = declared
        return base

    def _dir_metas(self) -> list[GgufMeta]:
        """直扫显式声明目录（进程内缓存一次）。

        让「添加模型目录」立刻能在列表里看到模型，而不必等全盘扫描。盘根声明
        （如 ``E:\\``）被排除——那属于全盘扫描的职责，同步直扫会变成分钟级阻塞。
        """
        with self._lock:
            if self._dir_metas_cache is not None:
                return list(self._dir_metas_cache)
        dirs = [d for d in self.declared_dirs() if not _is_drive_root(d)]
        metas: list[GgufMeta] = []
        if dirs:
            metas, stats = scan_gguf_files(dirs, parse_meta=True)
            if stats.get("permission_errors"):
                _LOG.debug("目录直扫跳过 %d 个无权限路径", stats["permission_errors"])
        with self._lock:
            self._dir_metas_cache = list(metas)
        return metas

    def local_metas(self) -> list[GgufMeta]:
        """本地 GGUF 元数据 = 全盘扫描结果（或缓存）∪ 显式目录直扫结果。

        两个来源合并而非二选一：全盘缓存可能过期或未覆盖用户新加的目录，
        而"用户在界面上明确加过的目录"永远应该出现在列表里。
        """
        with self._lock:
            disc = self._discovery
        base: list[GgufMeta] = []
        if disc is not None:
            metas, _gpu, _stats = disc.results()
            base = list(metas)
        if not base:
            cached = load_cache(self._cache_file, max_age_sec=DEFAULT_CACHE_TTL_SEC)
            if cached:
                base = cache_to_metas(cached)
        seen = {str(Path(m.path)).lower() for m in base}
        merged = list(base)
        for m in self._dir_metas():
            if str(Path(m.path)).lower() not in seen:
                seen.add(str(Path(m.path)).lower())
                merged.append(m)
        return merged

    def gpu_info(self) -> GpuInfo | None:
        """GPU 快照：本次扫描结果优先，其次缓存，最后实时探测。

        实时回退是必要的——扫描可能从未跑过（或缓存过期），而"哪些模型跑得动"
        完全依赖可用显存；少了这一步，列表里的显存判定会一直是 ``unknown``。
        """
        with self._lock:
            disc = self._discovery
        if disc is not None:
            _metas, gpu, _stats = disc.results()
            if gpu is not None:
                return gpu
        cached = load_cache(self._cache_file, max_age_sec=DEFAULT_CACHE_TTL_SEC)
        if cached and isinstance(cached.get("gpu"), dict):
            g = cached["gpu"]
            return GpuInfo(
                name=str(g.get("name") or ""),
                total_bytes=int(g.get("total_bytes") or 0),
                free_bytes=int(g.get("free_bytes") or 0),
            )
        return read_gpu_vram()

    # ---------- 服务探测 ----------

    def presets(self) -> list[tuple[str, int]]:
        """预设端点（host, port）；非法项跳过而非崩溃。"""
        out: list[tuple[str, int]] = []
        for item in self._cfg.service_presets:
            host, _, port = str(item).rpartition(":")
            if not host or not port.isdigit():
                _LOG.warning("忽略非法的 service_presets 项: %r", item)
                continue
            out.append((host, int(port)))
        return out

    def probe_all(self) -> list[ServiceProbe]:
        """并发探测全部预设端点（串行最坏 4×timeout，并发最坏 1×timeout）。"""
        presets = self.presets()
        if not presets:
            return []
        timeout = self._cfg.probe_timeout_sec
        env_name = self._cfg.api_key_env
        with ThreadPoolExecutor(max_workers=_PROBE_WORKERS) as pool:
            futures = [
                pool.submit(
                    probe_service,
                    host,
                    port,
                    timeout=timeout,
                    api_key_env=env_name,
                )
                for host, port in presets
            ]
            return [f.result() for f in futures]

    def probe_endpoint(self, host: str, port: int) -> ServiceProbe:
        return probe_service(
            host,
            port,
            timeout=self._cfg.probe_timeout_sec,
            api_key_env=self._cfg.api_key_env,
        )

    # ---------- 清单 ----------

    def list_models(
        self,
        *,
        probes: list[ServiceProbe] | None = None,
        include_services: bool = True,
        free_vram_bytes: int | None = None,
        n_ctx: int | None = None,
    ) -> list[LlmModelEntry]:
        """全部可选模型（本地 GGUF + 已有服务），**不做可行性过滤**。

        ``include_services=False`` 时只返回本地条目（不产生任何网络调用）。
        ``free_vram_bytes`` 为 None 时读实时 GPU 显存（无 NVIDIA 卡则显存判定为
        ``unknown``，不猜硬件）。
        """
        ctx = n_ctx if n_ctx is not None else self._cfg.n_ctx
        free = free_vram_bytes
        if free is None:
            gpu = self.gpu_info()
            free = gpu.free_bytes if gpu is not None and gpu.free_bytes > 0 else None

        with self._lock:
            active_id = self._state.get("active_id")

        entries: list[LlmModelEntry] = []
        entries.extend(self._local_entries(free, ctx))
        if include_services:
            entries.extend(
                self._service_entries(probes if probes is not None else self.probe_all())
            )
        for e in entries:
            e.active = e.id == active_id
        with self._lock:
            for e in entries:
                self._seen[e.id] = e
        return entries

    def _local_entries(self, free_vram_bytes: int | None, n_ctx: int) -> list[LlmModelEntry]:
        """本地条目：同模型多副本全部保留，主条目带 ``duplicate_count``。"""
        metas = self.local_metas()
        # 主条目选择：同 key 里路径最短者（通常是最"正规"的安装位置）
        primary_path: dict[str, str] = {}
        counts: dict[str, int] = {}
        for m in metas:
            k = model_key(m)
            counts[k] = counts.get(k, 0) + 1
            cur = primary_path.get(k)
            if cur is None or (len(m.path), m.path) < (len(cur), cur):
                primary_path[k] = m.path

        out: list[LlmModelEntry] = []
        for m in sorted(metas, key=lambda x: -x.size_bytes):
            k = model_key(m)
            is_primary = m.path == primary_path[k]
            entry = LlmModelEntry(
                id=_local_id(m.path),
                name=Path(m.path).stem,
                source=SOURCE_LOCAL,
                display_name=Path(m.path).stem,
                path=m.path,
                size_bytes=m.size_bytes,
                architecture=m.architecture,
                quant=m.quant,
                max_ctx=m.max_ctx,
                is_embedding=m.is_embedding,
                available=m.ok,
                error=m.error,
                primary=is_primary,
                duplicate_of=None if is_primary else _local_id(primary_path[k]),
                duplicate_count=counts[k] - 1 if is_primary else 0,
            )
            if m.ok:
                entry.vram = estimate_vram(m, n_ctx, free_vram_bytes=free_vram_bytes).to_dict()
            if m.is_embedding:
                entry.notes.append("向量/嵌入模型，不能替代对话模型。")
            if entry.duplicate_count:
                entry.notes.append(f"同模型另有 {entry.duplicate_count} 份副本（已列出，未隐藏）。")
            if not m.ok:
                entry.notes.append("头部无法解析，不可选中（文件可能损坏或非 GGUF）。")
            out.append(entry)
        return out

    def _service_entries(self, probes: list[ServiceProbe]) -> list[LlmModelEntry]:
        out: list[LlmModelEntry] = []
        for probe in probes:
            if not probe.reachable:
                continue
            for name in probe.models:
                entry = LlmModelEntry(
                    id=_service_id(probe.base_url, name),
                    name=name,
                    source=SOURCE_SERVICE,
                    display_name=name,
                    endpoint=probe.base_url,
                )
                if probe.needs_auth:
                    entry.available = False
                    entry.error = "auth_required"
                    entry.notes.append("该服务要求鉴权，请配置 API Key 环境变量后再选。")
                else:
                    entry.notes.append(f"由已有服务提供（{probe.base_url}），不占用本机额外显存。")
                out.append(entry)
        return out

    def _all_entries(self, *, include_services: bool = True) -> list[LlmModelEntry]:
        return self.list_models(include_services=include_services)

    def get(self, model_id: str, *, include_services: bool = True) -> LlmModelEntry | None:
        """按 id 取条目：先查实时清单，未命中再落到最近一次清单快照。

        回落是必要的——用户点的是刚才列表里的那一项，而探测结果不会缓存在
        ``LlmDiscovery`` 里（它只负责本轮扫描）。没有回落时，端点一次探测失败
        就会让用户"看着列表却选不中"。代价是条目可能略陈旧：``select`` 对本地
        条目会再校验文件是否仍在，避免选中已被删除的权重。
        """
        for e in self._all_entries(include_services=include_services):
            if e.id == model_id:
                return e
        with self._lock:
            return self._seen.get(model_id)

    # ---------- 选中 ----------

    @property
    def active_id(self) -> str | None:
        with self._lock:
            return self._state.get("active_id")

    def selection(self) -> dict[str, Any]:
        """当前选中快照（含推导出的加载方式），供 ``GET /llm/status`` 直接返回。"""
        with self._lock:
            st = dict(self._state)
        return {
            "active_id": st.get("active_id"),
            "mode": st.get("mode"),
            "endpoint": st.get("endpoint") or "",
            "model_path": st.get("model_path") or "",
        }

    def select(self, model_id: str) -> LlmModelEntry:
        """选中模型并持久化。

        - 本地条目 → ``mode=managed`` + ``model_file=<该 GGUF 路径>``；
        - 服务条目 → ``mode=external`` + ``external_endpoint=<端点>``；
        - 不可用条目（头部解析失败 / 服务要求鉴权）→ ``ValueError``，拒绝选中，
          不把"选了个跑不了的"静默当成成功。
        """
        entry = self.get(model_id)
        if entry is None:
            raise KeyError(model_id)
        if not entry.available:
            raise ValueError(entry.error or "模型不可用，无法选中")
        if entry.source == SOURCE_SERVICE:
            mode, endpoint, model_path = MODE_EXTERNAL, entry.endpoint, ""
        else:
            # 条目可能来自清单快照：选中前确认权重仍在（否则会把启动失败留到下次）
            if not Path(entry.path).is_file():
                raise ValueError(f"模型文件不存在: {entry.path}")
            mode, endpoint, model_path = MODE_MANAGED, "", entry.path
        with self._lock:
            self._state.update(
                {
                    "active_id": model_id,
                    "mode": mode,
                    "endpoint": endpoint,
                    "model_path": model_path,
                }
            )
        self._save_state()
        entry.active = True
        return entry

    def clear_selection(self) -> None:
        with self._lock:
            self._state.update({"active_id": None, "mode": None, "endpoint": "", "model_path": ""})
        self._save_state()

    def effective_llm_cfg(self, base: LlmCfg) -> LlmCfg:
        """按选中项推导生效配置（``LlamaServerManager`` 直接消费，无需认识注册表）。

        未选中任何模型时原样返回 ``base``——保持「配置即行为」的可预测性。
        """
        with self._lock:
            mode = self._state.get("mode")
            endpoint = self._state.get("endpoint") or ""
            model_path = self._state.get("model_path") or ""
        if mode == MODE_EXTERNAL:
            return base.model_copy(
                update={
                    "mode": MODE_EXTERNAL,
                    "external_endpoint": endpoint or base.external_endpoint,
                }
            )
        if mode == MODE_MANAGED and model_path:
            return base.model_copy(update={"mode": MODE_MANAGED, "model_file": model_path})
        return base


__all__ = [
    "MODE_EXTERNAL",
    "MODE_MANAGED",
    "SOURCE_LOCAL",
    "SOURCE_SERVICE",
    "LlmModelEntry",
    "LlmRegistry",
]
