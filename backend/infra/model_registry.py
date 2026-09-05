"""模型注册表：扫描权重目录、列出可用模型、活跃指针持久化、运行时热切换。

设计要点：
- 权重目录解析与检测器加载使用同一套双锚点语义（安装根 → backend 包根，
  见 infra.paths），保证 dev 布局（``<根>/models/weights``）与打包布局
  （``<根>/backend/models/weights``）都能扫到注册表。
- 模型 id 采用 ``<stem>::<sha256[:12]>``，既稳定又可区分同一文件名的不同版本（重训替换后
  id 变化，避免「指针指向旧权重」的歧义）。
- ``activate`` 失败（loader 抛错）时**保持原活跃指针不变**（fail-safe），由调用方转换为
  4xx/5xx 返回，不会把「加载坏模型」静默当成成功。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# backend/infra/model_registry.py -> parents[2] = 安装根目录（项目根 / 部署包根）
from backend.infra.paths import BACKEND_ROOT as _BACKEND_ROOT
from backend.infra.paths import INSTALL_ROOT as _INSTALL_ROOT

_LOG = logging.getLogger("scandetection.model_registry")

_SUFFIXES = (".onnx", ".pt", ".pth")


def _resolve(p: str) -> str:
    """模型路径双锚点解析（与 infra.paths.resolve_model_uri 同语义，dev/打包布局通吃）。

    依次尝试 安装根 → backend 包根（打包布局下权重随 backend 资源分发在
    ``<安装根>/backend/models/weights``，仅试安装根会漏扫注册表）；均未命中时
    锚定安装根返回确定路径，报错时给出期望位置而非随 CWD 漂移的相对路径。
    """
    if os.path.isabs(p):
        return p
    for anchor in (_INSTALL_ROOT, _BACKEND_ROOT):
        candidate = anchor / p
        if candidate.exists():
            return str(candidate)
    return str(_INSTALL_ROOT / p)


@dataclass
class ModelEntry:
    """注册表中的单个模型条目。"""

    id: str
    name: str
    uri: str
    version: str  # 权重 sha256 前 12 位（版本指纹）
    size_bytes: int
    active: bool = False


class ModelRegistry:
    """权重目录扫描 + 活跃指针管理。

    loader 由调用方注入（通常为 ``reg.detector.load``），本类不依赖具体检测器实现，
    保持 infra 不引入业务决策。
    """

    def __init__(self, weights_dir: str, state_file: str) -> None:
        self.weights_dir = _resolve(weights_dir)
        self.state_file = _resolve(state_file)
        self._active_id: str | None = self._load_state()
        # 版本指纹缓存：键=(路径, mtime_ns, size)。权重数百 MB 级，全文件哈希
        # 不能每次 scan()（GET /models、get、mark_active_by_uri 都走 scan）重算；
        # 文件被替换时 mtime_ns/size 通常随之变化 → 自动失配重算（保 mtime+同
        # size 的刻意替换属对抗场景，桌面单机威胁模型下不设防）。
        # _vc_lock 串行化 cache miss 时的哈希与淘汰（GET /models 并发线程安全）。
        self._version_cache: dict[tuple[str, int, int], str] = {}
        self._vc_lock = threading.Lock()

    # ---- 持久化 --------------------------------------------------------------
    def _load_state(self) -> str | None:
        try:
            return json.loads(Path(self.state_file).read_text(encoding="utf-8")).get("active_id")
        except (OSError, json.JSONDecodeError):
            return None

    def _save_state(self) -> None:
        Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
        Path(self.state_file).write_text(
            json.dumps({"active_id": self._active_id}), encoding="utf-8"
        )

    @staticmethod
    def _hash(path: str) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()[:12]

    # ---- 扫描 ----------------------------------------------------------------
    def scan(self) -> list[ModelEntry]:
        entries: list[ModelEntry] = []
        d = Path(self.weights_dir)
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in _SUFFIXES:
                    entries.append(self._entry(f))
        # 状态自洽：持久化的活跃 id 若已不在目录中（权重被删），清空指针。
        ids = {e.id for e in entries}
        if self._active_id is not None and self._active_id not in ids:
            self._active_id = None
        return entries

    def _entry(self, f: Path) -> ModelEntry:
        st = f.stat()
        key = (str(f), st.st_mtime_ns, st.st_size)
        with self._vc_lock:
            h = self._version_cache.get(key)
            if h is None:
                h = self._hash(str(f))
                # 同一路径的旧指纹键出缓存（替换权重后旧键永不再命中，防无界增长）。
                # pop 容错：GET /models 在线程池并发执行时，两个线程可能同时
                # cache miss 并各自持同一路径的旧键快照——后删者会碰到已被
                # 对方删除的键，dict.pop 缺省返回避免 KeyError→500。
                for stale in [k for k in self._version_cache if k[0] == key[0]]:
                    self._version_cache.pop(stale, None)
                self._version_cache[key] = h
        mid = f"{f.stem}::{h}"
        return ModelEntry(
            id=mid,
            name=f.stem,
            uri=str(f),
            version=h,
            size_bytes=st.st_size,
            active=(mid == self._active_id),
        )

    # ---- 查询 / 激活 ----------------------------------------------------------
    @property
    def active_id(self) -> str | None:
        return self._active_id

    def get(self, model_id: str) -> ModelEntry | None:
        for e in self.scan():
            if e.id == model_id:
                return e
        return None

    def mark_active_by_uri(self, uri: str) -> None:
        """启动期/回退后将活跃指针同步到当前实际加载的权重（按 uri 匹配）。

        未命中时留告警：静默 no-op 会让"模型卡显示的 active 与实际推理
        权重"悄然脱钩（双锚点解析在双权重目录并存时可能锚到不同根）。
        """
        for e in self.scan():
            if e.uri == uri:
                self._active_id = e.id
                self._save_state()
                return
        _LOG.warning("mark_active_by_uri: 权重目录中未找到 %s，活跃指针保持不变", uri)

    def activate(self, model_id: str, loader: Callable[[str], None]) -> ModelEntry:
        """热切换：定位条目 → 调 loader(uri) 重载检测器 → 持久化活跃指针。

        loader 抛错时不修改活跃指针（fail-safe）。
        """
        entry = self.get(model_id)
        if entry is None:
            raise KeyError(model_id)
        loader(entry.uri)  # 失败直接上抛，由调用方转 HTTP 错误
        self._active_id = model_id
        self._save_state()
        entry.active = True
        return entry
