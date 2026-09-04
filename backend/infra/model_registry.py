"""模型注册表：扫描权重目录、列出可用模型、活跃指针持久化、运行时热切换。

设计要点：
- 与检测器加载使用同一套「安装根目录锚点」解析（``parents[2]``），保证 ``weights_dir``
  在部署包（``_pkg/ScanDetection/``）内指向 ``models/weights``，与 ``model.default_uri``
  的解析语义完全一致（见 dependencies._resolve_model_uri）。
- 模型 id 采用 ``<stem>::<sha256[:12]>``，既稳定又可区分同一文件名的不同版本（重训替换后
  id 变化，避免「指针指向旧权重」的歧义）。
- ``activate`` 失败（loader 抛错）时**保持原活跃指针不变**（fail-safe），由调用方转换为
  4xx/5xx 返回，不会把「加载坏模型」静默当成成功。
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# backend/infra/model_registry.py -> parents[2] = 安装根目录（项目根 / 部署包根）
from backend.infra.paths import INSTALL_ROOT as _INSTALL_ROOT

_SUFFIXES = (".onnx", ".pt", ".pth")


def _resolve(p: str) -> str:
    """将相对路径解析为绝对路径：优先相对安装根目录，否则保持原样（便于报出原始错误）。"""
    if os.path.isabs(p):
        return p
    root_candidate = _INSTALL_ROOT / p
    if root_candidate.exists():
        return str(root_candidate)
    return p


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
        h = self._hash(str(f))
        mid = f"{f.stem}::{h}"
        return ModelEntry(
            id=mid,
            name=f.stem,
            uri=str(f),
            version=h,
            size_bytes=f.stat().st_size,
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
        """启动期将活跃指针同步到当前实际加载的权重（按 uri 匹配）。"""
        for e in self.scan():
            if e.uri == uri:
                self._active_id = e.id
                self._save_state()
                return

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
