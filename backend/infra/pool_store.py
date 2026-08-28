"""训练池存储 IO 实现。

domain/active_learning.py 只持有端口契约与纯逻辑（标注格式化 / manifest 结构）；
pool 目录的 mkdir / 写标注 / 列文件 / 指纹 / manifest 读写 IO 均落在 infra，
经调用方装配注入——domain 运行期不直接触碰文件系统。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.domain.interfaces import PoolStore
from backend.evaluation.harness import golden_set_fingerprint


class FilePoolStore(PoolStore):
    """文件系统训练池存储（YOLO 标注 + manifest，对齐原 domain 内嵌 IO 语义）。

    manifest 与 pool_dir 同级（data/active/pool_manifest.json），与原实现一致。
    """

    def __init__(self, pool_dir: str | Path) -> None:
        self._root = Path(pool_dir)

    def write_label(self, stem: str, content: str) -> Path:
        """写入 {stem}.txt（同 stem 覆盖，防重复导出旧标注残留）。

        仅取文件名成分，杜绝 stem 含 ".."/绝对路径导致的目录穿越（纵深防御）。
        """
        self._root.mkdir(parents=True, exist_ok=True)
        out = self._root / f"{Path(stem).name}.txt"
        out.write_text(content, encoding="utf-8")
        return out

    def list_labels(self) -> list[str]:
        """pool 内全部 YOLO 标注（相对路径，排序稳定）；目录缺失返回空列表。"""
        if not self._root.is_dir():
            return []
        return sorted(
            str(p.relative_to(self._root)) for p in self._root.rglob("*.txt") if p.is_file()
        )

    def fingerprint(self) -> str | None:
        """数据版本指纹；无标注返回 None。"""
        if not self.list_labels():
            return None
        return golden_set_fingerprint(self._root)

    def load_manifest(self) -> dict[str, Any] | None:
        """读取持久化 manifest；无则 None；损坏容忍（返回 None）。"""
        path = self._manifest_path()
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def save_manifest(self, manifest: dict[str, Any]) -> Path:
        """持久化 manifest 到 pool_dir 同级（data/active/pool_manifest.json）。"""
        path = self._manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _manifest_path(self) -> Path:
        return self._root.resolve().parent / "pool_manifest.json"
