"""模型存储/加载（§T4 / §7.4）。

M4 前为桩：只管理 URI/版本/哈希状态，不加载真实权重。
真实推理实现（YOLO/RT-DETR 等）在 M4 里程碑以 DefectDetector 实现注入。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

# 安装根目录锚点：backend/infra/model_store.py -> parents[2] = 安装根目录
# （dev/repo 布局下为仓库根；Tauri 打包后为本机安装目录）
_INSTALL_ROOT = Path(__file__).resolve().parents[2]
# backend 包根目录锚点：parents[1] = backend/。
# Tauri 打包时模型随 ``backend`` 资源一同分发，落在 <安装目录>/backend/models/weights/，
# 而非安装根目录下的 models/weights；因此解析时回退到此处，避免找不到权重而静默降级。
_BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _resolve_model_uri(uri: str) -> str:
    """将相对模型路径解析为绝对路径。

    依次尝试以下锚点（任一存在即采用），保证 dev 与打包两种布局都能命中权重：
      1. 安装根目录（dev/repo 布局：``<root>/models/weights/...``）
      2. backend 包根目录（Tauri 打包布局：``<root>/backend/models/weights/...``）
    """
    if os.path.isabs(uri):
        return uri
    for anchor in (_INSTALL_ROOT, _BACKEND_ROOT):
        candidate = anchor / uri
        if candidate.exists():
            return str(candidate)
    return uri


class LocalModelStore:
    """registry 中的模型状态容器（单例，经 backend.app.dependencies 访问）。"""

    def __init__(self, default_uri: str, backend: str = "onnx") -> None:
        self.default_uri = default_uri
        self.backend = backend
        self.active_version: str | None = None
        self._hash: str | None = None

    def load(self, model_uri: str | None = None) -> None:
        """加载权重并记录版本（M1 桩：仅当文件存在时计算哈希）。"""
        uri = _resolve_model_uri(model_uri or self.default_uri)
        path = Path(uri)
        if path.is_file():
            # 分块摘要：原实现 read_bytes() 会把整份权重读进内存（大模型下是数百 MB 峰值）
            digest = hashlib.sha256()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
            self._hash = digest.hexdigest()[:12]
            self.active_version = f"{path.stem}::{self._hash}"
        else:
            self._hash = None
            self.active_version = f"{path.stem}::stub"

    @property
    def status(self) -> dict:
        return {
            "uri": self.default_uri,
            "backend": self.backend,
            "active_version": self.active_version,
        }
