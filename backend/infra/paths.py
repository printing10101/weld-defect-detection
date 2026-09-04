"""安装根/包根锚点与路径解析的唯一实现。

口径：相对配置路径一律解析为 `<安装根>/<p>`（Tauri 打包布局下安装根
即应用安装目录），与启动方式的 CWD 解耦；模型路径额外尝试 backend
包根锚点（打包时模型随 backend 资源分发）。`resolve_config_path`
仍由 infra.config 对外，内部锚点统一取自本模块。
"""

from __future__ import annotations

import os
from pathlib import Path

# 安装根目录锚点：backend/infra/paths.py -> parents[2] = 安装根目录。
# dev/repo 布局 = 仓库根；Tauri 打包布局 = 应用安装目录（main.rs 以此为 CWD）。
INSTALL_ROOT = Path(__file__).resolve().parents[2]
# backend 包根目录：模型/资源随 ``backend`` 一起分发的备选锚点。
BACKEND_ROOT = Path(__file__).resolve().parents[1]


def resolve_model_uri(uri: str) -> str:
    """将配置中的相对模型路径解析为绝对路径（双锚点，dev/打包布局通吃）。

    依次尝试（任一存在即采用）：
      1. 安装根目录（dev 布局：``<root>/models/weights/...``）
      2. backend 包根目录（打包布局：``<root>/backend/models/weights/...``）
    绝对路径原样返回；两锚点均未命中时返回锚定安装根的路径——调用方
    报错时能给出确定的期望位置，而不是随 CWD 漂移的相对路径。
    """
    if os.path.isabs(uri):
        return uri
    for anchor in (INSTALL_ROOT, BACKEND_ROOT):
        candidate = anchor / uri
        if candidate.exists():
            return str(candidate)
    return str(INSTALL_ROOT / uri)
