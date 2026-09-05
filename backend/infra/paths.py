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

# 用户数据目录覆盖（Tauri 打包版）：壳经环境变量告知"用户数据目录"（Windows
# = %APPDATA%/<identifier>），data/ 前缀的相对路径改锚到 <该目录>/data/...。
# 动机：NSIS 卸载会清空安装目录，数据放 <安装目录>/data 意味着卸载即无声
# 删除全部检查记录/报告/影像副本；放用户数据目录则卸载保留（主密钥
# .crypto_key 同随，存量密文仍可解）。仅 data/ 前缀受影响——配置/权重等
# 程序资产仍锚定安装根。
_DATA_DIR_ENV = "SCANDETECTION_USER_DATA_DIR"


def data_dir_override() -> Path | None:
    """返回壳指定的用户数据目录；未设置（开发布局）返回 None。"""
    raw = os.environ.get(_DATA_DIR_ENV, "").strip()
    if not raw:
        return None
    return Path(raw)


def resolve_data_path(rel: str | Path) -> Path:
    """解析 data/ 前缀的相对路径：有覆盖时落到 <覆盖目录>/<rel>，否则锚定安装根。

    与 config.resolve_config_path 同语义，但供 infra 内部模块（crypto 等）
    使用而不引入 config 依赖。绝对路径原样返回。保留完整相对路径（含 data
    段）——卸载钩子按同名目录结构备份旧数据，新旧布局可无缝衔接。
    """
    p = Path(rel)
    if p.is_absolute():
        return p
    override = data_dir_override()
    if override is not None and p.parts and p.parts[0] == "data":
        return override.joinpath(*p.parts)
    return INSTALL_ROOT / p


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
