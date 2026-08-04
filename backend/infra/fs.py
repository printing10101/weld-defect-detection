"""安全文件访问（§13.9 / §T4）：临时目录 + 防路径穿越。"""
from __future__ import annotations

import tempfile
from pathlib import Path


def secure_temp_dir() -> Path:
    """创建安全临时目录（勿用 /tmp 或用户目录直写）。"""
    return Path(tempfile.mkdtemp(prefix="scan_"))


def safe_resolve(base: Path, name: str) -> Path:
    """将 name 解析到 base 之下，越界即抛错（防路径穿越）。"""
    resolved = (base / name).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError("path traversal blocked")
    return resolved
