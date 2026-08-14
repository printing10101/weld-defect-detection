"""安全文件访问（§13.9 / §T4）：临时目录 + 防路径穿越。"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def secure_temp_dir(base: Path | None = None) -> Iterator[Path]:
    """创建安全临时目录；退出时连同内容一并清理（防止上传请求泄漏临时目录）。

    base 指向配置 tmp_dir 时可把临时文件约束到应用数据区，否则沿用系统临时目录。
    """
    d = Path(tempfile.mkdtemp(prefix="scan_", dir=str(base) if base else None))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def safe_resolve(base: Path, name: str) -> Path:
    """将 name 解析到 base 之下，越界即抛错（防路径穿越）。

    空名 / '.' / '..' 直接拒绝，避免调用方 open() 到目录或越界。
    """
    if not name or name in (".", ".."):
        raise ValueError("empty or invalid name")
    resolved = (base / name).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError("path traversal blocked")
    return resolved
