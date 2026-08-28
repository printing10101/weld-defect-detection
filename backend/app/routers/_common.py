"""路由公共工具。

集中三件此前散落在各路由的事：
1. multipart 上传暂存（含大小/扩展名限额与临时目录清理）；
2. ROI 表单解析（非法输入显式 422，而非静默忽略）；
3. 未实现占位响应。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import NoReturn

from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.infra.config import AppConfig
from backend.infra.fs import secure_temp_dir

_CHUNK = 1 << 20  # 1 MiB 分块读取，避免整文件驻留内存


def not_implemented(stage: str) -> JSONResponse:
    """ 骨架期占位：返回 501 并标注计划实现里程碑。"""
    return JSONResponse(
        {
            "error": {
                "code": "NOT_IMPLEMENTED",
                "message": f"planned in {stage}",
                "detail": None,
            }
        },
        status_code=501,
    )


def _reject(status: int, code: str, message: str) -> NoReturn:
    raise HTTPException(status_code=status, detail={"code": code, "message": message})


@asynccontextmanager
async def staged_upload(upload: UploadFile, cfg: AppConfig) -> AsyncIterator[Path]:
    """把上传文件落到受控临时目录并产出路径，退出时连目录一并删除。

    相较各路由原先的 `write_bytes(await image.read)`：
    - 分块写盘 + 累计计数，超过 upload.max_bytes 立即 413（原实现无上限，单请求可打爆内存）；
    - 扩展名白名单，非影像类型 415（原实现任意后缀均落盘）；
    - 临时目录随上下文退出清理（原实现只删文件，目录持续泄漏）。
    """
    up = cfg.upload
    suffix = Path(upload.filename or "upload.png").suffix.lower() or ".png"
    if suffix not in tuple(up.allowed_suffixes):
        _reject(415, "UNSUPPORTED_MEDIA_TYPE", f"不支持的影像格式: {suffix}")

    base: Path | None = Path(cfg.paths.tmp_dir)
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        base = None  # 数据区不可写时回退系统临时目录，不阻断请求

    with secure_temp_dir(base) as tmp_dir:
        tmp_path = tmp_dir / f"upload{suffix}"
        size = 0
        with tmp_path.open("wb") as fh:
            while True:
                chunk = await upload.read(_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > up.max_bytes:
                    _reject(413, "PAYLOAD_TOO_LARGE", f"文件超过上限 {up.max_bytes} 字节")
                fh.write(chunk)
        if size == 0:
            _reject(422, "EMPTY_UPLOAD", "上传文件为空")
        yield tmp_path


def parse_roi(raw: str | None) -> tuple[int, int, int, int] | None:
    """解析 "x,y,w,h" 形式的 ROI；缺省返回 None（全图）。

    原实现 `int(v)` 未捕获异常（非数字直接 500），且四元组长度不符时静默返回 None
    —— 用户以为限定了 ROI，实际按全图计算，属静默错误结果。这里一律显式 422。
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        _reject(422, "INVALID_ROI", 'roi 需为 "x,y,w,h" 四个整数')
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError:
        _reject(422, "INVALID_ROI", "roi 各分量必须为整数")
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        _reject(422, "INVALID_ROI", "roi 需满足 x>=0, y>=0, w>0, h>0")
    return x, y, w, h
