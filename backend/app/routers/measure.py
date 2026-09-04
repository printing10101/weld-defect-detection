"""图像质量测量 API（DB50/T 1807-2025 ）。

POST /api/v1/measure/snr          —— 归一化信噪比 SNRn（GB/T 26141.1 口径）
POST /api/v1/measure/duplex-wire  —— 双丝像质计空间分辨率（ISO 17636-2 双丝法）
上传底片（或底片 ROI 裁剪）+ 标定参数，返回测量结果与合格判定。
"""

from __future__ import annotations

from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.domain.measure.image_quality import measure_duplex_wire, measure_snr

router = APIRouter(tags=["measure"])


def _checked_bytes(image: UploadFile) -> bytes:
    """读取上传内容并强制大小限额（与 _common.staged_upload 同口径）。

    无限额整读时单请求即可打爆内存，且会绕过统一上传限额。
    """
    from backend.infra.config import load_config

    max_bytes = load_config().upload.max_bytes
    size = image.size
    if size is not None and size > max_bytes:
        raise HTTPException(
            413,
            detail={"code": "FILE_TOO_LARGE", "message": f"文件超过大小上限 {max_bytes} 字节"},
        )
    content = image.file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            413,
            detail={"code": "FILE_TOO_LARGE", "message": f"文件超过大小上限 {max_bytes} 字节"},
        )
    return content


def _decode(gray_depth: bool, content: bytes) -> np.ndarray:
    """字节流 → 灰度数组（np.fromfile 解码，规避 Windows 非 ASCII 路径问题）。"""
    arr = np.frombuffer(content, dtype=np.uint8)
    flags = cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH
    if not gray_depth:
        flags = cv2.IMREAD_COLOR
    img = cv2.imdecode(arr, flags)
    if img is None:
        raise HTTPException(422, detail={"code": "BAD_IMAGE", "message": "影像解码失败"})
    return img


def _parse_roi(roi: str | None, shape: tuple[int, ...]) -> tuple[slice, slice] | None:
    """ROI 字符串 "x,y,w,h" → (yslice, xslice)；缺省 None（整图）。"""
    if roi is None:
        return None
    parts = roi.split(",")
    if len(parts) != 4:
        raise HTTPException(422, detail={"code": "BAD_ROI", "message": 'roi 格式须为 "x,y,w,h"'})
    try:
        x, y, w, h = (int(v) for v in parts)
    except ValueError:
        raise HTTPException(422, detail={"code": "BAD_ROI", "message": "roi 须为整数"}) from None
    h_img, w_img = shape[:2]
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > w_img or y + h > h_img:
        raise HTTPException(422, detail={"code": "BAD_ROI", "message": "roi 超出图像范围"})
    return slice(y, y + h), slice(x, x + w)


@router.post("/measure/snr")
def snr_endpoint(
    image: Annotated[UploadFile, File()],
    srb_mm: Annotated[float | None, Form()] = None,
    pixel_spacing_mm: Annotated[float | None, Form()] = None,
    min_snrn: Annotated[float, Form()] = 130.0,
    roi: Annotated[str | None, Form()] = None,
) -> dict:
    """SNRn 测量。srb_mm 建议用双丝测量结果；缺省用像素尺寸保守估计（偏严）。"""
    content = _checked_bytes(image)
    img = _decode(gray_depth=True, content=content)
    win = _parse_roi(roi, img.shape)
    if win is not None:
        img = img[win[0], win[1]]
    try:
        result = measure_snr(
            img, srb_mm=srb_mm, pixel_spacing_mm=pixel_spacing_mm, min_snrn=min_snrn
        )
    except ValueError as e:
        raise HTTPException(422, detail={"code": "MEASURE_FAIL", "message": str(e)}) from e
    return result.to_dict()


@router.post("/measure/duplex-wire")
def duplex_wire_endpoint(
    image: Annotated[UploadFile, File()],
    pixel_spacing_mm: Annotated[float, Form()],
    wire_axis_deg: Annotated[float, Form()] = 0.0,
    roi: Annotated[str | None, Form()] = None,
) -> dict:
    """双丝像质计空间分辨率：ROI 对准双丝组，wire_axis_deg 为丝方向角（度）。"""
    content = _checked_bytes(image)
    img = _decode(gray_depth=True, content=content)
    win = _parse_roi(roi, img.shape)
    if win is not None:
        img = img[win[0], win[1]]
    try:
        result = measure_duplex_wire(
            img, pixel_spacing_mm=pixel_spacing_mm, wire_axis_deg=wire_axis_deg
        )
    except ValueError as e:
        raise HTTPException(422, detail={"code": "MEASURE_FAIL", "message": str(e)}) from e
    return result.to_dict()
