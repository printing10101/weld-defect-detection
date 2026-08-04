"""影像接入（§4.1）：DICOM / 通用图像 → ndarray + ImageMeta。

属于基础设施层（I/O）：只做解码与元数据抽取，不含业务判定。
输出约定：uint8 单通道灰度（供算法层使用）；ImageMeta 携带 modality / pixel_spacing。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from backend.domain.dto import ImageMeta, Modality

_DICOM_SUFFIXES = {".dcm", ".dicom"}


def load_image(path: str | Path, modality: str | None = None) -> tuple[np.ndarray, ImageMeta]:
    """读取影像并返回 (uint8 灰度图, 元数据)。modality 未指定时按扩展名推断。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {p}")
    mode = (modality or _detect_modality(p)).upper()
    if mode == Modality.DICOM.value or p.suffix.lower() in _DICOM_SUFFIXES:
        return _load_dicom(p)
    return _load_generic(p, Modality(mode) if _is_known(mode) else Modality.GENERIC)


def _detect_modality(p: Path) -> str:
    return Modality.DICOM.value if p.suffix.lower() in _DICOM_SUFFIXES else Modality.GENERIC.value


def _is_known(mode: str) -> bool:
    return mode in {m.value for m in Modality}


def _load_dicom(p: Path) -> tuple[np.ndarray, ImageMeta]:
    import pydicom

    ds = pydicom.dcmread(str(p))
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + inter
    # MONOCHROME1：值越低代表透过越少（底片黑度越高），反转以统一"高值=亮"
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).strip() == "MONOCHROME1":
        arr = arr.max() - arr
    gray = _to_uint8(arr)
    return gray, ImageMeta(modality=Modality.DICOM, pixel_spacing_mm=_pixel_spacing(ds))


def _pixel_spacing(ds) -> float | None:
    ps = getattr(ds, "PixelSpacing", None)
    if ps is None or len(ps) < 1 or not ps[0]:
        return None
    try:
        return float(ps[0])
    except (TypeError, ValueError):
        return None


def _load_generic(p: Path, mode: Modality) -> tuple[np.ndarray, ImageMeta]:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot decode image: {p}")
    return img, ImageMeta(modality=mode)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr.astype(np.uint8)
