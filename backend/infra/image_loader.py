"""影像接入：DICOM / 通用图像 → ndarray + ImageMeta。

属于基础设施层（I/O）：只做解码与元数据抽取，不含业务判定。
输出约定：uint8 单通道灰度（供算法层使用）；ImageMeta 携带 modality / pixel_spacing。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from backend.domain.dto import ImageMeta, Modality
from backend.domain.errors import ImageUnreadableError

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
    # 解压炸弹预检：多帧 DICOM 在 pixel_array 才实际展开，先按声明尺寸拦截。
    _check_declared_size(
        int(getattr(ds, "Columns", 0) or 0),
        int(getattr(ds, "Rows", 0) or 0),
        int(getattr(ds, "NumberOfFrames", 1) or 1),
        source=f"(DICOM {p.name})",
    )
    stored = ds.pixel_array
    gray, raw_frame = _dicom_frame(stored, ds)
    # 保留原始存储像素（未做 rescale/归一化）供黑度测量，
    # 否则逐图 min-max 拉伸会抹掉黑度这一绝对量。
    bits = int(getattr(ds, "BitsStored", None) or (16 if stored.dtype == np.uint16 else 8))
    return gray, ImageMeta(
        modality=Modality.DICOM,
        pixel_spacing_mm=_pixel_spacing(ds),
        bit_depth=bits,
        density_array=np.asarray(raw_frame) if raw_frame is not None else None,
    )


def _dicom_frame(stored: np.ndarray, ds) -> tuple[np.ndarray, np.ndarray | None]:
    """DICOM 存储像素 → (单帧 2D uint8 灰阶, 原始存储帧或 None)。

    含 Rescale/Slope-Intercept、MONOCHROME1 反转与多帧/彩色选帧。
    """
    arr = stored.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + inter
    # MONOCHROME1：值越低代表透过越少（底片黑度越高），反转以统一"高值=亮"
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).strip() == "MONOCHROME1":
        arr = arr.max() - arr
    # 多帧/彩色 DICOM按 SamplesPerPixel / NumberOfFrames 判定，
    # 不依赖最后维是否为 3/4（否则宽=3/4 的单通道多帧会被误判为彩色）。
    spp = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    nframes = int(getattr(ds, "NumberOfFrames", 1) or 1)
    frame, raw_frame = _select_dicom_frame(arr, stored, samples_per_pixel=spp, num_frames=nframes)
    return _to_uint8(frame), raw_frame


def _select_dicom_frame(
    arr: np.ndarray,
    stored: np.ndarray,
    samples_per_pixel: int = 1,
    num_frames: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """从 DICOM pixel_array 选出单帧 2D 灰阶与其原始像素（供黑度测量）。

    判定依据 SamplesPerPixel / NumberOfFrames（而非末维是否为 3/4，避免宽=3/4 的
    单通道多帧被误判为彩色）：
    - 单通道单帧 (H,W)：直接返回；
    - 单通道多帧 (F,H,W)：选空间标准差最大帧（最清晰/对比最强，利于 IQI 与评片）；
    - 彩色（无论单/多帧）：转灰度，密度回退灰阶（不做彩色黑度）。
    返回 (gray2d, raw_frame_or_None)。
    """
    if samples_per_pixel == 3:
        if num_frames > 1:  # 多帧彩色 (F,H,W,C)
            idx = int(np.argmax(arr.std(axis=(1, 2, 3))))
            return _color_to_gray(arr[idx]), None
        return _color_to_gray(arr), None  # 彩色 2D (H,W,C)
    if num_frames > 1:  # 单通道多帧 (F,H,W)
        idx = int(np.argmax(arr.std(axis=(1, 2))))
        return arr[idx], stored[idx]
    return arr, stored  # 单通道单帧 (H,W)


def _color_to_gray(arr: np.ndarray) -> np.ndarray:
    """RGB→灰阶（DICOM RGB 为 sRGB；亮度加权，不依赖 cv2 颜色空间假设）。"""
    if arr.dtype != np.uint8:
        arr = _to_uint8(arr)
    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    b = arr[..., 2].astype(np.float32)
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


def _pixel_spacing(ds) -> float | None:
    ps = getattr(ds, "PixelSpacing", None)
    if ps is None or len(ps) < 1 or not ps[0]:
        return None
    try:
        return float(ps[0])
    except (TypeError, ValueError):
        return None


def _load_generic(p: Path, mode: Modality) -> tuple[np.ndarray, ImageMeta]:
    # IMREAD_ANYDEPTH：保留 16bit 原始位深供黑度测量；cv2.imread 在 Windows 上对
    # 非 ASCII 路径会失败，故走 np.fromfile + imdecode 的 unicode 安全路径。
    img = _imread_unicode(p, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    if img is None:
        # cv2 不支持的格式（GIF/HEIC/AVIF/PNM 等）经 imageio / Pillow 解码回退。
        img = _imread_fallback(p)
    if img is None:
        raise ImageUnreadableError(f"无法解码图像: {p.name}")
    is_16 = img.dtype == np.uint16
    gray = img if not is_16 else _to_uint8(img)
    return gray, ImageMeta(
        modality=mode,
        bit_depth=16 if is_16 else 8,
        density_array=img if is_16 else None,
    )


def _imread_unicode(p: Path, flags: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray | None:
    """Unicode 安全解码：cv2.imread 在 Windows 上对非 ASCII 路径返回 None。

    解码前按声明尺寸（PNG IHDR / JPEG SOF 头）做像素总量预检——上传侧仅有
    压缩后体积上限，恶意构造的稀疏压缩图可声明数万像素宽高，解码即 OOM
    （解压炸弹）。预检不通过的文件抛 ImageUnreadableError。
    """
    buf = np.fromfile(str(p), dtype=np.uint8)
    if buf.size == 0:
        return None
    dims = _declared_dimensions(buf.tobytes())
    if dims is not None:
        _check_declared_size(*dims, source=f"({p.name})")
    return cv2.imdecode(buf, flags)


# 解压炸弹防线：像素总量上限（8K 底片约 48MP，200MP 已远超业务上限）与帧数上限。
_MAX_PIXELS = 200_000_000
_MAX_FRAMES = 64


def _check_declared_size(width: int, height: int, frames: int = 1, source: str = "") -> None:
    if width <= 0 or height <= 0:
        return
    if frames > _MAX_FRAMES:
        raise ImageUnreadableError(f"图像帧数超限（{frames} > {_MAX_FRAMES}）{source}")
    if width * height * max(1, frames) > _MAX_PIXELS:
        raise ImageUnreadableError(
            f"图像像素总量超限（{width}x{height}x{frames} > {_MAX_PIXELS}）{source}"
        )


def _declared_dimensions(buf: bytes) -> tuple[int, int, int] | None:
    """从文件头解析声明尺寸 (w, h, frames)；非 PNG/JPEG 返回 None（不预检）。"""
    if buf[:8] == b"\x89PNG\r\n\x1a\n" and buf[12:16] == b"IHDR" and len(buf) >= 24:
        return int.from_bytes(buf[16:20], "big"), int.from_bytes(buf[20:24], "big"), 1
    if buf[:2] == b"\xff\xd8":
        i, n = 2, len(buf)
        while i + 9 < n:
            if buf[i] != 0xFF:
                i += 1
                continue
            marker = buf[i + 1]
            if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h = int.from_bytes(buf[i + 5 : i + 7], "big")
                w = int.from_bytes(buf[i + 7 : i + 9], "big")
                return w, h, 1
            i += 2 + int.from_bytes(buf[i + 2 : i + 4], "big")
    return None


def _imread_fallback(p: Path) -> np.ndarray | None:
    """cv2 解码失败时的回退链：imageio（GIF/PNM/罕见 PNG/JPEG）→ Pillow+HEIF。

    多帧影像（动图 GIF / 多页）选取对比度最强的一帧，与 DICOM 多帧策略一致。
    返回 2D uint8/uint16 数组；全部解码器失败返回 None（由调用方抛可读错误）。
    """
    for decoder in (_decode_imageio, _decode_pillow):
        try:
            arr = decoder(p)
        except Exception:  # noqa: BLE001, S112 — 单个解码器失败继续尝试下一个
            continue
        if arr is None:
            continue
        arr = _pick_sharpest_frame(np.asarray(arr))
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = _rgb_to_gray(arr)
        if arr.ndim == 2:
            return arr
    return None


def _decode_imageio(p: Path) -> np.ndarray | None:
    import imageio.v3 as iio

    return iio.imread(str(p))


def _decode_pillow(p: Path) -> np.ndarray | None:
    from PIL import Image

    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pillow_heif_available = False
    else:
        pillow_heif_available = True

    with Image.open(p) as im:
        if pillow_heif_available:
            im.load()
        return np.asarray(im)


def _pick_sharpest_frame(arr: np.ndarray) -> np.ndarray:
    """(F,...) 多帧输入选空间标准差最大的帧（对比最强），单帧原样返回。"""
    if arr.ndim >= 3 and not (arr.ndim == 3 and arr.shape[-1] in (3, 4)):
        return arr[int(np.argmax(arr.std(axis=tuple(range(1, arr.ndim)))))]
    return arr


def _rgb_to_gray(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return arr.astype(np.uint8)


def read_gray(image_path: str) -> np.ndarray | None:
    """以 unicode 安全方式读取灰度图（cv2.imread 在中文路径上会失败）。

    静态加密兼容：影像副本可能为国密密文（SM4，魔数 b"SDC2"）或历史
    AES-256-GCM 密文（b"SDC1"），检测到任一魔数则解密后再解码；明文旧
    数据直接解码。密钥缺失/解密失败时返回 None（调用方降级，不抛 500）。
    DICOM 副本（.dcm/.dicom/.ima 或 DICM 魔数）走 pydicom 解码——否则
    cv2.imdecode 对 DICOM 返回 None，复核重出报告会静默丢附图。
    """
    try:
        with open(image_path, "rb") as fh:
            buf = fh.read()
    except OSError:
        return None
    if not buf:
        return None
    # C-01 国密化：新副本 SDC2（SM4-CTR+HMAC-SM3），存量副本 SDC1（AES-GCM）
    if buf.startswith((b"SDC2", b"SDC1")):
        try:
            from backend.infra.crypto import default_crypto_provider

            buf = default_crypto_provider().decrypt(buf)
        except Exception:  # noqa: BLE001  # 解密失败（密钥不符/密文损坏）→ 读图降级
            return None
    is_dicom = buf[128:132] == b"DICM" or Path(image_path).suffix.lower() in (
        ".dcm",
        ".dicom",
        ".ima",
    )
    if is_dicom:
        try:
            import io as _io

            import pydicom

            ds = pydicom.dcmread(_io.BytesIO(buf))
            stored = ds.pixel_array
            gray, _raw = _dicom_frame(stored, ds)
            return gray
        except Exception:  # noqa: BLE001 - DICOM 解码失败 → 读图降级
            return None
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
