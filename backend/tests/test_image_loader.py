"""影像接入测试：通用图像 + 最小 DICOM。"""

from __future__ import annotations

import cv2
import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import UID, generate_uid

from backend.domain.dto import Modality
from backend.infra.image_loader import _select_dicom_frame, load_image


def _write_min_dicom(path, shape=(16, 16), spacing=(0.1, 0.1)) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID("1.2.840.10008.5.1.4.1.1.7")
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = UID("1.2.840.10008.1.2.1")
    ds = FileDataset(str(path), Dataset(), file_meta=file_meta, preamble=b"\x00" * 128)
    ds.PixelData = np.zeros(shape, np.uint16).tobytes()
    ds.Rows, ds.Columns = shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    ds.PixelSpacing = list(spacing)
    ds.SOPClassUID = UID("1.2.840.10008.5.1.4.1.1.7")
    ds.SOPInstanceUID = generate_uid()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(path))


def _write_multiframe_dicom(path, frames, shape=(16, 16), spacing=(0.1, 0.1)) -> None:
    """写多帧单通道 DICOM。frames: list[np.ndarray] 同 shape。"""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID("1.2.840.10008.5.1.4.1.1.7")
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = UID("1.2.840.10008.1.2.1")
    ds = FileDataset(str(path), Dataset(), file_meta=file_meta, preamble=b"\x00" * 128)
    arr = np.stack(frames).astype(np.uint16)
    ds.PixelData = arr.tobytes()
    ds.NumberOfFrames = len(frames)
    ds.Rows, ds.Columns = shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    ds.PixelSpacing = list(spacing)
    ds.SOPClassUID = UID("1.2.840.10008.5.1.4.1.1.7")
    ds.SOPInstanceUID = generate_uid()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(path))


def test_generic_png(tmp_path) -> None:
    p = tmp_path / "img.png"
    cv2.imwrite(str(p), np.full((32, 48), 128, np.uint8))
    gray, meta = load_image(p)
    assert gray.dtype == np.uint8
    assert gray.shape == (32, 48)
    assert meta.modality is Modality.GENERIC
    assert meta.pixel_spacing_mm is None


def test_dicom_loads_with_spacing(tmp_path) -> None:
    p = tmp_path / "img.dcm"
    _write_min_dicom(p)
    gray, meta = load_image(p)
    assert gray.shape == (16, 16)
    assert gray.dtype == np.uint8
    assert meta.modality is Modality.DICOM
    assert meta.pixel_spacing_mm == 0.1


def test_missing_file_raises(tmp_path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.png")


def test_multiframe_dicom_selects_best_frame(tmp_path) -> None:
    """多帧 DICOM应降为单帧 2D 返回，且选中对比最强（标准差最大）帧。"""
    flat = np.full((16, 16), 100, np.uint16)  # 低对比
    noisy = (np.random.default_rng(1).normal(100, 40, (16, 16))).astype(np.uint16)  # 高对比
    p = tmp_path / "mf.dcm"
    _write_multiframe_dicom(p, [flat, noisy, flat])
    gray, meta = load_image(p)
    assert gray.ndim == 2
    assert gray.shape == (16, 16)
    assert meta.modality is Modality.DICOM
    # 选中的应是高对比帧（其像素分布覆盖更广）
    assert float(gray.std()) > 20.0


def test_select_dicom_frame_passthrough_2d() -> None:
    arr = np.zeros((8, 8), np.float32)
    out, raw = _select_dicom_frame(arr, arr.astype(np.uint16))
    assert out.shape == (8, 8)
    assert raw is not None


def test_select_dicom_frame_picks_max_std() -> None:
    a = np.zeros((3, 4, 4))
    a[0] += 1.0  # 帧0：常量（std=0）
    a[1] += 50.0 + np.random.default_rng(2).normal(0, 5, (4, 4))  # 帧1：含噪声→std 最大（应选）
    a[2] += 2.0  # 帧2：常量（std=0）
    out, _ = _select_dicom_frame(a, a.astype(np.uint16), num_frames=3)
    assert out.shape == (4, 4)
    assert float(out.mean()) > 40.0  # 选中帧1（均值≈50）


def test_select_dicom_frame_color_returns_gray_and_none() -> None:
    rgb = np.zeros((4, 4, 3), np.float32)
    rgb[..., 0] = 30.0
    out, raw = _select_dicom_frame(rgb, rgb.astype(np.uint8), samples_per_pixel=3)
    assert out.ndim == 2
    assert raw is None  # 彩色不提供密度原始像素（密度回退灰阶）
