"""影像接入测试（§4.1）：通用图像 + 最小 DICOM。"""
from __future__ import annotations

import cv2
import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import UID, generate_uid

from backend.domain.dto import Modality
from backend.infra.image_loader import load_image


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
