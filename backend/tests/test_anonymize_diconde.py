"""DICONDE 元数据 + 数据脱敏（§8.3.2）单测。"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydicom import dcmwrite
from pydicom.dataset import Dataset, FileMetaDataset

from backend.app.main import app
from backend.infra.diconde import parse_diconde
from backend.training.anonymize_images import (
    anonymize_tree,
    audit_directory,
    strip_dicom_phi,
    strip_jpeg_metadata,
)
from backend.training.anonymize_images import (
    main as cli_main,
)


def _dicom_bytes() -> bytes:
    """带透照工艺字段与患者隐私字段的合成 DICONDE。"""
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    # FileMetaDataset 的属性为动态生成，pyright 静态解析不了，走 setattr
    for attr, value in {
        "MediaStorageSOPClassUID": "1.2.840.10008.5.1.4.1.1.1.1",
        "MediaStorageSOPInstanceUID": "1.2.3.4",
        "TransferSyntaxUID": "1.2.840.10008.1.2",  # Implicit VR LE
    }.items():
        setattr(ds.file_meta, attr, value)
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "DX"
    ds.KVP = 200
    ds.Manufacturer = "TestRT"
    ds.AcquisitionDate = "20260829"
    ds.PatientName = "ZHANG^SAN"
    ds.PatientID = "P-001"
    ds.PatientBirthDate = "19800101"
    buf = io.BytesIO()
    dcmwrite(buf, ds, enforce_file_format=True)
    return buf.getvalue()


def _jpeg_with_exif() -> bytes:
    """合成 JPEG 并插入 APP1(Exif) 与 COM 段。"""
    ok, raw = cv2.imencode(".jpg", np.full((32, 32), 100, np.uint8))
    assert ok
    jpg = raw.tobytes()
    exif_payload = b"Exif\x00\x00" + b"GPSInfo..."
    seg = b"\xff\xe1" + (len(exif_payload) + 2).to_bytes(2, "big") + exif_payload
    com = b"\xff\xfe" + (len(b"shot by cam-001") + 2).to_bytes(2, "big") + b"shot by cam-001"
    return jpg[:2] + seg + com + jpg[2:]


# ---------------------------------------------------------------------------
# DICONDE 解析
# ---------------------------------------------------------------------------


def test_parse_diconde_technique_and_phi():
    meta = parse_diconde(_dicom_bytes())
    assert float(meta["technique"]["kvp"]) == 200
    assert meta["technique"]["manufacturer"] == "TestRT"
    assert meta["phi_present"] is True
    assert meta["phi"]["patient_name"] == "ZHANG^SAN"


def test_parse_diconde_rejects_non_dicom():
    with pytest.raises(ValueError, match="非 DICOM"):
        parse_diconde(b"\x89PNG not a dicom at all......")


def test_parse_diconde_clean_file_has_no_phi():
    from pydicom import dcmread

    data = _dicom_bytes()
    stripped, removed = strip_dicom_phi(data)
    assert set(removed) >= {"PatientName", "PatientID", "PatientBirthDate"}
    ds = dcmread(io.BytesIO(stripped))
    assert "PatientName" not in ds
    assert ds.KVP == 200  # 工艺字段保留


# ---------------------------------------------------------------------------
# 脱敏工具
# ---------------------------------------------------------------------------


def test_strip_jpeg_metadata_removes_exif_and_com():
    jpg = _jpeg_with_exif()
    assert b"Exif\x00\x00" in jpg
    out, removed = strip_jpeg_metadata(jpg)
    assert "APP1(Exif/XMP)" in removed and "COM" in removed
    assert b"Exif\x00\x00" not in out
    assert b"cam-001" not in out
    # 解码后像素仍在（SOS 数据段未动）
    img = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape == (32, 32, 3)


def test_strip_jpeg_passthrough_plain():
    ok, raw = cv2.imencode(".jpg", np.full((16, 16), 100, np.uint8))
    assert ok
    out, removed = strip_jpeg_metadata(raw.tobytes())
    assert removed == [] and out == raw.tobytes()


def test_anonymize_tree_end_to_end(tmp_path: Path):
    src = tmp_path / "src"
    (src / "films").mkdir(parents=True)
    (src / "films" / "a.dcm").write_bytes(_dicom_bytes())
    (src / "films" / "b.jpg").write_bytes(_jpeg_with_exif())
    (src / "films" / "c.png").write_bytes(
        np.full((8, 8), 50, np.uint8).tobytes()
    )  # 非法 png 也会被复制
    assert audit_directory(src), "源目录应检出隐私残留"

    dst = tmp_path / "dst"
    report = anonymize_tree(src, dst)
    assert len(report) == 3
    assert audit_directory(dst) == []  # 脱敏后无残留
    assert not audit_file_still_has_phi(dst / "films" / "a.dcm")


def audit_file_still_has_phi(p: Path) -> bool:
    from backend.infra.diconde import audit_dicom_phi

    return bool(audit_dicom_phi(p))


def test_cli_audit_only_exit_codes(tmp_path: Path, capsys):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.jpg").write_bytes(_jpeg_with_exif())
    assert cli_main(["--src", str(src), "--audit-only"]) == 1  # 有残留
    dst = tmp_path / "dst"
    assert cli_main(["--src", str(src), "--dst", str(dst)]) == 0  # 脱敏后干净
    assert cli_main(["--src", str(dst), "--audit-only"]) == 0


def test_cli_requires_dst_or_audit(tmp_path: Path):
    with pytest.raises(SystemExit):
        cli_main(["--src", str(tmp_path)])


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


def test_diconde_route(tmp_path: Path):
    dcm_path = tmp_path / "film.dcm"
    dcm_path.write_bytes(_dicom_bytes())
    png_path = tmp_path / "plain.png"
    cv2.imwrite(str(png_path), np.full((8, 8), 100, np.uint8))

    with TestClient(app) as client:
        from backend.app.dependencies import get_registry

        repo = get_registry().repository
        for iid, p in (("dcm-1", dcm_path), ("png-1", png_path)):
            repo.create_inspection(
                {"id": iid, "path": str(p), "source_type": "dicom", "modality": "DICOM"}, [], None
            )
        r = client.get("/api/v1/images/dcm-1/diconde")
        assert r.status_code == 200, r.text
        body = r.json()
        assert float(body["technique"]["kvp"]) == 200
        assert body["phi_present"] is True
        r2 = client.get("/api/v1/images/png-1/diconde")
        assert r2.status_code == 422
        r3 = client.get("/api/v1/images/nope/diconde")
        assert r3.status_code == 404
