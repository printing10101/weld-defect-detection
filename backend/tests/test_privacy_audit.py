"""脱敏残留审计测试（C-13）：DICONDE PHI / EXIF 残留扫描 + JSON/PDF 报告。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.infra.privacy_audit import audit_directory_phi, build_phi_audit_pdf


def _jpg_with_exif(path: Path) -> None:
    from PIL import Image

    img = Image.new("L", (8, 8), 128)
    exif = Image.Exif()
    exif[0x010F] = "TestCamera"  # Make（厂商）——典型敏感残留
    img.save(path, exif=exif.tobytes())


def test_audit_directory_flags_exif_residue(tmp_path: Path) -> None:
    _jpg_with_exif(tmp_path / "with_exif.jpg")
    clean = tmp_path / "clean.png"
    from PIL import Image

    Image.new("L", (8, 8), 100).save(clean)
    report = audit_directory_phi(tmp_path)
    assert report["scanned"] == 2
    assert report["n_findings"] == 1
    assert not report["clean"]
    finding = report["findings"][0]
    assert finding["kind"] == "exif"
    assert "Make" in finding["residues"]


def test_audit_directory_clean_when_no_metadata(tmp_path: Path) -> None:
    from PIL import Image

    Image.new("L", (8, 8), 100).save(tmp_path / "a.png")
    report = audit_directory_phi(tmp_path)
    assert report["scanned"] == 1
    assert report["clean"] is True
    assert report["findings"] == []


def test_audit_report_files_written(tmp_path: Path) -> None:
    _jpg_with_exif(tmp_path / "x.jpg")
    report = audit_directory_phi(tmp_path)
    from backend.infra.privacy_audit import write_audit_report

    paths = write_audit_report(report, tmp_path / "out")
    assert Path(paths["json"]).is_file()
    assert Path(paths["pdf"]).is_file()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["n_findings"] == 1
    assert Path(paths["pdf"]).stat().st_size > 0


def test_audit_pdf_smoke(tmp_path: Path) -> None:
    report = {
        "generated_at": "2026-01-01 00:00:00",
        "directory": str(tmp_path),
        "scanned": 3,
        "n_findings": 1,
        "clean": False,
        "findings": [{"file": "a.jpg", "kind": "exif", "residues": ["Make"]}],
        "errors": [],
    }
    out = build_phi_audit_pdf(report, tmp_path / "r.pdf")
    assert out.is_file() and out.stat().st_size > 0


def test_privacy_audit_endpoint(tmp_path: Path, monkeypatch) -> None:
    """端点：指定目录扫描 → JSON+PDF 报告落盘 + 审计留痕。"""
    _jpg_with_exif(tmp_path / "leak.jpg")
    with TestClient(app) as c:
        resp = c.post("/api/v1/privacy/audit", json={"directory": str(tmp_path)})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["scanned"] == 1
        assert body["n_findings"] == 1
        assert Path(body["report_files"]["json"]).is_file()
        assert Path(body["report_files"]["pdf"]).is_file()
        # 审计留痕
        entries = c.get("/api/v1/audit?action=privacy_audit").json()["entries"]
        assert entries


@pytest.mark.parametrize("kind", ["dicom_phi", "exif"])
def test_audit_kind_classification(kind: str, tmp_path: Path) -> None:
    """残留来源分类标注正确（exif/dicom_phi）。"""
    if kind == "exif":
        _jpg_with_exif(tmp_path / "a.jpg")
    else:
        pytest.importorskip("pydicom")
        from pydicom.dataset import Dataset
        from pydicom.uid import generate_uid

        ds = Dataset()
        ds.PatientName = "张三"  # 患者隐私标签（PHI 残留）
        ds.is_little_endian = True
        ds.is_implicit_VR = True
        ds.file_meta = Dataset()
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.file_meta.TransferSyntaxUID = "1.2.840.10008.1.2"
        ds.save_as(tmp_path / "b.dcm", write_like_original=False)
    report = audit_directory_phi(tmp_path)
    assert report["findings"], "应发现残留"
    assert report["findings"][0]["kind"] == kind
