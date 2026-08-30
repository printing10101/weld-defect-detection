"""报告数字签名校验（POST /api/v1/report/{id}/verify）。

覆盖：正常报告指纹一致（valid=true）；DB 内容被篡改后校验失败（valid=false）；
无指纹的旧报告返回 legacy（valid=null）；SM2 签名（C-03）双结果：
签名落 sidecar 且验签通过 / sidecar 篡改检出 / 无签名旧报告 signature=null。
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.infra.crypto import AesCrypto

# SM2 签名/验签测试用主密钥（base64 32B）
_SIG_KEY = AesCrypto.generate_key()


@pytest.fixture(scope="module", autouse=True)
def _authorized_grader(auth_table) -> Iterator[None]:
    """与 test_report_api 同构：注入 authorized 表 + 放宽黑度 + 关质量门禁。"""
    from backend.app import dependencies as deps
    from backend.domain.grade.nb47013 import Nb47013Grader
    from backend.domain.standards.tables.loader import load_standard_tables

    deps._registry = None
    reg = deps.get_registry()
    reg.grader = Nb47013Grader(load_standard_tables("NB/T47013.2-2015", filename=str(auth_table)))
    orig_low = reg.config.density.low
    orig_block = reg.config.quality.block_on_quality
    reg.config.density.low = 0.0
    reg.config.quality.block_on_quality = False
    try:
        yield
    finally:
        reg.config.density.low = orig_low
        reg.config.quality.block_on_quality = orig_block
        deps._registry = None


def _make_film(tmp_path) -> str:
    n, h, w = 19, 190, 640
    rng = np.random.default_rng(0)
    img = rng.normal(128.0, 2.0, (h, w)).astype(np.uint8)
    for i in range(n):
        y = round((i + 0.5) / n * h)
        cv2.line(img, (0, y), (w - 1, y), int(128 + 40.0), 3)
    cv2.circle(img, (120, 30), 10, 80, -1)
    cv2.circle(img, (420, 150), 7, 85, -1)
    path = tmp_path / "film.png"
    cv2.imwrite(str(path), img)
    return str(path)


def _post_report(client: TestClient, path: str) -> dict:
    with open(path, "rb") as f:
        resp = client.post(
            "/api/v1/report",
            files={"image": ("film.png", f, "image/png")},
            data={"pixel_spacing_mm": "0.1", "base_metal_thickness_mm": "20", "force": "true"},
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_report_verify_valid(tmp_path) -> None:
    """生成报告 → 校验通过：valid=true、指纹 64 位、带签发人。"""
    path = _make_film(tmp_path)
    with TestClient(app) as client:
        report = _post_report(client, path)
        resp = client.post(f"/api/v1/report/{report['report_id']}/verify")
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is True
    assert body["hash"] and len(body["hash"]) == 64
    assert body["signer"] is not None
    assert body["reason"] is None


def test_report_verify_tamper_detected(tmp_path) -> None:
    """DB 内容被篡改（级别改写）→ 校验失败 valid=false + reason=mismatch。"""
    path = _make_film(tmp_path)
    with TestClient(app) as client:
        report = _post_report(client, path)
        image_id = report["image_id"]
        # 模拟篡改：把影像记录级别改写（fingerprint 覆盖 joint_level）
        from backend.app import dependencies as deps

        reg = deps.get_registry()
        from sqlalchemy.orm import Session

        from backend.infra.db import ImageRecord

        with Session(reg.repository._engine) as session, session.begin():
            rec = session.get(ImageRecord, image_id)
            assert rec is not None
            rec.joint_level = "I" if rec.joint_level != "I" else "II"
        resp = client.post(f"/api/v1/report/{report['report_id']}/verify")
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert body["reason"] == "mismatch"


def test_report_verify_unknown_404() -> None:
    with TestClient(app) as client:
        resp = client.post("/api/v1/report/nope/verify")
    assert resp.status_code == 404


def test_report_verify_legacy_when_no_hash(tmp_path) -> None:
    """无指纹的旧报告 → valid=null + reason=legacy（不误判为篡改）。"""
    path = _make_film(tmp_path)
    with TestClient(app) as client:
        report = _post_report(client, path)
        from backend.app import dependencies as deps

        reg = deps.get_registry()
        from sqlalchemy.orm import Session

        from backend.infra.db import ReportRecord

        with Session(reg.repository._engine) as session, session.begin():
            rec = session.get(ReportRecord, report["report_id"])
            assert rec is not None
            rec.report_hash = None
            rec.signed_at = None
        resp = client.post(f"/api/v1/report/{report['report_id']}/verify")
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is None
    assert body["reason"] == "legacy"


# ---------------------------------------------------------------------------
# SM2 签名双结果（C-03）
# ---------------------------------------------------------------------------


def _signature_sidecar(report: dict) -> Path | None:
    """定位报告签名 sidecar：<reports_dir>/<image_id>.pdf.sig。

    API 响应不含磁盘路径（仅 pdf_url），故经配置的 reports 目录 + 影像 id 推导。
    """
    from backend.app import dependencies as deps

    reports_dir = Path(deps.get_registry().config.paths.reports_dir)
    matches = sorted(reports_dir.glob(f"{report['image_id']}*.sig"))
    return matches[0] if matches else None


def test_report_sm2_signature_valid(tmp_path, monkeypatch) -> None:
    """配置密钥出报告 → sidecar 落盘，验签 signature.valid=true + 公钥 128 hex。"""
    monkeypatch.setenv("SCAN_CRYPTO_KEY", _SIG_KEY)
    path = _make_film(tmp_path)
    with TestClient(app) as client:
        report = _post_report(client, path)
        resp = client.post(f"/api/v1/report/{report['report_id']}/verify")
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is True  # 指纹比对仍通过
    sig = body["signature"]
    assert sig is not None and sig["valid"] is True and sig["reason"] is None
    assert sig["algo"] == "SM2"
    assert sig["public_key"] and len(sig["public_key"]) == 128
    sidecar = _signature_sidecar(report)
    assert sidecar is not None and sidecar.exists()
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["signature"] and meta["fingerprint"] == body["hash"]


def test_report_sm2_signature_tamper_detected(tmp_path, monkeypatch) -> None:
    """sidecar 内签名值被替换 → 验签 signature.valid=false。"""
    monkeypatch.setenv("SCAN_CRYPTO_KEY", _SIG_KEY)
    path = _make_film(tmp_path)
    with TestClient(app) as client:
        report = _post_report(client, path)
        sidecar = _signature_sidecar(report)
        assert sidecar is not None and sidecar.exists()
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        meta["signature"] = "11" * 128  # 模拟攻击者改写签名值
        sidecar.write_text(json.dumps(meta), encoding="utf-8")
        resp = client.post(f"/api/v1/report/{report['report_id']}/verify")
    body = resp.json()
    assert body["valid"] is True  # 指纹不受 sidecar 篡改影响
    assert body["signature"]["valid"] is False
    assert body["signature"]["reason"] == "mismatch"


def test_report_sm2_signature_swapped_fingerprint_detected(tmp_path, monkeypatch) -> None:
    """重签换内容：sidecar 内部自洽（对另一份指纹合法签名）但与当前报告内容
    不符 → fingerprint_mismatch（拼包/换内容检出）。"""
    monkeypatch.setenv("SCAN_CRYPTO_KEY", _SIG_KEY)
    path = _make_film(tmp_path)
    with TestClient(app) as client:
        report = _post_report(client, path)
        sidecar = _signature_sidecar(report)
        assert sidecar is not None
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        # 攻击者持同一密钥对另一指纹重签（模拟签发后调换报告内容）
        from backend.infra.crypto import SoftSmProvider

        provider = SoftSmProvider(base64.b64decode(_SIG_KEY))
        meta["fingerprint"] = "ab" * 32
        meta["signature"] = provider.sign(b"ab" * 32)
        sidecar.write_text(json.dumps(meta), encoding="utf-8")
        resp = client.post(f"/api/v1/report/{report['report_id']}/verify")
    body = resp.json()
    assert body["signature"]["valid"] is False
    assert body["signature"]["reason"] == "fingerprint_mismatch"


def test_report_sm2_signature_missing_is_legacy(tmp_path, monkeypatch) -> None:
    """未配置密钥（无 sidecar）→ signature.valid=null，指纹校验不受影响。"""
    monkeypatch.delenv("SCAN_CRYPTO_KEY", raising=False)
    path = _make_film(tmp_path)
    with TestClient(app) as client:
        report = _post_report(client, path)
        sidecar = _signature_sidecar(report)
        assert sidecar is None or not sidecar.exists()
        resp = client.post(f"/api/v1/report/{report['report_id']}/verify")
    body = resp.json()
    assert body["valid"] is True
    assert body["signature"] is not None
    assert body["signature"]["valid"] is None
    assert body["signature"]["reason"] == "missing"
