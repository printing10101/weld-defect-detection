"""报告数字签名校验（POST /api/v1/report/{id}/verify）。

覆盖：正常报告指纹一致（valid=true）；DB 内容被篡改后校验失败（valid=false）；
无指纹的旧报告返回 legacy（valid=null）。
"""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


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
