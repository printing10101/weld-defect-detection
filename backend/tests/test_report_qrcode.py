"""报告二维码追溯（G01）单测 + API 集成测试。"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.infra.reporting.qrcode import parse_trace_code, qr_png_bytes, trace_code


@pytest.fixture(scope="module", autouse=True)
def _relax_quality_gates() -> Iterator[None]:
    """合成底片黑度/RQI 天然不过硬门禁（同 test_review_api 的放宽做法）。"""
    from backend.app import dependencies as deps

    deps._registry = None
    reg = deps.get_registry()
    original_low = reg.config.density.low
    original_block = reg.config.quality.block_on_quality
    reg.config.density.low = 0.0
    reg.config.quality.block_on_quality = False
    try:
        yield
    finally:
        reg.config.density.low = original_low
        reg.config.quality.block_on_quality = original_block
        deps._registry = None


# ---------------------------------------------------------------------------
# 追溯码编解码
# ---------------------------------------------------------------------------
def test_trace_code_roundtrip():
    code = trace_code("rpt123", "a" * 64)
    assert parse_trace_code(code) == ("rpt123", "a" * 16)


def test_parse_rejects_bad_formats():
    assert parse_trace_code("bad") is None
    assert parse_trace_code("RT-TRACE|only-two") is None
    assert parse_trace_code("WRONG|r1|aaaaaaaaaaaaaaaa") is None
    assert parse_trace_code("RT-TRACE|r1|short") is None
    assert parse_trace_code("RT-TRACE||aaaaaaaaaaaaaaaa") is None
    assert parse_trace_code("RT-TRACE|r1|zzzzzzzzzzzzzzzz") is None  # 非 hex


def test_qr_png_generation_and_failsoft(monkeypatch):
    payload = trace_code("r1", "a" * 64)
    png = qr_png_bytes(payload)
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    # 生成器不可用时 fail-soft 返回 None，不抛（出片不阻断）
    import sys

    monkeypatch.setitem(sys.modules, "qrcode", None)
    assert qr_png_bytes(payload) is None


# ---------------------------------------------------------------------------
# 追溯查询端点（与报告链路集成）
# ---------------------------------------------------------------------------
def _synthetic(path) -> None:
    import cv2
    import numpy as np

    n, h, w = 19, 190, 640
    rng = np.random.default_rng(2)
    img = rng.normal(128.0, 2.0, (h, w)).astype(np.uint8)
    for i in range(n):
        y = round((i + 0.5) / n * h)
        cv2.line(img, (0, y), (w - 1, y), 168, 3)
    cv2.circle(img, (120, 30), 10, 80, -1)
    cv2.imwrite(str(path), img)


def test_trace_endpoint_found_and_matched(tmp_path):
    from backend.app import dependencies as deps
    from backend.app.main import app

    img = tmp_path / "qr1.png"
    _synthetic(img)
    with TestClient(app) as client:
        with open(img, "rb") as f:
            resp = client.post(
                "/api/v1/report",
                files={"image": ("qr1.png", f, "image/png")},
                data={"pixel_spacing_mm": "0.1", "base_metal_thickness_mm": "20"},
            )
        assert resp.status_code == 200, resp.text
        rep = resp.json()
        deps._registry = None  # 重建后再查库，模拟跨请求
        with TestClient(app) as client2:
            from backend.app.dependencies import get_registry

            reg = get_registry()
            image = reg.repository.get_image(rep["image_id"])
            assert image is not None
            report_id = (image.get("report") or {}).get("report_id")
            stored_hash = (image.get("report") or {}).get("report_hash") or ""
            assert report_id and stored_hash

            code = trace_code(report_id, stored_hash)
            out = client2.get(f"/api/v1/report/trace/{code}")
            assert out.status_code == 200, out.text
            body = out.json()
            assert body["archive_found"] is True
            assert body["hash_match"] is True


def test_trace_endpoint_missing_archive():
    from backend.app.main import app

    code = trace_code("nonexistent", "0" * 64)
    with TestClient(app) as client:
        out = client.get(f"/api/v1/report/trace/{code}")
        assert out.status_code == 200
        body = out.json()
        assert body["archive_found"] is False
        assert body["hash_match"] is False


def test_trace_endpoint_rejects_garbage():
    from backend.app.main import app

    with TestClient(app) as client:
        out = client.get("/api/v1/report/trace/hello-world")
        assert out.status_code == 422
