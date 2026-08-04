"""verify 端点集成测试（§4.2 / §T5）。"""
from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app

_N_WIRES = 19


def _upload_image_bytes(amps: list[float]) -> bytes:
    rng = np.random.default_rng(0)
    img = rng.normal(128.0, 2.0, (190, 640)).astype(np.uint8)
    for i, amp in enumerate(amps):
        y = round((i + 0.5) / _N_WIRES * 190)
        cv2.line(img, (0, y), (639, y), int(128 + amp), 3)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_verify_endpoint_shape() -> None:
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/verify",
            files={"image": ("x.png", _upload_image_bytes([40.0] * _N_WIRES), "image/png")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["iqi"]["iqi_type"] == "wire"
    assert body["iqi"]["achieved"] == str(_N_WIRES)
    assert body["iqi"]["passed"] is True
    assert "density" in body
    assert "density_ok" in body
    assert "evaluable" in body


def test_verify_rejects_non_image() -> None:
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/api/v1/verify",
            files={"image": ("x.png", b"not an image", "image/png")},
        )
    assert resp.status_code == 500  # 解码失败由全局异常处理（M2 暂用默认）


def test_verify_missing_file_422() -> None:
    with TestClient(app) as client:
        resp = client.post("/api/v1/verify")
    assert resp.status_code == 422
