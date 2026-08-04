"""judge 端点集成测试（§6 / §T8 熔断）。"""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


def _payload() -> dict:
    return {
        "base_metal_thickness_mm": 20.0,
        "pixel_spacing_mm": 0.1,
        "standard_id": "NB/T47013.2-2015",
        "defects": [
            {
                "id": "d1",
                "class_id": 0,  # POROSITY
                "bbox": [0, 0, 100, 10],  # L=10mm
                "confidence": 0.6,
                "uncertainty": 0.4,
            }
        ],
    }


def test_judge_unauthorized_fuses_422() -> None:
    """标准数值未授权（authorized=false）→ 熔断 422，不输出级别。"""
    with TestClient(app) as client:
        resp = client.post("/api/v1/judge", json=_payload())
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "GRADING_AMBIGUOUS"


def test_judge_missing_thickness_422() -> None:
    body = _payload()
    body.pop("base_metal_thickness_mm")
    with TestClient(app) as client:
        resp = client.post("/api/v1/judge", json=body)
    assert resp.status_code == 422  # pydantic 校验


def test_judge_empty_payload_422() -> None:
    with TestClient(app) as client:
        resp = client.post("/api/v1/judge", json={})
    assert resp.status_code == 422
