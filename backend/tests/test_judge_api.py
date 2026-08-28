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


def test_judge_authorized_returns_level() -> None:
    """NB/T47013 数值表已授权（authorized=true）→ 产出真实级别，不再熔断。

    载荷：10mm×1mm 缺陷（长宽比>3 → 按条形判定）@ T=20mm。
    条形单条 10mm：II 限值 T/3≈6.67、III 限值 2T/3≈13.33 → 落在 III 级；
    组内 12T 区累计 10mm ≤ glim2≈13.33 → 组 II；取最差 → III。
    尺寸临界（长径≈T/2）→ need_review=True。
    """
    with TestClient(app) as client:
        resp = client.post("/api/v1/judge", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["standard_id"] == "NB/T47013.2-2015"
    assert body["joint_level"] == "III"
    assert body["per_defect_grade"], "应给出逐缺陷级别"
    assert isinstance(body["need_review"], bool)
    # T1：未持有授权正本（authorized_copy=false）→ 响应必须带强免责声明，
    # 明确"非标准授权正本 / 不替代责任工程师法定评定"。
    assert body["disclaimer"] is not None
    assert "非标准授权正本" in body["disclaimer"]
    assert "法定判定" in body["disclaimer"]


def test_judge_missing_thickness_422() -> None:
    body = _payload()
    body.pop("base_metal_thickness_mm")
    with TestClient(app) as client:
        resp = client.post("/api/v1/judge", json=body)
    assert resp.status_code == 422  # pydantic 校验


def test_judge_missing_spacing_fuses_422() -> None:
    """未提供像素标定 → 不再静默 1.0 mm/px（伪物理），grader 熔断 422（§6/§T8）。"""
    body = _payload()
    body.pop("pixel_spacing_mm")
    with TestClient(app) as client:
        resp = client.post("/api/v1/judge", json=body)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "GRADING_AMBIGUOUS"


def test_judge_empty_payload_422() -> None:
    with TestClient(app) as client:
        resp = client.post("/api/v1/judge", json={})
    assert resp.status_code == 422


def test_judge_skeleton_standard_fuses_422() -> None:
    """§6.1 骨架标准（GB/T 3323 已注册未实现）→ grade 熔断 422。"""
    body = _payload()
    body["standard_id"] = "GB/T3323-2019"
    with TestClient(app) as client:
        resp = client.post("/api/v1/judge", json=body)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "GRADING_AMBIGUOUS"


def test_judge_unknown_standard_422() -> None:
    """§6.1 未注册标准 → 装配即 422。"""
    body = _payload()
    body["standard_id"] = "ISO-UNKNOWN-XXXX"
    with TestClient(app) as client:
        resp = client.post("/api/v1/judge", json=body)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "GRADING_AMBIGUOUS"
