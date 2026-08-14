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
    """解码失败 → ImageUnreadableError → 400 IMG_UNREADABLE（§14 错误码表）。

    此前无全局处理器时该路径漏成 500，属实现缺陷而非契约。
    """
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/api/v1/verify",
            files={"image": ("x.png", b"not an image", "image/png")},
        )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "IMG_UNREADABLE"


def test_verify_rejects_unsupported_suffix() -> None:
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/verify",
            files={"image": ("x.exe", b"MZ\x00\x00", "application/octet-stream")},
        )
    assert resp.status_code == 415


def test_verify_rejects_bad_roi() -> None:
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/verify",
            files={"image": ("x.png", _upload_image_bytes([40.0] * _N_WIRES), "image/png")},
            data={"iqi_roi": "1,2,3"},
        )
    assert resp.status_code == 422


def test_verify_missing_file_422() -> None:
    with TestClient(app) as client:
        resp = client.post("/api/v1/verify")
    assert resp.status_code == 422


def _upload_dark_lines(amps: list[float], n: int = 19) -> bytes:
    """生成孔型像质计测试图（暗线于亮背景）。"""
    rng = np.random.default_rng(0)
    img = rng.normal(128.0, 2.0, (190, 640)).astype(np.uint8)
    for i, amp in enumerate(amps):
        y = round((i + 0.5) / n * 190)
        cv2.line(img, (0, y), (639, y), int(max(0, 128 - amp)), 3)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_verify_hole_type() -> None:
    """iqi_type=hole 应走孔型识别，返回 iqi_type=='hole'（§4.2 线型或孔型）。"""
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/verify",
            files={"image": ("x.png", _upload_dark_lines([40.0] * 19), "image/png")},
            data={"iqi_type": "hole"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["iqi"]["iqi_type"] == "hole"


def test_verify_grade_with_thickness() -> None:
    """传入透照厚度 → iqi.grade 由可达丝号+厚度映射（§4.2 A/AB/B）。

    全幅 19 丝、厚度 1.0mm（≤2mm 要求丝号 14）→ 最高等级 A。
    """
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/verify",
            files={"image": ("x.png", _upload_image_bytes([40.0] * _N_WIRES), "image/png")},
            data={"thickness_mm": "1.0"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["iqi"]["achieved"] == str(_N_WIRES)
    assert body["iqi"]["grade"] == "A"


def test_verify_grade_none_without_thickness() -> None:
    """未传厚度 → grade 为 None（不臆造等级）。"""
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/verify",
            files={"image": ("x.png", _upload_image_bytes([40.0] * _N_WIRES), "image/png")},
        )
    assert resp.status_code == 200
    assert resp.json()["iqi"]["grade"] is None


def test_verify_pseudo_defect_field_present() -> None:
    """响应须含 pseudo_defect 段（passed:bool, notes:list）。"""
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/verify",
            files={"image": ("x.png", _upload_image_bytes([40.0] * _N_WIRES), "image/png")},
        )
    assert resp.status_code == 200
    pd = resp.json()["pseudo_defect"]
    assert isinstance(pd["passed"], bool)
    assert isinstance(pd["notes"], list)


def _upload_realistic_film() -> bytes:
    """1000x1000 大底片 + 角落非全宽 IQI（真实场景：IQI 仅占角落）。"""
    rng = np.random.default_rng(5)
    img = rng.normal(128.0, 2.0, (1000, 1000)).astype(np.uint8)
    for i in range(_N_WIRES):
        y = round((i + 0.5) / _N_WIRES * 190)
        cv2.line(img, (20, y), (180, y), 128 + 42, 3)  # 160px 宽角落 IQI
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_verify_pseudo_defect_passes_on_realistic_film() -> None:
    """真实大底片（IQI 仅占角落）→ 伪缺陷不误阻断（passed=True）。"""
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/verify",
            files={"image": ("x.png", _upload_realistic_film(), "image/png")},
            data={"thickness_mm": "1.0"},
        )
    assert resp.status_code == 200
    pd = resp.json()["pseudo_defect"]
    assert pd["passed"] is True
