""" 设备标定档案（API 级测试）。

覆盖：注册/列表/详情、标定记录与相对偏差计算、跨设备一致率 ≤5% 边界
（≤5% → ok，>5% → over）、未知设备 404。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


def _register(client: TestClient, name: str = "CR-01", **extra) -> dict:
    resp = client.post(
        "/api/v1/devices",
        json={"name": name, "model": "X-Ray 3000", "serial_no": "SN-001", **extra},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_device_register_list_detail() -> None:
    with TestClient(app) as client:
        dev = _register(client, name="CR-01")
        dev_id = dev["device_id"]
        assert dev["name"] == "CR-01"
        assert dev["calibration_count"] == 0

        rows = client.get("/api/v1/devices").json()
        assert rows[0]["device_id"] == dev_id  # 最近注册在前
        assert rows[0]["last_calibration"] is None

        detail = client.get(f"/api/v1/devices/{dev_id}").json()
        assert detail["calibrations"] == []
        assert detail["calibration_count"] == 0


def test_calibration_within_5pct_ok() -> None:
    """参考值 0.1000，实测 0.1030（偏差 3%）→ status=ok。"""
    with TestClient(app) as client:
        dev = _register(client, name="CR-02")
        dev_id = dev["device_id"]
        resp = client.post(
            f"/api/v1/devices/{dev_id}/calibrations",
            json={
                "calibrator": "alice",
                "pixel_spacing_mm": 0.103,
                "ref_pixel_spacing_mm": 0.1,
                "density_ref": 2.4,
                "notes": "routine",
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "ok"
        assert body["deviation_pct"] == 3.0

        detail = client.get(f"/api/v1/devices/{dev_id}").json()
        assert detail["calibration_count"] == 1
        assert detail["calibrations"][0]["status"] == "ok"


def test_calibration_over_5pct_over() -> None:
    """参考值 0.1000，实测 0.107（偏差 7%）→ status=over（跨设备一致率超标）。"""
    with TestClient(app) as client:
        dev = _register(client, name="CR-03")
        dev_id = dev["device_id"]
        resp = client.post(
            f"/api/v1/devices/{dev_id}/calibrations",
            json={
                "calibrator": "bob",
                "pixel_spacing_mm": 0.107,
                "ref_pixel_spacing_mm": 0.1,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "over"
        assert resp.json()["deviation_pct"] == 7.0

        # 列表形态：最近标定摘要带 over 状态
        rows = client.get("/api/v1/devices").json()
        assert rows[0]["last_calibration"]["status"] == "over"


def test_calibration_without_ref_no_deviation() -> None:
    """未填参考值时无偏差判定，status 保持 ok。"""
    with TestClient(app) as client:
        dev = _register(client, name="CR-04")
        resp = client.post(
            f"/api/v1/devices/{dev['device_id']}/calibrations",
            json={"calibrator": "carol", "pixel_spacing_mm": 0.1},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["deviation_pct"] is None


def test_device_unknown_404() -> None:
    with TestClient(app) as client:
        assert client.get("/api/v1/devices/nope").status_code == 404
        resp = client.post(
            "/api/v1/devices/nope/calibrations",
            json={"calibrator": "x", "pixel_spacing_mm": 0.1},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"]["code"] == "NOT_FOUND"
