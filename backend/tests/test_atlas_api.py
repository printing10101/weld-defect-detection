"""集成测试：缺陷图谱样本库（发布/检索/局部图/撤销/审计）。"""

from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


def _write_film(path) -> None:
    """带一个暗斑缺陷的合成底片（图谱局部图裁剪源）。"""
    rng = np.random.default_rng(7)
    img = rng.normal(128.0, 3.0, (200, 300)).astype(np.uint8)
    img[80:110, 120:160] = 60  # 暗斑（气孔类语境）
    cv2.imwrite(str(path), img)


@pytest.fixture()
def inspection(tmp_path):
    """向测试库写入一条影像+缺陷记录（底片留在本测试的 tmp_path 内，
    发布裁图发生在测试执行期，文件必须活过整个测试），返回 (image_id, defect_id)。"""
    from backend.app import dependencies as deps

    reg = deps.get_registry()
    image_id = uuid.uuid4().hex
    defect_id = f"{image_id}:det1"
    film = tmp_path / "atlas_film.png"
    _write_film(film)
    image_row = {
        "id": image_id,
        "path": str(film),
        "source_type": "image",
        "modality": "GENERIC",
        "workpiece_no": "WP-ATLAS",
        "weld_no": "W01",
        "joint_level": "II",
        "standard_id": "NB/T47013.2-2015",
    }
    defect_row = {
        "id": defect_id,
        "image_id": image_id,
        "class_id": 0,
        "bbox_px": [120.0, 80.0, 40.0, 30.0],
        "shape": "round",
        "length_mm": 4.0,
        "width_mm": 3.0,
        "confidence": 0.87,
        "source": "auto",
        "joint_level": "II",
        "standard_id": "NB/T47013.2-2015",
    }
    reg.repository.create_inspection(image_row, [defect_row])
    yield image_id, defect_id


def test_publish_list_crop_delete(inspection) -> None:
    """发布 → 检索 → 详情 → 局部图 → 重复发布 409 → 撤销 → 404。"""
    _, defect_id = inspection
    with TestClient(app) as client:
        resp = client.post("/api/v1/atlas", json={"defect_id": defect_id, "note": "典型气孔样本"})
        assert resp.status_code == 200, resp.text
        atlas_id = resp.json()["atlas_id"]

        # 检索：按类过滤命中
        lst = client.get("/api/v1/atlas", params={"class": 0}).json()
        assert lst["total"] >= 1
        sample = next(it for it in lst["items"] if it["id"] == atlas_id)
        assert sample["workpiece_no"] == "WP-ATLAS"
        assert sample["joint_level"] == "II"
        assert sample["source"] == "auto"
        assert sample["note"] == "典型气孔样本"

        # 详情与局部图（含外扩边距，小于整图且可解码）
        detail = client.get(f"/api/v1/atlas/{atlas_id}")
        assert detail.status_code == 200
        crop = client.get(f"/api/v1/atlas/{atlas_id}/crop.png")
        assert crop.status_code == 200
        assert crop.headers["content-type"] == "image/png"
        decoded = cv2.imdecode(np.frombuffer(crop.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert decoded is not None
        assert decoded.shape[0] < 200 and decoded.shape[1] < 300
        crop_path = resp.json()["crop_path"]  # 发布响应携带落盘路径
        assert Path(crop_path).is_file()

        # 同一缺陷重复发布 → 409
        dup = client.post("/api/v1/atlas", json={"defect_id": defect_id})
        assert dup.status_code == 409

        # 撤销 → 详情/局部图 404 → 落盘文件一并删除（无孤儿密文）
        rm = client.delete(f"/api/v1/atlas/{atlas_id}", params={"reason": "误选样本"})
        assert rm.status_code == 200
        assert client.get(f"/api/v1/atlas/{atlas_id}").status_code == 404
        assert client.get(f"/api/v1/atlas/{atlas_id}/crop.png").status_code == 404
        assert not Path(crop_path).is_file()


def test_publish_unknown_defect_404() -> None:
    with TestClient(app) as client:
        resp = client.post("/api/v1/atlas", json={"defect_id": "no-such-defect"})
        assert resp.status_code == 404


def test_delete_requires_reason(inspection) -> None:
    """撤销必须留原因（审计语义），缺参 422。"""
    _, defect_id = inspection
    with TestClient(app) as client:
        atlas_id = client.post("/api/v1/atlas", json={"defect_id": defect_id}).json()["atlas_id"]
        resp = client.delete(f"/api/v1/atlas/{atlas_id}")
        assert resp.status_code == 422
