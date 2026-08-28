"""人工复核缺陷增删改集成测试（DB50/T 1807-2025 §6.1.4 + 审计哈希链）。"""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module", autouse=True)
def _authorized_grader(auth_table) -> Iterator[None]:
    """缺陷增删改后需重评级：用 authorized 测试表 + 放宽黑度/质量门禁。"""
    from backend.app import dependencies as deps
    from backend.domain.grade.nb47013 import Nb47013Grader
    from backend.domain.standards.tables.loader import load_standard_tables

    deps._registry = None
    reg = deps.get_registry()
    reg.grader = Nb47013Grader(load_standard_tables("NB/T47013.2-2015", filename=str(auth_table)))
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


def _synthetic(path) -> None:
    n, h, w = 19, 190, 640
    rng = np.random.default_rng(7)
    img = rng.normal(128.0, 2.0, (h, w)).astype(np.uint8)
    for i in range(n):
        y = round((i + 0.5) / n * h)
        cv2.line(img, (0, y), (w - 1, y), int(128 + 40.0), 3)
    cv2.circle(img, (120, 30), 10, 80, -1)  # 气孔样暗点 → 检出若干缺陷
    cv2.imwrite(str(path), img)


@pytest.fixture()
def client(tmp_path) -> Iterator[tuple[TestClient, dict]]:
    """带一张已检影像（含缺陷与报告）的客户端。"""
    img = tmp_path / "edit1.png"
    _synthetic(img)
    with TestClient(app) as client:
        with open(img, "rb") as f:
            resp = client.post(
                "/api/v1/report",
                files={"image": ("edit1.png", f, "image/png")},
                data={"pixel_spacing_mm": "0.1", "base_metal_thickness_mm": "20"},
            )
        assert resp.status_code == 200, resp.text
        yield client, resp.json()


def _defects(client: TestClient, image_id: str) -> list[dict]:
    from backend.app.dependencies import get_registry

    return (get_registry().repository.get_image(image_id) or {}).get("defects") or []


def test_add_manual_defect_triggers_regrade_and_audit(client):
    http, rep = client
    n0 = rep["defect_count"]
    resp = http.post(
        f"/api/v1/review/{rep['image_id']}/defects",
        json={"class_id": 0, "bbox_px": [300, 100, 20, 20], "reason": "复核补录漏检气孔"},
        headers={"X-Operator-Name": "reviewer-a"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["defect"]["source"] == "manual"
    assert body["defect_count"] == n0 + 1  # 重评级计入新缺陷
    rows = _defects(http, rep["image_id"])
    assert any(d["source"] == "manual" for d in rows)
    # 审计留痕
    audit = http.get("/api/v1/audit", params={"object_id": body["defect"]["id"]}).json()
    assert any(a["action"] == "defect.add" for a in audit["entries"])
    # 审计哈希链未被破坏
    from backend.app.dependencies import get_registry

    assert get_registry().repository.verify_chain() is True


def test_edit_defect_type_to_crack_forces_iv(client):
    http, rep = client
    rows = _defects(http, rep["image_id"])
    target = rows[0]["id"]
    resp = http.patch(
        f"/api/v1/review/defects/{target}",
        json={"class_id": 4, "reason": "复核确认为裂纹"},
        headers={"X-Operator-Name": "reviewer-b"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # 裂纹零容忍 → 综合级别 IV
    assert body["joint_level"] == "IV"
    updated = next(d for d in _defects(http, rep["image_id"]) if d["id"] == target)
    assert updated["class_id"] == 4 and updated["joint_level"] == "IV"
    audit = http.get("/api/v1/audit", params={"object_id": target}).json()
    entry = next(a for a in audit["entries"] if a["action"] == "defect.edit")
    assert entry["before"]["class_id"] != entry["after"]["class_id"]


def test_edit_defect_bbox_updates_geometry(client):
    http, rep = client
    target = _defects(http, rep["image_id"])[0]["id"]
    resp = http.patch(
        f"/api/v1/review/defects/{target}",
        json={"bbox_px": [10, 10, 40, 40], "reason": "位置修正"},
        headers={"X-Operator-Name": "reviewer-b"},
    )
    assert resp.status_code == 200, resp.text
    updated = next(d for d in _defects(http, rep["image_id"]) if d["id"] == target)
    assert updated["bbox_px"] == [10.0, 10.0, 40.0, 40.0]


def test_delete_defect_soft_and_audit(client):
    http, rep = client
    n0 = len(_defects(http, rep["image_id"]))
    target = _defects(http, rep["image_id"])[0]["id"]
    resp = http.delete(
        f"/api/v1/review/defects/{target}",
        params={"reason": "确认为伪影像，非缺陷"},
        headers={"X-Operator-Name": "reviewer-c"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["defect"]["deleted"] is True
    rows = _defects(http, rep["image_id"])
    assert len(rows) == n0 - 1  # 软删除后列表不再展示
    audit = http.get("/api/v1/audit", params={"object_id": target}).json()
    assert any(a["action"] == "defect.delete" for a in audit["entries"])
    from backend.app.dependencies import get_registry

    assert get_registry().repository.verify_chain() is True


def test_reason_required(client):
    http, rep = client
    target = _defects(http, rep["image_id"])[0]["id"]
    # 缺 reason → FastAPI 422
    assert http.delete(f"/api/v1/review/defects/{target}").status_code == 422
    # 空 reason → 422
    r = http.patch(
        f"/api/v1/review/defects/{target}",
        json={"reason": "  "},
        headers={"X-Operator-Name": "x"},
    )
    assert r.status_code == 422


def test_patch_requires_at_least_one_field(client):
    http, rep = client
    target = _defects(http, rep["image_id"])[0]["id"]
    r = http.patch(
        f"/api/v1/review/defects/{target}",
        json={"reason": "只写理由"},
        headers={"X-Operator-Name": "x"},
    )
    assert r.status_code == 422


def test_add_defect_validation_errors(client):
    http, rep = client
    r = http.post(
        f"/api/v1/review/{rep['image_id']}/defects",
        json={"class_id": 99, "bbox_px": [1, 1, 5, 5], "reason": "越界类别"},
    )
    assert r.status_code == 422
    r = http.post(
        "/api/v1/review/nonexistent/defects",
        json={"class_id": 0, "bbox_px": [1, 1, 5, 5], "reason": "影像不存在"},
    )
    assert r.status_code == 404


def test_edit_missing_defect_404(client):
    http, _rep = client
    r = http.patch(
        "/api/v1/review/defects/no-such-id",
        json={"class_id": 0, "reason": "不存在"},
    )
    assert r.status_code == 404
