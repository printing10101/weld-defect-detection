"""M7 测试：主动学习闭环（§5.5/§5.6）。"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.domain.active_learning import (
    export_training_labels,
    high_value_score,
    select_high_value,
    to_yolo_label,
    training_pool_manifest,
)
from backend.domain.dto import BBox, DefectClass, Detection
from backend.infra.pool_store import FilePoolStore


def _det(
    class_id: DefectClass,
    x: float = 10.0,
    y: float = 10.0,
    w: float = 20.0,
    h: float = 20.0,
    score: float = 0.9,
    uncertainty: float = 0.1,
) -> Detection:
    return Detection(
        id=f"{class_id.name}-{x}",
        bbox=BBox(x, y, w, h),
        class_id=class_id,
        score=score,
        uncertainty=uncertainty,
    )


# ---------------------------------------------------------------------------
# 采样器（§5.6 高价值样本）
# ---------------------------------------------------------------------------


def test_high_value_score_uncertainty_dominant() -> None:
    """高不确定性 → 高采样价值（即使高置信近边界也应人工确认）。"""
    d = _det(DefectClass.POROSITY, uncertainty=0.95, score=0.8)
    assert high_value_score(d) >= 0.9
    # 低不确定性高置信 → 低价值（不需要人工）
    d2 = _det(DefectClass.POROSITY, uncertainty=0.05, score=0.9)
    assert high_value_score(d2) < 0.3


def test_high_value_score_safety_class_base() -> None:
    """安全关键/稀有类（裂纹/未熔合）→ 即使高置信低不确定也保底采样。"""
    d = _det(DefectClass.CRACK, uncertainty=0.05, score=0.95)
    assert high_value_score(d) >= 0.5
    # 普通类同条件无保底
    d2 = _det(DefectClass.POROSITY, uncertainty=0.05, score=0.95)
    assert high_value_score(d2) < 0.5


def test_select_high_value_orders_and_filters() -> None:
    detections = [
        _det(DefectClass.POROSITY, uncertainty=0.95),  # 最高价值
        _det(DefectClass.POROSITY, uncertainty=0.05, score=0.9),
        _det(DefectClass.CRACK, uncertainty=0.05, score=0.95),  # 安全类保底
    ]
    top = select_high_value(detections, top_k=2)
    assert len(top) == 2
    assert top[0].value_score >= top[1].value_score  # 降序
    # min_value 过滤
    filtered = select_high_value(detections, top_k=10, min_value=0.4)
    assert all(c.value_score >= 0.4 for c in filtered)
    assert len(filtered) < len(detections)  # 低价值被滤掉


# ---------------------------------------------------------------------------
# 标注回流（§5.5 人工确认 → YOLO 训练池）
# ---------------------------------------------------------------------------


def test_to_yolo_label_normalized_and_clipped() -> None:
    d = _det(DefectClass.POROSITY, x=10, y=20, w=100, h=50)
    line = to_yolo_label(d, 1000, 800)
    parts = line.split()
    assert parts[0] == "0"  # class_id
    cx, cy, w, h = (float(v) for v in parts[1:])
    assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
    assert abs(w - 0.1) < 1e-6 and abs(h - 0.0625) < 1e-6
    # 越界裁剪：不出 >1 坐标
    d_big = _det(DefectClass.POROSITY, x=900, y=700, w=500, h=500)
    line_big = to_yolo_label(d_big, 1000, 800)
    assert all(0.0 <= float(v) <= 1.0 for v in line_big.split()[1:])


def test_export_training_labels_writes_pool(tmp_path: Path) -> None:
    pool = tmp_path / "training_pool"
    detections = [_det(DefectClass.POROSITY), _det(DefectClass.SLAG, x=50)]
    out = export_training_labels(
        "PG101-1-1", detections, 1000, 800, store=FilePoolStore(pool)
    )
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("0 ")
    assert lines[1].startswith("1 ")


def test_export_with_class_override(tmp_path: Path) -> None:
    """人工改判：把误检气孔(0) 改判为裂纹(4)。"""
    d = _det(DefectClass.POROSITY)
    out = export_training_labels(
        "x",
        [d],
        1000,
        800,
        store=FilePoolStore(tmp_path),
        class_overrides={d.id: 4},
    )
    assert out.read_text(encoding="utf-8").strip().startswith("4 ")


def test_pool_manifest_counts_and_fingerprint(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    store = FilePoolStore(pool)
    manifest = training_pool_manifest(store)  # 目录不存在 → 0/None（不臆造）
    assert manifest["sample_count"] == 0
    assert manifest["fingerprint"] is None

    export_training_labels("a", [_det(DefectClass.POROSITY)], 100, 100, store=store)
    export_training_labels("b", [_det(DefectClass.CRACK)], 100, 100, store=store)
    m2 = training_pool_manifest(store)
    assert m2["sample_count"] == 2
    assert m2["fingerprint"]  # 数据版本指纹
    # 内容变化 → 指纹变
    export_training_labels("c", [_det(DefectClass.SLAG)], 100, 100, store=store)
    assert training_pool_manifest(store)["fingerprint"] != m2["fingerprint"]


# ---------------------------------------------------------------------------
# API 编排
# ---------------------------------------------------------------------------


def test_active_api_sample() -> None:
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/active/sample",
            json={
                "defects": [
                    {
                        "id": "d1",
                        "class_id": 0,
                        "bbox": [0, 0, 10, 10],
                        "confidence": 0.8,
                        "uncertainty": 0.95,
                    },
                    {
                        "id": "d2",
                        "class_id": 4,
                        "bbox": [0, 0, 10, 10],
                        "confidence": 0.95,
                        "uncertainty": 0.05,
                    },
                ]
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    # 高不确定 d1 排前（0.95 > 安全类保底 0.5）
    assert body["candidates"][0]["detection_id"] == "d1"
    assert body["candidates"][0]["value_score"] >= 0.9


def test_active_api_sample_invalid_class_422() -> None:
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/active/sample",
            json={"defects": [{"id": "x", "class_id": 99, "bbox": [0, 0, 1, 1]}]},
        )
    assert resp.status_code == 422


def test_active_api_export_and_pool(monkeypatch, tmp_path: Path) -> None:
    from backend.app import dependencies as deps

    reg = deps.get_registry()
    # 把训练池指向临时目录（避免污染 data/active/）
    monkeypatch.setattr(reg.config.paths, "data_dir", str(tmp_path / "data"))

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/active/export",
            json={
                "image_stem": "PG101-9-1",
                "image_w": 1000,
                "image_h": 800,
                "defects": [
                    {
                        "id": "d1",
                        "class_id": 0,
                        "bbox": [10, 10, 20, 20],
                        "confidence": 0.9,
                        "uncertainty": 0.1,
                    },
                    {
                        "id": "d2",
                        "class_id": 4,
                        "bbox": [50, 50, 15, 15],
                        "confidence": 0.85,
                        "uncertainty": 0.3,
                    },
                ],
            },
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["sample_count"] == 2
    assert body["fingerprint"]  # 版本指纹
    assert body["total_in_pool"] == 1

    with TestClient(app) as client:
        pool = client.get("/api/v1/active/pool")
    assert pool.status_code == 200
    assert pool.json()["sample_count"] == 1
    assert "PG101-9-1.txt" in pool.json()["files"]
