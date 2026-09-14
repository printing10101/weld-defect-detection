"""复核结论自动回流训练池（G21）集成测试。

共识复核（级别落定）后，训练池应自动出现该影像的 YOLO 标注文件，响应
带 training_pool_synced=True——此前回流依赖人工补调 /active/export。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.tests.test_review_api import _post_report, _review, _synthetic  # 复用设施


@pytest.fixture(scope="module", autouse=True)
def _authorized_grader(auth_table) -> Iterator[None]:
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


def _pool_dir() -> Path:
    from backend.app.dependencies import get_registry
    from backend.infra.config import resolve_config_path

    reg = get_registry()
    return Path(
        resolve_config_path(str(Path(reg.config.paths.data_dir) / "active" / "training_pool"))
    )


def test_consensus_review_auto_exports_to_pool(tmp_path) -> None:
    img = tmp_path / "revpool1.png"
    _synthetic(img)
    with TestClient(app) as client:
        rep = _post_report(client, img, pixel_spacing_mm="0.1", base_metal_thickness_mm="20")
        image_id = rep["image_id"]
        pool = _pool_dir()
        label = pool / f"{image_id}.txt"
        assert not label.exists(), "复核前不应有训练池标注"

        out = _review(client, image_id=image_id, reviewer="alice", role="initial", defect_grades=[])
    assert out["joint_level"] is not None  # 级别落定才触发回流
    assert out["training_pool_synced"] is True
    assert label.exists(), "共识复核后训练池应自动出现标注文件"
    lines = [ln for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines, "标注文件不应为空（合成底片应产生检出）"
    for ln in lines:
        parts = ln.split()
        assert len(parts) == 5  # YOLO: class cx cy w h
        assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])


def test_deferred_review_does_not_export(tmp_path) -> None:
    """级别未落定（强分歧升级仲裁）时不回流——缺陷集还不是"人工确认标注"。"""
    from backend.app.dependencies import get_registry

    img = tmp_path / "revpool2.png"
    _synthetic(img)
    with TestClient(app) as client:
        rep = _post_report(client, img, pixel_spacing_mm="0.1", base_metal_thickness_mm="20")
        image_id = rep["image_id"]
        reg = get_registry()
        image = reg.repository.get_image(image_id)
        defect_ids = [d["id"] for d in (image.get("defects") or [])]

        out = _review(
            client,
            image_id=image_id,
            reviewer="bob",
            role="initial",
            defect_grades=[{"defect_id": d, "joint_level": "IV"} for d in defect_ids],
        )
    assert out["joint_level"] is None
    assert out["needs_arbitration"] is True
    assert not (_pool_dir() / f"{image_id}.txt").exists()
