"""集成测试：人工复核闭环+ 审计日志+ PDF/A。

依赖 auth_table fixture（与生产表不同：authorized=true 才能正常评级），
覆盖：初评达成共识并落地级别/清空 need_review/重生成 PDF/A、分歧升级仲裁、
仲裁结案，以及审计链可查、PDF 为 PDF/A-1b。
"""

from __future__ import annotations

import io
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from backend.app.main import app
from backend.infra.reporting.pdfa import is_pdfa_compliant


@pytest.fixture(scope="module", autouse=True)
def _authorized_grader(auth_table) -> Iterator[None]:
    """复核测试需要正常评级：用 authorized 测试表替换全局 grader。"""
    from backend.app import dependencies as deps
    from backend.domain.grade.nb47013 import Nb47013Grader
    from backend.domain.standards.tables.loader import load_standard_tables

    deps._registry = None
    reg = deps.get_registry()
    reg.grader = Nb47013Grader(load_standard_tables("NB/T47013.2-2015", filename=str(auth_table)))
    # 合成底片是 8bit 噪声图，D=log10(2^8/(G+1)) 天然远低于 AB 级 2.0 下限，
    # 会被"底片不可评"硬前置（409 IQI_FAIL）拦下。复核闭环必须基于一张可评
    # 底片（否则自动级别恒为 None，κ/共识无意义），故本模块放宽黑度下限；
    # 同理，合成底片 RQI/硬门禁天然不过质量门槛（本模块测复核流程而非质量
    # 门禁，后者由 test_preprocess 单测覆盖），故同时关闭 block_on_quality。
    # 退出时还原并清空注册表，避免污染后续测试模块。
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
    rng = np.random.default_rng(1)
    img = rng.normal(128.0, 2.0, (h, w)).astype(np.uint8)
    for i in range(n):
        y = round((i + 0.5) / n * h)
        cv2.line(img, (0, y), (w - 1, y), int(128 + 40.0), 3)
    cv2.circle(img, (120, 30), 10, 80, -1)
    cv2.circle(img, (420, 150), 7, 85, -1)
    cv2.imwrite(str(path), img)


def _post_report(client: TestClient, image_path, **form) -> dict:
    with open(image_path, "rb") as f:
        resp = client.post(
            "/api/v1/report", files={"image": (image_path.name, f, "image/png")}, data=form
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _review(client: TestClient, **body) -> dict:
    resp = client.post("/api/v1/review", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_review_consensus_closes_loop(tmp_path) -> None:
    img = tmp_path / "rev1.png"
    _synthetic(img)
    with TestClient(app) as client:
        rep = _post_report(client, img, pixel_spacing_mm="0.1", base_metal_thickness_mm="20")
        image_id = rep["image_id"]
        # 初评：与自动级完全一致 → 共识、落地、清空 need_review
        out = _review(client, image_id=image_id, reviewer="alice", role="initial", defect_grades=[])
    assert out["consensus"] is True
    assert out["needs_arbitration"] is False
    assert out["stage"] == "consensus"
    assert out["joint_level"] is not None
    assert out["need_review"] is False
    assert out["reviewed_by"] == "alice"
    assert out["review_count"] == 1
    assert 0.0 <= out["kappa"] <= 1.0

    # PDF 现应为 PDF/A-1b
    pdf_url = rep["pdf_url"]
    with TestClient(app) as client:
        pdf = client.get(pdf_url)
        pdf_bytes = pdf.content
    assert pdf_bytes[:4] == b"%PDF"
    p = tmp_path / "out.pdf"
    p.write_bytes(pdf_bytes)
    compliant, info = is_pdfa_compliant(p)
    assert compliant, info


def test_review_disagreement_then_arbitration(tmp_path) -> None:
    img = tmp_path / "rev2.png"
    _synthetic(img)
    with TestClient(app) as client:
        rep = _post_report(client, img, pixel_spacing_mm="0.1", base_metal_thickness_mm="20")
        image_id = rep["image_id"]
        # 取一个缺陷的 id 与自动级，制造强分歧（全部推翻为 IV）
        detail = client.get(f"/api/v1/report/{rep['report_id']}/pdf")  # 触发存在性
        assert detail.status_code == 200
        # 直接查库（pipeline 同一 DB）拿到缺陷 id
        from backend.app.dependencies import get_registry

        reg = get_registry()
        image = reg.repository.get_image(image_id)
        assert image is not None
        defect_ids = [d["id"] for d in (image.get("defects") or [])]
        assert defect_ids, "synthetic image should yield defects"

        # 初评：全部推翻为 IV → 与自动级分歧 → 升级仲裁
        out1 = _review(
            client,
            image_id=image_id,
            reviewer="bob",
            role="initial",
            defect_grades=[{"defect_id": d, "joint_level": "IV"} for d in defect_ids],
        )
        assert out1["consensus"] is False
        assert out1["needs_arbitration"] is True
        assert out1["joint_level"] is None
        assert out1["need_review"] is True
        assert out1["review_count"] == 1

        # 仲裁：定 II 级 → 结案、落地、清空 need_review
        out2 = _review(
            client, image_id=image_id, reviewer="carol", role="arbitrator", overall_level="II"
        )
        assert out2["consensus"] is True
        assert out2["needs_arbitration"] is False
        assert out2["stage"] == "arbitrated"
        assert out2["joint_level"] == "II"
        assert out2["need_review"] is False
        assert out2["reviewed_by"] == "carol"
        assert out2["review_count"] == 2


def test_review_audit_chain(tmp_path) -> None:
    img = tmp_path / "rev3.png"
    _synthetic(img)
    with TestClient(app) as client:
        rep = _post_report(client, img, pixel_spacing_mm="0.1", base_metal_thickness_mm="20")
        image_id = rep["image_id"]
        _review(client, image_id=image_id, reviewer="alice", role="initial", defect_grades=[])
        # 审计：该影像的 review 动作应可查，且哈希链连续
        aud = client.get("/api/v1/audit", params={"object_id": image_id, "action": "review"})
        assert aud.status_code == 200
        body = aud.json()
        assert body["total"] >= 1
        entries = body["entries"]
        # 链：除首条外，每条 prev_hash == 上一条 hash
        for i in range(1, len(entries)):
            assert entries[i]["prev_hash"] == entries[i - 1]["hash"]


def test_review_404(tmp_path) -> None:
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/review",
            json={"image_id": "nope", "reviewer": "x", "role": "initial", "defect_grades": []},
        )
    assert resp.status_code == 404


def test_review_bad_role(tmp_path) -> None:
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/review",
            json={"image_id": "x", "reviewer": "x", "role": "watcher", "defect_grades": []},
        )
    assert resp.status_code == 422


def test_report_embedds_comparison_images(tmp_path) -> None:
    """报告应并排嵌入『送检原始影像』与『检测标注影像』两张图，便于人工判断。"""
    img = tmp_path / "cmp.png"
    _synthetic(img)
    with TestClient(app) as client:
        rep = _post_report(client, img, pixel_spacing_mm="0.1", base_metal_thickness_mm="20")
        pdf = client.get(rep["pdf_url"])
        reader = PdfReader(io.BytesIO(pdf.content))
        total = sum(len(p.images) for p in reader.pages)
    assert total >= 2, f"report should embed original+annotated images, got {total}"


def test_report_reads_unicode_path_image() -> None:
    """cv2.imread 在中文路径上会失败；改用 open+imdecode 后须能正常读取。"""
    from backend.infra.reporting.pdf_reporter import (
        _build_graph_bytes,
        _build_original_bytes,
    )

    cn_dir = Path(os.environ.get("TMP", "/tmp")) / "扫描检测_演示目录"
    cn_dir.mkdir(parents=True, exist_ok=True)
    try:
        cn_img = cn_dir / "焊缝底片.png"
        h, w = 190, 640
        rng = np.random.default_rng(7)
        arr = rng.normal(128.0, 2.0, (h, w)).astype(np.uint8)
        cv2.circle(arr, (120, 30), 10, 80, -1)
        ok, buf = cv2.imencode(".png", arr)
        assert ok
        cn_img.write_bytes(buf.tobytes())  # Python open 写，绕开 cv2.imwrite 中文路径坑

        defects = [{"class_name": "气孔", "bbox_px": [115, 20, 20, 20], "joint_level": "I"}]
        orig = _build_original_bytes(str(cn_img))
        anno = _build_graph_bytes(str(cn_img), defects)
        assert orig is not None, "中文路径原图读取失败"
        assert anno is not None, "中文路径标注图读取失败"
    finally:
        shutil.rmtree(cn_dir, ignore_errors=True)
