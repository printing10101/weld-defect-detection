"""集成测试：report 全链路（评片→落库→PDF）与 records 检索统计。"""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module", autouse=True)
def _authorized_grader(auth_table) -> Iterator[None]:
    """report 全链路需要正常评级：用 authorized 测试表替换全局 grader。

    生产表 authorized=false 会熔断，故本模块注入测试表副本；
    注：本机 pytest 按文件名字母序执行，test_judge_api（依赖熔断）先于本模块。

    合成底片为低黑度噪声图，RQI/硬门禁天然不过质量门槛——本模块测的是
    评片/复核**流程**而非质量门禁（后者由 test_preprocess 单测直接覆盖），
    故关闭 block_on_quality，退出时还原（不改变生产配置）。
    """
    from backend.app import dependencies as deps
    from backend.domain.grade.nb47013 import Nb47013Grader
    from backend.domain.standards.tables.loader import load_standard_tables

    deps._registry = None  # 强制按测试 env 重建（paths 隔离）
    reg = deps.get_registry()
    reg.grader = Nb47013Grader(load_standard_tables("NB/T47013.2-2015", filename=str(auth_table)))
    orig_block = reg.config.quality.block_on_quality
    reg.config.quality.block_on_quality = False
    try:
        yield
    finally:
        reg.config.quality.block_on_quality = orig_block


def _synthetic(path) -> None:
    """噪声图 + 19 根 IQI 丝（亮线）+ 2 个暗斑缺陷。"""
    n, h, w = 19, 190, 640
    rng = np.random.default_rng(0)
    img = rng.normal(128.0, 2.0, (h, w)).astype(np.uint8)
    for i in range(n):
        y = round((i + 0.5) / n * h)
        cv2.line(img, (0, y), (w - 1, y), int(128 + 40.0), 3)
    cv2.circle(img, (120, 30), 10, 80, -1)
    cv2.circle(img, (420, 150), 7, 85, -1)
    cv2.imwrite(str(path), img)


def _post_report(client: TestClient, image_path, **form) -> dict:
    """提交评片。

    合成底片黑度不达标（evaluable=False），按设计文档"不通过则阻断评片"
    默认会被 409 拦截；测试链路显式带 force=true 走"出片但不定级"分支。
    """
    form.setdefault("force", "true")
    with open(image_path, "rb") as f:
        resp = client.post(
            "/api/v1/report",
            files={"image": (image_path.name, f, "image/png")},
            data=form,
        )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_report_blocks_unevaluable_film(tmp_path) -> None:
    """129 行：黑度/IQI 不通过是硬前置，必须阻断评片（409 IQI_FAIL）。"""
    img = tmp_path / "syn_block.png"
    _synthetic(img)
    with TestClient(app) as client, open(img, "rb") as f:
        resp = client.post(
            "/api/v1/report",
            files={"image": (img.name, f, "image/png")},
            data={"pixel_spacing_mm": "0.1", "base_metal_thickness_mm": "20"},
        )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "IQI_FAIL"


def test_report_full_pipeline(tmp_path) -> None:
    img = tmp_path / "syn.png"
    _synthetic(img)
    with TestClient(app) as client:
        body = _post_report(
            client,
            img,
            pixel_spacing_mm="0.1",
            base_metal_thickness_mm="20",
            workpiece_no="WP-M6-001",
            weld_no="W-1",
            signer="tester",
        )
    assert body["report_id"]
    assert body["image_id"]
    assert body["evaluable"] is False  # 合成图黑度不达标
    # 不合格底片不构成评定依据：force 出片但不得输出级别，改走人工复核
    assert body["joint_level"] is None
    assert body["need_review"] is True
    assert body["defect_count"] >= 1
    assert body["pdf_url"].startswith("/api/v1/report/")
    # 报告端点必须随结果返回强免责声明（authorized_copy=false），
    # 与 judge 端点同源，明确"非标准授权正本 / 不替代责任工程师法定评定"。
    assert body["disclaimer"] is not None
    assert "非标准授权正本" in body["disclaimer"]
    assert "法定判定" in body["disclaimer"]

    # PDF 可下载且为合法 PDF
    with TestClient(app) as client:
        pdf = client.get(body["pdf_url"])
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content[:4] == b"%PDF"
    assert len(pdf.content) > 1000


def test_report_grades_when_film_evaluable(tmp_path, monkeypatch) -> None:
    """底片可评时应正常输出级别（保留判定链路覆盖）。

    合成图为 8bit，按 D=log10(2^8/(G+1)) 其黑度天然远低于 AB 级 2.0 门限，
    故放宽下限模拟一张合格底片，再验证全链路评级。
    """
    from backend.app import dependencies as deps

    reg = deps.get_registry()
    monkeypatch.setattr(reg.config.density, "low", 0.0)

    img = tmp_path / "syn_ok.png"
    _synthetic(img)
    with TestClient(app) as client, open(img, "rb") as f:
        resp = client.post(
            "/api/v1/report",
            files={"image": (img.name, f, "image/png")},
            data={"pixel_spacing_mm": "0.1", "base_metal_thickness_mm": "20"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["evaluable"] is True
    assert body["joint_level"] is not None  # authorized 测试表 → 正常评级


def test_report_without_spacing_refuses_to_grade(tmp_path, monkeypatch) -> None:
    """缺少像素标定时禁止定级：原实现会伪造 1.0 mm/px 后照常出级别。"""
    from backend.app import dependencies as deps

    reg = deps.get_registry()
    monkeypatch.setattr(reg.config.density, "low", 0.0)

    img = tmp_path / "syn_nospacing.png"
    _synthetic(img)
    with TestClient(app) as client, open(img, "rb") as f:
        resp = client.post(
            "/api/v1/report",
            files={"image": (img.name, f, "image/png")},
            data={"base_metal_thickness_mm": "20"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["joint_level"] is None
    assert body["need_review"] is True


def test_report_rejects_non_positive_spacing(tmp_path) -> None:
    img = tmp_path / "syn_badspacing.png"
    _synthetic(img)
    with TestClient(app) as client, open(img, "rb") as f:
        resp = client.post(
            "/api/v1/report",
            files={"image": (img.name, f, "image/png")},
            data={"pixel_spacing_mm": "0", "base_metal_thickness_mm": "20"},
        )
    assert resp.status_code == 422


def test_report_regenerate_by_image_id(tmp_path) -> None:
    img = tmp_path / "syn2.png"
    _synthetic(img)
    with TestClient(app) as client:
        first = _post_report(client, img, pixel_spacing_mm="0.1", base_metal_thickness_mm="20")
        second = client.post(
            "/api/v1/report",
            data={"image_id": first["image_id"], "template": "standard"},
        )
    assert second.status_code == 200
    body = second.json()
    assert body["image_id"] == first["image_id"]
    assert body["report_id"]  # 新报告（同影像重新生成）
    assert body["joint_level"] == first["joint_level"]
    # 合规处置建议随报告输出（新评片与重新生成两条路径都应有）
    for resp in (first, body):
        assert resp["disposition"] in {"accept", "conditional", "rework", "recheck"}
        assert isinstance(resp["disposition_label"], str) and resp["disposition_label"]
        assert isinstance(resp["disposition_actions"], list) and resp["disposition_actions"]


def test_report_missing_input() -> None:
    with TestClient(app) as client:
        resp = client.post("/api/v1/report")
    assert resp.status_code == 422


def test_report_pdf_404(tmp_path) -> None:
    with TestClient(app) as client:
        resp = client.get("/api/v1/report/nope/pdf")
    assert resp.status_code == 404


def test_records_list_and_stats(tmp_path) -> None:
    img = tmp_path / "syn3.png"
    _synthetic(img)
    with TestClient(app) as client:
        _post_report(client, img, pixel_spacing_mm="0.1", base_metal_thickness_mm="20")
        resp = client.get("/api/v1/records")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] >= 1
    assert len(body["items"]) >= 1
    it = body["items"][0]
    assert "image_id" in it and "joint_level" in it and "created_at" in it
    assert "defect_count" in it
    assert body["stats"]["total"] >= 1
    assert "by_level" in body["stats"] and "by_class" in body["stats"]


def test_records_filter_by_level(tmp_path) -> None:
    """级别过滤语义：命中项级别必须全等于过滤值，未定级记录不得混入。

    注：测试库在模块内共享，其他用例可能已写入任意级别的记录，故不能断言
    某级别计数为 0（脆弱），改为断言过滤器语义本身。
    """
    img = tmp_path / "syn4.png"
    _synthetic(img)
    with TestClient(app) as client:
        created = _post_report(client, img, pixel_spacing_mm="0.1", base_metal_thickness_mm="20")
        resp = client.get("/api/v1/records", params={"level": "IV", "size": 100})
    assert resp.status_code == 200
    body = resp.json()
    assert all(it["joint_level"] == "IV" for it in body["items"])
    assert created["joint_level"] is None  # force 出片但不定级
    assert created["image_id"] not in [it["image_id"] for it in body["items"]]


def test_records_rejects_invalid_level() -> None:
    """非法级别应在入口 422，而非落到仓储层 ValueError→500。"""
    with TestClient(app) as client:
        resp = client.get("/api/v1/records", params={"level": "V"})
    assert resp.status_code == 422
