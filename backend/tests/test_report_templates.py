"""P2 报告模板数据化测试（§7.2）。

覆盖：
- 默认模板 standard.yaml 加载：关键版式文案与 v1 内联版式等价；
- 自定义模板部分覆盖 → 与 standard 深合并（继承其余）；
- 未知/损坏/非法模板名 → 宽容回退 standard（不阻断出片，含路径消毒）；
- 端到端：report API 传未知模板名仍 200 出片（回退路径验证）。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.infra.reporting.templates import (
    ReportTemplate,
    _TEMPLATES_DIR,
    load_report_template,
)


# ---------------------------------------------------------------------------
# 加载器（单元）
# ---------------------------------------------------------------------------


def test_load_standard_template_matches_v1_content() -> None:
    tpl = load_report_template("standard")
    assert isinstance(tpl, ReportTemplate)
    assert tpl.name == "standard"
    # 与 v1 内联版式等价（§7.2 数据化不改变默认输出）
    assert tpl.cover_title == "射线焊缝缺陷智能检测评片报告"
    assert tpl.doc_title_prefix == "射线检测评片报告"
    assert tpl.section_workpiece == "一、工件信息"
    assert tpl.section_defects == "四、缺陷清单与当量尺寸"
    assert tpl.defect_columns[0] == "#" and "评级" in tpl.defect_columns
    assert tpl.workpiece_fields["workpiece_no"] == "工件号"
    assert tpl.params_fields["pixel_spacing"] == "像素标定"
    assert tpl.iqi_fields["evaluable"] == "可评片性"
    assert tpl.disclaimer_label == "免责声明：{text}"
    assert tpl.fallback == "—"
    assert tpl.not_provided == "未提供"


def test_custom_partial_template_merges_over_standard(tmp_path: Path) -> None:
    """自定义模板只覆盖部分键 → 深合并继承 standard 其余。"""
    (tmp_path / "brand.yaml").write_text(
        "cover_title: 某某检测中心射线评片报告\n"
        "section_conclusion: 八、结论与签署\n"
        "name: brand\n",
        encoding="utf-8",
    )
    tpl = load_report_template("brand", templates_dir=tmp_path)
    assert tpl.name == "brand"
    assert tpl.cover_title == "某某检测中心射线评片报告"  # 覆盖生效
    assert tpl.section_conclusion == "八、结论与签署"
    # 未覆盖键继承 standard
    assert tpl.section_workpiece == "一、工件信息"
    assert tpl.defect_columns[0] == "#"
    assert tpl.workpiece_fields["workpiece_no"] == "工件号"


def test_unknown_template_falls_back_to_standard() -> None:
    tpl = load_report_template("no_such_template")
    assert tpl.name == "standard"
    assert tpl.cover_title == "射线焊缝缺陷智能检测评片报告"


def test_corrupt_template_falls_back_to_standard(tmp_path: Path) -> None:
    (tmp_path / "broken.yaml").write_text("cover_title: [unclosed", encoding="utf-8")
    tpl = load_report_template("broken", templates_dir=tmp_path)
    assert tpl.name == "standard"


def test_unsafe_template_name_sanitized() -> None:
    """模板名可能来自 API → 路径穿越/非法字符一律回退 standard。"""
    assert load_report_template("../../etc/passwd").name == "standard"
    assert load_report_template("..\\..\\x").name == "standard"
    assert load_report_template("a/b").name == "standard"
    assert load_report_template("").name == "standard"


def test_template_name_with_yaml_suffix_accepted() -> None:
    tpl = load_report_template("standard.yaml")
    assert tpl.name == "standard"
    assert tpl.section_workpiece == "一、工件信息"


# ---------------------------------------------------------------------------
# 端到端：report API 传未知模板名 → 宽容回退 standard，正常出片
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _authorized_grader(auth_table) -> Iterator[None]:
    """与 test_report_api 相同：注入 authorized 测试表 + 关质量门禁（流程而非门禁）。"""
    from backend.app import dependencies as deps
    from backend.domain.grade.nb47013 import Nb47013Grader
    from backend.domain.standards.tables.loader import load_standard_tables

    deps._registry = None
    reg = deps.get_registry()
    reg.grader = Nb47013Grader(load_standard_tables("NB/T47013.2-2015", filename=str(auth_table)))
    orig_block = reg.config.quality.block_on_quality
    reg.config.quality.block_on_quality = False
    try:
        yield
    finally:
        reg.config.quality.block_on_quality = orig_block


def test_report_api_unknown_template_falls_back(tmp_path: Path) -> None:
    """未知模板名 → build 回退 standard，report API 仍 200 出片（不阻断）。"""
    import cv2
    import numpy as np

    img = tmp_path / "syn_tpl.png"
    rng = np.random.default_rng(0)
    arr = rng.normal(128.0, 2.0, (190, 640)).astype(np.uint8)
    cv2.imwrite(str(img), arr)

    with TestClient(app) as client, open(img, "rb") as f:
        resp = client.post(
            "/api/v1/report",
            files={"image": (img.name, f, "image/png")},
            data={
                "pixel_spacing_mm": "0.1",
                "base_metal_thickness_mm": "20",
                "force": "true",
                "template": "no_such_template",
            },
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["report_id"]


def test_default_template_file_ships_with_package() -> None:
    """默认模板随包分发（Tauri 打包/任意 CWD 均可用）。"""
    assert (_TEMPLATES_DIR / "standard.yaml").exists()
