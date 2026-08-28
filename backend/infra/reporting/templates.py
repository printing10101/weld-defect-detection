"""报告模板（数据驱动，P2）。

模板 = YAML 数据文件，渲染引擎（reportlab）不变；新增/换版式 = 增加模板文件，
不改代码。默认模板
``templates/standard.yaml`` 与 v1 内联版式等价。

加载语义（宽容，不阻断出片）：
- 模板名先做路径消毒（仅 ``[A-Za-z0-9_-]``，拒绝目录穿越）；
- 未知/缺失/损坏 → 回退 standard 并告警；
- 自定义模板与 standard **深合并**（只覆盖部分键即可继承其余）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_LOG = logging.getLogger("scandetection.reporting.templates")

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")

# 兼容：带 .yaml 后缀的模板名也接受（如 "standard.yaml"）
_DEFAULT_NAME = "standard"


@dataclass(frozen=True)
class ReportTemplate:
    """报告版式/文案数据（数据驱动模板，供 _render 消费）。

    所有 ``*_fields`` 为语义键 → 显示文案的映射（值仍由渲染引擎按内容计算）；
    模板数据只控制"说什么、怎么标"，不控制排版算法。
    """

    name: str
    doc_title_prefix: str
    author: str
    cover_title: str
    meta_report_no: str
    meta_image_id: str
    meta_generated_at: str

    section_workpiece: str
    workpiece_fields: dict[str, str] = field(default_factory=dict)
    section_params: str = ""
    params_fields: dict[str, str] = field(default_factory=dict)
    section_iqi: str = ""
    iqi_fields: dict[str, str] = field(default_factory=dict)
    section_defects: str = ""
    defects_empty: str = ""
    defect_columns: tuple[str, ...] = ()
    section_comparison: str = ""
    section_basis: str = ""
    basis_empty: str = ""
    section_conclusion: str = ""
    level_text: str = ""
    no_level_text: str = ""
    review_warn: str = ""
    signer_text: str = ""
    fingerprint_text: str = ""
    disclaimer_label: str = ""
    fallback: str = "—"
    not_provided: str = "未提供"


def _deep_merge(base: dict, over: dict) -> dict:
    """递归合并：over 覆盖 base；dict 嵌套逐层合并，其余类型直接替换。"""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _tpl_from(raw: dict) -> ReportTemplate:
    """由合并后的原始 dict 构造模板（逐键取默认，防御缺失键）。"""
    fields = {
        "workpiece_fields": dict(raw.get("workpiece_fields") or {}),
        "params_fields": dict(raw.get("params_fields") or {}),
        "iqi_fields": dict(raw.get("iqi_fields") or {}),
    }
    return ReportTemplate(
        name=str(raw.get("name") or _DEFAULT_NAME),
        doc_title_prefix=str(raw.get("doc_title_prefix") or "射线检测评片报告"),
        author=str(raw.get("author") or "ScanDetection"),
        cover_title=str(raw.get("cover_title") or "射线焊缝缺陷智能检测评片报告"),
        meta_report_no=str(raw.get("meta_report_no") or "报告编号：{v}"),
        meta_image_id=str(raw.get("meta_image_id") or "影像编号：{v}"),
        meta_generated_at=str(raw.get("meta_generated_at") or "生成时间：{v}"),
        section_workpiece=str(raw.get("section_workpiece") or "一、工件信息"),
        section_params=str(raw.get("section_params") or "二、检测参数"),
        section_iqi=str(raw.get("section_iqi") or "三、影像质量校验（IQI / 黑度）"),
        section_defects=str(raw.get("section_defects") or "四、缺陷清单与当量尺寸"),
        defects_empty=str(raw.get("defects_empty") or "未检出缺陷。"),
        defect_columns=tuple(str(v) for v in (raw.get("defect_columns") or [])),
        section_comparison=str(
            raw.get("section_comparison") or "五、检测影像对比（送检原始影像 vs 检测标注影像）"
        ),
        section_basis=str(raw.get("section_basis") or "六、判定依据条款"),
        basis_empty=str(raw.get("basis_empty") or "无自动判定依据（标准数值未授权或无需判定）。"),
        section_conclusion=str(raw.get("section_conclusion") or "七、结论"),
        level_text=str(raw.get("level_text") or "综合评定级别：{level} 级"),
        no_level_text=str(raw.get("no_level_text") or "无法自动评级，需人工复核。"),
        review_warn=str(raw.get("review_warn") or "⚠ 本报告标注需要人工复核。"),
        signer_text=str(raw.get("signer_text") or "签字：{signer}"),
        fingerprint_text=str(
            raw.get("fingerprint_text") or "数字指纹：SHA-256:{fp}（报告内容防篡改校验）"
        ),
        disclaimer_label=str(raw.get("disclaimer_label") or "免责声明：{text}"),
        fallback=str(raw.get("fallback") or "—"),
        not_provided=str(raw.get("not_provided") or "未提供"),
        **fields,
    )


def _read_template(name: str, templates_dir: Path) -> dict:
    """读取单模板文件 → dict；缺失/损坏抛 ValueError（由调用方回退）。"""
    path = templates_dir / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"报告模板不存在: {name}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"报告模板解析失败: {name}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TypeError(f"报告模板结构非法（应为映射）: {name}")
    return raw


def load_report_template(
    name: str = _DEFAULT_NAME, templates_dir: Path | None = None
) -> ReportTemplate:
    """按名加载报告模板（深合并 standard；未知/损坏回退 standard，不阻断出片）。

    ``name`` 允许带 ``.yaml`` 后缀；路径消毒：仅 ``[A-Za-z0-9_-]`` 字符，
    拒绝 ``..``/绝对路径（模板名可能来自 API 查询参数）。

    合并基准始终取**包内** standard.yaml（随包分发）；``templates_dir`` 仅用于
    读取自定义模板（默认同包目录），因此自定义目录缺失 standard 也不影响回退。
    """
    base_dir = templates_dir or _TEMPLATES_DIR
    stem = Path(name).name.removesuffix(".yaml")
    if not _SAFE_NAME.match(stem) or stem == ".." or "/" in name or "\\" in name:
        _LOG.warning("非法模板名 %r，回退 %s", name, _DEFAULT_NAME)
        stem = _DEFAULT_NAME
    base = _read_template(_DEFAULT_NAME, _TEMPLATES_DIR)
    if stem == _DEFAULT_NAME:
        return _tpl_from(base)
    try:
        custom = _read_template(stem, base_dir)
    except ValueError as exc:
        _LOG.warning("自定义模板不可用（%s），回退 %s", exc, _DEFAULT_NAME)
        return _tpl_from(base)
    merged = _deep_merge(base, custom)
    merged["name"] = stem
    return _tpl_from(merged)
