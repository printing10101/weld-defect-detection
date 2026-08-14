"""报告内容组装（§7.2，纯逻辑无 I/O）。

模板章节：封面 / 工件信息 / 检测参数 / IQI 与黑度校验结论 / 缺陷清单与当量尺寸
/ 判定依据条款 / 结论。v1 统一模板（template 参数预留，后续可扩展多模板）。
本模块只做"数据 → 报告内容"的纯组装；PDF 渲染在 infra/reporting（读图/读写文件）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_DISCLAIMER_DEFAULT = (
    "⚠ 标准来源声明：本系统评级数值转录自公开解读文本，非标准授权正本；"
    "未经授权原文逐条复核与正式签核前，报告结论仅用于 AI 辅助预筛与质量追溯参考，"
    "不构成合格/不合格的法定判定依据，最终级别须经责任工程师复核并签核后方可采信。"
)


@dataclass(frozen=True)
class ReportContent:
    """报告渲染所需全部数据（与渲染实现解耦）。"""

    report_id: str
    image_id: str
    generated_at: str
    workpiece_no: str | None
    weld_no: str | None
    source_type: str
    modality: str
    pixel_spacing_mm: float | None
    base_metal_thickness_mm: float | None
    iqi_pass: bool | None
    iqi_detail: dict[str, Any] | None
    density: float | None
    density_ok: bool | None
    evaluable: bool
    defects: tuple[dict[str, Any], ...]  # 缺陷行（类别/几何/评级）
    joint_level: str | None
    need_review: bool
    basis: tuple[str, ...]
    standard_ref: str
    signer: str | None
    disclaimer: str
    fingerprint: str | None = None  # 报告内容指纹（SHA-256，§7.2 数字签名，PDF 页脚展示）


def build_report_content(
    image: dict[str, Any],
    defects: list[dict[str, Any]],
    report: dict[str, Any] | None,
    disclaimer: str | None = None,
    fingerprint: str | None = None,
) -> ReportContent:
    """从检查记录（repository 返回的 dict）组装报告内容。

    - joint_level 为 None（未授权熔断/不可评级）时结论标注"无法自动评级，需人工复核"；
    - evaluable=False（IQI/黑度不达标）时报告带免责声明，不冒充正式评片。
    """
    return ReportContent(
        report_id=str((report or {}).get("report_id") or ""),
        image_id=str(image["image_id"]),
        generated_at=str((report or {}).get("generated_at") or image.get("created_at") or ""),
        workpiece_no=image.get("workpiece_no"),
        weld_no=image.get("weld_no"),
        source_type=str(image.get("source_type") or "image"),
        modality=str(image.get("modality") or "GENERIC"),
        pixel_spacing_mm=image.get("pixel_spacing_mm"),
        base_metal_thickness_mm=image.get("base_metal_thickness_mm"),
        iqi_pass=image.get("iqi_pass"),
        iqi_detail=image.get("iqi_detail"),
        density=image.get("density"),
        density_ok=image.get("density_ok"),
        evaluable=bool(image.get("evaluable", True)),
        defects=tuple(defects),
        joint_level=image.get("joint_level"),
        need_review=bool(image.get("need_review", False)),
        basis=tuple((report or {}).get("basis") or []),
        standard_ref=str(image.get("standard_id") or ""),
        signer=(report or {}).get("signer"),
        disclaimer=disclaimer or _DISCLAIMER_DEFAULT,
        fingerprint=fingerprint,
    )
