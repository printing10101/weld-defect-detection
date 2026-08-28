"""合规处置建议引擎（独立适配器，P0-E）。

双轨纪律：本模块**独立于 domain/grade 主干**，只消费评级输出（级别 + 缺陷类别），
输出「验收 / 有条件 / 返修 / 判废 / 复核」处置建议，**不参与任何评级计算**、
不读取评级数值表、不改动 GradeResult 契约。

规则（通用工业语义，NB/T47013.2 体系）：
- 零容忍类（裂纹 / 未熔合 / 未焊透）存在 → 不合格（返修或判废），与级别无关；
- 深孔（黑度>母材）→ 直判不合格（返修或判废）；
- 级别 IV → 不合格（返修或判废）；
- 系统判定 need_review（高不确定性 / 尺寸临界 / 复核兜底）→ 暂缓处置，人工复核后定；
- 级别 III → 有条件验收（按设计 / 合同文件评估）；
- 级别 I / II → 验收合格；
- 级别不可得（方法标准 / 未授权熔断 / 未知）→ 复核（人工判定），不臆造结论。

容错承诺：本引擎**永不抛错**——任何输入（含熔断场景）都退化为「需人工复核」，
并附免责声明，绝不输出误导性的合格/不合格结论。
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.domain.dto import DefectClass, Detection, GradeResult, JointLevel

# 零容忍缺陷（NB/T47013.2-2015：I-III 级均不允许，存在即 IV 级）
_ZERO_TOLERANCE = {
    DefectClass.CRACK,
    DefectClass.LACK_OF_FUSION,
    DefectClass.INCOMPLETE_PENETRATION,
}

# 处置代码（机器可读，前端据此渲染徽标/样式）
DISPOSITION_ACCEPT = "accept"  # 验收合格
DISPOSITION_CONDITIONAL = "conditional"  # 有条件验收
DISPOSITION_REWORK = "rework"  # 不合格（返修/判废）
DISPOSITION_RECHECK = "recheck"  # 暂缓处置，人工复核

_LABELS = {
    DISPOSITION_ACCEPT: "验收合格",
    DISPOSITION_CONDITIONAL: "有条件验收",
    DISPOSITION_REWORK: "不合格（返修/判废）",
    DISPOSITION_RECHECK: "需人工复核",
}

_DEFAULT_DISCLAIMER = (
    "⚠ 处置建议仅依据评级结果与通用验收语义生成，供 AI 辅助预筛与质量追溯参考；"
    "不构成合格/不合格的法定判定依据。最终处置须由持证人/责任工程师依授权标准原文"
    "复核并签核后方可采信。"
)


@dataclass(frozen=True)
class Recommendation:
    """一条合规处置建议（机器可读 + 人类可读动作）。"""

    disposition: str  # accept | conditional | rework | recheck
    disposition_label: str  # 中文标签
    actions: tuple[str, ...]  # 具体处置动作（按序执行）
    basis: tuple[str, ...]  # 建议依据（人类可读）
    standard_id: str
    disclaimer: str | None = None


def _rework(
    standard_id: str, zero_tolerance: bool = False, disclaimer: str | None = None
) -> Recommendation:
    basis = (
        "存在零容忍类缺陷（裂纹/未熔合/未焊透），NB/T47013.2-2015 规定 I-III 级均不允许"
        if zero_tolerance
        else "评级为 IV 级（不合格），不满足验收要求",
    )
    return Recommendation(
        disposition=DISPOSITION_REWORK,
        disposition_label=_LABELS[DISPOSITION_REWORK],
        actions=(
            "判为不合格",
            "按标准要求进行返修（打磨补焊等）",
            "返修后重新拍片并重新评级",
            "无法返修或返修后仍不合格时判废",
        ),
        basis=basis,
        standard_id=standard_id,
        disclaimer=disclaimer,
    )


def recommend(
    level: str | JointLevel | None,
    defects: list[Detection],
    *,
    need_review: bool = False,
    standard_id: str = "NB/T47013.2-2015",
    disclaimer: str | None = None,
) -> Recommendation:
    """由评级输出生成处置建议（纯函数，可单测）。

    level      : 综合级别（'I'~'IV' 或 JointLevel；None=级别不可得/方法标准）
    defects    : 参与评级的缺陷（用于零容忍/深孔判定）
    need_review: 系统人工复核标志（高不确定性/尺寸临界/复核兜底）
    """
    dl = disclaimer if disclaimer is not None else _DEFAULT_DISCLAIMER

    # 1) 零容忍类：无论级别一律不合格（与评级引擎同语义，避免级别口径不一致）
    if any(d.class_id in _ZERO_TOLERANCE for d in defects):
        return _rework(standard_id, zero_tolerance=True, disclaimer=dl)

    # 2) 深孔（黑度>母材，§6.2 直判 IV）：不合格
    if any(d.deep_hole for d in defects):
        return _rework(standard_id, zero_tolerance=False, disclaimer=dl)

    if level is None:
        return Recommendation(
            disposition=DISPOSITION_RECHECK,
            disposition_label=_LABELS[DISPOSITION_RECHECK],
            actions=(
                "当前标准不输出缺陷级别（方法标准）或评级数值未授权",
                "由持证人/责任工程师依标准原文人工判定",
                "按人工判定结论执行验收或返修",
            ),
            basis=("级别不可得：方法标准或未授权熔断，禁止自动处置",),
            standard_id=standard_id,
            disclaimer=dl,
        )

    if isinstance(level, JointLevel):
        level = level.value

    # 3) IV 级：不合格
    if level == "IV":
        return _rework(standard_id, zero_tolerance=False, disclaimer=dl)

    # 4) 需人工复核：暂缓处置（即使级别 I/II，临界/高不确定也不直接放行）
    if need_review:
        return Recommendation(
            disposition=DISPOSITION_RECHECK,
            disposition_label=_LABELS[DISPOSITION_RECHECK],
            actions=(
                "暂不直接判合格：系统检测到高不确定性/尺寸临界缺陷",
                "由评片员复核并确认级别",
                "按复核后的级别执行验收或返修",
            ),
            basis=(f"系统评级 {level} 级，但触发人工复核兜底（need_review）",),
            standard_id=standard_id,
            disclaimer=dl,
        )

    # 5) III 级：有条件验收
    if level == "III":
        return Recommendation(
            disposition=DISPOSITION_CONDITIONAL,
            disposition_label=_LABELS[DISPOSITION_CONDITIONAL],
            actions=(
                "不单独构成直接合格判定",
                "按设计文件/合同验收等级评估是否可接受",
                "如设计验收等级为 II 级，则本底片不合格，须返修",
            ),
            basis=("系统评级 III 级，是否合格取决于设计/合同验收等级",),
            standard_id=standard_id,
            disclaimer=dl,
        )

    # 6) I / II 级：验收合格
    if level in ("I", "II"):
        return Recommendation(
            disposition=DISPOSITION_ACCEPT,
            disposition_label=_LABELS[DISPOSITION_ACCEPT],
            actions=(
                "验收合格（按 NB/T47013.2-2015 级别体系）",
                "允许放行；建议按质量管理要求留存底片与报告",
            ),
            basis=(f"系统评级 {level} 级，满足一般构件验收要求",),
            standard_id=standard_id,
            disclaimer=dl,
        )

    # 7) 未知级别字符串：不臆造结论，交人工
    return Recommendation(
        disposition=DISPOSITION_RECHECK,
        disposition_label=_LABELS[DISPOSITION_RECHECK],
        actions=("未知级别，由责任工程师人工判定",),
        basis=(f"无法识别的级别 {level!r}，禁止自动处置",),
        standard_id=standard_id,
        disclaimer=dl,
    )


def recommend_from_grade(
    result: GradeResult,
    defects: list[Detection],
    *,
    disclaimer: str | None = None,
) -> Recommendation:
    """由 GradeResult 便捷生成建议（供 judge/report 链路复用）。"""
    return recommend(
        result.joint_level,
        defects,
        need_review=result.need_review,
        standard_id=result.standard_id,
        disclaimer=disclaimer if disclaimer is not None else result.disclaimer,
    )


__all__ = [
    "DISPOSITION_ACCEPT",
    "DISPOSITION_CONDITIONAL",
    "DISPOSITION_RECHECK",
    "DISPOSITION_REWORK",
    "Recommendation",
    "recommend",
    "recommend_from_grade",
]
