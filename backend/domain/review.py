"""人工复核闭环核心逻辑。

设计文档  要求"初评→复评→仲裁"双人评片一致性工作流：系统自动评级作为
"评片员 A"，人工复核提交作为"评片员 B"，用 Cohen's κ 量化一致性；κ 低于阈值
即分歧过大，升级第三名仲裁。本模块只做判定计算，不碰数据库/文件系统，便于
单测与在 app 层装配。

不替代  标准判定逻辑，仅在其外层加合规壳。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

# 级别严重度排序（IV 最严重），用于由逐缺陷级别推算综合级别
_LEVEL_ORDER: dict[str, int] = {"I": 1, "II": 2, "III": 3, "IV": 4}
# 未评级缺陷在 κ 计算中作为独立类别，避免"未评=自动级"的虚假一致
_UNRATED = "NR"
_DEFAULT_KAPPA_THRESHOLD = 0.8  # §15.3 高度一致阈值（κ≥0.8）


class ReviewRole(str, enum.Enum):
    """复核角色。"""

    INITIAL = "initial"  # 初评
    SECONDARY = "secondary"  # 复评
    ARBITRATOR = "arbitrator"  # 仲裁（最终权威）


class ReviewStage(str, enum.Enum):
    """复核阶段（响应 stage 字段）。"""

    PENDING = "pending"  # 待复核
    CONSENSUS = "consensus"  # 初评/复评达成一致
    ARBITRATED = "arbitrated"  # 仲裁结案
    NEEDS_ARBITRATION = "needs_arbitration"  # 分歧过大，待仲裁


def cohen_kappa(rater_a: list[str], rater_b: list[str]) -> float:
    """Cohen's κ（名义类别一致性），范围 [-1, 1]，1.0 = 完全一致。

    rater_a / rater_b 为两位评片员对每个缺陷给出的类别标签（等长）。
    未评级缺陷以 _UNRATED 表示，作为独立类别参与计算。
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a 与 rater_b 长度必须一致")
    n = len(rater_a)
    if n == 0:
        return 1.0  # 无缺陷 → 视为完全一致
    categories = sorted({*rater_a, *rater_b})
    observed = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n
    pa = {c: sum(1 for x in rater_a if x == c) / n for c in categories}
    pb = {c: sum(1 for x in rater_b if x == c) / n for c in categories}
    expected = sum(pa[c] * pb[c] for c in categories)
    if expected >= 1.0:
        return 1.0  # 所有样本同属一类且两评片员完全一致 → 满分
    return (observed - expected) / (1.0 - expected)


def classify_agreement(kappa: float, threshold: float = _DEFAULT_KAPPA_THRESHOLD) -> bool:
    """κ 是否达到高度一致阈值。"""
    return kappa >= threshold


@dataclass(frozen=True)
class ReviewDecision:
    """一次复核的判定结果（由 resolve_review 产出，供 app 层落库）。"""

    final_level: str | None  # 复核后综合级别（None=尚未定案，待仲裁）
    per_defect_level: dict[str, str | None]  # defect_id -> 最终逐缺陷级别
    consensus: bool  # 是否达成共识
    needs_arbitration: bool  # 是否升级仲裁
    stage: ReviewStage
    kappa: float  # 与自动评级一致性
    reviewed_by: str | None  # 最终定案人（达成共识/仲裁时填）
    need_review: bool  # 影像是否仍需人工关注


def resolve_review(
    *,
    auto_grades: list[str | None],
    defect_ids: list[str],
    reviewer_grades: dict[str, str],
    overall_level: str | None,
    reviewer: str,
    role: ReviewRole,
    kappa_threshold: float = _DEFAULT_KAPPA_THRESHOLD,
) -> ReviewDecision:
    """根据自动级别与人工复核提交，计算一致性并判定共识/仲裁。

    参数：
    - auto_grades: 各缺陷系统自动评级（None=未评级，计为 _UNRATED）；
    - defect_ids: 与 auto_grades 对齐的缺陷主键；
    - reviewer_grades: 复核对部分缺陷的级别覆盖（{defect_id: 级别}）；
    - overall_level: 复核显式综合级别（优先于按缺陷推算）；
    - reviewer: 评片员标识；role: 复核角色；kappa_threshold: 一致阈值。

    规则：
    - 仲裁(arbitrator)为最终权威，直接落地（consensus=True, 清空 need_review）；
    - 初评/复评：κ≥阈值 → 达成共识落地；否则升级仲裁（不改最终级别，保持 need_review）。
    """
    if len(auto_grades) != len(defect_ids):
        raise ValueError("auto_grades 与 defect_ids 长度必须一致")

    # 逐缺陷最终级别（含未复核缺陷，沿用自动级），供综合级别推算。
    per_defect: dict[str, str | None] = {}
    for d_id, ag in zip(defect_ids, auto_grades):
        a = ag if ag is not None else _UNRATED
        b = reviewer_grades.get(d_id, a)  # 未提交 → 沿用自动级（含未评级）
        per_defect[d_id] = b if b != _UNRATED else None

    # κ 仅基于"成对缺陷"：评片员实际复核（出现在 reviewer_grades 中）的缺陷，
    # 才代表其独立判断。未复核缺陷若计为"与自动级一致"会虚增一致性、掩盖真实
    # 分歧。reviewer_grades 为空 → 空列表 → cohen_kappa
    # 返回 1.0，即"无复核证据"时按现行约定视为一致（与既有测试/流程一致）。
    paired_a: list[str] = []
    paired_b: list[str] = []
    for d_id, ag in zip(defect_ids, auto_grades):
        if d_id in reviewer_grades:
            paired_a.append(ag if ag is not None else _UNRATED)
            paired_b.append(reviewer_grades[d_id])

    kappa = cohen_kappa(paired_a, paired_b)
    agreed = classify_agreement(kappa, kappa_threshold)

    if role is ReviewRole.ARBITRATOR:
        # 仲裁为最终权威：直接落地，清空复核标记
        return ReviewDecision(
            final_level=_derive_level(per_defect, overall_level),
            per_defect_level=per_defect,
            consensus=True,
            needs_arbitration=False,
            stage=ReviewStage.ARBITRATED,
            kappa=kappa,
            reviewed_by=reviewer,
            need_review=False,
        )

    if agreed:
        return ReviewDecision(
            final_level=_derive_level(per_defect, overall_level),
            per_defect_level=per_defect,
            consensus=True,
            needs_arbitration=False,
            stage=ReviewStage.CONSENSUS,
            kappa=kappa,
            reviewed_by=reviewer,
            need_review=False,
        )

    # 分歧过大 → 升级仲裁，暂不改动最终级别与缺陷级别
    return ReviewDecision(
        final_level=None,
        per_defect_level=per_defect,
        consensus=False,
        needs_arbitration=True,
        stage=ReviewStage.NEEDS_ARBITRATION,
        kappa=kappa,
        reviewed_by=None,
        need_review=True,
    )


def _derive_level(per_defect: dict[str, str | None], overall: str | None) -> str | None:
    """由逐缺陷最终级别推算综合级别：overall 优先，否则取最严重级别；无缺陷取 I。"""
    if overall:
        return overall
    rated = [lv for lv in per_defect.values() if lv is not None]
    if not rated:
        return "I"  # 无任何缺陷 → I 级（可接受）
    return max(rated, key=lambda lv: _LEVEL_ORDER.get(lv, 0))
