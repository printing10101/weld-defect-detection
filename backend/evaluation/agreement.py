"""评级一致率 harness（技术规格 §15.3 / DB50/T 1807-2025）。

规格承诺：与高级评片员比对的评级一致率 ≥95%、Cohen's κ ≥0.8。

输入为配对的（自动评级, 人工评级）序列——数据经人工复核工作流产生
（每条复核记录的 defect 级别对），可由 run_spec_eval CLI 以 JSONL 喂入。
未评级（None，如判定熔断）作为独立类别参与计算，不做剔除——剔除会
虚高一致率，违反诚实评估口径。
"""

from __future__ import annotations

from typing import Any

from backend.domain.review import cohen_kappa

_UNRATED = "<无级别>"

_DEFAULT_MIN_AGREEMENT = 0.95
_DEFAULT_MIN_KAPPA = 0.8

# 供 CLI 默认值引用（与 review.kappa_threshold 配置默认保持同步，配置为权威）。
DEFAULT_MIN_AGREEMENT = _DEFAULT_MIN_AGREEMENT
DEFAULT_MIN_KAPPA = _DEFAULT_MIN_KAPPA


def grading_agreement(
    auto_grades: list[str | None],
    human_grades: list[str | None],
    *,
    min_agreement: float = DEFAULT_MIN_AGREEMENT,
    min_kappa: float = DEFAULT_MIN_KAPPA,
) -> dict[str, Any]:
    """配对评级的一致率 + Cohen's κ + 阈值判定。

    两个序列必须等长（同一缺陷集合的两个评级来源）。返回 dict 可 JSON 化，
    结构与 std501807 的输出风格一致（数值 + verdict 布尔）。

    诚实评估口径：空输入（n=0）不构成达标证据，verdict 判不通过——
    与 calibration/quant_agreement 两个 harness 的空输入语义一致。
    """
    if len(auto_grades) != len(human_grades):
        raise ValueError("auto_grades 与 human_grades 长度必须一致")
    n = len(auto_grades)
    a = [g if g is not None else _UNRATED for g in auto_grades]
    h = [g if g is not None else _UNRATED for g in human_grades]

    agree = sum(1 for x, y in zip(a, h) if x == y)
    agreement_rate = agree / n if n else 0.0
    kappa = cohen_kappa(a, h)

    # 混淆矩阵：{自动级别: {人工级别: 计数}}（分歧的追溯清单）
    confusion: dict[str, dict[str, int]] = {}
    for x, y in zip(a, h):
        confusion.setdefault(x, {}).setdefault(y, 0)
        confusion[x][y] += 1

    return {
        "n_pairs": n,
        "n_agree": agree,
        # 空输入按 0.0 呈现而非 1.0：无数据不虚报一致率。
        "agreement_rate": round(agreement_rate, 4),
        "cohens_kappa": round(kappa, 4),
        "confusion": confusion,
        "thresholds": {"min_agreement": min_agreement, "min_kappa": min_kappa},
        "verdict": {
            "agreement_pass": n > 0 and agreement_rate >= min_agreement,
            "kappa_pass": n > 0 and kappa >= min_kappa,
            "passed": n > 0 and agreement_rate >= min_agreement and kappa >= min_kappa,
        },
    }
