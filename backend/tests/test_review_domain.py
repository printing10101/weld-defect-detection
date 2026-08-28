"""单元测试：复核一致性计算（Cohen's κ）与共识/仲裁判定。"""

from __future__ import annotations

import pytest

from backend.domain.review import (
    ReviewRole,
    ReviewStage,
    classify_agreement,
    cohen_kappa,
    resolve_review,
)


def test_kappa_perfect_agreement() -> None:
    a = ["I", "II", "III", "IV"]
    assert cohen_kappa(a, a) == 1.0


def test_kappa_empty_is_one() -> None:
    assert cohen_kappa([], []) == 1.0


def test_kappa_total_disagreement_negative() -> None:
    # 两位评片员在所有样本上都不同（两类交替）→ 一致性应 < 0
    a = ["I", "II", "I", "II"]
    b = ["II", "I", "II", "I"]
    k = cohen_kappa(a, b)
    assert k < 0


def test_kappa_partial_agreement_between_bounds() -> None:
    a = ["I", "I", "II", "II"]
    b = ["I", "II", "II", "I"]  # 2/4 一致
    k = cohen_kappa(a, b)
    assert -1.0 < k < 1.0


def test_classify_agreement_threshold() -> None:
    assert classify_agreement(0.85)
    assert not classify_agreement(0.75)


def test_resolve_consensus_full_agreement() -> None:
    dec = resolve_review(
        auto_grades=["II", "III"],
        defect_ids=["d1", "d2"],
        reviewer_grades={},
        overall_level=None,
        reviewer="alice",
        role=ReviewRole.INITIAL,
        kappa_threshold=0.8,
    )
    assert dec.consensus is True
    assert dec.needs_arbitration is False
    assert dec.stage is ReviewStage.CONSENSUS
    assert dec.final_level == "III"  # 最严重级别
    assert dec.need_review is False
    assert dec.reviewed_by == "alice"


def test_resolve_disagreement_needs_arbitration() -> None:
    dec = resolve_review(
        auto_grades=["I", "I"],
        defect_ids=["d1", "d2"],
        reviewer_grades={"d1": "IV", "d2": "IV"},
        overall_level=None,
        reviewer="bob",
        role=ReviewRole.INITIAL,
        kappa_threshold=0.8,
    )
    assert dec.consensus is False
    assert dec.needs_arbitration is True
    assert dec.stage is ReviewStage.NEEDS_ARBITRATION
    assert dec.final_level is None  # 未定案
    assert dec.need_review is True
    assert dec.reviewed_by is None


def test_resolve_arbitrator_is_final_authority() -> None:
    dec = resolve_review(
        auto_grades=["I", "I"],
        defect_ids=["d1", "d2"],
        reviewer_grades={"d1": "IV", "d2": "IV"},
        overall_level="II",
        reviewer="carol",
        role=ReviewRole.ARBITRATOR,
        kappa_threshold=0.8,
    )
    assert dec.consensus is True
    assert dec.needs_arbitration is False
    assert dec.stage is ReviewStage.ARBITRATED
    assert dec.final_level == "II"  # 仲裁显式级别优先
    assert dec.need_review is False
    assert dec.reviewed_by == "carol"


def test_resolve_unrated_defect_treated_as_category() -> None:
    # 自动未评级(None) + 复核给出级别 → 视为分歧（NR vs 实级）
    dec = resolve_review(
        auto_grades=[None],
        defect_ids=["d1"],
        reviewer_grades={"d1": "II"},
        overall_level=None,
        reviewer="alice",
        role=ReviewRole.INITIAL,
        kappa_threshold=0.8,
    )
    assert dec.needs_arbitration is True
    assert dec.per_defect_level["d1"] == "II"


@pytest.mark.parametrize("role", ["initial", "secondary", "arbitrator"])
def test_resolve_accepts_all_roles(role: str) -> None:
    dec = resolve_review(
        auto_grades=["II"],
        defect_ids=["d1"],
        reviewer_grades={},
        overall_level=None,
        reviewer="x",
        role=ReviewRole(role),
    )
    assert dec is not None
