"""契约测试（§T7 / §19.6）：验证 DTO/接口/异常符合 §T2 冻结契约。"""
from __future__ import annotations

from typing import Protocol

from backend.domain import dto, errors, interfaces


def test_protocols_are_runtime_checkable() -> None:
    for name in (
        "DefectDetector",
        "StandardGrader",
        "Preprocessor",
        "IQIVerifier",
        "Quantifier",
        "Reporter",
        "Syncer",
    ):
        assert issubclass(getattr(interfaces, name), Protocol)


def test_detection_dto_frozen() -> None:
    det = dto.Detection(
        id="d1",
        bbox=dto.BBox(0, 0, 10, 10),
        class_id=dto.DefectClass.CRACK,
        score=0.9,
        uncertainty=0.1,
    )
    assert det.class_id is dto.DefectClass.CRACK
    assert dto.DefectShape.ROUND.value == "round"
    assert dto.JointLevel.IV.value == "IV"


def test_error_codes_match_api_contract() -> None:
    assert errors.ImageUnreadableError().http_status == 400
    assert errors.IQIFailError().code == "IQI_FAIL"
    assert errors.ModelUnavailableError().http_status == 503
    assert errors.GradingAmbiguousError().http_status == 422


def test_interfaces_have_expected_methods() -> None:
    assert hasattr(interfaces.DefectDetector, "infer")
    assert hasattr(interfaces.StandardGrader, "grade")
    assert hasattr(interfaces.Syncer, "federate")
