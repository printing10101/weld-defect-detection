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


def test_standard_registry_contract() -> None:
    """§6.1 标准注册表：默认实现可装配；骨架标准已登记；未知熔断。"""
    from backend.domain.errors import GradingAmbiguousError
    from backend.domain.grade.gb3323 import Gb3323Grader
    from backend.domain.grade.registry import get_grader, supported_standard_ids

    ids = supported_standard_ids()
    assert "NB/T47013.2-2015" in ids
    assert "GB/T3323-2019" in ids  # 预留骨架已登记
    assert "ASME-V" in ids
    assert "ISO17636" in ids
    # 骨架标准可装配（tables 可缺省），grade() 时熔断
    assert isinstance(get_grader("GB/T3323-2019"), Gb3323Grader)
    try:
        get_grader("NOT-A-STANDARD")
    except GradingAmbiguousError:
        pass
    else:  # pragma: no cover
        raise AssertionError("未知标准必须熔断")
    # NB/T47013 缺表也熔断（防无授权数值表静默错判）
    try:
        get_grader("NB/T47013.2-2015", None)
    except GradingAmbiguousError:
        pass
    else:  # pragma: no cover
        raise AssertionError("NB/T47013 缺数值表必须熔断")
