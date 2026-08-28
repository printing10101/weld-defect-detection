"""P2 真插件机制测试。

覆盖：
- 三个注册表（检测器/标准/量化器）的公开注册 API（register_*）：登记、装配、冲突拒绝；
- get_detector 通用构造路径（插件检测器不再被 else-raise 拦截）；
- bootstrap_plugins 幂等发现 + 单插件损坏隔离（失败仅告警不阻断）。
"""

from __future__ import annotations

from typing import cast

import pytest

from backend.app import plugins
from backend.app.plugins import bootstrap_plugins
from backend.domain.detect.registry import (
    _DETECTOR_SPECS,
    DetectorSpec,
    get_detector,
    supported_detector_kinds,
)
from backend.domain.errors import ModelUnavailableError
from backend.domain.grade.registry import (
    _GRADERS,
    _STANDARD_META,
    GraderSpec,
    get_grader,
    standard_capabilities,
    supported_standard_ids,
)
from backend.domain.interfaces import DefectDetector, Quantifier
from backend.domain.quantify import (
    _QUANTIFIER_SPECS,
    QuantifierSpec,
    get_quantifier,
    supported_quantifier_kinds,
)

# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------


class _FakeDetector:
    """零参数构造 + load/infer，满足 DefectDetector 契约的最小插件检测器。"""

    def __init__(self) -> None:
        self.loaded: tuple | None = None

    def load(self, model_uri: str, backend: str = "onnx") -> None:
        self.loaded = (model_uri, backend)

    def infer(self, image, conf, iou, class_conf=None):
        return []


class _FakeGrader:
    """最小插件判定器（熔断语义：不读表）。"""

    def __init__(self, tables=None) -> None:
        self.tables = tables

    def grade(self, defects, context):
        raise NotImplementedError  # 测试不调用


_FAKE_DET_SPEC = DetectorSpec(
    kind="plugin_det", display_name="插件检测器 (P2)", cls=_FakeDetector, needs_model=False
)
_FAKE_GRADER_SPEC = GraderSpec(
    standard_id="FAKE-STD",
    cls=_FakeGrader,
    meta={
        "name": "FAKE 标准（测试插件）",
        "grades_defects": False,
        "levels": None,
        "table_required": False,
        "table_filename": None,
        "note": "测试用插件标准。",
    },
)
_FAKE_QUANT_SPEC = QuantifierSpec(
    kind="plugin_q",
    display_name="插件量化器 (P2)",
    cls=cast(type[Quantifier], type("PluginQ", (), {})),
    needs_image=False,
)


def _cleanup() -> None:
    """从全局注册表摘除测试插件（防污染其它用例）。"""
    _DETECTOR_SPECS.pop("plugin_det", None)
    _GRADERS.pop("FAKE-STD", None)
    _STANDARD_META.pop("FAKE-STD", None)
    _QUANTIFIER_SPECS.pop("plugin_q", None)


# ---------------------------------------------------------------------------
# 注册 API（register_*）
# ---------------------------------------------------------------------------


def test_register_detector_kind_and_build_via_generic_path() -> None:
    from backend.domain.detect.registry import register_detector_kind

    try:
        register_detector_kind(_FAKE_DET_SPEC)
        assert "plugin_det" in supported_detector_kinds()
        det = get_detector("plugin_det", model_uri="models/x.onnx", backend="onnx")
        assert isinstance(det, _FakeDetector)
        assert det.loaded == ("models/x.onnx", "onnx")  # 通用构造路径生效
    finally:
        _cleanup()


def test_register_detector_kind_conflict_raises() -> None:
    from backend.domain.detect.registry import register_detector_kind

    try:
        register_detector_kind(_FAKE_DET_SPEC)
        with pytest.raises(ModelUnavailableError):
            # 同 kind 不同实现类 → 拒绝覆盖（防插件静默顶替内置）
            register_detector_kind(
                DetectorSpec(
                    kind="plugin_det",
                    display_name="x",
                    cls=cast(type[DefectDetector], _FakeDetector2),
                    needs_model=False,
                )
            )
    finally:
        _cleanup()


def test_register_standard_and_capabilities() -> None:
    from backend.domain.grade.registry import register_standard

    try:
        register_standard(_FAKE_GRADER_SPEC)
        assert "FAKE-STD" in supported_standard_ids()
        cap = standard_capabilities("FAKE-STD")
        assert cap["name"].startswith("FAKE")
        assert cap["status"] == "method_standard"  # table_required=False
        grader = get_grader("FAKE-STD")  # 非 Nb47013 → 通用构造 impl(tables)
        assert isinstance(grader, _FakeGrader)
    finally:
        _cleanup()


def test_register_standard_conflict_raises() -> None:
    from backend.domain.errors import GradingAmbiguousError
    from backend.domain.grade.registry import register_standard

    try:
        register_standard(_FAKE_GRADER_SPEC)
        with pytest.raises(GradingAmbiguousError):
            register_standard(
                GraderSpec(
                    standard_id="FAKE-STD",
                    cls=type("OtherGrader", (_FakeGrader,), {}),
                    meta={"name": "x", "table_required": False},
                )
            )
    finally:
        _cleanup()


def test_register_quantifier_kind_and_resolve() -> None:
    from backend.domain.quantify import register_quantifier_kind

    try:
        register_quantifier_kind(_FAKE_QUANT_SPEC)
        assert "plugin_q" in supported_quantifier_kinds()
        q = get_quantifier("plugin_q")
        assert type(q).__name__ == "PluginQ"  # 经 spec.cls() 通用构造
        with pytest.raises(ModelUnavailableError):
            register_quantifier_kind(
                QuantifierSpec(
                    kind="plugin_q",
                    display_name="x",
                    cls=cast(type[Quantifier], object),
                    needs_image=False,
                )
            )
    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# 发现机制（bootstrap_plugins + entry-point）
# ---------------------------------------------------------------------------


class _FakeEP:
    """伪 entry point：仅需 name/load 供加载器消费。"""

    def __init__(self, name: str, obj) -> None:
        self.name = name
        self._obj = obj

    def load(self):
        return self._obj


def _discover_all(monkeypatch) -> None:
    """monkeypatch entry_points 返回三组插件规格，并复位幂等标记。"""
    specs = {
        "scandetection.detectors": [_FakeEP("plugin_det", _FAKE_DET_SPEC)],
        "scandetection.graders": [_FakeEP("plugin_std", _FAKE_GRADER_SPEC)],
        "scandetection.quantifiers": [_FakeEP("plugin_q", _FAKE_QUANT_SPEC)],
    }

    def fake_entry_points(group: str):
        return specs.get(group, [])

    monkeypatch.setattr(plugins, "_entry_points", fake_entry_points)
    monkeypatch.setattr(plugins, "_PLUGINS_DISCOVERED", False)


def test_bootstrap_discovers_all_groups(monkeypatch) -> None:
    _discover_all(monkeypatch)
    try:
        n = bootstrap_plugins()
        assert n == 3
        assert "plugin_det" in supported_detector_kinds()
        assert "FAKE-STD" in supported_standard_ids()
        assert "plugin_q" in supported_quantifier_kinds()
        # 幂等：再次调用不重复注册、不报错
        assert bootstrap_plugins() == 0
    finally:
        _cleanup()
        monkeypatch.setattr(plugins, "_PLUGINS_DISCOVERED", False)


def test_bootstrap_broken_plugin_isolated(monkeypatch) -> None:
    """单个插件 load 失败仅告警，不阻断其余插件与启动。"""
    specs = {
        "scandetection.detectors": [
            _FakeEP("broken", _raise_load()),
            _FakeEP("plugin_det", _FAKE_DET_SPEC),
        ],
    }

    def fake_entry_points(group: str):
        return specs.get(group, [])

    monkeypatch.setattr(plugins, "_entry_points", fake_entry_points)
    monkeypatch.setattr(plugins, "_PLUGINS_DISCOVERED", False)
    try:
        n = bootstrap_plugins()  # 不抛
        assert n == 1  # 仅健康的注册成功
        assert "plugin_det" in supported_detector_kinds()
        assert "broken" not in supported_detector_kinds()
    finally:
        _cleanup()
        monkeypatch.setattr(plugins, "_PLUGINS_DISCOVERED", False)


class _RaiseLoad:
    def __init__(self) -> None:
        pass

    def load(self):
        raise ValueError("broken plugin")


def _raise_load() -> _RaiseLoad:
    return _RaiseLoad()


class _FakeDetector2:
    """与 _FakeDetector 不同的实现类（冲突测试用）。"""

    def load(self, model_uri: str, backend: str = "onnx") -> None: ...
    def infer(self, image, conf, iou, class_conf=None): ...
