"""插件发现与装配（§19.4 扩展菜谱 → 真插件机制，P2）。

第三方/扩展包经 setuptools entry-point 声明实现，启动期发现并登记进 domain
注册表，核心代码零改动（"接口不动、实现可插拔"）：

- entry-point 组 ``scandetection.detectors``  → DetectorSpec 实例
- entry-point 组 ``scandetection.graders``    → GraderSpec 实例
- entry-point 组 ``scandetection.quantifiers``→ QuantifierSpec 实例

插件包示例（pyproject.toml）：

    [project.entry-points."scandetection.detectors"]
    rt_detr = "my_pkg.detector:SPEC"      # SPEC 为 DetectorSpec 实例

``bootstrap_plugins()`` 进程级幂等（once 标志），未安装任何插件时静默无操作；
单个插件加载/注册失败仅告警，不阻断启动（与同步"尽力而为"同哲学）。
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable

from backend.domain.detect.registry import DetectorSpec, register_detector_kind
from backend.domain.grade.registry import GraderSpec, register_standard
from backend.domain.quantify import QuantifierSpec, register_quantifier_kind

_LOG = logging.getLogger("scandetection.plugins")

# entry-point 组 → 注册函数（新扩展面在此登记，与 §19.4 扩展菜谱对齐）
_GROUPS: list[tuple[str, Callable[[object], None]]] = [
    ("scandetection.detectors", register_detector_kind),
    ("scandetection.graders", register_standard),
    ("scandetection.quantifiers", register_quantifier_kind),
]

# 进程级幂等标记（测试可 monkeypatch 重置以重复发现）
_PLUGINS_DISCOVERED = False


def _entry_points(group: str):
    """按组取 entry points（py3.10+ 关键字 API；兜底旧式 dict 接口）。"""
    try:
        return importlib.metadata.entry_points(group=group)
    except TypeError:  # pragma: no cover - 旧 Python 兜底
        return importlib.metadata.entry_points().get(group, ())


def _register(group: str, register: Callable[[object], None]) -> int:
    """加载并注册单组 entry points；失败仅告警（单插件坏不拖垮启动）。"""
    registered = 0
    for ep in _entry_points(group):
        try:
            register(ep.load())
            registered += 1
            _LOG.info("插件已注册: %s (%s)", ep.name, group)
        except Exception as exc:  # noqa: BLE001 - 单插件失败不阻断启动
            _LOG.warning("插件加载失败 %s (%s): %s", ep.name, group, exc)
    return registered


def bootstrap_plugins() -> int:
    """启动期发现并注册全部插件（幂等）；返回本次注册数。"""
    global _PLUGINS_DISCOVERED
    if _PLUGINS_DISCOVERED:
        return 0
    _PLUGINS_DISCOVERED = True
    total = sum(_register(group, register) for group, register in _GROUPS)
    if total:
        _LOG.info("插件发现完成：共注册 %d 项", total)
    return total
