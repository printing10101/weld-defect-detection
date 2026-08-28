"""检测器注册表（§5.1 模型无关，ADR-002）。

将"检测器种类 → 实现类 + 装配方式"收敛到单一真源，使 application 层
（dependencies._build_detector）按 config.detect.kind 查表装配，兑现
"换检测器不改主干"：新增检测器只需在此登记，主干编排层无需改动。

get_detector(kind, *, model_uri, backend, blob_cfg) 按 kind 路由实现类，
完成 ``cls() + load(model_uri, backend)`` 的标准装配；未知 kind 抛
ModelUnavailableError（复用 §14 错误码，不新增契约）。

detector_capabilities 输出检测器能力目录（GET /detectors 数据源）：
needs_model = True（训练模型，须权重）/ False（基线，权重可选占位）。
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.domain.detect.blob_detector import BlobConfig, BlobDetector
from backend.domain.detect.yolo_detector import YoloDetector
from backend.domain.errors import ModelUnavailableError
from backend.domain.interfaces import DefectDetector


@dataclass(frozen=True)
class DetectorSpec:
    """某检测器种类的装配元数据。"""

    kind: str
    display_name: str
    cls: type[DefectDetector]
    needs_model: bool  # True=训练模型（须权重）；False=基线（权重可选占位）


# kind → 装配元数据（键与 configs/default.yaml 的 detect.kind 保持一致）
_DETECTOR_SPECS: dict[str, DetectorSpec] = {
    "trained_yolo": DetectorSpec(
        kind="trained_yolo",
        display_name="YOLO 训练模型检测器 (M4b)",
        cls=YoloDetector,
        needs_model=True,
    ),
    "baseline_blob": DetectorSpec(
        kind="baseline_blob",
        display_name="连通域基线检测器 (M4a)",
        cls=BlobDetector,
        needs_model=False,
    ),
}


def supported_detector_kinds() -> list[str]:
    """已注册检测器种类（含骨架与插件）。"""
    return list(_DETECTOR_SPECS)


def register_detector_kind(spec: DetectorSpec) -> None:
    """注册/覆盖检测器种类（§19.4 插件发现入口，P2）。

    同 kind 已注册且实现类不同 → 抛 ModelUnavailableError（防插件静默顶替内置）；
    相同实现（幂等重发现）→ 无操作。
    """
    existing = _DETECTOR_SPECS.get(spec.kind)
    if existing is not None and existing.cls is not spec.cls:
        raise ModelUnavailableError(
            f"检测器种类 {spec.kind!r} 已注册（{existing.cls.__name__}），拒绝覆盖"
        )
    _DETECTOR_SPECS[spec.kind] = spec


def detector_capabilities(kind: str) -> dict:
    """单检测器能力目录（注册表元数据；供 GET /detectors 展示）。"""
    spec = _DETECTOR_SPECS.get(kind)
    if spec is None:
        raise ModelUnavailableError(
            f"不支持的检测器种类 {kind}（支持：{', '.join(_DETECTOR_SPECS)}）"
        )
    return {
        "kind": spec.kind,
        "name": spec.display_name,
        "needs_model": spec.needs_model,
    }


def get_detector(
    kind: str,
    *,
    model_uri: str | None = None,
    backend: str = "onnx",
    blob_cfg: BlobConfig | None = None,
) -> DefectDetector:
    """按 kind 装配检测器：``cls() + load(model_uri, backend)``。

    - trained_yolo：须提供真实 model_uri（权重缺失由调用方策略处理）；
    - baseline_blob：权重可选占位，装配 BlobConfig（缺省用默认配置）。
    未知 kind 抛 ModelUnavailableError（503，需人工复核配置）。
    """
    spec = _DETECTOR_SPECS.get(kind)
    if spec is None:
        raise ModelUnavailableError(
            f"不支持的检测器种类 {kind}（支持：{', '.join(_DETECTOR_SPECS)}），需人工复核配置"
        )
    # BlobDetector 需构造参数；其余（含插件检测器）走通用零参数构造 + load，
    # 兑现 §19.4"接口不动、实现可插拔"。
    if spec.cls is BlobDetector:
        det: DefectDetector = BlobDetector(blob_cfg or BlobConfig())
    else:
        det = spec.cls()
    det.load(model_uri or "", backend)
    return det
