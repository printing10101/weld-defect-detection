"""领域层：缺陷检测器。

检测器实现满足 DefectDetector 契约（interfaces.py），经 detect/registry 装配。
application 层禁止自行 new 检测器——统一走 ``get_detector``。
"""

from __future__ import annotations

from backend.domain.detect.blob_detector import BlobConfig, BlobDetector
from backend.domain.detect.registry import (
    DetectorSpec,
    detector_capabilities,
    get_detector,
    supported_detector_kinds,
)
from backend.domain.detect.yolo_detector import YoloDetector

__all__ = [
    "BlobConfig",
    "BlobDetector",
    "DetectorSpec",
    "YoloDetector",
    "detector_capabilities",
    "get_detector",
    "supported_detector_kinds",
]
