"""领域接口唯一真源（冻结契约，§T2 / §19.3）。

实现规则：
- 任何实现类必须满足对应 Protocol（含签名、异常语义）。
- 禁止为满足实现而改动签名或新增必需参数；确需演进先写 ADR。
- `image: np.ndarray` 约定：单通道灰度，shape (H,W) 或 (H,W,1)，
  dtype uint8(0-255) 或 float32(0-1)。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .dto import (
    Detection,
    Geometry,
    GradeResult,
    ImageMeta,
    IQIResult,
)


@runtime_checkable
class DefectDetector(Protocol):
    """缺陷检测器（模型无关，§5.1）。"""

    def load(self, model_uri: str, backend: str = "onnx") -> None: ...
    def infer(
        self, image: np.ndarray, conf: float, iou: float, class_conf: dict[int, float] | None = None
    ) -> list[Detection]: ...


@runtime_checkable
class StandardGrader(Protocol):
    """标准判定器（多标准适配，§6.1）。"""

    def grade(self, defects: list[Detection], context: ImageMeta) -> GradeResult: ...


@runtime_checkable
class Preprocessor(Protocol):
    """预处理算法族（§4.3）。"""

    def denoise(self, image: np.ndarray) -> np.ndarray: ...
    def enhance(self, image: np.ndarray, gamma: float) -> np.ndarray: ...
    def edges(self, image: np.ndarray, roi) -> np.ndarray: ...
    def morph(self, edges: np.ndarray, k_open: int, k_close: int) -> np.ndarray: ...


@runtime_checkable
class IQIVerifier(Protocol):
    """像质计/底片质量校验（§4.2）。"""

    def verify(self, image: np.ndarray) -> IQIResult: ...


@runtime_checkable
class Quantifier(Protocol):
    """缺陷量化（§5.4）。"""

    def measure(self, detection: Detection, pixel_spacing_mm: float) -> Geometry: ...


@runtime_checkable
class Reporter(Protocol):
    """报告生成（§7.2）。返回 PDF 路径。"""

    def build(self, image_id: str, template: str) -> str: ...


@runtime_checkable
class Syncer(Protocol):
    """端边云同步适配器（v1=LocalAdapter，§7.6）。"""

    def push(self, record) -> None: ...
    def pull(self) -> list: ...
    def federate(self, weights) -> None: ...
