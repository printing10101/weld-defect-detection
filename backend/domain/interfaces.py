"""领域接口唯一真源（冻结契约， / ）。

实现规则：
- 任何实现类必须满足对应 Protocol（含签名、异常语义）。
- 禁止为满足实现而改动签名或新增必需参数；确需演进先写 ADR。
- `image: np.ndarray` 约定：单通道灰度，shape (H,W) 或 (H,W,1)，
  dtype uint8(0-255) 或 float32(0-1)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

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
    """缺陷检测器（模型无关，）。"""

    def load(self, model_uri: str, backend: str = "onnx") -> None: ...
    def infer(
        self, image: np.ndarray, conf: float, iou: float, class_conf: dict[int, float] | None = None
    ) -> list[Detection]: ...


@runtime_checkable
class StandardGrader(Protocol):
    """标准判定器（多标准适配，）。"""

    def grade(self, defects: list[Detection], context: ImageMeta) -> GradeResult: ...


@runtime_checkable
class Preprocessor(Protocol):
    """预处理算法族。"""

    def denoise(self, image: np.ndarray) -> np.ndarray: ...
    def enhance(self, image: np.ndarray, gamma: float) -> np.ndarray: ...
    def edges(self, image: np.ndarray, roi) -> np.ndarray: ...
    def morph(self, edges: np.ndarray, k_open: int, k_close: int) -> np.ndarray: ...


@runtime_checkable
class IQIVerifier(Protocol):
    """像质计/底片质量校验。"""

    def verify(self, image: np.ndarray) -> IQIResult: ...


@runtime_checkable
class Quantifier(Protocol):
    """缺陷量化。

    方法说明：
    - `measure`: 仅用检测框几何近似量化（契约必需，供无图场景）；
    - `quantify`: 可选图像感知量化（有图时调用，可复用 measure 回退）。
      所有生产量化器均实现 quantify，供 pipeline 统一入口调用。
    """

    def measure(self, detection: Detection, pixel_spacing_mm: float) -> Geometry: ...

    def quantify(
        self,
        detection: Detection,
        pixel_spacing_mm: float,
        *,
        image: np.ndarray | None = None,
        cfg: Any = None,
    ) -> Geometry: ...


@runtime_checkable
class Reporter(Protocol):
    """报告生成。返回 PDF 路径。

    `gray` 为可选优化：pipeline 已解码的灰度底片，复用以避免报告渲染二次解码；
    调用方可省略（实现需自行加载）。`witness`（S-22）为可选军代表/见证人署名，
    实现应支持关键字透传（缺省 None，版式不变）。
    """

    def build(
        self,
        image_id: str,
        template: str,
        gray: np.ndarray | None = None,
        witness: str | None = None,
    ) -> str: ...


@runtime_checkable
class Syncer(Protocol):
    """端边云同步适配器（v1=LocalAdapter，）。"""

    def push(self, record) -> None: ...
    def pull(self) -> list: ...
    def federate(self, weights) -> None: ...


@runtime_checkable
class QueuePort(Protocol):
    """本地待同步队列端口。

    domain 只声明"持久化追加 + 观测计数"，不触碰文件系统；
    JSONL 落盘等 IO 由 infra 实现（JsonlQueue）经构造注入。
    """

    def append(self, record) -> None: ...
    def count(self) -> int: ...


@runtime_checkable
class HttpPushPort(Protocol):
    """HTTP 推送端口。

    domain 只声明"尽力而为 POST"，失败不得抛（同步不阻断主流程）；
    urllib 传输由 infra 实现（UrllibJsonPoster）经构造注入。
    """

    def post(self, endpoint: str, token: str | None, record) -> None: ...


@runtime_checkable
class PoolStore(Protocol):
    """训练池存储端口。

    domain 只声明"写标注 / 列标注 / 指纹 / manifest 读写"，文件系统 IO 由
    infra 实现（FilePoolStore）经调用方装配注入；domain 运行期不触碰磁盘。
    """

    def write_label(self, stem: str, content: str) -> Path: ...
    def list_labels(self) -> list[str]: ...  # 相对文件名（如 "a.txt"），排序稳定
    def fingerprint(self) -> str | None: ...  # 数据版本指纹；无标注则 None
    def load_manifest(self) -> dict[str, Any] | None: ...
    def save_manifest(self, manifest: dict[str, Any]) -> Path: ...
