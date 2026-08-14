"""跨层 DTO（冻结契约，§T2 / 规格书 §13.3）。

修改任何字段/枚举值须先走 ADR 流程（§19.8），并同步：
- backend/domain/interfaces.py
- backend/app/routers（Pydantic schema）
- src/src/types/api.ts（前端类型）
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class DefectClass(Enum):
    """焊缝缺陷类别（§1.3，ADR-010 扩展为 6 类）。

    0 POROSITY 气孔
    1 SLAG 夹渣/夹杂
    2 INCOMPLETE_PENETRATION 未焊透
    3 LACK_OF_FUSION 未熔合
    4 CRACK 裂纹
    5 UNDERCUT 咬边/内凹（ADR-010 新增：竞赛评分项 + SWRD 第 6 类）
    """

    POROSITY = 0
    SLAG = 1
    INCOMPLETE_PENETRATION = 2
    LACK_OF_FUSION = 3
    CRACK = 4
    UNDERCUT = 5


class DefectShape(Enum):
    """按长宽比 L/W 归类（NB/T47013）：<=3 圆形，>3 条形。"""

    ROUND = "round"
    LINEAR = "linear"


class JointLevel(Enum):
    """评级 I–IV（§5）。"""

    I = "I"
    II = "II"
    III = "III"
    IV = "IV"


class Modality(Enum):
    """影像模态（§4.1）。"""

    CR = "CR"
    DR = "DR"
    DICOM = "DICOM"
    GENERIC = "GENERIC"


@dataclass(frozen=True)
class BBox:
    """边界框（像素坐标，左上角 + 宽高）。"""

    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class Detection:
    """单个缺陷检测结果（§14 Detection schema 的领域形态）。"""

    id: str
    bbox: BBox
    class_id: DefectClass
    score: float
    uncertainty: float
    shape: DefectShape | None = None
    mask_ref: str | None = None
    deep_hole: bool = False  # 深孔（黑度>母材，§6.2 直判 IV）；由检测器/管线提供，默认 False


@dataclass(frozen=True)
class Geometry:
    """量化几何属性（§5.4）。"""

    length_mm: float
    width_mm: float
    area_mm2: float
    perimeter_mm: float
    aspect_ratio: float
    position_x_mm: float
    position_y_mm: float


@dataclass(frozen=True)
class GradeResult:
    """标准判定结果（§6.5）。"""

    joint_level: JointLevel
    per_defect_grade: tuple[JointLevel, ...]
    basis: tuple[str, ...]
    need_review: bool
    standard_id: str
    standard_version: str
    # 标准来源免责声明（工业过渡路径，T1）：authorized_copy=false 时为强声明，
    # 提示"数值转录自公开解读、非授权正本、不替代责任工程师法定评定"。
    disclaimer: str | None = None


@dataclass(frozen=True)
class IQIResult:
    """像质计校验结果（§4.2）。"""

    iqi_type: str
    achieved: str | None
    required: str
    passed: bool
    grade: str | None = None  # A/AB/B 影像质量等级（厚度+可达丝号映射，None=未定/无厚度）


@dataclass(frozen=True)
class ImageMeta:
    """判定上下文（母材厚度等，§6.1）。"""

    modality: Modality
    pixel_spacing_mm: float | None = None
    base_metal_thickness_mm: float | None = None
    bit_depth: int | None = None
    density_array: np.ndarray | None = None
