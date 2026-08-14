"""ASME BPVC Sec.V Art.2 适配器（§6.1 注册，方法标准语义）。

ASME BPVC Section V Article 2 是**无损检测方法标准**（底片质量、操作、像质
要求），**不定义缺陷验收级别**——缺陷验收在构件标准（ASME VIII Div.1 /
B31 系列等）中给出，且多采用"验收/不合格"判定而非 I-IV 分级。

因此本适配器的正确语义是：**不输出缺陷级别**（grade() 熔断 422 → 需人工
按构件标准判定），而不是套用 NB/GB 体系的 I-IV 级别（会构成标准错配的
静默错判）。
"""

from __future__ import annotations

from backend.domain.dto import Detection, GradeResult, ImageMeta
from backend.domain.errors import GradingAmbiguousError
from backend.domain.interfaces import StandardGrader


class AsmeSecVGrader(StandardGrader):
    """ASME BPVC Section V Article 2 射线检测（方法标准，不评级）。"""

    standard_id = "ASME-V"

    def __init__(self, tables) -> None:
        self.tables = tables

    def grade(self, defects: list[Detection], context: ImageMeta) -> GradeResult:
        raise GradingAmbiguousError(
            f"标准 {self.standard_id} 为无损检测方法标准，不定义缺陷验收级别；"
            "验收须依构件标准（如 ASME VIII Div.1 / B31 系列）人工判定"
        )
