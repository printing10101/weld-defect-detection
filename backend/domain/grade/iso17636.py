"""ISO 17636 适配器。

ISO 17636（焊缝无损检测 射线检测）定义**成像质量等级 A/B**（IQI 灵敏度）
与工艺要求，**不输出缺陷验收级别**——缺陷验收等级在 ISO 10675-1 中给出
（验收等级 1/2/3，配合 ISO 5817 的缺陷质量分级）。

本适配器正确语义：不套用 NB/GB 的 I-IV 级别（标准错配即静默错判）；
grade 熔断 422 → 需人工按 ISO 10675-1 判定。成像质量等级 A/B 的 IQI
灵敏度映射表未转录前返回空。
"""

from __future__ import annotations

from backend.domain.dto import Detection, GradeResult, ImageMeta
from backend.domain.errors import GradingAmbiguousError
from backend.domain.interfaces import StandardGrader


class Iso17636Grader(StandardGrader):
    """ISO 17636 焊缝射线检测（成像质量等级 A/B，不输出缺陷级别）。"""

    standard_id = "ISO17636"

    def __init__(self, tables) -> None:
        self.tables = tables

    def grade(self, defects: list[Detection], context: ImageMeta) -> GradeResult:
        raise GradingAmbiguousError(
            f"标准 {self.standard_id} 定义成像质量等级 A/B（IQI 灵敏度），"
            "不输出缺陷级别；缺陷验收等级须依 ISO 10675-1 人工判定"
        )
