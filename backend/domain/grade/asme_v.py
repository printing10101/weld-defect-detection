"""ASME BPVC Sec.V Art.2 适配器骨架（§6.1 预留）。

v1 仅登记标准身份与数值表入口，评级逻辑未实现：
grade() 一律抛 GradingAmbiguousError（熔断 422），禁止静默错判——
宁可"需人工复核"也不输出一个未经验证的级别。
"""

from __future__ import annotations

from backend.domain.dto import Detection, GradeResult, ImageMeta
from backend.domain.errors import GradingAmbiguousError
from backend.domain.interfaces import StandardGrader


class AsmeSecVGrader(StandardGrader):
    """ASME BPVC Section V Article 2 射线检测评级（预留骨架）。"""

    standard_id = "ASME-V"

    def __init__(self, tables) -> None:
        self.tables = tables

    def grade(self, defects: list[Detection], context: ImageMeta) -> GradeResult:
        raise GradingAmbiguousError(
            f"标准 {self.standard_id} 适配器未实现（预留骨架）：禁止输出级别，需人工复核"
        )
