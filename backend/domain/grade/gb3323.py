"""GB/T 3323-2019 适配器（§6.1 注册，语义化熔断）。

GB/T 3323-2019 与 NB/T47013.2 同为承压设备焊缝射线检测评级体系（缺陷分级
I-IV，等效点数法同源），但本系统尚未转录/获得其授权评级数值表：
grade() 一律抛 GradingAmbiguousError（熔断 422），禁止静默错判——
宁可"需人工复核"也不输出一个未经验证的级别。
"""

from __future__ import annotations

from backend.domain.dto import Detection, GradeResult, ImageMeta
from backend.domain.errors import GradingAmbiguousError
from backend.domain.interfaces import StandardGrader


class Gb3323Grader(StandardGrader):
    """GB/T 3323-2019 熔焊焊接接头射线检测评级（缺表熔断）。"""

    standard_id = "GB/T3323-2019"

    def __init__(self, tables) -> None:
        self.tables = tables

    def grade(self, defects: list[Detection], context: ImageMeta) -> GradeResult:
        raise GradingAmbiguousError(
            f"标准 {self.standard_id} 评级数值表未转录/未授权，禁止输出级别；"
            "同源体系可参考 NB/T47013.2 判定（需人工复核）"
        )
