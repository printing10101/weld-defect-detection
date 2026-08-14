"""标准判定器注册表（§6.1 多标准适配）。

v1 实现 NB/T47013.2-2015；其余标准为预留骨架（GB/T 3323 / ASME-V / ISO 17636），
grade() 一律抛 GradingAmbiguousError 熔断（422 → 需人工复核），禁止静默错判。

``get_grader`` 按 standard_id 路由实现类并装配数值表；未知标准抛
GradingAmbiguousError（复用 §14 错误码，不新增契约）。
"""

from __future__ import annotations

from backend.domain.errors import GradingAmbiguousError
from backend.domain.grade.asme_v import AsmeSecVGrader
from backend.domain.grade.gb3323 import Gb3323Grader
from backend.domain.grade.iso17636 import Iso17636Grader
from backend.domain.grade.nb47013 import Nb47013Grader
from backend.domain.interfaces import StandardGrader
from backend.domain.standards.tables.loader import StandardTables

# standard_id → 实现类（键与规格书 §6.1 / 表文件 standard_id 一致）
_GRADERS: dict[str, type[StandardGrader]] = {
    "NB/T47013.2-2015": Nb47013Grader,
    "GB/T3323-2019": Gb3323Grader,
    "ASME-V": AsmeSecVGrader,
    "ISO17636": Iso17636Grader,
}


def supported_standard_ids() -> list[str]:
    """已注册标准 id（含骨架）。"""
    return list(_GRADERS)


def get_grader(
    standard_id: str,
    tables: StandardTables | None = None,
) -> StandardGrader:
    """按 standard_id 装配判定器。

    - NB/T47013.2-2015：必须提供已授权数值表（authorized 熔断由 Nb47013Grader 自身执行）；
    - 骨架标准：tables 可为 None（grade() 直接熔断，不读表）；
    - 未知标准：抛 GradingAmbiguousError（422，需人工复核）。
    """
    impl = _GRADERS.get(standard_id)
    if impl is None:
        raise GradingAmbiguousError(
            f"不支持的标准 {standard_id}（支持：{', '.join(_GRADERS)}），禁止输出级别，需人工复核"
        )
    if impl is Nb47013Grader and tables is None:
        raise GradingAmbiguousError(f"标准 {standard_id} 数值表缺失，禁止输出级别，需人工复核")
    return impl(tables)  # type: ignore[arg-type]  # 骨架类不读表，tables=None 合法


# 重新导出骨架类，便于测试直接引用
__all__ = [
    "AsmeSecVGrader",
    "Gb3323Grader",
    "Iso17636Grader",
    "Nb47013Grader",
    "get_grader",
    "supported_standard_ids",
]
