"""标准判定器注册表（§6.1 多标准适配）。

v1 实现 NB/T47013.2-2015 评级引擎；GB/T 3323 / ASME-V / ISO 17636 为已注册
适配器（语义化熔断：标准语义下不输出级别，禁止静默错判）。

``get_grader`` 按 standard_id 路由实现类并装配数值表；未知标准抛
GradingAmbiguousError（复用 §14 错误码，不新增契约）。

``standard_capabilities`` 输出标准能力目录（GET /standards 数据源）：
status = enabled（表存在且 authorized）/ unauthorized（表存在未授权）
| tables_missing（缺表）/ method_standard（方法标准不评级）。
"""

from __future__ import annotations

from typing import Any

from backend.domain.errors import GradingAmbiguousError
from backend.domain.grade.asme_v import AsmeSecVGrader
from backend.domain.grade.gb3323 import Gb3323Grader
from backend.domain.grade.iso17636 import Iso17636Grader
from backend.domain.grade.nb47013 import Nb47013Grader
from backend.domain.interfaces import StandardGrader
from backend.domain.standards.tables.loader import StandardTables, load_standard_tables

# standard_id → 实现类（键与规格书 §6.1 / 表文件 standard_id 一致）
_GRADERS: dict[str, type[StandardGrader]] = {
    "NB/T47013.2-2015": Nb47013Grader,
    "GB/T3323-2019": Gb3323Grader,
    "ASME-V": AsmeSecVGrader,
    "ISO17636": Iso17636Grader,
}

# 标准能力目录（§6.1 多标准适配元数据，供 GET /standards 展示）。
# status 为动态值（见 standard_capabilities），此处仅给静态语义。
_STANDARD_META: dict[str, dict[str, Any]] = {
    "NB/T47013.2-2015": {
        "name": "NB/T 47013.2-2015 承压设备无损检测 第2部分：射线检测",
        "grades_defects": True,  # 输出缺陷级别 I-IV
        "levels": ["I", "II", "III", "IV"],
        "table_required": True,
        "table_filename": "nb47013.yaml",
        "note": "圆形/条形缺陷评定（等效点数法）；裂纹/未熔合/未焊透零容忍。",
    },
    "GB/T3323-2019": {
        "name": "GB/T 3323-2019 承压设备金属材料熔化焊焊接接头射线照相检测",
        "grades_defects": True,
        "levels": ["I", "II", "III", "IV"],
        "table_required": True,
        "table_filename": None,  # 评级数值表未转录/未授权
        "note": "与 NB/T47013.2 同源评级体系；数值表未转录/未授权前禁止输出级别。",
    },
    "ASME-V": {
        "name": "ASME BPVC Section V Article 2 无损检测（射线）",
        "grades_defects": False,  # 方法标准：不定义缺陷验收级别
        "levels": None,
        "table_required": False,
        "table_filename": None,
        "note": "无损检测方法标准（底片质量/工艺要求）；缺陷验收依构件标准"
        "（如 ASME VIII Div.1 / B31 系列）人工判定，本系统不输出 ASME 缺陷级别。",
    },
    "ISO17636": {
        "name": "ISO 17636 焊缝无损检测 射线检测",
        "grades_defects": False,  # 定义成像质量等级 A/B，不输出缺陷级别
        "levels": None,
        "table_required": False,
        "table_filename": None,
        "note": "定义成像质量等级 A/B（IQI 灵敏度）与工艺要求；缺陷验收等级"
        "依 ISO 10675-1 人工判定。",
    },
}


def supported_standard_ids() -> list[str]:
    """已注册标准 id（含骨架）。"""
    return list(_GRADERS)


def standard_capabilities(standard_id: str) -> dict[str, Any]:
    """单标准能力目录（注册表元数据 + 动态表状态）。"""
    meta = _STANDARD_META.get(standard_id)
    if meta is None:
        raise GradingAmbiguousError(
            f"不支持的标准 {standard_id}（支持：{', '.join(_GRADERS)}），禁止输出级别，需人工复核"
        )
    out: dict[str, Any] = {"standard_id": standard_id}
    out.update(dict(meta))
    if meta.get("table_required"):
        fn = meta.get("table_filename")
        try:
            tables = load_standard_tables(standard_id, filename=fn)
            out["status"] = "enabled" if tables.authorized else "unauthorized"
        except (FileNotFoundError, ValueError, KeyError):
            out["status"] = "tables_missing"
    else:
        out["status"] = "method_standard"
    return out


def all_standard_capabilities() -> list[dict[str, Any]]:
    """全部注册标准的能力清单（按注册顺序）。"""
    return [standard_capabilities(sid) for sid in _GRADERS]


def get_grader(
    standard_id: str,
    tables: StandardTables | None = None,
) -> StandardGrader:
    """按 standard_id 装配判定器。

    - NB/T47013.2-2015：必须提供已授权数值表（authorized 熔断由 Nb47013Grader 自身执行）；
    - 语义化熔断标准：tables 可为 None（grade() 直接熔断，不读表）；
    - 未知标准：抛 GradingAmbiguousError（422，需人工复核）。
    """
    impl = _GRADERS.get(standard_id)
    if impl is None:
        raise GradingAmbiguousError(
            f"不支持的标准 {standard_id}（支持：{', '.join(_GRADERS)}），禁止输出级别，需人工复核"
        )
    if impl is Nb47013Grader and tables is None:
        raise GradingAmbiguousError(f"标准 {standard_id} 数值表缺失，禁止输出级别，需人工复核")
    return impl(tables)  # type: ignore[arg-type]  # 熔断类不读表，tables=None 合法


# 重新导出骨架类，便于测试直接引用
__all__ = [
    "AsmeSecVGrader",
    "Gb3323Grader",
    "Iso17636Grader",
    "Nb47013Grader",
    "all_standard_capabilities",
    "get_grader",
    "standard_capabilities",
    "supported_standard_ids",
]
