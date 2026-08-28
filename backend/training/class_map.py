"""缺陷类别映射。

统一把**不同来源数据集**的类别名映射到本项目冻结的 :class:`DefectClass`
枚举，保证训练、推理、判定三层使用同一套类别语义。

来源：
- SWRD（北理工，CC BY 4.0）：气孔/夹杂/裂纹/咬边/未熔合/未焊透
- Kaggle Weld Quality（CC0，焊缝外观，仅预训练用）：Bad Welding/Crack/...
- 用户 165 张 定检 标注（Label Studio 导出，中文名）
"""

from __future__ import annotations

from enum import Enum

from backend.domain.dto import DefectClass


class SourceClass(str, Enum):
    """各数据集原始类别名（小写键）。"""

    # SWRD（论文 Table 2）
    POROSITY = "porosity"  # 气孔
    INCLUSION = "inclusion"  # 夹杂/夹渣
    CRACK = "crack"  # 裂纹
    UNDERCUT = "undercut"  # 咬边
    LACK_OF_FUSION = "lack_of_fusion"  # 未熔合
    LACK_OF_PENETRATION = "lack_of_penetration"  # 未焊透
    # 用户数据可能用到
    SLAG = "slag"  # 夹渣（与 Inclusion 同义）
    CONCAVITY = "concavity"  # 内凹（SWRD 未单列）
    # Kaggle CC0 外观（仅预训练）
    BAD_WELD = "bad_welding"
    GOOD_WELD = "good_welding"
    EXCESS_REINFORCEMENT = "excess_reinforcement"
    SPATTERS = "spatters"


# 主训练集（SWRD + 用户 X 光）使用的 6 类，与 DefectClass 一一对应（ 扩展 UNDERCUT）
SWRD_TO_DEFECTCLASS: dict[str, DefectClass] = {
    "porosity": DefectClass.POROSITY,
    "inclusion": DefectClass.SLAG,
    "slag": DefectClass.SLAG,
    "lack_of_penetration": DefectClass.INCOMPLETE_PENETRATION,
    "lack_of_fusion": DefectClass.LACK_OF_FUSION,
    "crack": DefectClass.CRACK,
    # 咬边： 起为独立第 6 类 DefectClass.UNDERCUT（竞赛评分项 + SWRD 第 6 类）。
    # 内凹：DB50/T 1807 单面焊重点关注缺陷，2026-08 起拆分为独立第 7 类 CONCAVITY
    # （追加索引 6，历史 0-5 索引不变）。
    "undercut": DefectClass.UNDERCUT,
    "concavity": DefectClass.CONCAVITY,
}

# Label Studio / 用户标注常用中文名 → DefectClass
ZH_TO_DEFECTCLASS: dict[str, DefectClass] = {
    "气孔": DefectClass.POROSITY,
    "夹渣": DefectClass.SLAG,
    "夹杂": DefectClass.SLAG,
    "未焊透": DefectClass.INCOMPLETE_PENETRATION,
    "未熔合": DefectClass.LACK_OF_FUSION,
    "裂纹": DefectClass.CRACK,
    "咬边": DefectClass.UNDERCUT,
    "内凹": DefectClass.CONCAVITY,
}

# YOLO 训练用的有序类别列表（index = class_id），必须与 DefectClass 枚举值一致
YOLO_CLASSES: list[str] = [c.name for c in DefectClass]
# DefectClass: POROSITY=0, SLAG=1, INCOMPLETE_PENETRATION=2, LACK_OF_FUSION=3, CRACK=4, UNDERCUT=5, CONCAVITY=6
assert YOLO_CLASSES == [
    "POROSITY",
    "SLAG",
    "INCOMPLETE_PENETRATION",
    "LACK_OF_FUSION",
    "CRACK",
    "UNDERCUT",
    "CONCAVITY",
], "YOLO_CLASSES 必须与 DefectClass 枚举（含 UNDERCUT=5, CONCAVITY=6）严格一致"


def defectclass_to_yolo_idx(cls: DefectClass) -> int:
    """DefectClass → YOLO class index（与 data.yaml 顺序一致）。"""
    return cls.value


def map_source_label(label: str) -> DefectClass | None:
    """将任意来源标签归一化到 DefectClass；无法映射返回 None。"""
    key = label.strip().lower()
    if key in SWRD_TO_DEFECTCLASS:
        return SWRD_TO_DEFECTCLASS[key]
    if key in ZH_TO_DEFECTCLASS:
        return ZH_TO_DEFECTCLASS[key]
    # 容错：包含关键字
    if "poros" in key or "气孔" in key:
        return DefectClass.POROSITY
    if "slag" in key or "inclus" in key or "夹" in key or "渣" in key:
        return DefectClass.SLAG
    if "penetr" in key or "焊透" in key:
        return DefectClass.INCOMPLETE_PENETRATION
    if "fusion" in key or "熔合" in key:
        return DefectClass.LACK_OF_FUSION
    if "crack" in key or "裂纹" in key:
        return DefectClass.CRACK
    if "concav" in key or "内凹" in key:
        return DefectClass.CONCAVITY
    return None
