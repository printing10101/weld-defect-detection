"""标准数值表加载器（§T8）。

带结构校验；authorized=false（占位/未获授权）时**禁止**用于级别输出——
这是防"静默错判"的熔断（§11 风险表），任何判定实现必须先检查 authorized。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_TABLES_DIR = Path(__file__).parent

_REQUIRED_KEYS = {
    "standard_id",
    "version",
    "authorized",
    "round_points",  # {{STD_TABLE_点数}}
    "round_ignore_size_mm",  # {{STD_TABLE_不计}}
    "round_grade_limits",  # {{STD_TABLE_圆形上限}}
    "round_rating_zone_mm",  # {{STD_TABLE_评定区}}
    "linear_limits",  # {{STD_TABLE_条形}}
}


@dataclass(frozen=True)
class StandardTables:
    standard_id: str
    version: str
    authorized: bool
    data: dict[str, Any]
    # 工业过渡路径（T1）：两个独立概念，避免"数值完整可用"与"持有授权正本"混为一谈。
    # - authorized：数值完整、评级运算可运行（AI 预筛）。False 时熔断不输出级别。
    # - authorized_copy：是否依法持有标准授权正本并完成逐条签核。默认 False。
    authorized_copy: bool = False
    # 数值来源说明（转录自公开解读 / 授权正本副本等），随免责声明输出。
    source_note: str = ""


def disclaimer_for(tables: StandardTables) -> str:
    """生成标准来源免责声明（工业过渡路径，T1）。

    authorized_copy=False（未持有授权正本）→ 强声明：评级数值转录自公开解读，
    仅供 AI 辅助预筛，不替代责任工程师法定评定。authorized_copy=True → 轻声明。
    声明只依赖标准表（standard-level），判定器 / API / PDF 报告共用同一实现。
    """

    if tables.authorized_copy:
        return (
            "评级数值已依授权标准正本逐条复核并签核（authorized_copy=true）。"
            "本评级仍须由责任工程师复核确认后生效。"
        )
    note = (tables.source_note or "").strip()
    head = (
        "⚠ 标准来源声明：本系统评级数值转录自公开解读文本，非标准授权正本；"
        "当前未持有授权副本，数值未经授权原文逐条复核与正式签核。"
    )
    tail = (
        "本评级仅用于 AI 辅助预筛与质量追溯参考，不构成合格/不合格的法定判定依据；"
        "最终级别须经持证人/责任工程师依授权标准原文复核并签核后方可采信。"
    )
    if note:
        return f"{head}\n{note}\n{tail}"
    return f"{head}\n{tail}"


def load_standard_tables(standard_id: str, filename: str | None = None) -> StandardTables:
    """加载并校验标准数值表。文件缺失或结构不符直接抛错（启动即失败）。

    filename 缺省时由 standard_id 派生（含 '/' 等字符时需显式传入，如
    'nb47013.yaml' 对应 'NB/T47013.2-2015'）。
    """
    name = filename or f"{standard_id}.yaml"
    # 支持绝对路径（测试注入 authorized 表副本用）；否则按 tables/ 目录解析
    path = Path(name) if Path(name).is_absolute() else _TABLES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"no standard tables for: {standard_id} ({name})")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    _validate(raw)
    if raw["standard_id"] != standard_id:
        raise ValueError(f"table standard_id mismatch: {raw['standard_id']} != {standard_id}")
    return StandardTables(
        standard_id=str(raw["standard_id"]),
        version=str(raw["version"]),
        authorized=bool(raw["authorized"]),
        data=raw,
        authorized_copy=bool(raw.get("authorized_copy", False)),
        source_note=str(raw.get("source_note", "")).strip(),
    )


def _validate(raw: dict[str, Any]) -> None:
    missing = _REQUIRED_KEYS - set(raw)
    if missing:
        raise ValueError(f"standard table missing keys: {sorted(missing)}")
    # 嵌套结构校验：防止运行时才在评级中 KeyError/TypeError（§T8 要求启动即失败）
    for key, field in (
        ("round_points", "max_d_mm"),
        ("round_ignore_size_mm", "max_t"),
        ("round_grade_limits", "max_t"),
        ("round_rating_zone_mm", "max_t"),
    ):
        rows = raw.get(key)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{key} 必须是非空列表")
        keys = [r.get(field) for r in rows]
        if any(k is None for k in keys):
            raise ValueError(f"{key} 的每行必须包含 '{field}'")
        if keys != sorted(keys):
            raise ValueError(f"{key} 必须按 '{field}' 升序排列")
    for lv in ("level2", "level3"):
        sub = (raw.get("linear_limits") or {}).get(lv, {})
        if not {"t_factor", "min_mm", "max_mm"} <= set(sub):
            raise ValueError(f"linear_limits.{lv} 缺少 t_factor/min_mm/max_mm")
    # 条形组内累计（§6.2，12T 评定区）：可选键，存在则必须结构完整
    group = (raw.get("linear_limits") or {}).get("group")
    if group is not None:
        if not isinstance(group.get("zone_t_factor"), (int, float)) or group["zone_t_factor"] <= 0:
            raise ValueError("linear_limits.group.zone_t_factor 必须为正数")
        for lv in ("level2", "level3"):
            sub = group.get(lv, {})
            if not {"t_factor", "min_mm", "max_mm"} <= set(sub):
                raise ValueError(f"linear_limits.group.{lv} 缺少 t_factor/min_mm/max_mm")
