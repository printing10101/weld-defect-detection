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
    "round_points",           # {{STD_TABLE_点数}}
    "round_ignore_size_mm",   # {{STD_TABLE_不计}}
    "linear_limits",          # {{STD_TABLE_条形}}
}


@dataclass(frozen=True)
class StandardTables:
    standard_id: str
    version: str
    authorized: bool
    data: dict[str, Any]


def load_standard_tables(standard_id: str, filename: str | None = None) -> StandardTables:
    """加载并校验标准数值表。文件缺失或结构不符直接抛错（启动即失败）。

    filename 缺省时由 standard_id 派生（含 '/' 等字符时需显式传入，如
    'nb47013.yaml' 对应 'NB/T47013.2-2015'）。
    """
    name = filename or f"{standard_id}.yaml"
    path = _TABLES_DIR / name
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
    )


def _validate(raw: dict[str, Any]) -> None:
    missing = _REQUIRED_KEYS - set(raw)
    if missing:
        raise ValueError(f"standard table missing keys: {sorted(missing)}")
