"""标准数值表文件数据源（基础设施层实现， 依赖倒置）。

实现 domain 声明的 TableSource 端口：从本地 YAML 加载并校验标准数值表。
domain 因此不接触文件系统——本实现由 app（dependencies）在启动时注入为默认数据源。

如需数据库/云端表，新增一个 TableSource 实现并在此装配即可，domain 主干不变。
"""

from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from backend.domain.standards.tables.loader import StandardTables, _validate

# 标准表随 domain 包分发（backend/domain/standards/tables/*.yaml）；
# 经 domain 包定位目录，避免相对 infra 路径的脆弱拼接。
_tables_pkg = importlib.import_module("backend.domain.standards.tables")
_TABLES_DIR = Path(_tables_pkg.__file__).parent  # type: ignore[arg-type]


class FileTableSource:
    """本地 YAML 标准表数据源（TableSource 实现）。"""

    def __init__(self, tables_dir: str | Path | None = None) -> None:
        self._dir = Path(tables_dir) if tables_dir else _TABLES_DIR

    def load(self, standard_id: str, filename: str | None = None) -> StandardTables:
        """加载并校验标准数值表。文件缺失或结构不符直接抛错（启动即失败）。

        filename 缺省时由 standard_id 派生；绝对路径（测试注入授权副本）优先。
        """
        name = filename or f"{standard_id}.yaml"
        path = Path(name) if Path(name).is_absolute() else self._dir / name
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
