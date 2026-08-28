"""标准数值表数据源端口（§T8 依赖倒置）。

领域层只声明"需要按 standard_id 取一张标准表"，不关心表从何而来
（本地 YAML / 数据库 / 云端）。具体实现（`FileTableSource` 等）由基础设施层提供，
经 DI 注入；生产环境 domain 因此完全不接触文件系统（§T8 验收硬化）。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TableSource(Protocol):
    """标准数值表数据源（模型无关，ADR-002 同构）。"""

    def load(self, standard_id: str, filename: str | None = None) -> "StandardTables":
        """加载并校验标准数值表；文件缺失/结构不符抛错（启动即失败）。"""
        ...
