"""启动时 schema 迁移（§13.6 / P2-5）。

用 Alembic 管理 schema 版本，替代"仅靠 create_all 裸建表"：
- 全新 DB（无 images 表）→ `alembic upgrade head` 建表；
- 历史 DB（create_all 已建表、但无 alembic_version）→ `alembic stamp head`
  （打上基线版本、不执行 DDL，避免 CREATE TABLE 与已存在表冲突）；
- 已带版本表 → `alembic upgrade head` 应用任何新增迁移。

无论哪条路径，仓储层保留的 `Base.metadata.create_all` 作为幂等兜底仍安全。
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import create_engine, inspect

_LOG = logging.getLogger("scandetection.migrate")

_ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"
_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


def ensure_migrations(db_path: str) -> str:
    """确保 DB schema 处于 Alembic 头版本；返回最终所在版本号。

    注意：探测与最终读版本使用独立短连接（context 退出即归还连接池），
    执行 alembic 命令前**必须**先释放探测连接——SQLite 连接池在 alembic
    命令持有 DDL 连接时复用会触发 ResourceClosedError。
    """
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(_ALEMBIC_INI))
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")

    engine = create_engine(f"sqlite:///{db_path}", future=True)
    with engine.connect() as conn:
        insp = inspect(conn)
        has_version = insp.has_table("alembic_version")
        has_images = insp.has_table("images")
        version_row = (
            conn.exec_driver_sql("SELECT version_num FROM alembic_version").fetchone()
            if has_version
            else None
        )
        current = version_row[0] if version_row else None
    engine.dispose()  # 释放探测连接，避免与 alembic 的 DDL 连接冲突

    if has_version:
        # 已受版本管理：应用任何待执行迁移（通常是 no-op）
        if current is None and has_images:
            # 历史 create_all 遗留 + 空版本表（早期 upgrade 冲突被吞、仅 create_all 兜底）：
            # 先 stamp 到当前头版本之前的基线，再 upgrade head，只执行新增迁移（0003 起），
            # 避免 0001 的 CREATE TABLE 与既有表冲突。
            _LOG.info("empty alembic_version + legacy tables: stamp baseline then upgrade")
            command.stamp(cfg, "0002_devices_report_hash")
        command.upgrade(cfg, "head")
    elif has_images:
        # 历史 create_all DB：标记到基线，不重建表
        _LOG.info("legacy create_all DB detected, stamping baseline (no DDL)")
        command.stamp(cfg, "head")
    else:
        # 全新 DB：执行初始迁移建表
        _LOG.info("fresh DB, running initial migration")
        command.upgrade(cfg, "head")

    # 读取最终版本（新短连接）
    with engine.connect() as conn:
        row = conn.exec_driver_sql("SELECT version_num FROM alembic_version").fetchone()
    engine.dispose()
    return row[0] if row else "?"


if __name__ == "__main__":
    # 手动执行：python -m backend.infra.migrate <db_path>
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "data/scan.db"
    print(ensure_migrations(target))
