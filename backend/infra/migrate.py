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
    """确保 DB schema 处于 Alembic 头版本；返回最终所在版本号。"""
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
    engine.dispose()

    if has_version:
        # 已受版本管理：应用任何待执行迁移（通常是 no-op）
        command.upgrade(cfg, "head")
    elif has_images:
        # 历史 create_all DB：标记到基线，不重建表
        _LOG.info("legacy create_all DB detected, stamping baseline (no DDL)")
        command.stamp(cfg, "head")
    else:
        # 全新 DB：执行初始迁移建表
        _LOG.info("fresh DB, running initial migration")
        command.upgrade(cfg, "head")

    # 读取最终版本
    with engine.connect() as conn:
        row = conn.exec_driver_sql("SELECT version_num FROM alembic_version").fetchone()
    return row[0] if row else "?"


if __name__ == "__main__":
    # 手动执行：python -m backend.infra.migrate <db_path>
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "data/scan.db"
    print(ensure_migrations(target))
