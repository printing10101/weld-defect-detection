"""P2-5：启动时 schema 迁移（Alembic 兼容历史 create_all DB）。

- 全新 DB → upgrade head 建表；
- 历史 create_all DB（无 alembic_version）→ stamp head，不执行 DDL（避免建表冲突）；
- 幂等：重复运行版本不变。
"""

from __future__ import annotations

from sqlalchemy import inspect

from backend.infra.db import Base, create_db_engine
from backend.infra.migrate import ensure_migrations


def test_migrate_fresh_db_creates_tables(tmp_path) -> None:
    p = str(tmp_path / "fresh.db")
    version = ensure_migrations(p)
    eng = create_db_engine(p)
    with eng.connect() as c:
        tables = set(inspect(c).get_table_names())
    assert version == "0001_initial"
    assert {"images", "defects", "reports", "reviews", "audit_log", "alembic_version"} <= tables


def test_migrate_legacy_create_all_db_stamps_head(tmp_path) -> None:
    p = str(tmp_path / "legacy.db")
    # 模拟历史 DB：create_all 已建表，但无 alembic_version
    Base.metadata.create_all(create_db_engine(p))
    version = ensure_migrations(p)
    eng = create_db_engine(p)
    with eng.connect() as c:
        has_version = inspect(c).get_table_names().__contains__("alembic_version")
        ver = c.exec_driver_sql("SELECT version_num FROM alembic_version").fetchone()
    assert has_version
    assert ver[0] == "0001_initial"
    assert version == "0001_initial"
    # 幂等
    assert ensure_migrations(p) == "0001_initial"
