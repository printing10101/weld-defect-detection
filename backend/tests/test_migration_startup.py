"""P2-5：启动时 schema 迁移（Alembic 兼容历史 create_all DB）。

- 全新 DB → upgrade head 建表；
- 历史 create_all DB（无 alembic_version）→ stamp head，不执行 DDL（避免建表冲突）；
- 幂等：重复运行版本不变。
"""

from __future__ import annotations

from sqlalchemy import inspect

from backend.infra.db import Base, create_db_engine
from backend.infra.migrate import ensure_migrations

# schema 演进：0001 基线 + 0002（devices/calibrations + reports 数字签名字段）+ 0003（ 审计增强）
# + 0004（删除 users 表：移除用户/认证系统，改操作员姓名机制）
_HEAD = "0005_defect_review_source"


def test_migrate_fresh_db_creates_tables(tmp_path) -> None:
    p = str(tmp_path / "fresh.db")
    version = ensure_migrations(p)
    eng = create_db_engine(p)
    with eng.connect() as c:
        tables = set(inspect(c).get_table_names())
    assert version == _HEAD
    assert {"images", "defects", "reports", "reviews", "audit_log", "alembic_version"} <= tables
    assert {"devices", "calibrations"} <= tables


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
    assert ver is not None
    assert ver[0] == _HEAD
    assert version == _HEAD
    # 幂等
    assert ensure_migrations(p) == _HEAD


def test_migrate_empty_version_legacy_db_applies_only_new_migrations(tmp_path) -> None:
    """历史 create_all DB + 空 alembic_version（早期 upgrade 被吞）→
    stamp 到上一基线再 upgrade，只执行 0003，且新列存在。"""
    p = str(tmp_path / "legacy_empty_version.db")
    # 模拟 0002 时代旧库：images/defects 无 batch_no/disposition + alembic_version 空表
    eng = create_db_engine(p)
    with eng.connect() as c:
        c.exec_driver_sql("CREATE TABLE images (id VARCHAR(64) PRIMARY KEY)")
        c.exec_driver_sql("CREATE TABLE defects (id VARCHAR(64) PRIMARY KEY)")
        c.exec_driver_sql("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
    eng.dispose()

    version = ensure_migrations(p)
    assert version == _HEAD

    eng = create_db_engine(p)
    with eng.connect() as c:
        cols_images = {r[1] for r in c.exec_driver_sql("PRAGMA table_info(images)").fetchall()}
        cols_defects = {r[1] for r in c.exec_driver_sql("PRAGMA table_info(defects)").fetchall()}
        ver = c.exec_driver_sql("SELECT version_num FROM alembic_version").fetchone()
    eng.dispose()
    assert ver is not None and ver[0] == _HEAD
    assert "batch_no" in cols_images  # P1-F：images.batch_no
    assert "disposition" in cols_defects  # P0-E/P1-F：defects.disposition


def test_new_columns_present_in_orm(tmp_path) -> None:
    """ORM 模型与迁移列一致（模型即 schema 真源，）。"""
    from backend.infra.db import DefectRecord, ImageRecord

    cols_img = {c.name for c in ImageRecord.__table__.columns}
    cols_def = {c.name for c in DefectRecord.__table__.columns}
    assert "batch_no" in cols_img
    assert "disposition" in cols_def
