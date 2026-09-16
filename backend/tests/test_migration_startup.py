"""P2-5：启动时 schema 迁移（Alembic 兼容历史 create_all DB）。

- 全新 DB → upgrade head 建表；
- 历史 create_all DB（无 alembic_version）→ stamp head，不执行 DDL（避免建表冲突）；
- 幂等：重复运行版本不变；
- 列自愈：stamp head / upgrade 中途失败只推进版本号不执行 DDL，ORM 元数据
  对"表在、列缺"做幂等 ADD COLUMN 补齐（安装版 images.film_no 事故回归）。
"""

from __future__ import annotations

import pytest
import sqlalchemy.exc as sa_exc
from sqlalchemy import inspect

from backend.infra.db import Base, create_db_engine
from backend.infra.migrate import ensure_migrations

# schema 演进：0001 基线 + 0002（devices/calibrations + reports 数字签名字段）+ 0003（ 审计增强）
# + 0004（删除 users 表：移除用户/认证系统，改操作员姓名机制）
# + 0005（defects 复核留痕）+ 0006（gate_rejects 不合格底片留档台账）
# + 0010（images.content_hash 影像内容摘要：批量上传查重）
# + 0011（images.stamp_* 底片印字识别快照：扫描日期/编号，正/镜像）
# + 0013（defect_atlas 缺陷图谱样本库）
# + 0014（images.film_no 独立片号：印字 OCR 结构化抽取，G05）
_HEAD = "0014_film_no"


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
    assert "content_hash" in cols_img
    assert "disposition" in cols_def


def test_migrate_legacy_db_missing_column_healed(tmp_path) -> None:
    """回归（安装版 film_no 事故）：遗留库（无 alembic_version、images 只有
    旧时代列）→ stamp head 不执行 DDL，版本到头而物理列缺失；列自愈必须把
    ORM 新增列补齐，否则评定归档 INSERT 新列即 OperationalError。"""
    p = str(tmp_path / "legacy_missing_col.db")
    eng = create_db_engine(p)
    with eng.connect() as c:
        # 只建 images 一张 0001 时代最小表：无 film_no/batch_no/content_hash
        c.exec_driver_sql(
            "CREATE TABLE images (id VARCHAR(64) PRIMARY KEY, path VARCHAR(512))"
        )
    eng.dispose()

    version = ensure_migrations(p)
    assert version == _HEAD

    eng = create_db_engine(p)
    with eng.connect() as c:
        cols = {r[1] for r in c.exec_driver_sql("PRAGMA table_info(images)").fetchall()}
    eng.dispose()
    assert {"film_no", "batch_no", "content_hash"} <= cols
    # 幂等：重复运行版本不变、不抛异常
    assert ensure_migrations(p) == _HEAD


def test_migrate_versioned_db_missing_column_healed(tmp_path) -> None:
    """回归（安装版事故的精确状态）：alembic_version 已到 head、物理列缺失
    （历史版本曾被 stamp head 跳过 DDL）→ upgrade 是 no-op，列自愈必须仍把
    缺失列补齐。此状态若不修复，业务 INSERT 新列即 OperationalError。"""
    p = str(tmp_path / "versioned_missing_col.db")
    eng = create_db_engine(p)
    with eng.connect() as c:
        c.exec_driver_sql(
            "CREATE TABLE images (id VARCHAR(64) PRIMARY KEY, path VARCHAR(512))"
        )
        c.exec_driver_sql(
            "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"
        )
        c.exec_driver_sql(f"INSERT INTO alembic_version VALUES ('{_HEAD}')")
        c.commit()  # SQLAlchemy 2.0 commit-as-you-go：不显式提交则 INSERT 被回滚
    eng.dispose()

    version = ensure_migrations(p)
    assert version == _HEAD

    eng = create_db_engine(p)
    with eng.connect() as c:
        cols = {r[1] for r in c.exec_driver_sql("PRAGMA table_info(images)").fetchall()}
    eng.dispose()
    assert {"film_no", "batch_no", "content_hash"} <= cols


def test_migrate_upgrade_failure_still_heals_columns(tmp_path) -> None:
    """upgrade 中途撞表失败（dev 库 0013 defect_atlas 已存在场景）：异常照常
    上抛（调用方 create_all 兜底），但 finally 的列自愈仍补齐缺失列——
    版本卡住不能阻止列漂移被修复。"""
    p = str(tmp_path / "upgrade_fails.db")
    eng = create_db_engine(p)
    with eng.connect() as c:
        # 版本停在 0012，但 0013 要建的 defect_atlas 已被 create_all 抢建
        c.exec_driver_sql("CREATE TABLE images (id VARCHAR(64) PRIMARY KEY)")
        c.exec_driver_sql("CREATE TABLE defect_atlas (id VARCHAR(64) PRIMARY KEY)")
        c.exec_driver_sql(
            "CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"
        )
        c.exec_driver_sql("INSERT INTO alembic_version VALUES ('0012_report_meta')")
    eng.dispose()

    with pytest.raises(sa_exc.OperationalError):
        ensure_migrations(p)

    eng = create_db_engine(p)
    with eng.connect() as c:
        cols = {r[1] for r in c.exec_driver_sql("PRAGMA table_info(images)").fetchall()}
    eng.dispose()
    assert "film_no" in cols
