"""启动时 schema 迁移。

用 Alembic 管理 schema 版本，替代"仅靠 create_all 裸建表"：
- 全新 DB（无 images 表）→ `alembic upgrade head` 建表；
- 历史 DB（create_all 已建表、但无 alembic_version）→ `alembic stamp head`
  （打上基线版本、不执行 DDL，避免 CREATE TABLE 与已存在表冲突）；
- 已带版本表 → `alembic upgrade head` 应用任何新增迁移。

无论哪条路径，仓储层保留的 `Base.metadata.create_all` 作为幂等兜底仍安全。

列级自愈（_reconcile_missing_columns）：stamp head 与"upgrade 中途失败"都只
推进版本号、不执行 DDL，而 create_all 只建缺失表、不改已有表——此后业务
INSERT 引用新增列即 OperationalError（安装版 images.film_no 事故：遗留库被
stamp 到 0014 却没有该列，单幅评定归档必炸）。故每次迁移收尾都以 ORM 元数据
为真源对已存在的表补齐缺失列（仅 ADD COLUMN，幂等），堵住这一类静默漂移。
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from sqlalchemy import create_engine, inspect

_LOG = logging.getLogger("scandetection.migrate")

_ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"
_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"

# 进程内 DDL 串行化锁（create_all 与 alembic 命令共用）：alembic 的 env.py 代理机制（util.langhelpers 的
# _proxy 全局）不支持同进程并发 upgrade/stamp；同时 Base.metadata.create_all
# 与迁移线程并发建表会撞 "table already exists"。主应用 lifespan 后台线程、
# get_registry 装配、测试重建 Registry 都可能并发触发 DDL，统一在此串行化。
# 加锁代价可忽略（DDL 本就是低频操作）。
DDL_LOCK = threading.Lock()


def _schema_has(db_path: str, table: str) -> bool:
    """独立短连接探测表是否已存在（用于识别 create_all 裸建的遗留库）。"""
    import sqlite3

    con = sqlite3.connect(db_path)
    try:
        row = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        return row is not None
    finally:
        con.close()


def _reconcile_missing_columns(db_path: str) -> list[str]:
    """列级自愈：以 ORM 元数据为真源，为"表在、列缺"的表补 ADD COLUMN（幂等）。

    只做 SQLite 支持的 ADD COLUMN：可空列直接加；非空列必须带 server_default
    （SQLite ADD COLUMN 不回填存量行，无默认值的非空列加不上）——这类跳过并
    error 留痕，宁缺勿造伪值，由人工迁移脚本处理。缺表场景不归这里管
    （alembic 建新表 / create_all 兜底）。
    """
    from sqlalchemy import create_engine, inspect

    from backend.infra.db import Base

    engine = create_engine(f"sqlite:///{db_path}", future=True)
    added: list[str] = []
    try:
        with engine.connect() as conn:
            insp = inspect(conn)
            actual_tables = set(insp.get_table_names())
            for table in Base.metadata.sorted_tables:
                if table.name not in actual_tables:
                    continue
                actual_cols = {c["name"] for c in insp.get_columns(table.name)}
                for col in table.columns:
                    if col.name in actual_cols:
                        continue
                    if not col.nullable and col.server_default is None:
                        _LOG.error(
                            "列自愈跳过 %s.%s：非空列且无 server_default，"
                            "SQLite ADD COLUMN 无法安全回填，请补人工迁移脚本",
                            table.name,
                            col.name,
                        )
                        continue
                    ddl = (
                        f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" '
                        f"{col.type.compile(engine.dialect)}"
                    )
                    if col.server_default is not None:
                        ddl += f" DEFAULT {col.server_default.arg}"
                    conn.exec_driver_sql(ddl)
                    added.append(f"{table.name}.{col.name}")
            if added:
                conn.commit()
    finally:
        engine.dispose()
    if added:
        _LOG.warning("schema 列自愈完成，补齐缺失列：%s", ", ".join(added))
    return added


def ensure_migrations(db_path: str) -> str:
    """确保 DB schema 处于 Alembic 头版本；返回最终所在版本号。

    S-03：``db_path`` 可为 sqlite 文件路径或 paths.db_url 提供的完整 URL。
    - sqlite（默认）：路径语义不变，alembic URL 为 ``sqlite:///<path>``；
    - 非 sqlite URL（达梦/人大金仓等）：**未真机验证**，迁移链仅在 SQLite 上
      联调过——此处诚实跳过 alembic（返回 "skipped-non-sqlite"），由仓储层
      create_all 兜底建表，避免未验证方言上盲跑 DDL。
    - 环境变量覆盖：``SCAN_DB_URL`` 优先于入参（容器/部署注入用），
      alembic.ini 中 sqlalchemy.url 仅是占位。

    注意：探测与最终读版本使用独立短连接（context 退出即归还连接池），
    执行 alembic 命令前**必须**先释放探测连接——SQLite 连接池在 alembic
    命令持有 DDL 连接时复用会触发 ResourceClosedError。
    进程级 DDL_LOCK 串行化（见其注释）。
    """
    target = os.environ.get("SCAN_DB_URL", "").strip() or db_path
    if "://" in target and not target.startswith("sqlite"):
        _LOG.warning(
            "非 SQLite 方言（%s…）的 alembic 迁移未真机验证，跳过迁移（create_all 兜底）",
            target.split("://")[0],
        )
        return "skipped-non-sqlite"

    with DDL_LOCK:
        return _ensure_migrations_locked(target)


def _ensure_migrations_locked(db_path: str) -> str:
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

    try:
        if has_version:
            # 已受版本管理：应用任何待执行迁移（通常是 no-op）
            if current is None and has_images:
                # 历史 create_all 遗留 + 空版本表（早期 upgrade 冲突被吞、仅 create_all 兜底）：
                # 先 stamp 到当前头版本之前的基线，再 upgrade head，只执行新增迁移（0003 起），
                # 避免 0001 的 CREATE TABLE 与既有表冲突。
                _LOG.info("empty alembic_version + legacy tables: stamp baseline then upgrade")
                command.stamp(cfg, "0002_devices_report_hash")
            if current == "0001_initial" and _schema_has(db_path, "devices"):
                # 打包版遗留库：create_all 已建出 0002/0003 的表（devices 等），
                # 版本却停在 0001。直接 upgrade 会在 0002 的 CREATE TABLE 撞表。
                # 先 stamp 到 0003，再 upgrade 只执行 0004 起的新迁移。
                _LOG.info("stamped 0001 legacy DB has newer tables, stamping 0003 then upgrade")
                command.stamp(cfg, "0003_audit_batch_disposition")
            command.upgrade(cfg, "head")
        elif has_images:
            # 历史 create_all DB：标记到基线，不重建表
            # （不执行 DDL 意味着此后的新增列不会落库——由 finally 的列自愈兜底）
            _LOG.info("legacy create_all DB detected, stamping baseline (no DDL)")
            command.stamp(cfg, "head")
        else:
            # 全新 DB：执行初始迁移建表
            _LOG.info("fresh DB, running initial migration")
            command.upgrade(cfg, "head")
    finally:
        # 无论 upgrade 成功、stamp 跳过 DDL 还是中途失败，都以 ORM 元数据
        # 补齐缺失列——版本号推进了而物理列没跟上，比版本落后更危险
        # （业务 INSERT 直接 OperationalError，且 create_all 永远不会修列）。
        _reconcile_missing_columns(db_path)

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
