"""Alembic 环境（§13.6 / P2-5）。

- target_metadata 取自 backend.infra.db.Base.metadata（模型即 schema 真源，
  支持 `alembic revision --autogenerate`）；
- sqlalchemy.url 由 backend/infra/migrate.py 在启动时按 AppConfig.paths.db_path
  覆盖，env.py 不再自行读取配置（避免 CWD/相对路径问题）；
- 关闭 transactional_ddl 以兼容 SQLite（SQLite 不支持 DDL 事务）。
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from backend.infra.db import Base

config = context.config
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # SQLite 改名/删列需 batch 模式
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
