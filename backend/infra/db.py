"""SQLite 引擎/会话（SQLAlchemy，§7.1 / §T4）。

M1 仅提供引擎工厂；表结构随 M5 迁移补齐（migrations/，Alembic）。
后续云端扩展同 schema 换 PostgreSQL，SQLAlchemy 屏蔽差异。
"""
from __future__ import annotations

from sqlalchemy import Engine, create_engine


def create_db_engine(path: str) -> Engine:
    return create_engine(f"sqlite:///{path}", future=True)
