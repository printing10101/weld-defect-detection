"""密级标识字段（C-10）

Revision ID: 0008_secret_level
Revises: 0007_accounts_auth
Create Date:

- images / reports 增加 secret_level（0=非密 1=内部 2=秘密 3=机密，默认非密）
  与 classification_basis（定密依据）。SQLite ALTER TABLE ADD COLUMN 直兼容；
  历史行由 server_default='0' 回填为非密，行为不变。
- reviews 不单列密级：复核行隶属影像，密级随 images.secret_level 读取
  （评审动作本身不改变定密属性，避免双写漂移）。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_secret_level"
down_revision: str | None = "0007_accounts_auth"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """带存在性守卫的加列：遗留库（stamp 后 upgrade 路径）可能缺 reports 等表，
    缺表时跳过——该路径下 create_all 兜底建表会带上新列（ORM 为 schema 真源）。"""
    bind = op.get_bind()
    insp = sa.inspect(bind)
    for table in ("images", "reports"):
        if table not in insp.get_table_names():
            continue
        cols = {c["name"] for c in insp.get_columns(table)}
        if "secret_level" not in cols:
            op.add_column(
                table,
                sa.Column("secret_level", sa.Integer(), nullable=False, server_default="0"),
            )
        if "classification_basis" not in cols:
            op.add_column(
                table,
                sa.Column("classification_basis", sa.String(length=256), nullable=True),
            )


def downgrade() -> None:
    for table in ("reports", "images"):
        bind = op.get_bind()
        insp = sa.inspect(bind)
        if table not in insp.get_table_names():
            continue
        cols = {c["name"] for c in insp.get_columns(table)}
        if "classification_basis" in cols:
            op.drop_column(table, "classification_basis")
        if "secret_level" in cols:
            op.drop_column(table, "secret_level")
