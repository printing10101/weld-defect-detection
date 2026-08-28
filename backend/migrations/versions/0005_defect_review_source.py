"""defects.source / defects.deleted_at（人工复核缺陷增删改留痕）

Revision ID: 0005_defect_review_source
Revises: 0004_drop_users
Create Date:

DB50/T 1807-2025  复核功能补全：
- defects.source     : 来源（auto=检测器 | manual=人工添加），全程可追溯；
- defects.deleted_at : 软删除时间（复核删除不物理清除，供审计追溯）。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_defect_review_source"
down_revision: str | None = "0004_drop_users"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # SQLite 加列需 batch 模式（env.py 已开 render_as_batch）
    with op.batch_alter_table("defects") as batch_op:
        batch_op.add_column(sa.Column("source", sa.String(length=16), nullable=True))
        batch_op.add_column(sa.Column("deleted_at", sa.DateTime(), nullable=True))
    op.create_index("ix_defects_deleted_at", "defects", ["deleted_at"])


def downgrade() -> None:
    op.drop_index("ix_defects_deleted_at", table_name="defects")
    with op.batch_alter_table("defects") as batch_op:
        batch_op.drop_column("deleted_at")
        batch_op.drop_column("source")
