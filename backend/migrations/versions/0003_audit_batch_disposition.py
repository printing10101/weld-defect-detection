"""add batch_no / disposition

Revision ID: 0003_audit_batch_disposition
Revises: 0002_devices_report_hash
Create Date: 2026-08-15

P1-F 审计增强（可空列，向后兼容）：
- images.batch_no   : 批量追溯（batch 导入/复评批次）
- defects.disposition: 处置建议快照（accept | conditional | rework | recheck）
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_audit_batch_disposition"
down_revision: str | None = "0002_devices_report_hash"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # SQLite 加列需 batch 模式（env.py 已开 render_as_batch）
    with op.batch_alter_table("images") as batch_op:
        batch_op.add_column(sa.Column("batch_no", sa.String(length=64), nullable=True))
        batch_op.create_index("ix_images_batch_no", ["batch_no"])
    with op.batch_alter_table("defects") as batch_op:
        batch_op.add_column(sa.Column("disposition", sa.String(length=16), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("images") as batch_op:
        batch_op.drop_index("ix_images_batch_no")
        batch_op.drop_column("batch_no")
    with op.batch_alter_table("defects") as batch_op:
        batch_op.drop_column("disposition")
