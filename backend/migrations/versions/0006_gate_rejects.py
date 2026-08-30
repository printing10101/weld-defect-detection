"""gate_rejects（不合格底片留档台账，DB50/T 1807-2025 §5）

Revision ID: 0006_gate_rejects
Revises: 0005_defect_review_source
Create Date:

评片门禁（黑度/IQI/伪缺陷/质量/扫描参数）拦截的底片此前直接抛 IQIFailError、
早于原图落盘，无任何留档。本迁移建台账表：
- gate_rejects: 拒绝原因/明细 JSON/dpi/位深/操作员/时间（image_id 可空，
  拦截发生在 images 行写入之前）。
影像副本由 pipelines._persist_reject 按 security.encrypt 密文归档到
gate.rejects_dir；审计动作 gate_reject 走仓储 append_audit 哈希链。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_gate_rejects"
down_revision: str | None = "0005_defect_review_source"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "gate_rejects",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("image_id", sa.String(length=64), nullable=True),
        sa.Column("reject_reason", sa.String(length=256), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column("dpi", sa.Float(), nullable=True),
        sa.Column("bit_depth", sa.Integer(), nullable=True),
        sa.Column("operator", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_gate_rejects_created_at", "gate_rejects", ["created_at"])
    op.create_index("ix_gate_rejects_image_id", "gate_rejects", ["image_id"])


def downgrade() -> None:
    op.drop_index("ix_gate_rejects_image_id", table_name="gate_rejects")
    op.drop_index("ix_gate_rejects_created_at", table_name="gate_rejects")
    op.drop_table("gate_rejects")
