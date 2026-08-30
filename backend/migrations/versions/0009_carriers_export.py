"""载体台账与导出审批（C-12/C-14）

Revision ID: 0009_carriers_export
Revises: 0008_secret_level
Create Date:

- carriers: 涉密载体（底片/报告/备份）登记、借还与销毁全生命周期台账；
  销毁需保密员发起 + 系统管理员双确认，记录销毁方式与经办人。
- export_requests: 导出审批流——申请 → 保密员批准 → 一次性令牌 → 凭令下载，
  全部动作入审计。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009_carriers_export"
down_revision: str | None = "0008_secret_level"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "carriers",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("kind", sa.String(length=16), nullable=False),  # film | report | backup
        sa.Column("object_id", sa.String(length=64), nullable=True),
        sa.Column("secret_level", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("owner", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="in_stock"),
        sa.Column("borrow_history", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("destroy_method", sa.String(length=64), nullable=True),
        sa.Column("destroy_note", sa.Text(), nullable=True),
        sa.Column("destroy_requested_by", sa.String(length=64), nullable=True),
        sa.Column("destroy_confirmed_by", sa.String(length=64), nullable=True),
        sa.Column("destroyed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_carriers_object_id", "carriers", ["object_id"])
    op.create_index("ix_carriers_status", "carriers", ["status"])
    op.create_index("ix_carriers_created_at", "carriers", ["created_at"])

    op.create_table(
        "export_requests",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("subject", sa.String(length=128), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("requested_by", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="pending"),
        sa.Column("decided_by", sa.String(length=64), nullable=True),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
        sa.Column("token_hash", sa.String(length=64), nullable=True),
        sa.Column("token_expires_at", sa.DateTime(), nullable=True),
        sa.Column("used_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_export_requests_subject", "export_requests", ["subject"])
    op.create_index("ix_export_requests_status", "export_requests", ["status"])
    op.create_index("ix_export_requests_created_at", "export_requests", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_export_requests_created_at", table_name="export_requests")
    op.drop_index("ix_export_requests_status", table_name="export_requests")
    op.drop_index("ix_export_requests_subject", table_name="export_requests")
    op.drop_table("export_requests")
    op.drop_index("ix_carriers_created_at", table_name="carriers")
    op.drop_index("ix_carriers_status", table_name="carriers")
    op.drop_index("ix_carriers_object_id", table_name="carriers")
    op.drop_table("carriers")
