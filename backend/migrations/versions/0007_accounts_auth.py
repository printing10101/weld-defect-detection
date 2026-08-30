"""三员身份体系（C-06/C-07/C-09/C-19）

Revision ID: 0007_accounts_auth
Revises: 0006_gate_rejects
Create Date:

- accounts: 三员账号（sysadmin/secadmin/auditor，一人一岗），SM2 公钥登录，
  无口令；failed_attempts/locked_until 支撑连续挑战失败锁定。
- sessions: Bearer 会话，库中仅存 token 的 SM3 哈希；last_seen_at 支撑空闲超时。
- alerts: 安全告警入库（账号锁定等；波次3扩展处置流转）。
- security_audit: 独立安全审计链（C-19 双链），结构同 audit_log、独立 SM3 哈希链。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_accounts_auth"
down_revision: str | None = "0006_gate_rejects"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_AUDIT_COLS = [
    sa.Column("actor", sa.String(length=64), nullable=False),
    sa.Column("action", sa.String(length=32), nullable=False),
    sa.Column("object_type", sa.String(length=32), nullable=False),
    sa.Column("object_id", sa.String(length=64), nullable=False),
    sa.Column("before", sa.JSON(), nullable=True),
    sa.Column("after", sa.JSON(), nullable=True),
    sa.Column("note", sa.Text(), nullable=True),
    sa.Column("prev_hash", sa.String(length=64), nullable=False),
    sa.Column("hash", sa.String(length=64), nullable=False),
    sa.Column("created_at", sa.DateTime(), nullable=False),
]


def upgrade() -> None:
    op.create_table(
        "accounts",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("username", sa.String(length=64), nullable=False, unique=True),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("sm2_public_key", sa.String(length=256), nullable=True),
        sa.Column("auth_mode", sa.String(length=16), nullable=False, server_default="soft"),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="active"),
        sa.Column("failed_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("locked_until", sa.DateTime(), nullable=True),
        sa.Column("created_by", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_accounts_username", "accounts", ["username"])
    op.create_index("ix_accounts_created_at", "accounts", ["created_at"])

    op.create_table(
        "sessions",
        sa.Column("token_hash", sa.String(length=64), primary_key=True),
        sa.Column(
            "account_id",
            sa.String(length=64),
            sa.ForeignKey("accounts.id"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("revoked", sa.Boolean(), nullable=False, server_default=sa.text("0")),
    )
    op.create_index("ix_sessions_account_id", "sessions", ["account_id"])

    op.create_table(
        "alerts",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("level", sa.String(length=8), nullable=False, server_default="warn"),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="open"),
        sa.Column("resolved_by", sa.String(length=64), nullable=True),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_alerts_kind", "alerts", ["kind"])
    op.create_index("ix_alerts_status", "alerts", ["status"])
    op.create_index("ix_alerts_created_at", "alerts", ["created_at"])

    op.create_table(
        "security_audit",
        sa.Column("seq", sa.Integer(), primary_key=True, autoincrement=True),
        *_AUDIT_COLS,
    )
    for col in ("actor", "action", "object_type", "object_id", "hash", "created_at"):
        op.create_index(f"ix_security_audit_{col}", "security_audit", [col])


def downgrade() -> None:
    for col in ("created_at", "hash", "object_id", "object_type", "action", "actor"):
        op.drop_index(f"ix_security_audit_{col}", table_name="security_audit")
    op.drop_table("security_audit")
    op.drop_index("ix_alerts_created_at", table_name="alerts")
    op.drop_index("ix_alerts_status", table_name="alerts")
    op.drop_index("ix_alerts_kind", table_name="alerts")
    op.drop_table("alerts")
    op.drop_index("ix_sessions_account_id", table_name="sessions")
    op.drop_table("sessions")
    op.drop_index("ix_accounts_created_at", table_name="accounts")
    op.drop_index("ix_accounts_username", table_name="accounts")
    op.drop_table("accounts")
