"""drop users table

Revision ID: 0004_drop_users
Revises: 0003_audit_batch_disposition
Create Date: 2026-08-28

移除用户/认证系统（单机科研自用，无用户系统）：
- users 表（历史遗留，含密码哈希）随认证功能一并删除。
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004_drop_users"
down_revision: str | None = "0003_audit_batch_disposition"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 用户表数据（含密码哈希）不可恢复，删除前请自行备份。
    op.execute("DROP TABLE IF EXISTS users")


def downgrade() -> None:
    # 降级不恢复 users 表：历史用户数据已删除，无法找回。
    pass
