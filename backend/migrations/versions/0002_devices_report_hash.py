"""devices/calibrations + report 数字签名字段

Revision ID: 0002_devices_report_hash
Revises: 0001_initial
Create Date:

增量迁移：
- 新增 devices（设备档案）与 calibrations（标定记录，跨设备一致率 ≤5% 校验）；
- reports 增加 report_hash（内容指纹 SHA-256）与 signed_at（签发时间），
  支撑  报告数字签名与 POST /report/{id}/verify 防篡改校验。

与 backend.infra.db.Base.metadata（create_all）对齐；默认值由 ORM 写入时设置，
不在迁移中加 server_default（与 0001 约定一致）。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_devices_report_hash"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "devices",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("serial_no", sa.String(length=128), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_devices_created_at"), "devices", ["created_at"], unique=False)
    op.create_table(
        "calibrations",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("device_id", sa.String(length=64), nullable=False),
        sa.Column("calibrator", sa.String(length=64), nullable=False),
        sa.Column("pixel_spacing_mm", sa.Float(), nullable=False),
        sa.Column("ref_pixel_spacing_mm", sa.Float(), nullable=True),
        sa.Column("deviation_pct", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=8), nullable=False),
        sa.Column("density_ref", sa.Float(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("calibrated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["device_id"], ["devices.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_calibrations_device_id"), "calibrations", ["device_id"], unique=False)
    op.create_index(
        op.f("ix_calibrations_calibrated_at"), "calibrations", ["calibrated_at"], unique=False
    )
    op.add_column("reports", sa.Column("report_hash", sa.String(length=64), nullable=True))
    op.add_column("reports", sa.Column("signed_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column("reports", "signed_at")
    op.drop_column("reports", "report_hash")
    op.drop_index(op.f("ix_calibrations_calibrated_at"), table_name="calibrations")
    op.drop_index(op.f("ix_calibrations_device_id"), table_name="calibrations")
    op.drop_table("calibrations")
    op.drop_index(op.f("ix_devices_created_at"), table_name="devices")
    op.drop_table("devices")
