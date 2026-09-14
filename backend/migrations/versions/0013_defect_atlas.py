"""defect_atlas（缺陷图谱样本库）

Revision ID: 0013_defect_atlas
Revises: 0012_report_meta
Create Date:

人工筛选沉淀的典型缺陷样本库（培训/比对/复核参考）：
一行对应一个源缺陷（defect_id 唯一），crop 为缺陷局部图路径。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013_defect_atlas"
down_revision: str | None = "0012_report_meta"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "defect_atlas",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("defect_id", sa.String(length=64), nullable=False),
        sa.Column("image_id", sa.String(length=64), nullable=False),
        sa.Column("class_id", sa.Integer(), nullable=False),
        sa.Column("joint_level", sa.String(length=8), nullable=True),
        sa.Column("bbox_px", sa.JSON(), nullable=True),
        sa.Column("length_mm", sa.Float(), nullable=True),
        sa.Column("width_mm", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=True),
        sa.Column("reviewed_by", sa.String(length=64), nullable=True),
        sa.Column("workpiece_no", sa.String(length=64), nullable=True),
        sa.Column("weld_no", sa.String(length=64), nullable=True),
        sa.Column("standard_id", sa.String(length=64), nullable=True),
        sa.Column("crop_path", sa.String(length=512), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("defect_id"),
    )
    op.create_index("ix_defect_atlas_defect_id", "defect_atlas", ["defect_id"])
    op.create_index("ix_defect_atlas_image_id", "defect_atlas", ["image_id"])
    op.create_index("ix_defect_atlas_class_id", "defect_atlas", ["class_id"])
    op.create_index("ix_defect_atlas_joint_level", "defect_atlas", ["joint_level"])
    op.create_index("ix_defect_atlas_created_at", "defect_atlas", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_defect_atlas_created_at", table_name="defect_atlas")
    op.drop_index("ix_defect_atlas_joint_level", table_name="defect_atlas")
    op.drop_index("ix_defect_atlas_class_id", table_name="defect_atlas")
    op.drop_index("ix_defect_atlas_image_id", table_name="defect_atlas")
    op.drop_index("ix_defect_atlas_defect_id", table_name="defect_atlas")
    op.drop_table("defect_atlas")
