"""initial schema (§7.1 / §12.2 / §12.5)

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-11

Baseline migration: 创建 images / defects / reports / reviews / audit_log 五表，
与 backend.infra.db.Base.metadata（create_all）完全对齐。后续 schema 演进走
`alembic revision --autogenerate` 生成增量迁移，启动时 `backend.infra.migrate`
统一升级。

注意：Python 侧默认值（utcnow / dict / list / bool 默认）由 ORM 模型在写入时
设置，不在迁移中加 server_default，确保与 create_all 产物一致。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "images",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("path", sa.String(512), nullable=False),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("modality", sa.String(16), nullable=False),
        sa.Column("workpiece_no", sa.String(64), nullable=True),
        sa.Column("weld_no", sa.String(64), nullable=True),
        sa.Column("pixel_spacing_mm", sa.Float(), nullable=True),
        sa.Column("base_metal_thickness_mm", sa.Float(), nullable=True),
        sa.Column("iqi_pass", sa.Boolean(), nullable=True),
        sa.Column("iqi_detail", sa.JSON(), nullable=True),
        sa.Column("density", sa.Float(), nullable=True),
        sa.Column("density_ok", sa.Boolean(), nullable=True),
        sa.Column("pseudo_defect_pass", sa.Boolean(), nullable=True),
        sa.Column("pseudo_defect_notes", sa.JSON(), nullable=True),
        sa.Column("quality_pass", sa.Boolean(), nullable=True),
        sa.Column("quality_metrics", sa.JSON(), nullable=True),
        sa.Column("evaluable", sa.Boolean(), nullable=False),
        sa.Column("preprocess_params", sa.JSON(), nullable=False),
        sa.Column("joint_level", sa.String(8), nullable=True),
        sa.Column("need_review", sa.Boolean(), nullable=False),
        sa.Column("standard_id", sa.String(64), nullable=True),
        sa.Column("standard_version", sa.String(16), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Index("ix_images_joint_level", "joint_level"),
        sa.Index("ix_images_created_at", "created_at"),
    )
    op.create_table(
        "defects",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("image_id", sa.String(64), sa.ForeignKey("images.id"), nullable=False),
        sa.Column("class_id", sa.Integer(), nullable=False),
        sa.Column("bbox_px", sa.JSON(), nullable=True),
        sa.Column("shape", sa.String(16), nullable=True),
        sa.Column("length_mm", sa.Float(), nullable=True),
        sa.Column("width_mm", sa.Float(), nullable=True),
        sa.Column("area_mm2", sa.Float(), nullable=True),
        sa.Column("perimeter_mm", sa.Float(), nullable=True),
        sa.Column("position_x", sa.Float(), nullable=True),
        sa.Column("position_y", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("uncertainty", sa.Float(), nullable=False),
        sa.Column("joint_level", sa.String(8), nullable=True),
        sa.Column("need_review", sa.Boolean(), nullable=False),
        sa.Column("reviewed_by", sa.String(64), nullable=True),
        sa.Column("standard_id", sa.String(64), nullable=True),
        sa.Column("standard_version", sa.String(16), nullable=True),
        sa.Index("ix_defects_image_id", "image_id"),
        sa.Index("ix_defects_joint_level", "joint_level"),
    )
    op.create_table(
        "reports",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("image_id", sa.String(64), sa.ForeignKey("images.id"), nullable=False),
        sa.Column("joint_level", sa.String(8), nullable=True),
        sa.Column("generated_at", sa.DateTime(), nullable=False),
        sa.Column("pdf_path", sa.String(512), nullable=False),
        sa.Column("standard_ref", sa.String(128), nullable=True),
        sa.Column("signer", sa.String(64), nullable=True),
        sa.Column("basis", sa.JSON(), nullable=False),
        sa.Index("ix_reports_image_id", "image_id"),
    )
    op.create_table(
        "reviews",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("image_id", sa.String(64), sa.ForeignKey("images.id"), nullable=False),
        sa.Column("reviewer", sa.String(64), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("overall_level", sa.String(8), nullable=True),
        sa.Column("kappa", sa.Float(), nullable=False),
        sa.Column("consensus", sa.Boolean(), nullable=False),
        sa.Column("needs_arbitration", sa.Boolean(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Index("ix_reviews_image_id", "image_id"),
        sa.Index("ix_reviews_created_at", "created_at"),
    )
    op.create_table(
        "audit_log",
        sa.Column("seq", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("actor", sa.String(64), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("object_type", sa.String(32), nullable=False),
        sa.Column("object_id", sa.String(64), nullable=False),
        sa.Column("before", sa.JSON(), nullable=True),
        sa.Column("after", sa.JSON(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("prev_hash", sa.String(64), nullable=False),
        sa.Column("hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("seq"),
        sa.Index("ix_audit_log_actor", "actor"),
        sa.Index("ix_audit_log_action", "action"),
        sa.Index("ix_audit_log_object_type", "object_type"),
        sa.Index("ix_audit_log_object_id", "object_id"),
        sa.Index("ix_audit_log_hash", "hash", unique=True),
        sa.Index("ix_audit_log_created_at", "created_at"),
    )


def downgrade() -> None:
    op.drop_table("audit_log")
    op.drop_table("reviews")
    op.drop_table("reports")
    op.drop_table("defects")
    op.drop_table("images")
