"""影像内容哈希列：批量上传查重（与历史已检影像比对）

Revision ID: 0010_image_content_hash
Revises: 0009_carriers_export
Create Date:

- images.content_hash: 上传/评片时对原文件流式计算的 SHA256（64 hex），
  建索引供批量查重的 IN 批量比对；历史数据该列为 NULL（无法回溯文件摘要，
  查重只对非 NULL 哈希生效）。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010_image_content_hash"
down_revision: str | None = "0009_carriers_export"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("images", sa.Column("content_hash", sa.String(length=64), nullable=True))
    op.create_index("ix_images_content_hash", "images", ["content_hash"])


def downgrade() -> None:
    op.drop_index("ix_images_content_hash", table_name="images")
    op.drop_column("images", "content_hash")
