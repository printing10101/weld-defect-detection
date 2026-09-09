"""底片印字识别快照列（扫描日期/编号，正/镜像 + 缺印字复核标记）

Revision ID: 0011_film_stamp
Revises: 0010_image_content_hash
Create Date:

- images.stamp_status: present | missing | unavailable | off（识别结论）
- images.stamp_text / stamp_orientation / stamp_confidence: 印字内容快照
- images.stamp_need_review: 缺印字复核标记（单图即时判定；批量场景延迟到
  批次收尾按印字占比裁决后回填，豁免时保持 False）
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011_film_stamp"
down_revision: str | None = "0010_image_content_hash"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("images", sa.Column("stamp_status", sa.String(length=16), nullable=True))
    op.add_column("images", sa.Column("stamp_text", sa.String(length=128), nullable=True))
    op.add_column("images", sa.Column("stamp_orientation", sa.String(length=8), nullable=True))
    op.add_column("images", sa.Column("stamp_confidence", sa.Float(), nullable=True))
    op.add_column(
        "images",
        sa.Column("stamp_need_review", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("images", "stamp_need_review")
    op.drop_column("images", "stamp_confidence")
    op.drop_column("images", "stamp_orientation")
    op.drop_column("images", "stamp_text")
    op.drop_column("images", "stamp_status")
