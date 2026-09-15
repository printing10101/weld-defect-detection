"""独立片号列（G05：数据标准化——片号实体化）

Revision ID: 0014_film_no
Revises: 0013_defect_atlas
Create Date:

- images.film_no: 片号（底片自身编号，来自印字 OCR 结构化抽取，见
  backend/domain/stamp.py extract_film_no）。此前片号靠报告侧从 image_id
  尾号推导，与底片实际印字无关联；独立列使按片号检索/报告直印成为可能。
  缺省 NULL（印字未识别到片号/历史影像），报告回退影像短号口径。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0014_film_no"
down_revision: str | None = "0013_defect_atlas"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("images", sa.Column("film_no", sa.String(length=64), nullable=True))


def downgrade() -> None:
    op.drop_column("images", "film_no")
