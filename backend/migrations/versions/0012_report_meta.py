"""报告补充信息列（委托单位/工程名称/工艺参数等样张汇总表字段）

Revision ID: 0012_report_meta
Revises: 0011_film_stamp
Create Date:

- images.report_meta: JSON（键白名单见 backend/domain/report/meta_fields.py），
  前端评片表单按《射线检测报告》样张分组录入，出片时填入汇总表对应空格；
  缺省为 NULL（历史影像与新表单均允许不填，汇总表相应栏留空待手工补填）。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012_report_meta"
down_revision: str | None = "0011_film_stamp"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("images", sa.Column("report_meta", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("images", "report_meta")
