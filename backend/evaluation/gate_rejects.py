"""不合格底片留档台账（DB50/T 1807-2025 §5 评片硬前置）。

门禁（黑度/IQI/伪缺陷/质量/扫描参数）拦截的底片此前直接抛 IQIFailError、
早于原图落盘，导致"拦截了什么、为何拦截"无据可查。本模块补齐留档：

- 影像副本由 pipelines._persist_reject 按 security.encrypt 密文归档到
  gate.rejects_dir（复用影像加密落盘路径模式），本模块只管台账；
- gate_rejects 表记录拒绝原因与 dpi/位深/操作员快照；
- 审计动作 gate_reject 由 pipelines 经仓储 append_audit 写哈希链，不在本模块。

ORM 模型挂在 infra.db.Base 上（模型即 schema 真源），schema 演进由
migrations/versions/0006_gate_rejects.py 管理；store 自建引擎时幂等 create_all
兜底（与 InspectionRepository 同模式）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String, func, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from backend.infra.db import Base, create_db_engine, utcnow


class GateRejectRecord(Base):
    """一次评片门禁拦截的留档行（影像行未落库，image_id 仅作关联参考）。"""

    __tablename__ = "gate_rejects"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    # 可空：拦截发生在 images 行写入之前，多数拦截没有正式影像 id
    image_id: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    reject_reason: Mapped[str] = mapped_column(String(256))
    detail: Mapped[dict] = mapped_column(JSON, default=dict)  # 原因清单/归档路径/影像来源
    dpi: Mapped[float | None] = mapped_column(Float, default=None)  # 无法确定时为 NULL
    bit_depth: Mapped[int | None] = mapped_column(Integer, default=None)
    operator: Mapped[str | None] = mapped_column(String(64), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class GateRejectStore:
    """gate_rejects 台账读写（与 InspectionRepository 同 Session 模式）。"""

    def __init__(self, db_path: str) -> None:
        self._engine = create_db_engine(db_path)
        # 幂等兜底：遗留 create_all 库经 ensure_migrations stamp head 不会执行
        # 0006 的 DDL，靠这里把缺表补齐（与仓储层兜底语义一致）。
        Base.metadata.create_all(self._engine)

    def add(
        self,
        *,
        reject_id: str,
        reject_reason: str,
        detail: dict[str, Any],
        image_id: str | None = None,
        dpi: float | None = None,
        bit_depth: int | None = None,
        operator: str | None = None,
    ) -> dict[str, Any]:
        """写入一条拦截留档，返回可 JSON 化 dict。"""
        with Session(self._engine) as session, session.begin():
            rec = GateRejectRecord(
                id=reject_id,
                image_id=image_id,
                reject_reason=reject_reason,
                detail=detail,
                dpi=dpi,
                bit_depth=bit_depth,
                operator=operator,
            )
            session.add(rec)
            session.flush()
            return _to_dict(rec)

    def list(self, *, limit: int = 50, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        """按时间降序检索拦截留档，返回 (当页条目, 匹配总数)。"""
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        with Session(self._engine) as session:
            total = int(session.scalar(select(func.count()).select_from(GateRejectRecord)) or 0)
            rows = list(
                session.scalars(
                    select(GateRejectRecord)
                    .order_by(GateRejectRecord.created_at.desc())
                    .offset(offset)
                    .limit(limit)
                )
            )
            return [_to_dict(r) for r in rows], total


def _to_dict(rec: GateRejectRecord) -> dict[str, Any]:
    return {
        "id": rec.id,
        "image_id": rec.image_id,
        "reject_reason": rec.reject_reason,
        "detail": rec.detail,
        "dpi": rec.dpi,
        "bit_depth": rec.bit_depth,
        "operator": rec.operator,
        "created_at": rec.created_at.isoformat(timespec="seconds") if rec.created_at else None,
    }
