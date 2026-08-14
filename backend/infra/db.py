"""SQLite 引擎/会话与 ORM 模型（SQLAlchemy 2.0，§7.1 / §T4）。

M6 补齐 images/defects/reports 三表（逻辑模型见规格书 §7.1）：
- images 增加 joint_level/need_review 冗余列（按级别检索，来源=判定结果快照）；
- images 增加 iqi_detail/density_ok 快照列（报告"IQI 与黑度校验结论"章节数据源）；
- reports 增加 basis(JSON) 列（报告"判定依据条款"章节快照）。
M7 新增 reviews（人工复核提交）/ audit_log（不可变审计日志，哈希链）：
- reviews 支撑 §12.2 双人复核/仲裁状态机与 κ 一致性记录；
- audit_log 支撑 §12.5 合规追溯（谁/何时/对何对象/前后值/哈希链）。
v3 换 PostgreSQL 时同 schema 迁移，SQLAlchemy 屏蔽差异。
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    """UTC 无时区 datetime（SQLite 存储友好）。"""
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class ImageRecord(Base):
    """一次检查影像记录（§7.1 images）。"""

    __tablename__ = "images"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    path: Mapped[str] = mapped_column(String(512))  # 原图副本（报告缺陷图谱数据源）
    source_type: Mapped[str] = mapped_column(String(32))  # dicom | image
    modality: Mapped[str] = mapped_column(String(16))  # CR | DR | DICOM | GENERIC
    workpiece_no: Mapped[str | None] = mapped_column(String(64), default=None)
    weld_no: Mapped[str | None] = mapped_column(String(64), default=None)
    pixel_spacing_mm: Mapped[float | None] = mapped_column(Float, default=None)
    base_metal_thickness_mm: Mapped[float | None] = mapped_column(Float, default=None)
    iqi_pass: Mapped[bool | None] = mapped_column(default=None)
    iqi_detail: Mapped[dict | None] = mapped_column(
        JSON, default=None
    )  # {type, achieved, required}
    density: Mapped[float | None] = mapped_column(Float, default=None)
    density_ok: Mapped[bool | None] = mapped_column(default=None)
    pseudo_defect_pass: Mapped[bool | None] = mapped_column(default=None)
    pseudo_defect_notes: Mapped[list | None] = mapped_column(
        JSON, default=None
    )  # 伪缺陷筛查结论（§4.2：划痕/尘点/显影不均，notes=告警项）
    quality_pass: Mapped[bool | None] = mapped_column(default=None)
    quality_metrics: Mapped[dict | None] = mapped_column(
        JSON, default=None
    )  # 质量门禁结论（§4.4：RQI 分数/子分/BRISQUE 特征）
    evaluable: Mapped[bool] = mapped_column(default=True)
    preprocess_params: Mapped[dict] = mapped_column(JSON, default=dict)
    joint_level: Mapped[str | None] = mapped_column(
        String(8), default=None, index=True
    )  # 冗余（判定结果）
    need_review: Mapped[bool] = mapped_column(default=False)
    standard_id: Mapped[str | None] = mapped_column(String(64), default=None)
    standard_version: Mapped[str | None] = mapped_column(String(16), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class DefectRecord(Base):
    """缺陷检测/量化/评级明细（§7.1 defects）。"""

    __tablename__ = "defects"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id"), index=True)
    class_id: Mapped[int] = mapped_column(Integer)
    bbox_px: Mapped[list | None] = mapped_column(
        JSON, default=None
    )  # [x,y,w,h] 像素（报告图谱画框）
    shape: Mapped[str | None] = mapped_column(String(16), default=None)  # round | linear
    length_mm: Mapped[float | None] = mapped_column(Float, default=None)
    width_mm: Mapped[float | None] = mapped_column(Float, default=None)
    area_mm2: Mapped[float | None] = mapped_column(Float, default=None)
    perimeter_mm: Mapped[float | None] = mapped_column(Float, default=None)
    position_x: Mapped[float | None] = mapped_column(Float, default=None)
    position_y: Mapped[float | None] = mapped_column(Float, default=None)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    uncertainty: Mapped[float] = mapped_column(Float, default=1.0)
    joint_level: Mapped[str | None] = mapped_column(String(8), default=None, index=True)
    need_review: Mapped[bool] = mapped_column(default=False)
    reviewed_by: Mapped[str | None] = mapped_column(String(64), default=None)
    standard_id: Mapped[str | None] = mapped_column(String(64), default=None)
    standard_version: Mapped[str | None] = mapped_column(String(16), default=None)


class ReportRecord(Base):
    """报告产出记录（§7.1 reports）。"""

    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id"), index=True)
    joint_level: Mapped[str | None] = mapped_column(String(8), default=None)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    pdf_path: Mapped[str] = mapped_column(String(512))
    standard_ref: Mapped[str | None] = mapped_column(String(128), default=None)
    signer: Mapped[str | None] = mapped_column(String(64), default=None)
    basis: Mapped[list] = mapped_column(JSON, default=list)  # 判定依据条款快照


class ReviewRecord(Base):
    """人工复核提交（§12.2 双人复核/仲裁状态机）。

    每次复核（初评/复评/仲裁）写入一行；κ 为本次提交与系统自动评级的一致性
    （Cohen's κ）。consensus/needs_arbitration 记录本次复核的判定走向。
    """

    __tablename__ = "reviews"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id"), index=True)
    reviewer: Mapped[str] = mapped_column(String(64))  # 评片员姓名/工号
    role: Mapped[str] = mapped_column(String(16))  # initial | secondary | arbitrator
    overall_level: Mapped[str | None] = mapped_column(String(8), default=None)  # 复核综合级别
    kappa: Mapped[float] = mapped_column(Float, default=1.0)  # 与自动评级一致性
    consensus: Mapped[bool] = mapped_column(default=False)  # 是否达成共识
    needs_arbitration: Mapped[bool] = mapped_column(default=False)  # 是否升级仲裁
    note: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class AuditRecord(Base):
    """不可变审计日志（§12.5，哈希链 append-only）。

    谁(actor)/何时(created_at)/对何对象(object_type+object_id)做了何操作(action)，
    以及前后值(before/after)；hash = sha256(prev_hash || payload)，prev_hash 指向上一条，
    形成防篡改链。写入后不更新、不删除。
    """

    __tablename__ = "audit_log"

    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String(64), index=True)
    action: Mapped[str] = mapped_column(String(32), index=True)
    object_type: Mapped[str] = mapped_column(String(32), index=True)
    object_id: Mapped[str] = mapped_column(String(64), index=True)
    before: Mapped[dict | None] = mapped_column(JSON, default=None)
    after: Mapped[dict | None] = mapped_column(JSON, default=None)
    note: Mapped[str | None] = mapped_column(Text, default=None)
    prev_hash: Mapped[str] = mapped_column(String(64), default="0" * 64)
    hash: Mapped[str] = mapped_column(String(64), index=True, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class UserRecord(Base):
    """系统用户（§T3，P0 用户权限与登录）。

    role ∈ {reviewer 评片员, auditor 审核员, admin 管理员}（RBAC，见 backend/app/auth.py）。
    password_hash 为 PBKDF2-HMAC-SHA256 派生串（algo$iters$salt$hash），明文永不入库。
    disabled=True 时禁止登录（离职/停用，但历史审计/签名留痕保留）。
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(128), default=None)
    role: Mapped[str] = mapped_column(String(16), default="reviewer")
    password_hash: Mapped[str] = mapped_column(String(256))
    disabled: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    created_by: Mapped[str | None] = mapped_column(String(64), default=None)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)


def create_db_engine(path: str) -> Engine:
    """SQLite 引擎（§T4；v3 换 PostgreSQL 仅改此函数）。

    加固：父目录不存在时自动创建；开启外键校验、WAL 写模式与锁等待超时，
    避免并发评片/复核/审计落库时出现 database is locked，并让外键约束真正生效。
    """
    from sqlalchemy import event

    p = Path(path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{p}",
        future=True,
        connect_args={"timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _rec):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    return engine
