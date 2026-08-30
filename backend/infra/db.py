"""SQLite 引擎/会话与 ORM 模型（SQLAlchemy 2.0， / ）。

 补齐 images/defects/reports 三表（逻辑模型见设计文档 ）：
- images 增加 joint_level/need_review 冗余列（按级别检索，来源=判定结果快照）；
- images 增加 iqi_detail/density_ok 快照列（报告"IQI 与黑度校验结论"章节数据源）；
- reports 增加 basis(JSON) 列（报告"判定依据条款"章节快照）。
 新增 reviews（人工复核提交）/ audit_log（不可变审计日志，哈希链）：
- reviews 支撑  双人复核/仲裁状态机与 κ 一致性记录；
- audit_log 支撑  合规追溯（谁/何时/对何对象/前后值/哈希链）。
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
    """一次检查影像记录。"""

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
    batch_no: Mapped[str | None] = mapped_column(
        String(64), default=None, index=True
    )  # 批量追溯（P1-F）：所属批次号（batch 导入/复评批次）
    # 密级标识（C-10）：0=非密 1=内部 2=秘密 3=机密；由安全保密管理员设定/变更
    secret_level: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    # 定密依据（C-10）：变更密级时必须登记的依据（文件/条款号）
    classification_basis: Mapped[str | None] = mapped_column(String(256), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class DefectRecord(Base):
    """缺陷检测/量化/评级明细。"""

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
    disposition: Mapped[str | None] = mapped_column(
        String(16), default=None
    )  # 处置建议（P0-E/P1-F）：accept | conditional | rework | recheck（机器可读）
    source: Mapped[str | None] = mapped_column(
        String(16), default=None
    )  # 来源（0005/DB50/T 1807 §6.1.4）：auto=检测器 | manual=人工复核添加
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime, default=None
    )  # 软删除时间（复核删除不物理清除，供审计追溯）


class ReportRecord(Base):
    """报告产出记录。"""

    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id"), index=True)
    joint_level: Mapped[str | None] = mapped_column(String(8), default=None)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    pdf_path: Mapped[str] = mapped_column(String(512))
    standard_ref: Mapped[str | None] = mapped_column(String(128), default=None)
    signer: Mapped[str | None] = mapped_column(String(64), default=None)
    basis: Mapped[list] = mapped_column(JSON, default=list)  # 判定依据条款快照
    # 数字签名：报告内容指纹（SHA-256）+ 签发时间；POST /report/{id}/verify 校验。
    report_hash: Mapped[str | None] = mapped_column(String(64), default=None)
    signed_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    # 密级标识（C-10）：生成报告时从影像快照带入，PDF 页眉/页脚嵌入
    secret_level: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    classification_basis: Mapped[str | None] = mapped_column(String(256), default=None)


class DeviceRecord(Base):
    """检测设备档案。"""

    __tablename__ = "devices"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))  # 设备名/编号，如 CR-01
    model: Mapped[str | None] = mapped_column(String(128), default=None)
    serial_no: Mapped[str | None] = mapped_column(String(128), default=None)
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    created_by: Mapped[str | None] = mapped_column(String(64), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class CalibrationRecord(Base):
    """一次设备标定记录。

    标定时比对实测像素标定与标定件参考值：相对偏差 ≤5% → status=ok，
    超差 → status=over（跨设备一致率 ≤5% 的量化门槛）。
    """

    __tablename__ = "calibrations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    device_id: Mapped[str] = mapped_column(ForeignKey("devices.id"), index=True)
    calibrator: Mapped[str] = mapped_column(String(64))  # 标定员
    pixel_spacing_mm: Mapped[float] = mapped_column(Float)  # 实测像素标定（mm/px）
    ref_pixel_spacing_mm: Mapped[float | None] = mapped_column(Float, default=None)  # 标定件参考值
    deviation_pct: Mapped[float | None] = mapped_column(Float, default=None)  # 相对偏差 %
    status: Mapped[str] = mapped_column(String(8), default="ok")  # ok | over
    density_ref: Mapped[float | None] = mapped_column(Float, default=None)  # 黑度校验值（可选）
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    calibrated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class ReviewRecord(Base):
    """人工复核提交。

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
    """不可变审计日志。

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


class AccountRecord(Base):
    """三员账号（C-06/C-07）：一人一岗，一个账号只绑定一个角色。

    role: sysadmin(系统管理员) | secadmin(安全保密管理员) | auditor(安全审计员)；
    认证方式：SM2 挑战-响应（软件模式公钥由管理员登记；auth_mode=ukey 时走
    Pkcs11Provider 硬件签名，未对接真机前配置即报错）。无口令——私钥不出载体。
    """

    __tablename__ = "accounts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    role: Mapped[str] = mapped_column(String(16))  # sysadmin | secadmin | auditor
    sm2_public_key: Mapped[str | None] = mapped_column(
        String(256), default=None
    )  # 128 hex（x||y），登录验签用
    auth_mode: Mapped[str] = mapped_column(String(16), default="soft")  # soft | ukey
    status: Mapped[str] = mapped_column(String(16), default="active")  # active | disabled
    failed_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    created_by: Mapped[str | None] = mapped_column(String(64), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class SessionRecord(Base):
    """登录会话：库中只存 token 的 SM3 哈希（明文 token 仅在签发时返回一次）。"""

    __tablename__ = "sessions"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("accounts.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    revoked: Mapped[bool] = mapped_column(default=False)


class AlertRecord(Base):
    """安全告警（账号锁定、越权尝试等；C-19 波次3扩展）。"""

    __tablename__ = "alerts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)  # account_locked | ...
    level: Mapped[str] = mapped_column(String(8), default="warn")  # info | warn | critical
    message: Mapped[str] = mapped_column(Text)
    detail: Mapped[dict | None] = mapped_column(JSON, default=None)
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)  # open | resolved
    resolved_by: Mapped[str | None] = mapped_column(String(64), default=None)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    note: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class SecurityAuditRecord(Base):
    """独立安全审计链（C-19 双链）：结构同 audit_log，独立 SM3 哈希链。

    记录管理员/保密员关键操作（账号增删、密级变更、授权、导出审批）与审计员
    自身操作；与主审计链相互独立，防单链被整体覆盖后无迹可查。
    """

    __tablename__ = "security_audit"

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


class CarrierRecord(Base):
    """涉密载体台账（C-12）：底片/报告/备份的登记、借还与销毁全生命周期。"""

    __tablename__ = "carriers"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # 载体编号（登记时给定）
    kind: Mapped[str] = mapped_column(String(16))  # film(底片) | report(报告) | backup(备份)
    object_id: Mapped[str | None] = mapped_column(
        String(64), default=None, index=True
    )  # 关联对象（image_id/report_id/备份归档名）
    secret_level: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    owner: Mapped[str | None] = mapped_column(String(64), default=None)  # 责任人
    status: Mapped[str] = mapped_column(
        String(16), default="in_stock", index=True
    )  # in_stock | borrowed | returned | pending_destroy | destroyed
    borrow_history: Mapped[list] = mapped_column(
        JSON, default=list
    )  # [{action, operator, at, note}]
    destroy_method: Mapped[str | None] = mapped_column(String(64), default=None)
    destroy_note: Mapped[str | None] = mapped_column(Text, default=None)
    destroy_requested_by: Mapped[str | None] = mapped_column(String(64), default=None)  # 保密员
    destroy_confirmed_by: Mapped[str | None] = mapped_column(String(64), default=None)  # 系统管理员
    destroyed_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class ExportRequestRecord(Base):
    """导出审批（C-14）：申请 → 保密员批准/拒绝 → 一次性令牌 → 凭令下载。"""

    __tablename__ = "export_requests"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    subject: Mapped[str] = mapped_column(String(128), index=True)  # report:{id} | std_eval:false_reports ...
    reason: Mapped[str | None] = mapped_column(Text, default=None)
    requested_by: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        String(16), default="pending", index=True
    )  # pending | approved | rejected | consumed
    decided_by: Mapped[str | None] = mapped_column(String(64), default=None)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    token_hash: Mapped[str | None] = mapped_column(String(64), default=None)  # 一次性令牌 SM3
    token_expires_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    used_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


def create_db_engine(path_or_url: str) -> Engine:
    """数据库引擎（S-03 方言可插拔）。

    入参二选一：
    - 文件路径（默认，语义不变）→ ``sqlite:///<path>``，自动建父目录，
      开启外键校验/WAL/锁等待超时；
    - 完整 SQLAlchemy URL（含 ``://``，来自 paths.db_url）→ 按方言直接建引擎，
      不施加 SQLite PRAGMA。

    国产数据库方言示例（S-03，诚实声明：本仓库未做真机验证，仅给出接入写法，
    连通性/迁移行为须在目标环境联调确认）：

    - 达梦 DM8（需 ``pip install sqlalchemy_dm``，或厂商 dmPython + 方言包）::

        paths:
          db_url: dm+sqlalchemy_dm://SYSDBA:SYSDBA@127.0.0.1:5236?schema=SCANDetection

    - 人大金仓 KingbaseES（兼容 PostgreSQL 协议，可用 pg8000/psycopg2 方言）::

        paths:
          db_url: postgresql+psycopg2://system:manager@127.0.0.1:54321/scandetection

    切换方言后 schema 迁移（alembic）与 JSON 列类型兼容性须另行验证，
    见 docs/国产化适配矩阵.md（DB 维度，未真机验证项）。
    """
    from sqlalchemy import event

    if "://" in path_or_url:
        # 方言 URL：直接透传 SQLAlchemy（连接参数由 URL/方言自行承载）。
        return create_engine(path_or_url, future=True)

    p = Path(path_or_url)
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
