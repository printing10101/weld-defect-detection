"""检查记录仓储（§7.1 / §7.3 / §12.2 / §12.5）。

提供 images/defects/reports 的写入、检索与统计能力（纯存储，无业务判定逻辑）；
返回值均为 dict（JSON 可序列化），禁止 ORM 对象跨层传递（§19.1）。
M7 扩展：apply_review（复核落库）/ list_reviews / append_audit（哈希链）/ list_audit。
千级记录检索目标 < 1s：created_at/joint_level 索引 + 单查询聚合。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.domain.dto import DefectClass
from backend.infra.db import (
    AuditRecord,
    Base,
    DefectRecord,
    ImageRecord,
    ReportRecord,
    ReviewRecord,
    create_db_engine,
)

_LOG = logging.getLogger("scandetection.repository")

_PAGE_MAX = 100
_LEVELS = ("I", "II", "III", "IV")


class InspectionRepository:
    """SQLite 检查记录仓储。"""

    def __init__(self, db_path: str) -> None:
        self._engine = create_db_engine(db_path)
        self._audit_lock = threading.Lock()  # 审计哈希链写串行化（防分叉，单进程内）
        Base.metadata.create_all(self._engine)

    # ---- 写入 ----
    def create_inspection(
        self,
        image: dict[str, Any],
        defects: list[dict[str, Any]],
        report: dict[str, Any] | None = None,
    ) -> str:
        """写入一次检查（影像+缺陷+可选报告），事务保证一致性，返回 image_id。

        白名单校验外部 dict，避免多余键/缺键导致的 TypeError/KeyError；
        IntegrityError 转译为 ValueError，防止 SQL 细节外泄给客户端。

        写入顺序：**必须先 flush images 行**再写 defects/reports。ORM 未声明
        relationship() 时，SQLAlchemy 跨 mapper 的 INSERT 顺序按 mapper 排序键
        （模块名.类名）决定，DefectRecord 字母序早于 ImageRecord，会先插子表；
        在 `PRAGMA foreign_keys=ON` 下直接触发 FOREIGN KEY constraint failed。
        """
        from sqlalchemy.exc import IntegrityError

        image_fields = {c.name for c in ImageRecord.__table__.columns}
        defect_fields = {c.name for c in DefectRecord.__table__.columns}
        report_fields = {c.name for c in ReportRecord.__table__.columns}
        if "id" not in image:
            raise ValueError("image['id'] is required")
        if unknown := set(image) - image_fields:
            raise ValueError(f"unknown image fields: {sorted(unknown)}")
        image_id = str(image["id"])
        for d in defects:
            if unknown := set(d) - defect_fields:
                raise ValueError(f"unknown defect fields: {sorted(unknown)}")
            if d.get("image_id", image_id) != image_id:
                raise ValueError(f"defect image_id mismatch: {d.get('image_id')!r} != {image_id!r}")
        if report is not None:
            if "id" not in report:
                raise ValueError("report['id'] is required")
            if unknown := set(report) - report_fields:
                raise ValueError(f"unknown report fields: {sorted(unknown)}")
            if report.get("image_id", image_id) != image_id:
                raise ValueError(
                    f"report image_id mismatch: {report.get('image_id')!r} != {image_id!r}"
                )
        try:
            with Session(self._engine) as session, session.begin():
                session.add(ImageRecord(**image))
                session.flush()  # 父表先落库，满足 defects/reports 外键
                for d in defects:
                    session.add(DefectRecord(**d))
                if report is not None:
                    session.add(ReportRecord(**report))
                session.flush()
        except IntegrityError as e:
            detail = str(getattr(e, "orig", None) or e)
            if "UNIQUE" in detail.upper() or "PRIMARY KEY" in detail.upper():
                raise ValueError(f"inspection already exists: {image_id}") from e
            # 其余约束错误保留 DBAPI 原因（不含 SQL 语句），否则无从定位
            raise ValueError(f"inspection integrity error [{image_id}]: {detail}") from e
        return image_id

    # ---- 查询 ----
    def get_image(self, image_id: str) -> dict[str, Any] | None:
        """检查详情：影像 + 缺陷列表 + 报告。"""
        with Session(self._engine) as session:
            rec = session.get(ImageRecord, image_id)
            if rec is None:
                return None
            out = self._image_to_dict(rec)
            out["defects"] = [
                self._defect_to_dict(d)
                for d in session.scalars(
                    select(DefectRecord).where(
                        DefectRecord.image_id == image_id,
                        DefectRecord.deleted_at.is_(None),
                    )
                )
            ]
            rep = session.scalars(
                select(ReportRecord).where(ReportRecord.image_id == image_id)
            ).first()
            out["report"] = self._report_to_dict(rep) if rep is not None else None
            return out

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        with Session(self._engine) as session:
            rec = session.get(ReportRecord, report_id)
            return self._report_to_dict(rec) if rec is not None else None

    def update_report(
        self,
        report_id: str,
        pdf_path: str | None = None,
        report_hash: str | None = None,
        signed_at: datetime | None = None,
    ) -> None:
        """回填报告字段（PDF 路径 / 数字指纹 / 签发时间，§7.2）。

        pdf_path 由报告生成后回填；report_hash/signed_at 由 PdfReporter.build
        写入（数字签名）。任一字段为 None 表示不改动对应列。
        """
        with Session(self._engine) as session, session.begin():
            rec = session.get(ReportRecord, report_id)
            if rec is None:
                raise KeyError(f"report not found: {report_id}")
            if pdf_path is not None:
                rec.pdf_path = pdf_path
            if report_hash is not None:
                rec.report_hash = report_hash
            if signed_at is not None:
                rec.signed_at = signed_at

    def list_records(
        self,
        level: str | None = None,
        class_id: int | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        workpiece: str | None = None,
        page: int = 1,
        size: int = 20,
    ) -> tuple[list[dict[str, Any]], int]:
        """多条件检索（§7.3）：级别/缺陷类别/日期范围/工件号，分页。"""
        size = max(1, min(size, _PAGE_MAX))
        page = max(1, page)
        conds: list[Any] = []
        if level is not None:
            if level not in _LEVELS:
                raise ValueError(f"invalid level: {level!r}, expected one of {_LEVELS}")
            conds.append(ImageRecord.joint_level == level)
        if class_id is not None:
            sub = select(DefectRecord.image_id).where(DefectRecord.class_id == class_id)
            conds.append(ImageRecord.id.in_(sub))
        if date_from:
            conds.append(ImageRecord.created_at >= _parse_dt(date_from))
        if date_to:
            conds.append(ImageRecord.created_at < _parse_dt_end(date_to))
        if workpiece:
            # 转义 LIKE 通配符，避免 %/_ 被当作掩码造成误匹配/全表扫描
            esc = workpiece.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            conds.append(ImageRecord.workpiece_no.ilike(f"%{esc}%", escape="\\"))

        with Session(self._engine) as session:
            total = int(
                session.scalar(select(func.count()).select_from(ImageRecord).where(*conds)) or 0
            )
            rows = list(
                session.scalars(
                    select(ImageRecord)
                    .where(*conds)
                    .order_by(ImageRecord.created_at.desc())
                    .offset((page - 1) * size)
                    .limit(size)
                )
            )
            items = [self._image_to_dict(r) for r in rows]
            # 批量附加缺陷计数（一次 IN 查询）
            if items:
                ids = [it["image_id"] for it in items]
                cnt_rows = session.execute(
                    select(DefectRecord.image_id, func.count())
                    .where(DefectRecord.image_id.in_(ids))
                    .group_by(DefectRecord.image_id)
                ).all()
                cnt_map = {rid: c for rid, c in cnt_rows}
                for it in items:
                    it["defect_count"] = cnt_map.get(it["image_id"], 0)
            else:
                for it in items:
                    it["defect_count"] = 0
            return items, total

    def stats(self) -> dict[str, Any]:
        """缺陷统计与分布（§7.3）：总数 / 级别分布 / 缺陷类别分布。"""
        with Session(self._engine) as session:
            total = int(session.scalar(select(func.count()).select_from(ImageRecord)) or 0)
            by_level = {
                lv: n
                for lv, n in session.execute(
                    select(ImageRecord.joint_level, func.count())
                    .where(ImageRecord.joint_level.is_not(None))
                    .group_by(ImageRecord.joint_level)
                ).all()
            }
            by_class = {
                _class_name(cid): n
                for cid, n in session.execute(
                    select(DefectRecord.class_id, func.count()).group_by(DefectRecord.class_id)
                ).all()
            }
            return {"total": total, "by_level": by_level, "by_class": by_class}

    # ---- 人工复核 / 审计（M7，§12.2 / §12.5）----
    def apply_review(
        self,
        *,
        image_id: str,
        reviewer: str,
        role: str,
        final_level: str | None,
        per_defect_level: dict[str, str | None],
        consensus: bool,
        kappa: float,
        needs_arbitration: bool,
        note: str | None,
    ) -> dict[str, Any]:
        """写入一次复核并提交：更新 images/defects/reports 的复核结果。

        - final_level=None（待仲裁）时不改动综合级别与缺陷级别，仅记录分歧；
        - 仅当达成共识/仲裁（final_level 非空）时才把逐缺陷级别覆盖落库；
        - reviewed_by 仅在达成共识/仲裁时写入，标记最终定案人。
        返回本次复核行摘要 + 复核计数。
        """
        with Session(self._engine) as session, session.begin():
            image = session.get(ImageRecord, image_id)
            if image is None:
                raise KeyError(f"image not found: {image_id}")

            # 缺陷级别 + 复核人 + 复核标记（仅定案时覆盖级别）
            defects = list(
                session.scalars(
                    select(DefectRecord).where(
                        DefectRecord.image_id == image_id,
                        DefectRecord.deleted_at.is_(None),
                    )
                )
            )
            finalizing = consensus or role == "arbitrator"
            known_ids = {d.id for d in defects}
            unknown_ids = set(per_defect_level) - known_ids
            if unknown_ids:
                raise KeyError(
                    f"defect ids not belonging to image {image_id}: {sorted(unknown_ids)}"
                )
            for d in defects:
                if finalizing and d.id in per_defect_level:
                    d.joint_level = per_defect_level[d.id]
                if finalizing:
                    d.reviewed_by = reviewer
                    d.need_review = False
                elif d.reviewed_by is None:
                    d.need_review = True

            # 影像综合级别 + 复核标记
            if final_level is not None:
                image.joint_level = final_level
            # 无综合级别且无既有级别时，视为仍需复核（§12.2 状态机兜底，避免无声消失）
            image.need_review = bool(
                needs_arbitration
                or (not consensus and role != "arbitrator")
                or (final_level is None and image.joint_level is None)
            )

            # 报告综合级别同步
            reports = list(
                session.scalars(select(ReportRecord).where(ReportRecord.image_id == image_id))
            )
            for r in reports:
                if final_level is not None:
                    r.joint_level = final_level

            # 复核提交行
            review = ReviewRecord(
                id=uuid.uuid4().hex,
                image_id=image_id,
                reviewer=reviewer,
                role=role,
                overall_level=final_level,
                kappa=kappa,
                consensus=consensus,
                needs_arbitration=needs_arbitration,
                note=note,
            )
            session.add(review)
            # 在会话关闭前捕获属性（commit 后 expire_on_commit 会使对象过期）
            review_id = review.id

            count = int(
                session.scalar(
                    select(func.count())
                    .select_from(ReviewRecord)
                    .where(ReviewRecord.image_id == image_id)
                )
                or 0
            )

        return {
            "review_id": review_id,
            "image_id": image_id,
            "reviewer": reviewer,
            "role": role,
            "joint_level": final_level,
            "consensus": consensus,
            "needs_arbitration": needs_arbitration,
            "kappa": kappa,
            "review_count": count,
        }

    def list_reviews(self, image_id: str) -> list[dict[str, Any]]:
        """某影像的全部复核提交（按时间升序）。"""
        with Session(self._engine) as session:
            rows = list(
                session.scalars(
                    select(ReviewRecord)
                    .where(ReviewRecord.image_id == image_id)
                    .order_by(ReviewRecord.created_at.asc())
                )
            )
            return [self._review_to_dict(r) for r in rows]

    # ---- 人工复核缺陷增删改（DB50/T 1807-2025 §6.1.4，比标准严：全程审计留痕） ----

    def add_manual_defect(
        self,
        *,
        image_id: str,
        class_id: int,
        bbox_px: list[float],
        operator: str,
        reason: str,
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        """人工复核添加缺陷框（来源标记 manual，审计 before=None/after=新行）。"""
        if not reason.strip():
            raise ValueError("reason is required for defect add")
        with Session(self._engine) as session, session.begin():
            if session.get(ImageRecord, image_id) is None:
                raise KeyError(f"image not found: {image_id}")
            rec = DefectRecord(
                id=uuid.uuid4().hex,
                image_id=image_id,
                class_id=int(class_id),
                bbox_px=[float(v) for v in bbox_px],
                confidence=float(confidence),
                uncertainty=0.0,
                need_review=True,
                source="manual",
            )
            session.add(rec)
            session.flush()
            row = self._defect_to_dict(rec)
            defect_id = rec.id  # commit 后对象脱管，主键先取出
        self.append_audit(
            actor=operator,
            action="defect.add",
            object_type="defect",
            object_id=defect_id,
            before=None,
            after=row,
            note=reason,
        )
        return row

    def edit_defect(
        self,
        *,
        defect_id: str,
        operator: str,
        reason: str,
        class_id: int | None = None,
        bbox_px: list[float] | None = None,
    ) -> dict[str, Any]:
        """人工复核修改缺陷类型/位置（软字段级覆盖，审计记录 before/after 快照）。"""
        if not reason.strip():
            raise ValueError("reason is required for defect edit")
        with Session(self._engine) as session, session.begin():
            rec = session.get(DefectRecord, defect_id)
            if rec is None or rec.deleted_at is not None:
                raise KeyError(f"defect not found: {defect_id}")
            before = self._defect_to_dict(rec)
            if class_id is not None:
                rec.class_id = int(class_id)
            if bbox_px is not None:
                rec.bbox_px = [float(v) for v in bbox_px]
            # 人工改动后机器几何/评级失效：标记待复核，重评级由管线层触发
            rec.need_review = True
            session.flush()
            after = self._defect_to_dict(rec)
        self.append_audit(
            actor=operator,
            action="defect.edit",
            object_type="defect",
            object_id=defect_id,
            before=before,
            after=after,
            note=reason,
        )
        return after

    def delete_defect(self, *, defect_id: str, operator: str, reason: str) -> dict[str, Any]:
        """人工复核删除缺陷（软删除，不物理清除，审计留痕）。"""
        if not reason.strip():
            raise ValueError("reason is required for defect delete")
        with Session(self._engine) as session, session.begin():
            rec = session.get(DefectRecord, defect_id)
            if rec is None or rec.deleted_at is not None:
                raise KeyError(f"defect not found: {defect_id}")
            before = self._defect_to_dict(rec)
            rec.deleted_at = datetime.now(UTC).replace(tzinfo=None)
            deleted_str = _fmt_dt(rec.deleted_at)  # 会话关闭前捕获，避免脱管刷新
            session.flush()
        self.append_audit(
            actor=operator,
            action="defect.delete",
            object_type="defect",
            object_id=defect_id,
            before=before,
            after={**before, "deleted_at": deleted_str},
            note=reason,
        )
        return {**before, "deleted": True}

    def store_regrade(
        self,
        image_id: str,
        *,
        joint_level: str | None,
        per_defect_levels: dict[str, str],
        need_review: bool,
        geometry: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """缺陷增删改后回写机器重评级结果（几何可选更新，评级列全量覆盖）。

        geometry: {defect_id: {shape,length_mm,width_mm,area_mm2,perimeter_mm,position_x,position_y}}
        """
        with Session(self._engine) as session, session.begin():
            image = session.get(ImageRecord, image_id)
            if image is None:
                raise KeyError(f"image not found: {image_id}")
            image.joint_level = joint_level
            image.need_review = bool(need_review)
            defects = list(
                session.scalars(
                    select(DefectRecord).where(
                        DefectRecord.image_id == image_id,
                        DefectRecord.deleted_at.is_(None),
                    )
                )
            )
            known = set(per_defect_levels) | set(geometry or {})
            unknown = {d.id for d in defects} - known
            if unknown:
                raise KeyError(f"defect ids not belonging to image {image_id}: {sorted(unknown)}")
            for d in defects:
                if d.id in per_defect_levels:
                    d.joint_level = per_defect_levels[d.id]
                g = (geometry or {}).get(d.id)
                if g:
                    for k in ("shape", "length_mm", "width_mm", "area_mm2", "perimeter_mm", "position_x", "position_y"):
                        if k in g:
                            setattr(d, k, g[k])

    def append_audit(
        self,
        *,
        actor: str,
        action: str,
        object_type: str,
        object_id: str,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        note: str | None = None,
    ) -> dict[str, Any]:
        """追加一条不可变审计日志（哈希链）。返回所写行。

        防分叉：单进程内用 self._audit_lock 串行化读-改-写；并将时间戳纳入
        哈希覆盖，确保“何时”这一追溯要素不可被篡改（§12.5）。
        """
        now = datetime.now(UTC).replace(tzinfo=None)
        payload = json.dumps(
            {
                "actor": actor,
                "action": action,
                "object_type": object_type,
                "object_id": object_id,
                "before": before,
                "after": after,
                "note": note,
                "created_at": now.isoformat(timespec="seconds"),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        with self._audit_lock, Session(self._engine) as session, session.begin():
            last = session.scalars(
                select(AuditRecord).order_by(AuditRecord.seq.desc()).limit(1)
            ).first()
            prev_hash = last.hash if last is not None else "0" * 64
            h = hashlib.sha256(f"{prev_hash}|{payload}".encode()).hexdigest()
            rec = AuditRecord(
                actor=actor,
                action=action,
                object_type=object_type,
                object_id=object_id,
                before=before,
                after=after,
                note=note,
                created_at=now,
                prev_hash=prev_hash,
                hash=h,
            )
            session.add(rec)
            session.flush()
            _LOG.info(
                "audit append seq=%s action=%s object=%s:%s actor=%s",
                rec.seq,
                action,
                object_type,
                object_id,
                actor,
            )
            return self._audit_to_dict(rec)

    def list_audit(
        self,
        *,
        actor: str | None = None,
        action: str | None = None,
        object_type: str | None = None,
        object_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """审计日志检索（§14 GET /api/v1/audit），按时间降序。

        返回 (当页条目, 匹配总数)。原实现只返回列表，调用方拿 len() 当 total，
        在超过 limit 时会低报总数，审计场景不可接受。
        """
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        conds = []
        if actor:
            conds.append(AuditRecord.actor == actor)
        if action:
            conds.append(AuditRecord.action == action)
        if object_type:
            conds.append(AuditRecord.object_type == object_type)
        if object_id:
            conds.append(AuditRecord.object_id == object_id)
        with Session(self._engine) as session:
            total = int(
                session.scalar(select(func.count()).select_from(AuditRecord).where(*conds)) or 0
            )
            rows = list(
                session.scalars(
                    select(AuditRecord)
                    .where(*conds)
                    .order_by(AuditRecord.seq.desc())
                    .offset(offset)
                    .limit(limit)
                )
            )
            return [self._audit_to_dict(r) for r in rows], total

    def create_report_row(self, report_id: str, image_id: str) -> None:
        """补建一条报告行（用于重生成模式但影像尚无报告时）。"""
        with Session(self._engine) as session, session.begin():
            session.add(ReportRecord(id=report_id, image_id=image_id))

    def verify_chain(self) -> bool:
        """校验审计哈希链的连续性与不可分叉性（§12.5）。

        原实现用 list(session.scalars(...)) 一次性把整张审计表 materialize 成
        ORM 对象，随表增长内存占用线性膨胀（O(N)），且每次 /audit 校验都会触发。
        改为 yield_per 流式迭代：逐批（每批 1000 行）从游标取数，常驻内存 O(1)，
        且一旦发现哈希断裂立即返回 False，无需遍历全表。
        """
        with Session(self._engine) as session:
            prev = "0" * 64
            stmt = (
                select(AuditRecord)
                .order_by(AuditRecord.seq.asc())
                .execution_options(yield_per=1000)
            )
            for r in session.scalars(stmt):
                payload = json.dumps(
                    {
                        "actor": r.actor,
                        "action": r.action,
                        "object_type": r.object_type,
                        "object_id": r.object_id,
                        "before": r.before,
                        "after": r.after,
                        "note": r.note,
                        "created_at": r.created_at.isoformat(timespec="seconds"),
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                )
                expected = hashlib.sha256(f"{prev}|{payload}".encode()).hexdigest()
                if r.prev_hash != prev or r.hash != expected:
                    return False
                prev = r.hash
            return True

    # ---- 内部序列化 ----
    def _image_to_dict(self, rec: ImageRecord) -> dict[str, Any]:
        return {
            "image_id": rec.id,
            "path": rec.path,
            "source_type": rec.source_type,
            "modality": rec.modality,
            "workpiece_no": rec.workpiece_no,
            "weld_no": rec.weld_no,
            "pixel_spacing_mm": rec.pixel_spacing_mm,
            "base_metal_thickness_mm": rec.base_metal_thickness_mm,
            "iqi_pass": rec.iqi_pass,
            "iqi_detail": rec.iqi_detail,
            "density": rec.density,
            "density_ok": rec.density_ok,
            "pseudo_defect_pass": rec.pseudo_defect_pass,
            "pseudo_defect_notes": rec.pseudo_defect_notes,
            "quality_pass": rec.quality_pass,
            "quality_metrics": rec.quality_metrics,
            "evaluable": rec.evaluable,
            "joint_level": rec.joint_level,
            "need_review": rec.need_review,
            "standard_id": rec.standard_id,
            "standard_version": rec.standard_version,
            "created_at": _fmt_dt(rec.created_at),
        }

    @staticmethod
    def _defect_to_dict(d: DefectRecord) -> dict[str, Any]:
        return {
            "id": d.id,
            "image_id": d.image_id,
            "class_id": d.class_id,
            "class_name": _class_name(d.class_id),
            "bbox_px": list(d.bbox_px or []),
            "shape": d.shape,
            "length_mm": d.length_mm,
            "width_mm": d.width_mm,
            "area_mm2": d.area_mm2,
            "perimeter_mm": d.perimeter_mm,
            "position_x": d.position_x,
            "position_y": d.position_y,
            "confidence": d.confidence,
            "uncertainty": d.uncertainty,
            "joint_level": d.joint_level,
            "need_review": d.need_review,
            "reviewed_by": d.reviewed_by,
            "standard_id": d.standard_id,
            "standard_version": d.standard_version,
            "source": d.source,
            "deleted_at": _fmt_dt(d.deleted_at),
        }

    @staticmethod
    def _report_to_dict(r: ReportRecord) -> dict[str, Any]:
        return {
            "report_id": r.id,
            "image_id": r.image_id,
            "joint_level": r.joint_level,
            "generated_at": _fmt_dt(r.generated_at),
            "pdf_path": r.pdf_path,
            "standard_ref": r.standard_ref,
            "signer": r.signer,
            "basis": list(r.basis or []),
            "report_hash": r.report_hash,
            "signed_at": _fmt_dt(r.signed_at) if r.signed_at else None,
        }

    @staticmethod
    def _review_to_dict(r: ReviewRecord) -> dict[str, Any]:
        return {
            "review_id": r.id,
            "image_id": r.image_id,
            "reviewer": r.reviewer,
            "role": r.role,
            "overall_level": r.overall_level,
            "kappa": r.kappa,
            "consensus": r.consensus,
            "needs_arbitration": r.needs_arbitration,
            "note": r.note,
            "created_at": _fmt_dt(r.created_at),
        }

    @staticmethod
    def _audit_to_dict(r: AuditRecord) -> dict[str, Any]:
        return {
            "seq": r.seq,
            "actor": r.actor,
            "action": r.action,
            "object_type": r.object_type,
            "object_id": r.object_id,
            "before": r.before,
            "after": r.after,
            "note": r.note,
            "prev_hash": r.prev_hash,
            "hash": r.hash,
            "created_at": _fmt_dt(r.created_at),
        }


def _class_name(cid: int) -> str:
    """缺陷类别名（容错：未知 class_id 不抛异常，返回 UNKNOWN_<id> 便于定位脏数据）。"""
    try:
        return DefectClass(cid).name
    except ValueError:
        return f"UNKNOWN_{cid}"


def _parse_dt(raw: str) -> datetime:
    """解析日期：优先完整 ISO-8601（保留时间部分），否则回退 YYYY-MM-DD。"""
    s = (raw or "").strip()
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d")  # noqa: DTZ007 - 仅作日期区间过滤，无需时区
    except ValueError as e:
        raise ValueError(f"invalid date: {raw!r}, expected YYYY-MM-DD or ISO-8601") from e


def _parse_dt_end(raw: str) -> datetime:
    """解析区间上界，返回开区间右端点。

    "2026-08-08"（纯日期）视为「含当天」→ 次日 00:00:00；
    "2026-08-08T12:00"（带时间）按字面取值，不再无脑 +1 天
    （原实现对带时间的上界会多纳入整整一天的记录）。
    """
    s = (raw or "").strip()
    dt = _parse_dt(s)
    date_only = len(s) <= 10 or (dt.hour, dt.minute, dt.second, dt.microsecond) == (0, 0, 0, 0)
    return dt + timedelta(days=1) if date_only else dt


def _fmt_dt(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")
