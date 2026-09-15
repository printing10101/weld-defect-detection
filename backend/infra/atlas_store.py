"""缺陷图谱样本库（defect_atlas 表读写 + 缺陷局部裁图）。

图谱库是**人工筛选沉淀的典型缺陷样本**（培训/比对/复核参考，对标商业
评片软件的"缺陷图谱"产品线），与 defects 事实记录分离：发布/撤销均为
显式操作并留主审计链。

结构分工（与 GateRejectStore 同模式）：
- 本模块：defect_atlas 表的纯存储 + 从落盘影像（支持静态加密副本）裁出
  缺陷局部 PNG；
- app/routers/atlas.py：编排（查源缺陷 → 裁图 → 落盘 → 入库 → 审计）；
- ORM 模型挂 infra.db.Base（schema 真源），演进由 0013_defect_atlas 管理。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.infra.db import Base, DefectAtlasRecord, create_db_engine
from backend.infra.image_loader import read_gray

_LOG = logging.getLogger("scandetection.atlas")

_PAGE_MAX = 100


class AtlasStore:
    """defect_atlas 台账读写（与 InspectionRepository 同 Session 模式）。"""

    def __init__(self, db_path: str) -> None:
        self._engine = create_db_engine(db_path)
        # 幂等兜底：遗留 create_all 库经 ensure_migrations stamp head 不会执行
        # 0013 的 DDL，靠这里把缺表补齐（与仓储层兜底语义一致）。
        Base.metadata.create_all(self._engine)

    def publish(self, row: dict[str, Any]) -> str:
        """写入一条图谱样本；defect_id 重复（同一缺陷重复发布）→ ValueError。

        白名单校验外部 dict，防多余键/缺键的 TypeError/KeyError；
        IntegrityError 转译为 ValueError，防 SQL 细节外泄。
        """
        fields = {c.name for c in DefectAtlasRecord.__table__.columns}
        if unknown := set(row) - fields:
            raise ValueError(f"unknown atlas fields: {sorted(unknown)}")
        for required in ("id", "defect_id", "crop_path", "created_by"):
            if not row.get(required):
                raise ValueError(f"atlas row missing required field: {required}")
        try:
            with Session(self._engine) as session, session.begin():
                session.add(DefectAtlasRecord(**row))
        except IntegrityError as e:
            detail = str(getattr(e, "orig", None) or e)
            if "UNIQUE" in detail.upper():
                raise ValueError(f"defect already published: {row['defect_id']}") from e
            raise ValueError(f"atlas integrity error: {detail}") from e
        return str(row["id"])

    def list_samples(
        self,
        *,
        class_id: int | None = None,
        level: str | None = None,
        workpiece: str | None = None,
        page: int = 1,
        size: int = 20,
    ) -> tuple[list[dict[str, Any]], int]:
        """分页检索（按类/级别/工件号过滤，created_at 降序）。"""
        size = min(size, _PAGE_MAX)
        with Session(self._engine) as session:
            cond = []
            if class_id is not None:
                cond.append(DefectAtlasRecord.class_id == class_id)
            if level:
                cond.append(DefectAtlasRecord.joint_level == level)
            if workpiece:
                cond.append(DefectAtlasRecord.workpiece_no.contains(workpiece))
            total = (
                session.scalar(select(func.count()).select_from(DefectAtlasRecord).where(*cond))
                or 0
            )
            rows = session.scalars(
                select(DefectAtlasRecord)
                .where(*cond)
                .order_by(DefectAtlasRecord.created_at.desc())
                .offset((page - 1) * size)
                .limit(size)
            ).all()
            return [self._to_dict(r) for r in rows], int(total)

    def get(self, atlas_id: str) -> dict[str, Any] | None:
        with Session(self._engine) as session:
            rec = session.get(DefectAtlasRecord, atlas_id)
            return self._to_dict(rec) if rec is not None else None

    def get_crop_path(self, atlas_id: str) -> str | None:
        with Session(self._engine) as session:
            rec = session.get(DefectAtlasRecord, atlas_id)
            return rec.crop_path if rec is not None else None

    def remove(self, atlas_id: str) -> bool:
        """撤销样本（物理删除行；撤销动作由调用方写审计链）。"""
        with Session(self._engine) as session, session.begin():
            rec = session.get(DefectAtlasRecord, atlas_id)
            if rec is None:
                return False
            session.delete(rec)
            return True

    @staticmethod
    def _to_dict(rec: DefectAtlasRecord) -> dict[str, Any]:
        return {
            "id": rec.id,
            "defect_id": rec.defect_id,
            "image_id": rec.image_id,
            "class_id": rec.class_id,
            "joint_level": rec.joint_level,
            "bbox_px": rec.bbox_px,
            "length_mm": rec.length_mm,
            "width_mm": rec.width_mm,
            "confidence": rec.confidence,
            "source": rec.source,
            "reviewed_by": rec.reviewed_by,
            "workpiece_no": rec.workpiece_no,
            "weld_no": rec.weld_no,
            "standard_id": rec.standard_id,
            "note": rec.note,
            "created_by": rec.created_by,
            "created_at": rec.created_at.isoformat() if rec.created_at else None,
        }


def crop_defect_png(
    image_path: str,
    bbox_px: list[float],
    *,
    margin_frac: float = 0.5,
    min_margin_px: int = 24,
) -> bytes | None:
    """从落盘影像裁出缺陷局部 PNG bytes（read_gray 自动解密静态加密副本）。

    外扩边距 = max(min_margin_px, 边长×margin_frac)（保留缺陷周边焊缝语境，
    便于比对），逐边裁到影像边界内。影像不可读/裁剪区为空返回 None
    （调用方转 404/409，不让裁图失败伪装成成功发布）。
    """
    gray = read_gray(image_path)
    if gray is None or not bbox_px or len(bbox_px) < 4:
        return None
    h_img, w_img = gray.shape[:2]
    x, y, w, h = (float(v) for v in bbox_px[:4])
    mx = max(float(min_margin_px), w * margin_frac)
    my = max(float(min_margin_px), h * margin_frac)
    x0 = max(0, int(x - mx))
    y0 = max(0, int(y - my))
    x1 = min(w_img, int(x + w + mx))
    y1 = min(h_img, int(y + h + my))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = gray[y0:y1, x0:x1]
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        return None
    return buf.tobytes()


def write_crop_bytes(data: bytes, dest, *, encrypt: bool) -> str:
    """图谱局部图落盘（encrypt=True 走与影像副本相同的静态加密路径）。

    明文不落盘的纪律与 pipelines._write_encrypted_copy 一致：密钥不可用
    时拒绝明文降级、原样上抛（宁可发布失败，不可静默明文）。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not encrypt:
        dest.write_bytes(data)
        return str(dest)
    from backend.infra.crypto import CryptoKeyError, default_crypto_provider

    try:
        cipher = default_crypto_provider()
    except CryptoKeyError as exc:
        _LOG.error("静态加密密钥不可用（%s）：拒绝将图谱样本以明文落盘", exc)
        raise
    dest.write_bytes(cipher.encrypt(data))
    return str(dest)


def read_crop_bytes(path: str) -> bytes | None:
    """读图谱局部图（明文与 SDC1/SDC2 密文信封均支持，与 read_gray 同口径）。

    文件不存在返回 None；密钥不可用/信封损坏抛 CryptoKeyError/
    CryptoIntegrityError 原样上抛——与"样本缺失"混为同一 404 会让密钥
    丢失演变成整库静默变砖，调用方须区分处置。
    """
    p = Path(path)
    if not p.is_file():
        return None
    buf = p.read_bytes()
    if buf.startswith((b"SDC1", b"SDC2")):
        from backend.infra.crypto import default_crypto_provider

        buf = default_crypto_provider().decrypt(buf)
    return buf
