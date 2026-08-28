"""设备标定档案。

跨设备一致性是工业大批量筛查的关键（设计文档：跨设备一致率 ≤5%）：
- 每台设备登记档案（devices 表）；
- 每次标定记录实测像素标定与标定件参考值，计算相对偏差
  deviation_pct = |实测 − 参考| / 参考 × 100，>5% 标记 status=over（超差），
  否则 ok——同一门槛即"跨设备一致率 ≤5%"的量化落地。
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.infra.db import (
    Base,
    CalibrationRecord,
    DeviceRecord,
    create_db_engine,
)

_LOG = logging.getLogger("scandetection.devices")

_CONSISTENCY_LIMIT_PCT = 5.0  # §12.4：跨设备一致率门槛（相对偏差 ≤5%）


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _fmt(dt: datetime | None) -> str | None:
    return dt.strftime("%Y-%m-%dT%H:%M:%S") if dt else None


class DeviceStore:
    """设备与标定记录存储（SQLite，SQLAlchemy 2.0）。"""

    def __init__(self, db_path: str) -> None:
        self._engine = create_db_engine(db_path)
        Base.metadata.create_all(self._engine)

    # ---- 设备 ----
    def register(
        self,
        *,
        name: str,
        model: str | None = None,
        serial_no: str | None = None,
        notes: str | None = None,
        created_by: str | None = None,
    ) -> dict[str, Any]:
        device_id = uuid.uuid4().hex
        with Session(self._engine) as session, session.begin():
            session.add(
                DeviceRecord(
                    id=device_id,
                    name=name,
                    model=model,
                    serial_no=serial_no,
                    notes=notes,
                    created_by=created_by,
                )
            )
        dev = self.get(device_id)
        if dev is None:  # pragma: no cover - 刚写入不应丢失
            raise RuntimeError("device register failed")
        return dev

    def list(self) -> list[dict[str, Any]]:
        """设备列表（含最近标定摘要与一致性状态，按注册时间倒序）。"""
        with Session(self._engine) as session:
            rows = session.scalars(
                select(DeviceRecord).order_by(DeviceRecord.created_at.desc())
            ).all()
        return [self._device_with_latest(dev.id) for dev in rows]

    def get(self, device_id: str) -> dict[str, Any] | None:
        with Session(self._engine) as session:
            dev = session.get(DeviceRecord, device_id)
            if dev is None:
                return None
            calibrations = session.scalars(
                select(CalibrationRecord)
                .where(CalibrationRecord.device_id == device_id)
                .order_by(CalibrationRecord.calibrated_at.desc())
            ).all()
            return self._device_to_dict(dev, list(calibrations))

    # ---- 标定 ----
    def calibrate(
        self,
        *,
        device_id: str,
        calibrator: str,
        pixel_spacing_mm: float,
        ref_pixel_spacing_mm: float | None = None,
        density_ref: float | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        """记录一次标定；参考值存在时计算相对偏差并判定一致性（ok/over）。"""
        with Session(self._engine) as session, session.begin():
            dev = session.get(DeviceRecord, device_id)
            if dev is None:
                raise KeyError(f"device not found: {device_id}")
        deviation_pct: float | None = None
        status = "ok"
        if ref_pixel_spacing_mm and ref_pixel_spacing_mm > 0:
            deviation_pct = round(
                abs(pixel_spacing_mm - ref_pixel_spacing_mm) / ref_pixel_spacing_mm * 100, 2
            )
            status = "over" if deviation_pct > _CONSISTENCY_LIMIT_PCT else "ok"
        calib_id = uuid.uuid4().hex
        with Session(self._engine) as session, session.begin():
            session.add(
                CalibrationRecord(
                    id=calib_id,
                    device_id=device_id,
                    calibrator=calibrator,
                    pixel_spacing_mm=pixel_spacing_mm,
                    ref_pixel_spacing_mm=ref_pixel_spacing_mm,
                    deviation_pct=deviation_pct,
                    status=status,
                    density_ref=density_ref,
                    notes=notes,
                )
            )
        with Session(self._engine) as session:
            rec = session.get(CalibrationRecord, calib_id)
        if rec is None:  # pragma: no cover
            raise RuntimeError("calibration persist failed")
        return self._calib_to_dict(rec)

    def list_calibrations(self, device_id: str) -> list[dict[str, Any]]:
        with Session(self._engine) as session:
            rows = session.scalars(
                select(CalibrationRecord)
                .where(CalibrationRecord.device_id == device_id)
                .order_by(CalibrationRecord.calibrated_at.desc())
            ).all()
        return [self._calib_to_dict(r) for r in rows]

    # ---- 内部 ----
    def _device_with_latest(self, device_id: str) -> dict[str, Any]:
        dev = self.get(device_id)
        if dev is None:  # pragma: no cover
            return {"device_id": device_id, "name": "?", "deleted": True}
        # 列表形态：只带最近一次标定摘要，不含全部档案（保持轻量）
        calibs = dev.pop("calibrations", [])
        dev["last_calibration"] = calibs[0] if calibs else None
        return dev

    @staticmethod
    def _device_to_dict(dev: DeviceRecord, calibrations: list[CalibrationRecord]) -> dict[str, Any]:
        return {
            "device_id": dev.id,
            "name": dev.name,
            "model": dev.model,
            "serial_no": dev.serial_no,
            "notes": dev.notes,
            "created_by": dev.created_by,
            "created_at": _fmt(dev.created_at),
            "calibration_count": len(calibrations),
            "calibrations": [DeviceStore._calib_to_dict(c) for c in calibrations],
        }

    @staticmethod
    def _calib_to_dict(c: CalibrationRecord) -> dict[str, Any]:
        return {
            "calibration_id": c.id,
            "device_id": c.device_id,
            "calibrator": c.calibrator,
            "pixel_spacing_mm": c.pixel_spacing_mm,
            "ref_pixel_spacing_mm": c.ref_pixel_spacing_mm,
            "deviation_pct": c.deviation_pct,
            "status": c.status,
            "density_ref": c.density_ref,
            "notes": c.notes,
            "calibrated_at": _fmt(c.calibrated_at),
        }
