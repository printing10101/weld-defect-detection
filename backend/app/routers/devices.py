"""设备标定档案。

设备向导数据源：注册设备 → 录入标定（实测像素标定 vs 标定件参考值）→
系统计算相对偏差并判定一致性（≤5% → ok，超差 → over）。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.app.dependencies import Registry, get_operator_name, get_registry

router = APIRouter(tags=["devices"])


class DeviceIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    model: str | None = Field(default=None, max_length=128)
    serial_no: str | None = Field(default=None, max_length=128)
    notes: str | None = None


class CalibrationIn(BaseModel):
    calibrator: str = Field(min_length=1, max_length=64, description="标定员")
    pixel_spacing_mm: float = Field(gt=0, description="实测像素标定（mm/px）")
    ref_pixel_spacing_mm: float | None = Field(
        default=None, gt=0, description="标定件参考像素标定（跨设备一致性基准）"
    )
    density_ref: float | None = Field(default=None, gt=0, description="黑度校验值（可选）")
    notes: str | None = None


class CalibrationOut(BaseModel):
    calibration_id: str
    device_id: str
    calibrator: str
    pixel_spacing_mm: float
    ref_pixel_spacing_mm: float | None
    deviation_pct: float | None
    status: str  # ok | over
    density_ref: float | None
    notes: str | None
    calibrated_at: str | None


class DeviceOut(BaseModel):
    device_id: str
    name: str
    model: str | None
    serial_no: str | None
    notes: str | None
    created_by: str | None
    created_at: str | None
    calibration_count: int
    last_calibration: CalibrationOut | None = None


class DeviceDetailOut(DeviceOut):
    calibrations: list[CalibrationOut] = Field(default_factory=list)


@router.get("/devices", response_model=list[DeviceOut])
def list_devices(reg: Annotated[Registry, Depends(get_registry)]) -> list[DeviceOut]:
    """设备列表（含最近标定摘要与一致性状态）。"""
    return [DeviceOut(**row) for row in reg.device_store.list()]


@router.post("/devices", response_model=DeviceOut)
def register_device(
    body: DeviceIn,
    reg: Annotated[Registry, Depends(get_registry)],
    operator: Annotated[str, Depends(get_operator_name)],
) -> DeviceOut:
    """注册检测设备。"""
    dev = reg.device_store.register(
        name=body.name,
        model=body.model,
        serial_no=body.serial_no,
        notes=body.notes,
        created_by=operator,
    )
    return DeviceOut(**dev)


@router.get("/devices/{device_id}", response_model=DeviceDetailOut)
def device_detail(
    device_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
) -> DeviceDetailOut:
    """设备详情：档案 + 完整标定档案 + 一致性状态。"""
    dev = reg.device_store.get(device_id)
    if dev is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": f"device not found: {device_id}"},
        )
    calibrations = dev.pop("calibrations", [])
    out = DeviceDetailOut(**dev)
    out.calibrations = [CalibrationOut(**c) for c in calibrations]
    return out


@router.post("/devices/{device_id}/calibrations", response_model=CalibrationOut)
def add_calibration(
    device_id: str,
    body: CalibrationIn,
    reg: Annotated[Registry, Depends(get_registry)],
) -> CalibrationOut:
    """记录一次标定：相对偏差 >5% → status=over（跨设备一致率超标）。"""
    try:
        calib = reg.device_store.calibrate(
            device_id=device_id,
            calibrator=body.calibrator,
            pixel_spacing_mm=body.pixel_spacing_mm,
            ref_pixel_spacing_mm=body.ref_pixel_spacing_mm,
            density_ref=body.density_ref,
            notes=body.notes,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND", "message": f"device not found: {device_id}"},
        ) from exc
    return CalibrationOut(**calib)
