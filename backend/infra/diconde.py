"""DICONDE 元数据读取（ASTM E2339，DICOM 无损检测扩展）。

DICONDE 本质是 DICOM：影像属性走标准 DICOM 元素（曝光参数、设备信息等），
NDT 专有信息通过私有组（常见 ASNT/DICONDE 创建者）承载。本模块提取透照
工艺与设备档案字段，并识别私有组的存在与创建者名，供查看/报告引用。

与数据脱敏（anonymize_images.PHI_TAGS）共用同一套患者隐私标签清单：
读取接口输出的 patient_* 字段属于受隐私约束的元数据。
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from pydicom import dcmread
from pydicom.datadict import tag_for_keyword

# 透照工艺 / 设备档案字段（标准 DICOM 元素，keyword → 输出键名）
_TECHNIQUE_TAGS: dict[str, str] = {
    "Modality": "modality",
    "Manufacturer": "manufacturer",
    "StationName": "station_name",
    "DeviceSerialNumber": "device_serial",
    "KVP": "kvp",
    "TubeCurrent": "tube_current_ma",
    "ExposureTime": "exposure_time_ms",
    "Exposure": "exposure_mas",
    "ExposureTimeInuS": "exposure_time_us",
    "FocalSpots": "focal_spot_mm",
    "DistanceSourceToDetector": "source_to_detector_mm",
    "DistanceSourceToPatient": "source_to_patient_mm",
    "AcquisitionDate": "acquisition_date",
    "AcquisitionTime": "acquisition_time",
    "BodyPartExamined": "body_part_examined",
    "ViewPosition": "view_position",
    "PixelSpacing": "pixel_spacing_mm",
}

# 患者隐私标签（与 anonymize_images.PHI_TAGS 保持一致）
PHI_TAGS: dict[str, str] = {
    "PatientName": "patient_name",
    "PatientID": "patient_id",
    "PatientBirthDate": "patient_birth_date",
    "PatientSex": "patient_sex",
    "InstitutionName": "institution_name",
    "ReferringPhysicianName": "referring_physician",
    "OtherPatientIDs": "other_patient_ids",
}



def read_diconde_bytes(path: str | Path) -> bytes:
    """读取 DICONDE 文件字节：静态加密副本（SDC2 国密 / SDC1 历史 AES）先解密。

    信封分流与 image_loader.read_gray 同口径（decrypt 按魔数自动路由）——
    此前只认 SDC1，国密化后新落盘的 SDC2 副本会把密文直接喂给 DICOM 解析。
    """
    with open(path, "rb") as fh:
        buf = fh.read()
    if buf.startswith((b"SDC2", b"SDC1")):
        from backend.infra.crypto import default_crypto_provider

        return default_crypto_provider().decrypt(buf)
    return buf


def parse_diconde(data: bytes) -> dict[str, Any]:
    """解析 DICONDE 元数据。

    返回 {"technique": {...}, "phi": {...}, "private_groups": [...], "phi_present": bool}。
    非 DICOM 字节流（无 DICM 前导）抛 ValueError。
    """
    try:
        ds = dcmread(io.BytesIO(data), stop_before_pixels=True, force=False)
    except Exception as exc:
        raise ValueError(f"非 DICOM/DICONDE 文件或已损坏: {exc}") from exc

    def _get(keyword: str) -> Any:
        tag = tag_for_keyword(keyword)
        if tag is None or tag not in ds:
            return None
        value = ds[tag].value
        if isinstance(value, bytes):
            return None
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        if hasattr(value, "original_string"):  # DS/IS 数值串保留原始精度
            return str(value)
        return value if isinstance(value, (int, float, str)) else str(value)

    technique = {out: v for kw, out in _TECHNIQUE_TAGS.items() if (v := _get(kw)) is not None}
    phi = {out: v for kw, out in PHI_TAGS.items() if (v := _get(kw)) is not None}

    private: dict[str, list[str]] = {}
    for elem in ds:
        if elem.tag.is_private:
            creator_group = elem.tag.group
            if elem.tag.element == 0x0010 and isinstance(elem.value, str):
                private.setdefault(f"({creator_group:04x},xxxx)", []).append(elem.value.strip())
    return {
        "technique": technique,
        "phi": phi,
        "phi_present": bool(phi),
        "private_groups": private,
    }


def parse_diconde_file(path: str | Path) -> dict[str, Any]:
    """便捷入口：读文件（含解密）并解析。"""
    return parse_diconde(read_diconde_bytes(path))


def audit_dicom_phi(path: str | Path) -> list[str]:
    """审计单个 DICOM 文件中的患者隐私标签，返回残留字段名清单（脱敏校验用）。"""
    data = read_diconde_bytes(path)
    try:
        parsed = parse_diconde(data)
    except ValueError:
        return []
    return list(parsed["phi"].keys())
