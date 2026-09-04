"""不合格底片留档 + 扫描参数门禁（DB50/T 1807-2025 §5，E-05）测试。

覆盖：
- dpi 门禁（DICOM PixelSpacing 推算 / 无法确定时按 require_dpi 处置）；
- 位深硬门禁（8bit 默认拦截，allow_8bit 降级放行）；
- 拦截留档：密文归档目录 + gate_rejects 台账 + gate_reject 审计动作。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import UID, generate_uid
from skimage.io import imsave

from backend.app.dependencies import get_registry
from backend.app.pipelines import (
    InspectionPipeline,
    _check_bit_depth,
    _check_dpi,
    _estimate_dpi,
)
from backend.domain.dto import ImageMeta, Modality
from backend.domain.errors import IQIFailError
from backend.evaluation.gate_rejects import GateRejectStore
from backend.infra.config import GateCfg, resolve_config_path


def _uniform_png(path: Path) -> Path:
    """均匀灰片：无 IQI/缺陷，质量门禁判定不可评（与历史阻断用例同源）。"""
    imsave(str(path), np.full((256, 256), 128, dtype=np.uint8))
    return path


def _write_dicom(path: Path, spacing: tuple[float, float], shape: tuple = (64, 64)) -> Path:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID("1.2.840.10008.5.1.4.1.1.7")
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = UID("1.2.840.10008.1.2.1")
    ds = FileDataset(str(path), Dataset(), file_meta=file_meta, preamble=b"\x00" * 128)
    ds.PixelData = np.zeros(shape, np.uint16).tobytes()
    ds.Rows, ds.Columns = shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    ds.PixelSpacing = list(spacing)
    ds.SOPClassUID = UID("1.2.840.10008.5.1.4.1.1.7")
    ds.SOPInstanceUID = generate_uid()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(path))
    return path


# ---------------------------------------------------------------------------
# 纯函数：dpi / 位深门禁
# ---------------------------------------------------------------------------


def test_estimate_dpi_from_pixel_spacing() -> None:
    meta = ImageMeta(modality=Modality.DICOM, pixel_spacing_mm=0.05)
    assert _estimate_dpi(meta) == pytest.approx(508.0)  # 25.4 / 0.05


def test_estimate_dpi_unknown_returns_none() -> None:
    assert _estimate_dpi(ImageMeta(modality=Modality.GENERIC)) is None


def test_check_dpi_known_below_limit_blocks() -> None:
    ok, reason = _check_dpi(508.0, GateCfg(min_dpi=600))
    assert ok is False and "600" in reason and "dpi" in reason


def test_check_dpi_unknown_follows_require_switch() -> None:
    ok, reason = _check_dpi(None, GateCfg(require_dpi=True))
    assert ok is False and "扫描分辨率" in reason
    ok2, reason2 = _check_dpi(None, GateCfg(require_dpi=False))
    assert ok2 is True and reason2 == ""


def test_check_bit_depth_blocks_8bit_by_default() -> None:
    ok, reason = _check_bit_depth(8, GateCfg())
    assert ok is False and "16bit" in reason
    ok2, _ = _check_bit_depth(8, GateCfg(allow_8bit=True))
    assert ok2 is True  # 可配置降级放行
    ok3, _ = _check_bit_depth(16, GateCfg())
    assert ok3 is True
    # 位深未知（异常元数据）不拦：交由黑度/IQI 门禁兜底
    assert _check_bit_depth(None, GateCfg()) == (True, "")


# ---------------------------------------------------------------------------
# 全链路：拦截 + 留档
# ---------------------------------------------------------------------------


def test_rejected_film_is_archived_with_ledger_and_audit(tmp_path: Path) -> None:
    """8bit 底片硬拦截（allow_8bit=False）：阻断评片且留档三件套齐全。"""
    reg = get_registry()
    monkeypatch_allow = reg.config.gate
    monkeypatch_allow.allow_8bit = False
    try:
        pipe = InspectionPipeline(reg)
        img = _uniform_png(tmp_path / "uniform.png")
        with pytest.raises(IQIFailError):
            pipe.run_inspection(img, pixel_spacing_mm=0.1, base_metal_thickness_mm=20)

        # 1. 影像副本归档到 rejects 目录（经本地密钥文件加密 → SDC2 密文留档）
        rejects_dir = Path(resolve_config_path(reg.config.gate.rejects_dir))
        archived = list(rejects_dir.iterdir())
        assert archived, "被拦截底片应归档留档"

        # 2. 台账：原因/位深/操作员可查（库为会话级共享，按来源过滤本用例的行）
        store = GateRejectStore(str(resolve_config_path(reg.config.paths.db_path)))
        rows, total = store.list(limit=500)
        assert total >= 1
        mine = [r for r in rows if r["detail"].get("source", "").endswith("uniform.png")]
        assert len(mine) == 1
        row = mine[0]
        assert "16bit" in row["reject_reason"] or "位深" in row["reject_reason"]
        assert row["bit_depth"] == 8
        assert row["dpi"] is None  # 通用图像无 dpi 元数据

        # 3. 审计哈希链有 gate_reject 动作
        entries, _ = reg.repository.list_audit(action="gate_reject")
        assert any(e["object_id"] == row["id"] for e in entries)
    finally:
        monkeypatch_allow.allow_8bit = True  # 恢复 conftest 默认，避免污染其他用例


def test_low_dpi_dicom_blocked(tmp_path: Path) -> None:
    """DICOM PixelSpacing 推算 dpi 低于下限 → 阻断并留档 dpi 快照。"""
    reg = get_registry()
    pipe = InspectionPipeline(reg)
    img = _write_dicom(tmp_path / "low_dpi.dcm", spacing=(0.1, 0.1))  # 254 dpi
    with pytest.raises(IQIFailError, match="低于下限"):
        pipe.run_inspection(img, pixel_spacing_mm=0.1, base_metal_thickness_mm=20)
    store = GateRejectStore(str(resolve_config_path(reg.config.paths.db_path)))
    rows, total = store.list(limit=500)
    assert total >= 1
    mine = [r for r in rows if r["detail"].get("source", "").endswith("low_dpi.dcm")]
    assert len(mine) == 1
    assert mine[0]["dpi"] == pytest.approx(254.0)


def test_require_dpi_blocks_unknown_dpi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """require_dpi=true：通用图像无法确定 dpi 时从严拦截。"""
    reg = get_registry()
    monkeypatch.setattr(reg.config.gate, "require_dpi", True)
    pipe = InspectionPipeline(reg)
    img = _uniform_png(tmp_path / "uniform.png")
    with pytest.raises(IQIFailError, match="无法确定扫描分辨率"):
        pipe.run_inspection(img, pixel_spacing_mm=0.1, base_metal_thickness_mm=20)


def test_8bit_degrade_pass_emits_warning(tmp_path: Path) -> None:
    """allow_8bit=true（测试环境默认）：8bit 底片放行但告警留痕。"""
    reg = get_registry()
    pipe = InspectionPipeline(reg)
    # 带焊缝带与圆形缺陷的合成底片（force 出片，IQI 不合格也走告警链路）
    arr = np.full((512, 512), 160, dtype=np.uint8)
    arr[:, 230:282] = 120
    ys, xs = np.ogrid[0:512, 0:512]
    mask = (xs - 256) ** 2 + (ys - 256) ** 2 <= 18**2
    arr[mask] = 55
    img = tmp_path / "film.png"
    imsave(str(img), arr)
    out = pipe.run_inspection(img, pixel_spacing_mm=0.1, base_metal_thickness_mm=20, force=True)
    assert any("降级放行" in w for w in out["warnings"]), "8bit 降级放行必须告警留痕"
    assert any("无法确定扫描分辨率" in w for w in out["warnings"])
