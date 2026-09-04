"""门禁配置适配：把 infra 层 pydantic 配置对象拷贝为 domain 数据类。

IqiConfig / PseudoDefectCfg 的字段拷贝在检查管线与复核校验两处逐字
重复，任何一侧增改字段都可能与另一侧失配（门禁口径分叉）。统一经
本模块适配；infra 配置对象以鸭子类型传入（domain 不 import infra）。
"""

from __future__ import annotations

from backend.domain.iqi import IqiConfig
from backend.domain.pseudo_defect import PseudoDefectCfg


def iqi_cfg_from_settings(s) -> IqiConfig:
    """由配置对象的 iqi 段构造 IqiConfig。"""
    return IqiConfig(
        type=s.type,
        wire_diameters_mm=tuple(s.wire_diameters_mm),
        required_wire_no=s.required_wire_no,
        hole_diameters_mm=tuple(s.hole_diameters_mm),
        required_hole_no=s.required_hole_no,
        min_contrast_ratio=s.min_contrast_ratio,
        auto_locate=s.auto_locate,
        locate_threshold=s.locate_threshold,
        sensitivity=tuple(s.sensitivity),
    )


def pseudo_cfg_from_settings(s) -> PseudoDefectCfg:
    """由配置对象的 pseudo_defect 段构造 PseudoDefectCfg。"""
    return PseudoDefectCfg(
        hough_threshold=s.hough_threshold,
        scratch_min_ratio=s.scratch_min_ratio,
        scratch_grating_min_lines=s.scratch_grating_min_lines,
        canny_lo=s.canny_lo,
        canny_hi=s.canny_hi,
        uniformity_low_freq=s.uniformity_low_freq,
        uniformity_max_ratio=s.uniformity_max_ratio,
        dust_tophat_k=s.dust_tophat_k,
        dust_min_area=s.dust_min_area,
        dust_max_count=s.dust_max_count,
        block_on_scratch=s.block_on_scratch,
        block_on_uniformity=s.block_on_uniformity,
        block_on_dust=s.block_on_dust,
    )
