"""缺陷量化（§5.4，M4a 实现）。

实现冻结的 Quantifier 契约。M4a 从检测框计算几何（矩形近似），
M4b 换掩膜精修实现（SAM）——接口不变、实现可插拔（ADR-002 精神）。
像素标定：物理尺寸 = 像素尺寸 × pixel_spacing_mm。
"""
from __future__ import annotations

from backend.domain.dto import Detection, Geometry


class BBoxQuantifier:
    """M4a 量化：检测框 → 几何属性（矩形近似，供全链路验证）。"""

    def measure(self, detection: Detection, pixel_spacing_mm: float) -> Geometry:
        w_px = float(detection.bbox.w)
        h_px = float(detection.bbox.h)
        length_mm = max(w_px, h_px) * pixel_spacing_mm
        width_mm = min(w_px, h_px) * pixel_spacing_mm
        return Geometry(
            length_mm=round(length_mm, 3),
            width_mm=round(width_mm, 3),
            area_mm2=round(w_px * h_px * pixel_spacing_mm**2, 3),
            perimeter_mm=round(2 * (w_px + h_px) * pixel_spacing_mm, 3),
            aspect_ratio=round(length_mm / max(width_mm, 1e-6), 3),
            position_x_mm=round(detection.bbox.x * pixel_spacing_mm, 3),
            position_y_mm=round(detection.bbox.y * pixel_spacing_mm, 3),
        )
