"""缺陷量化（§5.4，M4a/M4b 实现）。

提供两种量化器，均实现冻结的 Quantifier 契约（measure(detection, pixel_spacing_mm) -> Geometry）：

- ``BBoxQuantifier``（M4a）：检测框矩形近似，供契约测试与无图场景。
- ``MaskQuantifier``（M4b 掩膜精修）：从增强图 ROI 内自适应阈值提取真实缺陷
  轮廓 → 最小外接矩形（MinAreaRect）得**有向**长/短边、轮廓面积/周长，比包围盒
  对不规则夹渣、裂纹分支、贴边缺陷更准（§5.3/§5.4；NFR §15.2 量化误差≤5%）。

为何不直接上 SAM2：SAM2 需 torch(本环境 CPU-only)+ 权重下载，未随部署包捆绑；
轮廓法仅用 cv2/numpy、零新增权重即可达到"掩膜级量化"目标，且可被后续 SAM 类
分割器经 DefectDetector 式接口热插替换（§19.4 扩展菜谱）。

像素标定：物理尺寸 = 像素尺寸 × pixel_spacing_mm。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from backend.domain.dto import BBox, DefectShape, Detection, Geometry
from backend.domain.errors import ModelUnavailableError
from backend.domain.interfaces import Quantifier


# ---------------------------------------------------------------------------
# 配置（§T8：domain 默认值 + infra/config.py + default.yaml + schema.yaml 四地同步）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MaskRefineCfg:
    """掩膜精修量化配置（M4b，§T8 四地同步）。

    enabled=False 时 MaskQuantifier 退化为包围盒近似（与 BBoxQuantifier 等价）。
    """

    enabled: bool = True
    blur_k: int = 5  # ROI 高斯平滑核（奇数），抑制颗粒噪对阈值的扰动
    adaptive_block: int = 31  # 自适应阈值窗口（奇数，≤ROI 短边）
    adaptive_c: float = 8.0  # 自适应阈值常数 C（0–255 量纲，越大越保守）
    min_mask_abs_area_px: int = 4  # 轮廓面积小于此值丢弃（抗孤立噪点）
    min_mask_rel_area: float = 0.25  # 掩膜面积须 ≥ 此比例×框面积，否则回退包围盒（抗误检）
    close_k: int = 5  # 形态学闭运算核（填补断裂、连成整体）
    round_aspect_max: float = 3.0  # 圆形/条形分界（NB/T47013：L/W<=3 为圆形）


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """统一到 8bit：16bit 底片直接做自适应阈值会因量纲不同而出错。"""
    if image.dtype == np.uint8:
        return image
    arr = image.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - lo) * (255.0 / (hi - lo))).astype(np.uint8)


# ---------------------------------------------------------------------------
# M4a：包围盒量化（冻结接口实现）
# ---------------------------------------------------------------------------
class BBoxQuantifier:
    """M4a 量化：检测框 → 几何属性（矩形近似，供全链路验证与契约测试）。"""

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

    def quantify(
        self,
        detection: Detection,
        pixel_spacing_mm: float,
        *,
        image: np.ndarray | None = None,
        cfg: MaskRefineCfg | None = None,
    ) -> Geometry:
        """统一量化入口（§T8 装配）：包围盒近似，忽略 image/cfg。

        与 MaskQuantifier.quantify 同签名，使两链路调用点一致、可经注册表互换。
        """
        return self.measure(detection, pixel_spacing_mm)


# ---------------------------------------------------------------------------
# M4b：掩膜精修量化（图像感知）
# ---------------------------------------------------------------------------
class MaskQuantifier:
    """M4b 掩膜精修量化：轮廓法得准确 L/W/面积/周长（§5.3/§5.4）。

    实现冻结 Quantifier 契约（``measure`` 为包围盒近似，供无图/测试场景），
    并额外提供图像感知的 ``quantify_from_image`` 与 ``refine``。
    """

    def measure(self, detection: Detection, pixel_spacing_mm: float) -> Geometry:
        """冻结接口：包围盒近似（与 BBoxQuantifier 一致）。"""
        return BBoxQuantifier().measure(detection, pixel_spacing_mm)

    def quantify(
        self,
        detection: Detection,
        pixel_spacing_mm: float,
        *,
        image: np.ndarray | None = None,
        cfg: MaskRefineCfg | None = None,
    ) -> Geometry:
        """统一量化入口（§T8 装配）：有图则掩膜精修，无图回退包围盒近似。

        与 BBoxQuantifier.quantify 同签名，使两链路调用点一致、可经注册表互换。
        """
        if image is None:
            return self.measure(detection, pixel_spacing_mm)
        return self.quantify_from_image(image, detection, pixel_spacing_mm, cfg)

    # ---- 掩膜提取 ----------------------------------------------------------
    @staticmethod
    def _defect_mask(roi: np.ndarray, cfg: MaskRefineCfg) -> np.ndarray:
        """ROI 内提取缺陷掩膜：暗缺陷(THRESH_BINARY_INV) + 亮缺陷(THRESH_BINARY) 并集。

        焊缝缺陷（气孔/夹渣/未熔合/未焊透/裂纹）在透射数字化影像上多为暗区；
        少数（如夹钨）偏亮，故两者取并集、取最大连通域，兼容两类。
        """
        roi8 = _to_uint8(roi)
        blur = cv2.GaussianBlur(roi8, (cfg.blur_k, cfg.blur_k), 0) if cfg.blur_k > 1 else roi8
        # 自适应窗口须为奇数且 ≤ ROI 短边
        blk = int(cfg.adaptive_block)
        blk = max(3, min(blk, roi8.shape[0] // 2, roi8.shape[1] // 2))
        if blk % 2 == 0:
            blk += 1
        dark = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blk, cfg.adaptive_c
        )
        bright = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, cfg.adaptive_c
        )
        mask = cv2.bitwise_or(dark, bright)
        if cfg.close_k > 1:
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_k, cfg.close_k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
        return mask

    @staticmethod
    def _largest_contour(mask: np.ndarray):
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[-2]  # 兼容 opencv 不同版本返回 (img, cnts, hier) / (cnts, hier)
        if not cnts:
            return None
        return max(cnts, key=cv2.contourArea)

    @staticmethod
    def _crop_roi(image: np.ndarray, bb: BBox):
        h, w = image.shape[:2]
        pad = int(max(bb.w, bb.h) * 0.2)  # 留 20% 余量，避免轮廓被框边截断
        x0 = max(0, int(bb.x) - pad)
        y0 = max(0, int(bb.y) - pad)
        x1 = min(w, int(bb.x + bb.w) + pad)
        y1 = min(h, int(bb.y + bb.h) + pad)
        return image[y0:y1, x0:x1], (x0, y0)

    # ---- 精修：更新检测框为最小外接矩形 + shape -------------------------
    def refine(
        self, image: np.ndarray, detection: Detection, cfg: MaskRefineCfg | None = None
    ) -> Detection:
        """返回精修后的 Detection：bbox 更新为掩膜最小外接矩形的轴对齐框，shape 按有向长宽比。

        退化（无掩膜/面积过小/越界）时原样返回，保证调用方安全。
        mask_ref 默认不落盘（掩膜持久化需存储设计），保持 None；SAM 类插件可在此写入 URI。
        """
        cfg = cfg or MaskRefineCfg()
        if not cfg.enabled:
            return detection
        bb = detection.bbox
        roi, (ox, oy) = self._crop_roi(image, bb)
        if roi.size == 0:
            return detection
        mask = self._defect_mask(roi, cfg)
        cnt = self._largest_contour(mask)
        if cnt is None:
            return detection
        area_px = cv2.contourArea(cnt)
        bbox_area_px = max(bb.w * bb.h, 1.0)
        if area_px < cfg.min_mask_abs_area_px or area_px < cfg.min_mask_rel_area * bbox_area_px:
            return detection
        rect = cv2.minAreaRect(cnt)
        (_cx, _cy), (rw, rh), _ang = rect
        L = max(rw, rh)
        W = min(rw, rh)
        aspect = L / max(W, 1e-6)
        box = cv2.boxPoints(rect)
        xs = box[:, 0]
        ys = box[:, 1]
        nx = max(0.0, float(xs.min()) + ox)
        ny = max(0.0, float(ys.min()) + oy)
        nw = float(xs.max() - xs.min())
        nh = float(ys.max() - ys.min())
        img_h, img_w = image.shape[:2]
        nw = min(img_w, nx + nw) - nx
        nh = min(img_h, ny + nh) - ny
        if nw <= 0 or nh <= 0:
            return detection
        shape = DefectShape.ROUND if aspect <= cfg.round_aspect_max else DefectShape.LINEAR
        return Detection(
            id=detection.id,
            bbox=BBox(x=nx, y=ny, w=nw, h=nh),
            class_id=detection.class_id,
            score=detection.score,
            uncertainty=detection.uncertainty,
            shape=shape,
            mask_ref=detection.mask_ref,
        )

    # ---- 量化：掩膜级几何 ------------------------------------------------
    def quantify_from_image(
        self,
        image: np.ndarray,
        detection: Detection,
        pixel_spacing_mm: float,
        cfg: MaskRefineCfg | None = None,
    ) -> Geometry:
        """从图像计算掩膜级几何；退化时回退包围盒近似（measure）。"""
        cfg = cfg or MaskRefineCfg()
        if not cfg.enabled:
            return self.measure(detection, pixel_spacing_mm)
        bb = detection.bbox
        roi, (ox, oy) = self._crop_roi(image, bb)
        if roi.size == 0:
            return self.measure(detection, pixel_spacing_mm)
        mask = self._defect_mask(roi, cfg)
        cnt = self._largest_contour(mask)
        if cnt is None:
            return self.measure(detection, pixel_spacing_mm)
        area_px = cv2.contourArea(cnt)
        bbox_area_px = max(bb.w * bb.h, 1.0)
        if area_px < cfg.min_mask_abs_area_px or area_px < cfg.min_mask_rel_area * bbox_area_px:
            return self.measure(detection, pixel_spacing_mm)
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), _ang = rect
        L_px = max(rw, rh)
        W_px = min(rw, rh)
        perimeter_px = cv2.arcLength(cnt, True)
        s = pixel_spacing_mm
        length_mm = L_px * s
        width_mm = W_px * s
        return Geometry(
            length_mm=round(length_mm, 3),
            width_mm=round(width_mm, 3),
            area_mm2=round(area_px * s * s, 3),
            perimeter_mm=round(perimeter_px * s, 3),
            aspect_ratio=round(length_mm / max(width_mm, 1e-6), 3),
            position_x_mm=round((cx + ox) * s, 3),
            position_y_mm=round((cy + oy) * s, 3),
        )


def refine_detections(
    image: np.ndarray, detections: list[Detection], cfg: MaskRefineCfg | None = None
) -> list[Detection]:
    """批量精修检测框（M4b 入口，供 detect 路由与全链路复用）。"""
    mq = MaskQuantifier()
    return [mq.refine(image, d, cfg) for d in detections]


# ---------------------------------------------------------------------------
# 量化器注册表（§T8，对齐 detect/registry.py 与 grade/registry.py 模式）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuantifierSpec:
    """量化器规格（注册表条目）。"""

    kind: str
    display_name: str
    cls: type[Quantifier]
    needs_image: bool  # True=图像感知（掩膜精修），False=包围盒近似（仅需检测框）


_QUANTIFIER_SPECS: dict[str, QuantifierSpec] = {
    "bbox": QuantifierSpec(
        kind="bbox",
        display_name="包围盒近似量化器 (M4a)",
        cls=BBoxQuantifier,
        needs_image=False,
    ),
    "mask": QuantifierSpec(
        kind="mask",
        display_name="掩膜精修量化器 (M4b)",
        cls=MaskQuantifier,
        needs_image=True,
    ),
}


def supported_quantifier_kinds() -> list[str]:
    """返回已注册量化器种类（注册表键，含插件）。"""
    return sorted(_QUANTIFIER_SPECS)


def register_quantifier_kind(spec: QuantifierSpec) -> None:
    """注册/覆盖量化器种类（§19.4 插件发现入口，P2）。

    同 kind 已注册且实现类不同 → 抛 ModelUnavailableError（防插件静默顶替内置）；
    相同实现（幂等重发现）→ 无操作。
    """
    existing = _QUANTIFIER_SPECS.get(spec.kind)
    if existing is not None and existing.cls is not spec.cls:
        raise ModelUnavailableError(
            f"量化器种类 {spec.kind!r} 已注册（{existing.cls.__name__}），拒绝覆盖"
        )
    _QUANTIFIER_SPECS[spec.kind] = spec


def quantifier_capabilities(kind: str) -> dict:
    """返回某量化器能力描述；未知种类抛 ModelUnavailableError（§14，复用而非新增）。"""
    spec = _QUANTIFIER_SPECS.get(kind)
    if spec is None:
        raise ModelUnavailableError(f"未知量化器种类: {kind!r}")
    return {
        "kind": spec.kind,
        "display_name": spec.display_name,
        "needs_image": spec.needs_image,
    }


def get_quantifier(kind: str = "bbox") -> Quantifier:
    """按种类取得量化器实例（依赖倒置：调用方经注册表装配，不在 app 层 new 实现）。

    未知种类抛 ModelUnavailableError（§14，复用而非新增错误码）。量化参数
    （如掩膜精修 MaskRefineCfg）在调用 ``quantify(..., cfg=...)`` 时透传，
    此处仅负责构造与装配，保持构造签名一致。
    """
    spec = _QUANTIFIER_SPECS.get(kind)
    if spec is None:
        raise ModelUnavailableError(
            f"未知量化器种类: {kind!r}（可选: {supported_quantifier_kinds()}）"
        )
    return spec.cls()
