"""缺陷量化。

提供两种量化器，均实现冻结的 Quantifier 契约（measure(detection, pixel_spacing_mm) -> Geometry）：

- ``BBoxQuantifier``：检测框矩形近似，供契约测试与无图场景。
- ``MaskQuantifier``（ 掩膜精修）：从增强图 ROI 内自适应阈值提取真实缺陷
  轮廓 → 最小外接矩形（MinAreaRect）得**有向**长/短边、轮廓面积/周长，比包围盒
  对不规则夹渣、裂纹分支、贴边缺陷更准。

为何不直接上 SAM2：SAM2 需 torch(本环境 CPU-only)+ 权重下载，未随部署包捆绑；
轮廓法仅用 cv2/numpy、零新增权重即可达到"掩膜级量化"目标，且可被后续 SAM 类
分割器经 DefectDetector 式接口热插替换。

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
# 配置
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MaskRefineCfg:
    """掩膜精修量化配置（， 四地同步）。

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
# 包围盒量化（冻结接口实现）
# ---------------------------------------------------------------------------
class BBoxQuantifier:
    """量化：检测框 → 几何属性（矩形近似，供全链路验证与契约测试）。"""

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
        """统一量化入口：包围盒近似，忽略 image/cfg。

        与 MaskQuantifier.quantify 同签名，使两链路调用点一致、可经注册表互换。
        """
        return self.measure(detection, pixel_spacing_mm)


# ---------------------------------------------------------------------------
# 掩膜精修量化（图像感知）
# ---------------------------------------------------------------------------
class MaskQuantifier:
    """掩膜精修量化：轮廓法得准确 L/W/面积/周长。

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
        """统一量化入口：有图则掩膜精修，无图回退包围盒近似。

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

        OpenCV 的 adaptiveThreshold 对 BINARY/INV 都取 T = mean - C：暗通道传
        +C 得"显著暗于局部均值"（正确）；亮通道必须传 -C 才是"显著亮于局部
        均值"——传 +C 会把整个 ROI（全部 ≥ mean-C 的像素）标成"亮"，掩膜恒
        为全 ROI、量化恒膨胀 ~40%（2026-09 由量化一致性 harness 发现并修复）。
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
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, -cfg.adaptive_c
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
    def _analyze_mask(self, image: np.ndarray, bb: BBox, cfg: MaskRefineCfg) -> tuple | None:
        """ROI 裁剪 → 掩膜 → 主轮廓 → 面积门槛 → 最小外接矩形（refine/量化共用）。

        返回 ``(rect, area_px, cnt, mask, (ox, oy))``；ROI 空、无轮廓、面积
        不足任一退化即返回 None，调用方按各自语义回退（refine 原样返回、
        量化回退包围盒）——判据与拆分前的两份实现逐行一致。
        """
        roi, (ox, oy) = self._crop_roi(image, bb)
        if roi.size == 0:
            return None
        mask = self._defect_mask(roi, cfg)
        cnt = self._largest_contour(mask)
        if cnt is None:
            return None
        area_px = cv2.contourArea(cnt)
        bbox_area_px = max(bb.w * bb.h, 1.0)
        if area_px < cfg.min_mask_abs_area_px or area_px < cfg.min_mask_rel_area * bbox_area_px:
            return None
        return cv2.minAreaRect(cnt), area_px, cnt, mask, (ox, oy)

    @staticmethod
    def _geometry_from_analysis(
        rect,
        area_px: float,
        cnt,
        mask: np.ndarray,
        ox: int,
        oy: int,
        pixel_spacing_mm: float,
        cfg: MaskRefineCfg,
    ) -> Geometry:
        """从掩膜分析结果计算几何（与拆分前 quantify_from_image 的算式一致）。"""
        (cx, cy), (rw, rh), _ang = rect
        L_px = max(rw, rh)
        W_px = min(rw, rh)
        perimeter_px = cv2.arcLength(cnt, True)
        s = pixel_spacing_mm
        # 条形缺陷长度用中心线弧长（G13）：弯曲裂纹的矩形长边量"弦"偏短，
        # 低估条形缺陷累计长度会导致评级偏松；退化/圆形回退矩形长边口径。
        centerline_mm: float | None = None
        length_px = L_px
        if L_px / max(W_px, 1e-6) > cfg.round_aspect_max:
            cl_px = _centerline_length_px(cnt, mask)
            if cl_px is not None and cl_px > 0:
                length_px = cl_px
                centerline_mm = round(cl_px * s, 3)
        length_mm = length_px * s
        width_mm = W_px * s
        return Geometry(
            length_mm=round(length_mm, 3),
            width_mm=round(width_mm, 3),
            area_mm2=round(area_px * s * s, 3),
            perimeter_mm=round(perimeter_px * s, 3),
            aspect_ratio=round(L_px * s / max(W_px * s, 1e-6), 3),
            position_x_mm=round((cx + ox) * s, 3),
            position_y_mm=round((cy + oy) * s, 3),
            centerline_mm=centerline_mm,
        )

    @staticmethod
    def _refined_detection(
        detection: Detection,
        rect,
        ox: int,
        oy: int,
        image_shape: tuple[int, ...],
        cfg: MaskRefineCfg,
    ) -> Detection | None:
        """从掩膜最小外接矩形构造精修 Detection；越界退化返回 None。"""
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
        img_h, img_w = image_shape[:2]
        nw = min(img_w, nx + nw) - nx
        nh = min(img_h, ny + nh) - ny
        if nw <= 0 or nh <= 0:
            return None
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
        analyzed = self._analyze_mask(image, detection.bbox, cfg)
        if analyzed is None:
            return detection
        rect, _area_px, _cnt, _mask, (ox, oy) = analyzed
        refined = self._refined_detection(detection, rect, ox, oy, image.shape, cfg)
        return refined if refined is not None else detection

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
        analyzed = self._analyze_mask(image, detection.bbox, cfg)
        if analyzed is None:
            return self.measure(detection, pixel_spacing_mm)
        rect, area_px, cnt, mask, (ox, oy) = analyzed
        return self._geometry_from_analysis(rect, area_px, cnt, mask, ox, oy, pixel_spacing_mm, cfg)


def refine_detections(
    image: np.ndarray, detections: list[Detection], cfg: MaskRefineCfg | None = None
) -> list[Detection]:
    """批量精修检测框（ 入口，供 detect 路由与全链路复用）。"""
    mq = MaskQuantifier()
    return [mq.refine(image, d, cfg) for d in detections]


def refine_and_quantify(
    image: np.ndarray,
    detections: list[Detection],
    pixel_spacing_mm: float,
    cfg: MaskRefineCfg | None = None,
) -> tuple[list[Detection], list[Geometry]]:
    """单遍精修+量化（/detect 路径）：每缺陷掩膜流水线只算一次。

    此前 /detect 先 refine_detections 再逐个 quantify——同一缺陷的掩膜
    流水线（高斯+双重自适应阈值+形态学+轮廓）跑两遍，CPU 直接翻倍；且
    两遍 ROI 不同（原始框 vs 精修框），框与几何可能来自不同轮廓。本路径
    两者同源同一轮廓；refine 结果与 refine_detections 完全一致（同一 ROI、
    同一判据），几何则与「refine 后 quantify」存在 ROI 边界效应级的细微
    差异（自适应阈值随 ROI 窗口略变），属预期。
    """
    mq = MaskQuantifier()
    cfg = cfg or MaskRefineCfg()
    refined: list[Detection] = []
    geometries: list[Geometry] = []
    for d in detections:
        analyzed = mq._analyze_mask(image, d.bbox, cfg) if cfg.enabled else None
        if analyzed is None:
            # 退化：与拆分路径一致——原样返回 + 包围盒近似
            refined.append(d)
            geometries.append(mq.measure(d, pixel_spacing_mm))
            continue
        rect, area_px, cnt, mask, (ox, oy) = analyzed
        refined.append(mq._refined_detection(d, rect, ox, oy, image.shape, cfg) or d)
        geometries.append(
            mq._geometry_from_analysis(rect, area_px, cnt, mask, ox, oy, pixel_spacing_mm, cfg)
        )
    return refined, geometries


# ---------------------------------------------------------------------------
# 中心线弧长（G13：条形缺陷真实长度）
# ---------------------------------------------------------------------------
def _centerline_length_px(cnt, mask: np.ndarray) -> float | None:
    """条形缺陷中心线弧长（像素）：沿 PCA 主轴分 bin 取质心，折线求和。

    最小外接矩形长边量的是"弦"，对弯曲裂纹系统性偏短；质心折线近似骨架
    弧长，无需骨架化依赖（cv2 无内置 thinning，ximgproc 不随部署包捆绑）。
    bin 宽取掩膜平均宽度（面积/主轴跨度），保证横穿缺陷截面各取一点。
    退化（点过少/掩膜近圆）返回 None，调用方回退矩形长边。
    """
    pts = cnt.reshape(-1, 2).astype(np.float64)
    if len(pts) < 4:
        return None
    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    eigvecs = np.linalg.eigh(cov)[1]
    axis = eigvecs[:, -1]  # 最大特征值方向 = 主轴
    t = (pts - mean) @ axis
    span = float(t.max() - t.min())
    if span <= 0:
        return None
    area = float(cv2.contourArea(cnt))
    bin_w = max(2.0, area / span)
    nbins = max(2, int(span / bin_w))
    edges = np.linspace(float(t.min()), float(t.max()), nbins + 1)
    centers: list[np.ndarray] = []
    for i in range(nbins):
        sel = (t >= edges[i]) & (t <= edges[i + 1])
        if sel.any():
            centers.append(pts[sel].mean(axis=0))
    if len(centers) < 2:
        return None
    c = np.asarray(centers)
    return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())


# ---------------------------------------------------------------------------
# 位置语义（G14：钟点位 / 焊缝轴向）
# ---------------------------------------------------------------------------
def clock_position(weld_cx: float, weld_cy: float, x_mm: float, y_mm: float) -> str:
    """缺陷相对焊缝圆心的钟点位（"H:MM"，半小时精度，12:00 = 正上方）。

    图像坐标 y 向下，先翻转到数学坐标再取顺时针角；钟点位是角度语义，
    不依赖像素标定。cx/cy 来自请求提供的焊缝圆心（管对接偏心透照布局），
    不从底片反推——猜不准的几何宁可不输出。
    """
    ang = float(np.degrees(np.arctan2(x_mm - weld_cx, weld_cy - y_mm)))  # 0=正上，顺时针为正
    if ang < 0:
        ang += 360.0
    # 一圈 720"分钟"（12h×60min），30 分钟精度 = 每 15° 一档
    total_minutes = int(round(ang * 2 / 30.0) * 30) % 720
    hour = total_minutes // 60
    minute = total_minutes % 60
    if hour == 0:
        hour = 12
    return f"{hour}:{minute:02d}"


def weld_axis(film_box: tuple[float, float, float, float]) -> str:
    """由胶片有效区外接框判定焊缝走向："h"（水平条带）或 "v"（垂直条带）。

    RT 底片焊缝沿胶片长边布置是行业惯例；轴向位置即缺陷中心在该方向
    上的投影。胶片区缺失或近方形（长短边比 < 1.2，走向不可判）时保守
    默认 "h"（多数底片为水平条带）。
    """
    _, _, w, h = film_box
    if h > 0 and w / h <= 1 / 1.2:  # 长边在垂直方向才判垂直
        return "v"
    return "h"


def nearest_neighbor_gaps(detections: list[Detection]) -> dict[str, tuple[str, float]]:
    """每缺陷的最近邻（id, 边缘间距 px）：G17 间距输出。

    边缘间距 = 两框在 x/y 上不重叠间隙的欧氏距离（47013"同线合并"gap 的
    2D 推广）；相互取最小。单缺陷返回空表。
    """
    gaps: dict[str, tuple[str, float]] = {}
    for i, a in enumerate(detections):
        best: tuple[str, float] | None = None
        for j, b in enumerate(detections):
            if i == j:
                continue
            dx = max(0.0, a.bbox.x - (b.bbox.x + b.bbox.w), b.bbox.x - (a.bbox.x + a.bbox.w))
            dy = max(0.0, a.bbox.y - (b.bbox.y + b.bbox.h), b.bbox.y - (a.bbox.y + a.bbox.h))
            gap = float(np.hypot(dx, dy))
            if best is None or gap < best[1]:
                best = (b.id, gap)
        if best is not None:
            gaps[a.id] = best
    return gaps


# ---------------------------------------------------------------------------
# 量化器注册表
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
    """注册/覆盖量化器种类。

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
    """返回某量化器能力描述；未知种类抛 ModelUnavailableError。"""
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

    未知种类抛 ModelUnavailableError。量化参数
    （如掩膜精修 MaskRefineCfg）在调用 ``quantify(..., cfg=...)`` 时透传，
    此处仅负责构造与装配，保持构造签名一致。
    """
    spec = _QUANTIFIER_SPECS.get(kind)
    if spec is None:
        raise ModelUnavailableError(
            f"未知量化器种类: {kind!r}（可选: {supported_quantifier_kinds()}）"
        )
    return spec.cls()
