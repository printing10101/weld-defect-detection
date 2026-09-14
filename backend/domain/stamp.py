"""底片印字识别（扫描日期/编号，正向 / 镜像 / 倒置 / 翻转）。

工业底片（X 射线胶片）扫描件上常带有透照日期与底片编号印字；底片背面扫描
或装片方向不同时，印字可能呈水平镜像、180° 倒置或垂直翻转。本模块在评片
链路中识别这些印字，作为底片性质落库供后续检索/追溯，并返回印字位置框
（供检测链路屏蔽印字区误检，见 detect.mask_stamp_zone）：

- 正向优先：先按原始方向 OCR，命中日期/编号模式即判正向（多数底片，省推理）；
  未命中再依次尝试镜像/180° 倒置/垂直翻转，取置信度最高者；
- 印字判定保守：只有命中「日期」或「编号」文本模式且识别置信度达标才记
  present，避免把胶片纹理噪点当印字；
- fail-soft：OCR 引擎缺失/推理异常一律降级为 unavailable，绝不阻断评片主链路
  （与预处理/检测器的可选降级语义一致）。

引擎：rapidocr-onnxruntime（纯 pip 安装、模型随包分发、复用进程内
onnxruntime，无系统级外部依赖）。引擎不可用时功能整体降级而非报错。
"""

from __future__ import annotations

import dataclasses
import logging
import re
import threading

import cv2
import numpy as np

_LOG = logging.getLogger("scandetection.stamp")

__all__ = ["StampCfg", "StampResult", "filter_stamp_zone", "read_stamp", "read_stamp_aligned"]


@dataclasses.dataclass
class StampCfg:
    """印字识别配置（由 infra.config.stamp 装配）。"""

    enabled: bool = True  # false=整体关闭（落库 stamp_status="off"，不参与复核）
    min_conf: float = 0.6  # 命中日期/编号模式所需的最低识别置信度
    max_side: int = 1800  # OCR 前长边上限（大底片等比降采样，控推理耗时）


@dataclasses.dataclass
class StampResult:
    """单张底片的印字识别结论（落库 images.stamp_* 的数据源）。

    status: present=识别到印字 | missing=未识别到 | unavailable=引擎不可用/异常
            | off=功能未启用
    orientation: normal=正向 | mirrored=镜像 | rotated=180°倒置 | flipped=垂直翻转
                 （仅 present 时有值；DB 列 String(8) 容纳全部取值）
    boxes: 命中印字的包围框 [[x0,y0,x1,y1], ...]，输入灰度图坐标（供检测链路
           屏蔽印字区误检）；missing/off/unavailable 时为空。
    """

    status: str
    text: str | None = None
    orientation: str | None = None
    confidence: float | None = None
    note: str | None = None
    boxes: list[list[float]] = dataclasses.field(default_factory=list)
    # 命中日期/编号模式的印字框（[x0,y0,x1,y1]，输入灰度图坐标）
    text_boxes: list[list[float]] = dataclasses.field(default_factory=list)
    # OCR 读到的全部文本框（置信度达标，输入灰度图坐标）：编号/日期之外的
    # 铅字（中心标、片号序号等）同样是印字，同样不该被检测当作缺陷。

    def summary(self, *, need_review: bool = False) -> dict:
        """批量任务/报告接口携带的精简快照（JSON 安全）。

        need_review 由调用方传入（缺印字是否触发人工复核取决于单图/批量延迟
        裁决语义，本模块无从知晓），缺省 False。
        """
        return {
            "status": self.status,
            "text": self.text,
            "orientation": self.orientation,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "need_review": need_review,
        }


# ---------------------------------------------------------------------------
# 印字文本模式：日期 / 底片编号
# ---------------------------------------------------------------------------

# 日期：2023-08-12 / 2023.08.12 / 2023/08/12 / 2023年08月12日 / 20230812，
# 以及真实底片常见的两位年份中文格式（23年1月8日，透照年份省略世纪）。
_DATE_LIKE = re.compile(
    r"\d{4}[-./年]\d{1,2}[-./月]\d{1,2}日?|\d{2,4}年\d{1,2}月\d{1,2}日|(?:19|20)\d{6}"
)
_NUM_RUN = re.compile(r"\d{3,}")
# 镜像裁决余量：翻转后最高置信度须比正向高出该值才判镜像（防两向都可读时抖动）
_MIRROR_MARGIN = 0.05


def is_date_token(text: str) -> bool:
    """是否日期样印字；8 位紧凑日期做年月日合理性校验（防把编号误判为日期）。"""
    compact = re.fullmatch(r"(?:19|20)(\d{2})(\d{2})(\d{2})", text)
    if compact is not None:
        month, day = int(compact.group(2)), int(compact.group(3))
        return 1 <= month <= 12 and 1 <= day <= 31
    return _DATE_LIKE.fullmatch(text) is not None


def is_id_token(text: str) -> bool:
    """是否编号样印字：含 ≥3 位连续数字的字母数字串（0421 / B3-0421 / No.0421）。

    日期串不算编号（同一段文字不重复计入两种模式）。
    """
    stripped = text.strip()
    if is_date_token(stripped) or len(stripped) < 3:
        return False
    return _NUM_RUN.search(stripped) is not None


def _is_stamp_token(text: str) -> bool:
    return is_date_token(text) or is_id_token(text)


# ---------------------------------------------------------------------------
# OCR 引擎（线程局部懒建：批量 worker 各持一份会话，避免跨线程共享会话的
# 线程安全问题，也免去全局锁串行化推理）
# ---------------------------------------------------------------------------

_engine_tls = threading.local()

_ENGINE_NOTE = "OCR 引擎不可用（未安装 rapidocr-onnxruntime）"


def _get_engine():
    """返回当前线程的 RapidOCR 实例；未安装返回 None（模块级缓存降级结论）。"""
    engine = getattr(_engine_tls, "engine", "?")
    if engine != "?":  # 已初始化（实例或 None）
        return engine
    try:
        from rapidocr_onnxruntime import RapidOCR  # 延迟导入：重依赖不进关键路径

        engine = RapidOCR()
    except Exception as exc:  # noqa: BLE001 - 引擎缺失/初始化失败按不可用降级
        _LOG.warning("rapidocr 初始化失败，印字识别降级为 unavailable: %s", exc)
        engine = None
    _engine_tls.engine = engine
    return engine


# ---------------------------------------------------------------------------
# 预处理与推理
# ---------------------------------------------------------------------------


def _to_uint8(gray: np.ndarray) -> np.ndarray:
    """OCR 输入统一为 8bit（load_image 已保证 uint8，此处兜底防御）。"""
    if gray.dtype == np.uint8:
        return gray
    arr = gray.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - lo) * (255.0 / (hi - lo))).astype(np.uint8)


def _prepare(gray: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """裁剪/降采样/反相后的 3 通道 OCR 输入，附坐标映射比例。

    - 胶片区整体偏暗（黑背景亮印字）时反相为「亮底暗字」，贴合 OCR 训练分布；
    - 大底片长边降采样到 max_side，印字相对胶片足够大，降采样不伤可读性；
    - 返回 scale 供把 OCR 框映射回输入灰度图坐标（反相不改几何）。
    """
    img = _to_uint8(gray)
    if int(img.mean()) < 110:
        img = 255 - img
    h, w = img.shape[:2]
    side = max(h, w)
    scale = 1.0
    if side > max_side > 0:
        scale = max_side / side
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), scale


def _ocr(engine, img_bgr: np.ndarray) -> list[tuple[str, float, list[float]]]:
    """跑一次 OCR，返回 [(文本, 置信度, 框[x0,y0,x1,y1])]；异常按空结果。"""
    result, _ = engine(img_bgr)
    out: list[tuple[str, float, list[float]]] = []
    for item in result or []:
        try:
            box = item[0]
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            text, score = str(item[1]), float(item[2])
        except (IndexError, TypeError, ValueError):
            continue
        if text.strip():
            out.append((text.strip(), score, [min(xs), min(ys), max(xs), max(ys)]))
    return out


def _assemble(
    reads: list[tuple[str, float, list[float]]], min_conf: float
) -> tuple[str, float, list[list[float]]] | None:
    """把全部命中片段按识别序拼成一条印字文本（日期+编号常见并存），附框。"""
    hits = [(t, s, b) for t, s, b in reads if s >= min_conf and _is_stamp_token(t)]
    if not hits:
        return None
    text = " ".join(t for t, _, _ in hits)
    conf = max(s for _, s, _ in hits)
    boxes = [b for _, _, b in hits]
    return text, conf, boxes


def _unmap_boxes(
    boxes: list[list[float]], shape: tuple[int, ...], scale: float, mode: str
) -> list[list[float]]:
    """把翻转后图像上的 OCR 框映射回原始输入坐标（逆变换 + 降采样还原）。

    mirrored=水平翻转、flipped=垂直翻转、rotated=180°；轴对齐框的逆映射是
    端点交换，随后统一除以 _prepare 的降采样比例。
    """
    h, w = shape[:2]
    out: list[list[float]] = []
    for x0, y0, x1, y1 in boxes:
        if mode == "mirrored":
            x0, x1 = w - x1, w - x0
        elif mode == "flipped":
            y0, y1 = h - y1, h - y0
        elif mode == "rotated":
            x0, x1 = w - x1, w - x0
            y0, y1 = h - y1, h - y0
        out.append([x0 / scale, y0 / scale, x1 / scale, y1 / scale])
    return out


# 正/镜像之外的候选方向（背面装反=180°倒置、翻面扫描=垂直翻转）。
# 仅在正向与镜像均未命中后才尝试（常规底片不多花推理），取置信度最高者。
_ORIENTATIONS: tuple[tuple[str, object], ...] = (
    ("rotated", lambda img: cv2.flip(img, -1)),
    ("flipped", lambda img: cv2.flip(img, 0)),
)


def read_stamp(gray: np.ndarray, cfg: StampCfg | None = None) -> StampResult:
    """识别底片印字（正向/镜像/倒置/翻转 + 位置框），fail-soft，永不抛异常。

    判定顺序：正向命中 → 正向（镜像须高出正向 _MIRROR_MARGIN 才夺回）；
    正向未命中 → 依次尝试镜像/180° 倒置/垂直翻转，取置信度最高者；
    全部未命中 → missing。功能关闭/引擎不可用 → off/unavailable。
    """
    cfg = cfg or StampCfg()
    if not cfg.enabled:
        return StampResult(status="off", note="印字识别未启用（stamp.enabled=false）")
    try:
        img, scale = _prepare(gray, cfg.max_side)
        engine = _get_engine()
        if engine is None:
            return StampResult(status="unavailable", note=_ENGINE_NOTE)

        normal_reads = _ocr(engine, img)
        normal_hit = _assemble(normal_reads, cfg.min_conf)
        mirrored_reads = _ocr(engine, cv2.flip(img, 1))
        mirrored_hit = _assemble(mirrored_reads, cfg.min_conf)
        normal_text = _text_boxes_of(normal_reads, cfg.min_conf)
        mirrored_text = _text_boxes_of(mirrored_reads, cfg.min_conf)
        # 镜像底片的正向 OCR 常读出"编号样"乱码，单纯阈值挡不住——按余量
        # 比较才能稳定判镜像（真实印字翻转后分数显著更高）。
        if mirrored_hit is not None and (
            normal_hit is None or mirrored_hit[1] >= normal_hit[1] + _MIRROR_MARGIN
        ):
            return StampResult(
                status="present",
                text=mirrored_hit[0][:120],
                orientation="mirrored",
                confidence=mirrored_hit[1],
                boxes=_unmap_boxes(mirrored_hit[2], img.shape, scale, "mirrored"),
                text_boxes=_unmap_boxes(mirrored_text, img.shape, scale, "mirrored"),
            )
        if normal_hit is not None:
            return StampResult(
                status="present",
                text=normal_hit[0][:120],
                orientation="normal",
                confidence=normal_hit[1],
                boxes=_unmap_boxes(normal_hit[2], img.shape, scale, "normal"),
                text_boxes=_unmap_boxes(normal_text, img.shape, scale, "normal"),
            )
        # 正/镜像均未命中：补试 180° 倒置与垂直翻转（背面装反/翻面扫描），
        # 取置信度最高者。仅在双Miss后才多花两次推理，常规底片耗时不变。
        best: tuple[str, tuple[str, float, list[list[float]]], list[list[float]]] | None = None
        for name, transform in _ORIENTATIONS:
            reads = _ocr(engine, transform(img))
            hit = _assemble(reads, cfg.min_conf)
            if hit is not None and (best is None or hit[1] > best[1][1]):
                best = (name, hit, _text_boxes_of(reads, cfg.min_conf))
        if best is not None:
            name, (text, conf, boxes), text_boxes = best
            return StampResult(
                status="present",
                text=text[:120],
                orientation=name,
                confidence=conf,
                boxes=_unmap_boxes(boxes, img.shape, scale, name),
                text_boxes=_unmap_boxes(text_boxes, img.shape, scale, name),
            )
        # 全部方向均未命中日期/编号模式：OCR 已跑，仍返回正向通道读到的全部
        # 文本框供印字区过滤（铅字印字存在只是未匹配模式，同样不是缺陷）。
        return StampResult(
            status="missing",
            note="未识别到日期/编号印字（四方向均未命中）",
            text_boxes=_unmap_boxes(normal_text, img.shape, scale, "normal"),
        )
    except Exception as exc:  # noqa: BLE001 - 印字识别任何异常不阻断评片主链路
        _LOG.warning("印字识别异常，降级 unavailable: %s", exc)
        return StampResult(status="unavailable", note=str(exc)[:200])


def _text_boxes_of(
    reads: list[tuple[str, float, list[float]]], min_conf: float
) -> list[list[float]]:
    """全部置信度达标的 OCR 文本框（不限日期/编号模式，供印字区过滤）。"""
    return [box for _t, score, box in reads if score >= min_conf]


def read_stamp_aligned(gray: np.ndarray, cfg: StampCfg, film=None) -> StampResult:
    """在胶片区上识别印字，命中框映射回整图坐标（与检测框同一坐标系）。

    film 非 None 时裁剪胶片区送 OCR（排除灯箱亮背景对反相判定的稀释，与
    评片管道的历史行为一致），命中框按胶片区偏移平移回整图；film=None 时
    整图识别。检测链路（评片管道 / /detect 预检）统一经此取印字框，避免
    "裁剪坐标 vs 整图坐标"错位。
    """
    region = gray
    if film is not None:
        region = gray[film.y : film.y + film.h, film.x : film.x + film.w]
    result = read_stamp(region, cfg)
    if film is not None and (result.boxes or result.text_boxes):
        result.boxes = _offset_boxes(result.boxes, film)
        result.text_boxes = _offset_boxes(result.text_boxes, film)
    return result


def _offset_boxes(boxes: list[list[float]], film) -> list[list[float]]:
    """把胶片区局部坐标的框平移回整图坐标。"""
    return [[x0 + film.x, y0 + film.y, x1 + film.x, y1 + film.y] for x0, y0, x1, y1 in boxes]


def filter_stamp_zone(
    detections: list,
    boxes: list[list[float]],
    *,
    pad_frac: float = 0.6,
) -> tuple[list, list]:
    """印字区误检过滤：检测框中心落入印字框外扩区域即判印字误检。

    返回 (保留列表, 被屏蔽列表)。外扩按印字框短边的 pad_frac 比例（OCR 框
    通常略小于实际印字字符）；用中心点判定而非 IoU——印字框面积大，小缺陷
    框与其重叠的 IoU 天然偏低，IoU 判定会漏掉绝大多数误检。

    纯函数：只做几何判定，不接触 OCR/文件；被屏蔽的检出由调用方留痕
    （审计 + 响应 warnings），绝不静默丢弃。
    """
    kept, masked = [], []
    for d in detections:
        cx = d.bbox.x + d.bbox.w / 2
        cy = d.bbox.y + d.bbox.h / 2
        inside = False
        for x0, y0, x1, y1 in boxes:
            padx = (x1 - x0) * pad_frac
            pady = (y1 - y0) * pad_frac
            if x0 - padx <= cx <= x1 + padx and y0 - pady <= cy <= y1 + pady:
                inside = True
                break
        (masked if inside else kept).append(d)
    return kept, masked
