"""底片印字识别（扫描日期/编号，正向 / 镜像）。

工业底片（X 射线胶片）扫描件上常带有透照日期与底片编号印字；底片背面扫描
时印字呈水平镜像。本模块在评片链路中识别这些印字，作为底片性质落库供后续
检索/追溯：

- 正向优先：先按原始方向 OCR，命中日期/编号模式即判正向（多数底片，省一次
  翻转推理）；未命中再对水平翻转图 OCR，命中即判镜像；
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

__all__ = ["StampCfg", "StampResult", "read_stamp"]


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
    orientation: normal=正向 | mirrored=镜像（仅 present 时有值）
    """

    status: str
    text: str | None = None
    orientation: str | None = None
    confidence: float | None = None
    note: str | None = None

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

# 日期：2023-08-12 / 2023.08.12 / 2023/08/12 / 2023年08月12日 / 20230812
_DATE_LIKE = re.compile(r"\d{4}[-./年]\d{1,2}[-./月]\d{1,2}日?|(?:19|20)\d{6}")
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


def _prepare(gray: np.ndarray, max_side: int) -> np.ndarray:
    """裁剪/降采样/反相后的 3 通道 OCR 输入。

    - 胶片区整体偏暗（黑背景亮印字）时反相为「亮底暗字」，贴合 OCR 训练分布；
    - 大底片长边降采样到 max_side，印字相对胶片足够大，降采样不伤可读性。
    """
    img = _to_uint8(gray)
    if int(img.mean()) < 110:
        img = 255 - img
    h, w = img.shape[:2]
    side = max(h, w)
    if side > max_side > 0:
        scale = max_side / side
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _ocr(engine, img_bgr: np.ndarray) -> list[tuple[str, float]]:
    """跑一次 OCR，返回 [(文本, 置信度)]；引擎异常按空结果（由外层定级）。"""
    result, _ = engine(img_bgr)
    out: list[tuple[str, float]] = []
    for item in result or []:
        try:
            text, score = str(item[1]), float(item[2])
        except (IndexError, TypeError, ValueError):
            continue
        if text.strip():
            out.append((text.strip(), score))
    return out


def _assemble(reads: list[tuple[str, float]], min_conf: float) -> tuple[str, float] | None:
    """把全部命中片段按识别序拼成一条印字文本（日期+编号常见并存）。"""
    hits = [(t, s) for t, s in reads if s >= min_conf and _is_stamp_token(t)]
    if not hits:
        return None
    text = " ".join(t for t, _ in hits)
    conf = max(s for _, s in hits)
    return text, conf


def read_stamp(gray: np.ndarray, cfg: StampCfg | None = None) -> StampResult:
    """识别底片印字（含镜像判定），fail-soft，永不抛异常。

    判定顺序：正向命中 → 正向；正向未命中而镜像命中 → 镜像（印字文本取自
    翻转后的读数）；两者皆未命中 → missing。功能关闭/引擎不可用 → off/unavailable，
    不参与缺印字复核语义。
    """
    cfg = cfg or StampCfg()
    if not cfg.enabled:
        return StampResult(status="off", note="印字识别未启用（stamp.enabled=false）")
    try:
        img = _prepare(gray, cfg.max_side)
        engine = _get_engine()
        if engine is None:
            return StampResult(status="unavailable", note=_ENGINE_NOTE)

        # 双向各跑一次 OCR 后按置信度+余量裁决：镜像底片的正向 OCR 常读出
        # "编号样"乱码（如 S053-08-J5 @0.67），单纯阈值挡不住——翻转后真实
        # 印字（日期+编号）分数显著更高，按余量比较才能稳定判镜像。
        normal_hit = _assemble(_ocr(engine, img), cfg.min_conf)
        mirrored_hit = _assemble(_ocr(engine, cv2.flip(img, 1)), cfg.min_conf)
        if mirrored_hit is not None and (
            normal_hit is None or mirrored_hit[1] >= normal_hit[1] + _MIRROR_MARGIN
        ):
            return StampResult(
                status="present",
                text=mirrored_hit[0][:120],
                orientation="mirrored",
                confidence=mirrored_hit[1],
            )
        if normal_hit is not None:
            return StampResult(
                status="present",
                text=normal_hit[0][:120],
                orientation="normal",
                confidence=normal_hit[1],
            )
        return StampResult(status="missing", note="未识别到日期/编号印字（正/镜像均未命中）")
    except Exception as exc:  # noqa: BLE001 - 印字识别任何异常不阻断评片主链路
        _LOG.warning("印字识别异常，降级 unavailable: %s", exc)
        return StampResult(status="unavailable", note=str(exc)[:200])
