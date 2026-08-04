"""传统 CV 基线检测器（§5，M4a，零训练）。

实现冻结的 DefectDetector 契约（ADR-002）。通过连通域分析定位缺陷候选
（暗缺陷为主——气孔/夹渣/未焊透在透射数字化影像上偏暗；可选同时检亮斑）。

⚠️ 基线定位语义：仅用于**全链路验证**（软件先行）。
类别暂标 POROSITY 且置信度取启发式低值 + 高不确定性；**不可用于正式评级**，
须替换为训练模型（M4b YOLODetector），接口不变、主干零改动。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from backend.domain.dto import BBox, DefectClass, Detection
from backend.domain.preprocess.metrics import estimate_noise
from backend.domain.preprocess.pipeline import OpencvPreprocessor


@dataclass(frozen=True)
class BlobConfig:
    min_area_px: int = 30
    max_area_px: int = 200_000
    min_size_px: int = 3
    noise_sigma_ratio: float = 2.5  # 阈值 = 背景 ± k×原图噪声
    abs_threshold: float = 8.0  # 绝对阈值下限（低噪声图兜底，防过度检出）
    dark_only: bool = False  # False=暗+亮缺陷候选都检


class BlobDetector:
    """连通域基线检测器（M4a）。"""

    def __init__(
        self,
        cfg: BlobConfig | None = None,
        preprocessor: OpencvPreprocessor | None = None,
    ) -> None:
        self.cfg = cfg or BlobConfig()
        self.preprocessor = preprocessor or OpencvPreprocessor()
        self._loaded = False

    def load(self, model_uri: str, backend: str = "onnx") -> None:
        """基线无需权重；model_uri 仅作版本记录占位。"""
        self._loaded = True

    def infer(
        self,
        image: np.ndarray,
        conf: float = 0.3,
        iou: float = 0.5,
    ) -> list[Detection]:
        """预处理 → 自适应二值化（相对背景）→ 连通域 → 缺陷候选框。"""
        enhanced = self.preprocessor.enhance(self.preprocessor.denoise(image), 1.0)
        blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
        bg = float(np.median(blur))
        # 阈值用**原图噪声**估计（denoise 后噪声趋近 0 会致过度检出），
        # 并设绝对下限兜底低噪声图。
        original_noise = estimate_noise(image)
        margin = max(self.cfg.noise_sigma_ratio * max(original_noise, 1.0), self.cfg.abs_threshold)

        mask = np.zeros_like(image, np.uint8)
        mask[blur < bg - margin] = 255  # 暗缺陷
        if not self.cfg.dark_only:
            mask[blur > bg + margin] = 255  # 亮缺陷（夹钨等）

        n_labels, _labels, stats, _cents = cv2.connectedComponentsWithStats(mask)
        detections: list[Detection] = []
        for i in range(1, n_labels):
            x, y, w, h, area = stats[i]
            if area < self.cfg.min_area_px or area > self.cfg.max_area_px:
                continue
            if w < self.cfg.min_size_px or h < self.cfg.min_size_px:
                continue
            score = _heuristic_score(w, h, area)
            detections.append(
                Detection(
                    id=f"blob-{i}",
                    bbox=BBox(float(x), float(y), float(w), float(h)),
                    class_id=DefectClass.POROSITY,  # 基线占位，不可信
                    score=score,
                    uncertainty=round(1.0 - score, 3),
                )
            )
        return detections


def _heuristic_score(w: int, h: int, area: int) -> float:
    """启发式置信度：面积越大、形态越紧凑越可信（0.3–0.7，刻意保守）。"""
    size_term = min(area / 2000.0, 1.0)
    compactness = min(area / max(w * h, 1), 1.0)
    return round(0.3 + 0.4 * 0.5 * (size_term + compactness), 3)
