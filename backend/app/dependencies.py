"""DI / registry 单例（§T4）。

共享状态（模型、队列）唯一入口；线程安全。
禁止在 router 内直接 new 模型或绕过 registry（§19.3）。
"""
from __future__ import annotations

import threading

from backend.domain.detect.blob_detector import BlobConfig, BlobDetector
from backend.domain.grade.nb47013 import Nb47013Grader
from backend.domain.interfaces import DefectDetector, StandardGrader
from backend.domain.standards.tables.loader import load_standard_tables
from backend.infra.config import AppConfig, load_config
from backend.infra.model_store import LocalModelStore


class Registry:
    """应用共享状态容器（单例）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.config: AppConfig = load_config()
        self.model = LocalModelStore(self.config.model.default_uri, self.config.model.backend)
        self.detector: DefectDetector = self._build_detector()
        self.grader: StandardGrader = self._build_grader()

    def _build_detector(self) -> DefectDetector:
        """按配置装配检测器：M4a 基线 / M4b 训练模型（模型无关接口，ADR-002）。"""
        dc = self.config.detect
        if dc.baseline_enabled:
            return BlobDetector(
                BlobConfig(
                    min_area_px=dc.min_area_px,
                    max_area_px=dc.max_area_px,
                    min_size_px=dc.min_size_px,
                    noise_sigma_ratio=dc.noise_sigma_ratio,
                    abs_threshold=dc.abs_threshold,
                    dark_only=dc.dark_only,
                )
            )
        raise RuntimeError("detect.baseline_enabled=false 需先装配训练模型（M4b）")

    def _build_grader(self) -> StandardGrader:
        """按配置装配标准判定器（多标准适配，ADR-003）。"""
        sc = self.config.standard
        tables = load_standard_tables(sc.default_id, filename=sc.tables_filename)
        return Nb47013Grader(tables)

    @property
    def health(self) -> dict:
        with self._lock:
            return {"status": "ok", "app_version": "0.1.0", **self.model.status}


_registry: Registry | None = None
_registry_lock = threading.Lock()


def get_registry() -> Registry:
    """获取全局 registry（懒初始化单例）。"""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = Registry()
    return _registry
