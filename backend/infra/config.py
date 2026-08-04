"""配置加载（§13.6 / §T8）。

- 所有可调参数入 configs/*.yaml，环境变量以 SCAN_ 前缀覆盖；
- 禁止硬编码端口/路径/密钥；
- 新增配置键必须同时更新 schema.yaml 与本模块字段。
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings

_BASE = Path(__file__).resolve().parents[1] / "configs"


class ServerCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 18773


class ModelCfg(BaseModel):
    default_uri: str = "models/weights/best.onnx"
    backend: str = "onnx"  # onnx | torch | tensorrt


class SecurityCfg(BaseModel):
    encrypt: bool = True


class PathsCfg(BaseModel):
    data_dir: str = "data"
    tmp_dir: str = "data/tmp"


class DensityCfg(BaseModel):
    low: float = 2.0  # AB 级黑度下限
    high: float = 4.5  # AB 级黑度上限


class IqiCfg(BaseModel):
    # 线型像质计丝号 1..N 直径(mm)，递增（公开参考，待官方复核）
    wire_diameters_mm: tuple[float, ...] = (
        3.2, 2.5, 2.0, 1.6, 1.25, 1.0, 0.8, 0.63, 0.5, 0.4,
        0.32, 0.25, 0.2, 0.16, 0.125, 0.1, 0.08, 0.063, 0.05,
    )
    required_wire_no: int = 10
    min_contrast_ratio: float = 3.0


class PreprocessCfg(BaseModel):
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    median_k: int = 3
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    canny_kernel: int = 5
    morph_k_open: int = 3
    morph_k_close: int = 3


class DetectCfg(BaseModel):
    baseline_enabled: bool = True  # M4a 基线检测器开关
    min_area_px: int = 30
    max_area_px: int = 200_000
    min_size_px: int = 3
    noise_sigma_ratio: float = 2.5
    abs_threshold: float = 8.0
    dark_only: bool = False


class StandardCfg(BaseModel):
    default_id: str = "NB/T47013.2-2015"
    tables_filename: str = "nb47013.yaml"


class AppConfig(BaseSettings):
    server: ServerCfg = ServerCfg()
    model: ModelCfg = ModelCfg()
    security: SecurityCfg = SecurityCfg()
    paths: PathsCfg = PathsCfg()
    density: DensityCfg = DensityCfg()
    iqi: IqiCfg = IqiCfg()
    preprocess: PreprocessCfg = PreprocessCfg()
    detect: DetectCfg = DetectCfg()
    standard: StandardCfg = StandardCfg()

    model_config = {"env_prefix": "SCAN_"}


def load_config() -> AppConfig:
    """从 configs/default.yaml 加载，缺省回退到模型默认值（安全默认）。"""
    cfg_file = _BASE / "default.yaml"
    if cfg_file.exists():
        raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
        return AppConfig(**raw)
    return AppConfig()
