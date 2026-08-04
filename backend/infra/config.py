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


class AppConfig(BaseSettings):
    server: ServerCfg = ServerCfg()
    model: ModelCfg = ModelCfg()
    security: SecurityCfg = SecurityCfg()
    paths: PathsCfg = PathsCfg()

    model_config = {"env_prefix": "SCAN_"}


def load_config() -> AppConfig:
    """从 configs/default.yaml 加载，缺省回退到模型默认值（安全默认）。"""
    cfg_file = _BASE / "default.yaml"
    if cfg_file.exists():
        raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
        return AppConfig(**raw)
    return AppConfig()
