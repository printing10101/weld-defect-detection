"""DI / registry 单例（§T4）。

共享状态（模型、队列）唯一入口；线程安全。
禁止在 router 内直接 new 模型或绕过 registry（§19.3）。
"""
from __future__ import annotations

import threading

from backend.infra.config import AppConfig, load_config
from backend.infra.model_store import LocalModelStore


class Registry:
    """应用共享状态容器（单例）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.config: AppConfig = load_config()
        self.model = LocalModelStore(self.config.model.default_uri, self.config.model.backend)

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
