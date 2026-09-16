"""健康检查。

存活探针走 try_get_registry 非阻塞获取：启动期（registry 后台装配中）即返回
status=starting（HTTP 200），端口绑定后立即可应答；业务端点仍走 get_registry
阻塞等待装配完成。status 表达 liveness，starting 表示进程存活但模型仍在加载。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from backend.app.dependencies import Registry, try_get_registry

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(reg: Annotated[Registry | None, Depends(try_get_registry)]) -> dict:
    if reg is None:
        # registry 装配中（模型加载/迁移）：存活即应答，字段与就绪形态对齐。
        # sync 适配器名如实从配置读取——此前硬编码 "local"，配置为 http/cloud
        # 时启动窗口内监控会看到错误结论；配置不可得时如实标注 unknown。
        guest_mode = False
        try:
            from backend.infra.config import load_config

            cfg = load_config()
            adapter = cfg.sync.kind
            guest_mode = bool(cfg.auth.guest_mode)
        except Exception:  # noqa: BLE001 - health 探针不因配置读取失败而 500
            adapter = "unknown"
        return {
            "status": "starting",
            "degraded": False,
            "app_version": "1.0.0",
            "guest_mode": guest_mode,
            "detector": "loading",
            "detector_degraded": False,
            "sync": {"adapter": adapter, "pending": 0},
            "uri": "",
            "backend": "",
            "active_version": None,
        }
    return reg.health
