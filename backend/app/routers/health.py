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
        return {
            "status": "starting",
            "degraded": False,
            "app_version": "0.1.0",
            "detector": "loading",
            "detector_degraded": False,
            "sync": {"adapter": "local", "pending": 0},
            "uri": "",
            "backend": "",
            "active_version": None,
        }
    return reg.health
