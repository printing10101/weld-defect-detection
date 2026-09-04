"""C-15 纯离线自检结论——由静态配置得出"无外网依赖"判定。

判定只依赖 infra.config 的 AppConfig；启动自检与查询端点共用。
"""

from __future__ import annotations

from backend.infra.config import AppConfig


def offline_conclusion(config: AppConfig) -> dict:
    """C-15 纯离线自检：由静态配置得出"无外网依赖"结论（启动自检与端点共用）。

    判定（诚实边界：这是软件侧配置层的自证，物理断网须由 OS/主机防火墙
    出站默认拒绝兜底，见 docs/deployment-baseline.md）：
    - offline_mode = sync.kind == local（同步通道不留任何外发出口）；
    - egress_guard_enabled = egress.enabled（进程级外联拦截是否在岗）。
    """
    return {
        "offline_mode": config.sync.kind == "local",
        "sync_kind": config.sync.kind,
        "egress_guard_enabled": bool(config.egress.enabled),
    }
