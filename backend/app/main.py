"""FastAPI 应用入口。

- 挂载 /api/v1，端点清单见；
- CORS 仅允许本机来源（127.0.0.1）；
- 全局异常处理器：AppError -> 统一错误包， 起挂载。
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from backend.app.auth import AuthError

from backend.app.dependencies import get_registry
from backend.app.routers import (
    active,
    audit,
    auth,
    batch,
    carriers,
    classification,
    compliance,
    detect,
    devices,
    evaluation,
    explain,
    export,
    health,
    judge,
    measure,
    metrics,
    models,
    preprocess,
    privacy,
    recommend,
    records,
    report,
    review,
    standards,
    std_eval,
    system,
    verify,
)
from backend.domain.errors import AppError
from backend.infra.config import load_config, resolve_config_path
from backend.infra.fs import safe_resolve


def _envelope(code: str, message: str, detail=None) -> dict:
    return {"error": {"code": code, "message": message, "detail": detail}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 先统一日志；重装配（模型加载/DB 迁移，实测 ~2.5s，冷启动更久）移入后台线程，
    # 使 uvicorn 立即绑定端口——前端 /health 探测马上通过（status=starting），
    # 业务端点首个请求会经 get_registry 阻塞等待装配完成（语义与原先一致）。
    _configure_logging()

    # 孤儿兜底：若由 Tauri 壳启动（env 给了父 PID），监控父进程消失即自杀退出，
    # 避免壳被强杀/崩溃时遗留孤儿后端长期占用端口与内存。
    try:
        from backend.infra.orphan_guard import start_orphan_guard_if_spawned

        start_orphan_guard_if_spawned()
    except Exception as exc:  # noqa: BLE001 - 兜底武装失败不阻断启动
        _LOG.warning("orphan-guard arm failed: %s", exc)

    # 边界安全启动自检（C-15/C-16/C-17）：全部为配置驱动的轻量动作，
    # 失败不应阻断启动（日志显式留痕，见各模块）。
    try:
        boot_cfg = load_config()
        # C-16：进程级外联防护（先于任何请求处理；幂等）
        from backend.infra.egress_guard import configure_egress_guard

        configure_egress_guard(boot_cfg.egress.enabled, list(boot_cfg.egress.allow_cidrs))
        # C-17：IPC 一次性令牌（enforce 时签发并落盘，与中间件共用令牌槽）
        if boot_cfg.ipc.enforce:
            from backend.infra.ipc_token import ensure_token

            ensure_token(resolve_config_path(boot_cfg.paths.data_dir))
        # C-15：离线模式自检结论（静态配置检查，启动日志留痕；非 local 在
        # registry 就绪后补告警入库，见 _init_registry）
        from backend.app.routers.system import offline_conclusion

        conclusion = offline_conclusion(boot_cfg)
        if conclusion["offline_mode"]:
            _LOG.info(
                "离线模式自检：offline_mode=True (sync=%s, egress_guard=%s) —— 无外网依赖",
                conclusion["sync_kind"],
                conclusion["egress_guard_enabled"],
            )
        else:
            _LOG.warning(
                "离线模式自检：sync.kind=%s（非 local），数据将离开本机（C-15 告警留痕）",
                conclusion["sync_kind"],
            )
    except Exception as exc:  # noqa: BLE001 - 自检失败不阻断启动
        _LOG.warning("边界安全启动自检失败（不阻断）: %s", exc)

    def _init_registry() -> None:
        # P2-5：启动时把 DB schema 升到 Alembic 头版本（兼容历史 create_all DB：自动 stamp）。
        try:
            from backend.infra.migrate import ensure_migrations

            db_path = get_registry().config.paths.db_path
            version = ensure_migrations(db_path)
            _LOG.info("schema migrations applied (version=%s)", version)
        except Exception as exc:  # noqa: BLE001 - 迁移失败不应阻止启动；create_all 兜底
            _LOG.warning("schema migration skipped (create_all fallback): %s", exc)
        get_registry()
        # C-15：sync.kind 非 local（http/cloud）= 配置层显式选择了"数据出本机"，
        # 启动即落 high 级安全告警留痕（offline_mode=False 的持久证据链）。
        try:
            reg = get_registry()
            if reg.config.sync.kind != "local":
                reg.security_store.raise_alert(
                    kind="sync_nonlocal",
                    level="high",
                    message=f"同步通道为 {reg.config.sync.kind}（非 local），数据将离开本机",
                    detail={
                        "sync_kind": reg.config.sync.kind,
                        "endpoint": reg.config.sync.http_endpoint,
                    },
                )
        except Exception as exc:  # noqa: BLE001 - 告警失败不阻断装配
            _LOG.warning("sync_nonlocal 告警落库失败: %s", exc)
        # S-09 内存看门狗（默认关）：config.watchdog.enabled=true 时启动后台采样。
        try:
            wc = reg.config.watchdog
            if wc.enabled:
                from backend.infra.watchdog import MemoryWatchdog

                reg.watchdog = MemoryWatchdog(
                    interval_sec=wc.interval_sec,
                    rss_warn_mb=wc.rss_warn_mb,
                    rss_restart_mb=wc.rss_restart_mb,
                    graceful_restart=wc.graceful_restart,
                    data_dir=resolve_config_path(reg.config.paths.data_dir),
                    raise_alert=reg.security_store.raise_alert,
                    append_audit=reg.repository.append_audit,
                )
                reg.watchdog.start()
                _LOG.info(
                    "memory watchdog started (warn=%.0fMB restart=%.0fMB)",
                    wc.rss_warn_mb,
                    wc.rss_restart_mb,
                )
        except Exception as exc:  # noqa: BLE001 - 看门狗失败不阻断装配
            _LOG.warning("memory watchdog start failed: %s", exc)
        # S-20 磁盘水位看门狗（默认开）：data 分区剩余空间低水位告警+审计。
        try:
            dc = reg.config.disk_space
            if dc.enabled:
                from backend.infra.disk_space import DiskWatchdog

                reg.disk_watchdog = DiskWatchdog(
                    interval_sec=dc.interval_sec,
                    warn_ratio_pct=dc.warn_ratio_pct,
                    warn_min_bytes=dc.warn_min_bytes,
                    data_dir=resolve_config_path(reg.config.paths.data_dir),
                    raise_alert=reg.security_store.raise_alert,
                    append_audit=reg.repository.append_audit,
                )
                reg.disk_watchdog.start()
                _LOG.info(
                    "disk watchdog started (ratio<%.1f%% or free<%d bytes)",
                    dc.warn_ratio_pct,
                    dc.warn_min_bytes,
                )
        except Exception as exc:  # noqa: BLE001 - 看门狗失败不阻断装配
            _LOG.warning("disk watchdog start failed: %s", exc)
        # S-12a 定期备份（默认 0=关）：backup.interval_hours>0 时后台定时备份。
        try:
            interval = reg.config.backup.interval_hours
            if interval and interval > 0:
                from backend.infra.backup import BackupScheduler

                reg.backup_scheduler = BackupScheduler(
                    interval, lambda: _scheduled_backup(reg)
                )
                reg.backup_scheduler.start()
                _LOG.info("backup scheduler started (interval=%.2fh)", interval)
        except Exception as exc:  # noqa: BLE001 - 调度失败不阻断装配
            _LOG.warning("backup scheduler start failed: %s", exc)
        # P2-8：随主应用同进程拉起人工标注器（默认关；开启后主动学习闭环无需另开终端）。
        _start_annotator_if_enabled()
        _LOG.info("application startup complete (registry assembled)")

    threading.Thread(target=_init_registry, name="registry-init", daemon=True).start()
    yield
    # 应用退出时优雅关停批量线程池（等运行中任务结束），避免 worker 被硬杀
    try:
        from backend.app.dependencies import try_get_registry

        reg = try_get_registry()
        if reg is not None:
            reg.batch_manager.shutdown()
            # S-09/S-12a：看门狗与备份调度线程一并优雅退出。
            if getattr(reg, "watchdog", None) is not None:
                reg.watchdog.stop()
            if getattr(reg, "disk_watchdog", None) is not None:
                reg.disk_watchdog.stop()
            if getattr(reg, "backup_scheduler", None) is not None:
                reg.backup_scheduler.stop()
    except Exception as exc:  # noqa: BLE001 - 关停失败不应掩盖其它退出逻辑
        _LOG.warning("batch_manager shutdown skipped: %s", exc)


def _scheduled_backup(reg) -> None:
    """S-12a 定期备份任务体（审计 note 与手动备份区分）。"""
    from backend.app.routers.system import run_backup

    run_backup(reg, note="scheduled backup (backup.interval_hours)")


def _start_annotator_if_enabled() -> None:
    """标注器随主应用同进程启动。

    仅当 config.annotator.enabled=True；守护线程，主进程退出即终止。
    """
    try:
        from backend.app.dependencies import get_registry

        cfg = get_registry().config.annotator
        if not cfg.enabled:
            return
        import os
        import threading

        os.environ["ANNO_HOST"] = cfg.host
        os.environ["ANNO_PORT"] = str(cfg.port)

        def _run() -> None:
            from backend.annotator import server as ann

            ann.main()

        t = threading.Thread(target=_run, name="annotator", daemon=True)
        t.start()
        _LOG.info("annotator launched in-process on %s:%s", cfg.host, cfg.port)
    except Exception as exc:  # noqa: BLE001 - 标注器失败不应影响主应用
        _LOG.warning("annotator launch failed: %s", exc)


_LOG = logging.getLogger("scandetection")


def _configure_logging() -> None:
    """统一日志（关键路径可追溯，）。

    由配置驱动（observability.log_format）：text=人类可读（本地开发默认），
    json=结构化单行日志（接入采集/ELK 的可观测基础）。固定 INFO 级别。
    """
    from backend.infra.logging import configure_logging

    try:
        cfg = load_config()
        fmt = cfg.observability.log_format
    except Exception:  # noqa: BLE001 - 配置缺失时回退 text，日志不应成为启动阻断点
        fmt = "text"
    configure_logging(fmt)


def create_app() -> FastAPI:
    """构造 FastAPI 应用。

    抽为工厂函数：测试可经 create_app 获得全新实例以验证真实鉴权链路
    （不继承 conftest 注入的 admin 覆盖）；模块级 `app` 供现有测试/生产使用。
    """
    app = FastAPI(title="ScanDetection", version="0.1.0", lifespan=lifespan)

    # CORS 允许源由配置驱动：默认覆盖 Tauri webview
    # （tauri://localhost）+ 本地开发源（127.0.0.1 / :5173）。桌面应用仅监听
    # 本机，风险可控。不再使用 "*"，否则任意外部网站均可跨源读取本机 API
    # （含审计链 / 报告）；部署新增前端源改 configs/default.yaml 即可，不改代码。
    cfg = load_config()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.server.cors_origins),
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "X-Operator-Name", "Authorization", "X-Export-Token"],
    )

    # /：安全响应头 + 基础限流（P2-9）。中间件按添加顺序执行，
    # CORS 在最内层（先于安全头/限流处理），保证 OPTIONS 预检同样获得安全头。
    from backend.app.security import (
        IpcTokenMiddleware,
        RateLimitMiddleware,
        SecurityHeadersMiddleware,
    )

    app.add_middleware(RateLimitMiddleware)  # 外层：限流
    app.add_middleware(SecurityHeadersMiddleware)  # 内层：安全头

    # IPC 一次性令牌校验（C-17）：业务请求须带 X-IPC-Token 或会话凭据。
    # enforce 由配置驱动（测试经 conftest 置 false，最小侵入）。
    if cfg.ipc.enforce:
        app.add_middleware(
            IpcTokenMiddleware,
            enforce=True,
            data_dir=resolve_config_path(cfg.paths.data_dir),
        )

    # 可观测性：进程内指标中间件（最外层，采集所有 HTTP 请求计数/耗时）。
    from backend.infra.metrics import MetricsMiddleware, get_metrics

    get_metrics().enabled = cfg.observability.enable_metrics
    app.add_middleware(MetricsMiddleware)

    # 三员鉴权（C-06）：除存活/指标/认证端点外，全部业务路由要求已登录
    # （Bearer 会话）。测试经 conftest 的 dependency_overrides 统一注入测试
    # principal，不影响既有测试；角色级管控（require_role）在各敏感端点上强制。
    from backend.app.auth import get_principal

    open_routers = (health.router, metrics.router, auth.router)
    for router in (
        health.router,
        metrics.router,
        auth.router,
        verify.router,
        preprocess.router,
        standards.router,
        detect.router,
        devices.router,
        judge.router,
        recommend.router,
        batch.router,
        review.router,
        explain.router,
        report.router,
        records.router,
        models.router,
        audit.router,
        active.router,
        evaluation.router,
        std_eval.router,
        measure.router,
        system.router,
        classification.router,
        carriers.router,
        export.router,
        privacy.router,
        compliance.router,
        auth.alerts_router,  # C-22 告警通知（unread-count / ack，端点自带鉴权依赖）
    ):
        if router in open_routers:
            app.include_router(router, prefix="/api/v1")
        else:
            app.include_router(router, prefix="/api/v1", dependencies=[Depends(get_principal)])
    return app


app = create_app()


# ---------------------------------------------------------------------------
# 统一错误包：领域异常 → 对应 HTTP 状态；校验错误 → 422；其余 → 500（不透传细节）。
# ---------------------------------------------------------------------------
@app.exception_handler(AppError)
async def _app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(_envelope(exc.code, str(exc)), status_code=exc.http_status)


@app.exception_handler(AuthError)
async def _auth_error_handler(request: Request, exc: AuthError) -> JSONResponse:
    # C-22 异常行为告警：越权访问（require_role 403）落安全告警留痕。
    # 401（未登录/会话失效）不告警——未认证请求高频且无身份可归责，避免刷屏。
    if exc.status == 403:
        _record_unauthorized_access(request, exc)
    return JSONResponse(_envelope(exc.code, exc.message), status_code=exc.status)


def _record_unauthorized_access(request: Request, exc: AuthError) -> None:
    """越权 403 → 安全告警（best-effort，失败不影响 403 响应本身）。"""
    try:
        from backend.app.dependencies import try_get_registry

        reg = try_get_registry()
        if reg is None or not reg.config.alerts.unauthorized_access.enabled:
            return
        principal = getattr(request.state, "principal", None)
        reg.security_store.raise_alert(
            kind="unauthorized_access",
            level="warn",
            message=f"越权访问被拒绝: {request.method} {request.url.path}",
            detail={
                "path": request.url.path,
                "method": request.method,
                "actor": getattr(principal, "username", None),
                "role": getattr(principal, "role", None),
                "code": exc.code,
            },
        )
    except Exception as exc_:  # noqa: BLE001 - 告警失败不掩盖 403
        _LOG.warning("unauthorized_access 告警落库失败: %s", exc_)


@app.exception_handler(RequestValidationError)
async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        _envelope("VALIDATION_ERROR", "请求参数校验失败", exc.errors()),
        status_code=422,
    )


@app.exception_handler(Exception)
async def _unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
    _LOG.exception("unhandled exception")
    return JSONResponse(_envelope("INTERNAL", "服务器内部错误"), status_code=500)


# ---------------------------------------------------------------------------
# 静态前端托管（打包 / 浏览器启动回退模式）。
# 当 dist 存在时，根路径与未命中静态文件的路由回退到 index.html（SPA）。
# 仅当 dist 缺失（纯开发未构建）时跳过，避免影响 API。
# ---------------------------------------------------------------------------
_DIST_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "dist",  # 打包后：<安装目录>/dist
    Path(__file__).resolve().parents[2] / "src" / "dist",  # 开发：<项目>/src/dist
    Path.cwd() / "dist",
    Path.cwd() / "src" / "dist",
]
DIST = next((p for p in _DIST_CANDIDATES if p.is_dir()), None)


@app.get("/")
async def _serve_root():
    if DIST is None:
        return HTMLResponse(
            "<h2>ScanDetection</h2><p>前端未构建（dist 缺失）。"
            "请使用 Tauri 桌面端，或在开发模式下执行 <code>pnpm build</code>。</p>"
        )
    return FileResponse(DIST / "index.html")


@app.get("/{full_path:path}")
async def _serve_spa(full_path: str):
    # 不拦截 API 路由（已由上方路由器处理）。
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404)
    if DIST is None:
        raise HTTPException(status_code=404)
    # 防路径穿越：越界一律 404（手写 handler 必须显式校验，StaticFiles 已内置此防护）。
    try:
        candidate = safe_resolve(DIST, full_path)
    except ValueError:
        raise HTTPException(status_code=404) from None
    if candidate.is_file():
        return FileResponse(candidate)
    index = DIST / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404)
    # SPA 回退：未知前端路由交给 index.html 处理。
    return FileResponse(index)
