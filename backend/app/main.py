"""FastAPI 应用入口（§T5）。

- 挂载 /api/v1，端点清单见 §14；
- CORS 仅允许本机来源（127.0.0.1）；
- 全局异常处理器：AppError -> 统一错误包（§13.4），M2 起挂载。
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from backend.app.dependencies import get_registry
from backend.app.routers import (
    active,
    audit,
    batch,
    detect,
    devices,
    evaluation,
    explain,
    health,
    judge,
    measure,
    metrics,
    models,
    preprocess,
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
from backend.infra.config import load_config
from backend.infra.fs import safe_resolve


def _envelope(code: str, message: str, detail=None) -> dict:
    return {"error": {"code": code, "message": message, "detail": detail}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 先统一日志；重装配（模型加载/DB 迁移，实测 ~2.5s，冷启动更久）移入后台线程，
    # 使 uvicorn 立即绑定端口——前端 /health 探测马上通过（status=starting），
    # 业务端点首个请求会经 get_registry() 阻塞等待装配完成（语义与原先一致）。
    _configure_logging()

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
        # P2-8：随主应用同进程拉起人工标注器（默认关；开启后主动学习闭环无需另开终端）。
        _start_annotator_if_enabled()
        _LOG.info("application startup complete (registry assembled)")

    threading.Thread(target=_init_registry, name="registry-init", daemon=True).start()
    yield
    # F11：应用退出时优雅关停批量线程池（等运行中任务结束），避免 worker 被硬杀
    try:
        from backend.app.dependencies import try_get_registry

        reg = try_get_registry()
        if reg is not None:
            reg.batch_manager.shutdown()
    except Exception as exc:  # noqa: BLE001 - 关停失败不应掩盖其它退出逻辑
        _LOG.warning("batch_manager shutdown skipped: %s", exc)


def _start_annotator_if_enabled() -> None:
    """标注器随主应用同进程启动（§12.2 主动学习闭环，P2-8）。

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
    """统一日志（关键路径可追溯，§13.5）。

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
    """构造 FastAPI 应用（§T5）。

    抽为工厂函数：测试可经 create_app() 获得全新实例以验证真实鉴权链路
    （不继承 conftest 注入的 admin 覆盖）；模块级 `app` 供现有测试/生产使用。
    """
    app = FastAPI(title="ScanDetection", version="0.1.0", lifespan=lifespan)

    # CORS 允许源由配置驱动（§13.6 配置中心化，P2）：默认覆盖 Tauri webview
    # （tauri://localhost）+ 本地开发源（127.0.0.1 / :5173）。桌面应用仅监听
    # 本机，风险可控。不再使用 "*"，否则任意外部网站均可跨源读取本机 API
    # （含审计链 / 报告）；部署新增前端源改 configs/default.yaml 即可，不改代码。
    cfg = load_config()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.server.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Operator-Name"],
    )

    # §7.5 / §13.9：安全响应头 + 基础限流（P2-9）。中间件按添加顺序执行，
    # CORS 在最内层（先于安全头/限流处理），保证 OPTIONS 预检同样获得安全头。
    from backend.app.security import RateLimitMiddleware, SecurityHeadersMiddleware

    app.add_middleware(RateLimitMiddleware)  # 外层：限流
    app.add_middleware(SecurityHeadersMiddleware)  # 内层：安全头

    # 可观测性：进程内指标中间件（最外层，采集所有 HTTP 请求计数/耗时）。
    from backend.infra.metrics import MetricsMiddleware, get_metrics

    get_metrics().enabled = cfg.observability.enable_metrics
    app.add_middleware(MetricsMiddleware)

    for router in (
        health.router,
        metrics.router,
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
    ):
        app.include_router(router, prefix="/api/v1")
    return app


app = create_app()


# ---------------------------------------------------------------------------
# 统一错误包（§13.4）：领域异常 → 对应 HTTP 状态；校验错误 → 422；其余 → 500（不透传细节）。
# ---------------------------------------------------------------------------
@app.exception_handler(AppError)
async def _app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(_envelope(exc.code, str(exc)), status_code=exc.http_status)


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
