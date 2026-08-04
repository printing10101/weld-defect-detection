"""FastAPI 应用入口（§T5）。

- 挂载 /api/v1，端点清单见 §14；
- CORS 仅允许本机来源（127.0.0.1）；
- 全局异常处理器：AppError -> 统一错误包（§13.4），M2 起挂载。
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routers import (
    audit,
    batch,
    detect,
    explain,
    health,
    judge,
    models,
    preprocess,
    records,
    report,
    review,
    verify,
)

app = FastAPI(title="ScanDetection", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in (
    health.router,
    verify.router,
    preprocess.router,
    detect.router,
    judge.router,
    batch.router,
    review.router,
    explain.router,
    report.router,
    records.router,
    models.router,
    audit.router,
):
    app.include_router(router, prefix="/api/v1")
