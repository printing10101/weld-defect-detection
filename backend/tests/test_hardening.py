"""基础设施加固回归测试（日志/异常包/转换工具）。

覆盖本轮优化的关键路径：
- 全局未捕获异常 → 统一 500 错误包（不泄露内部细节）；
- SWRD 转换器坐标展平（修复 F821 未定义变量 p 的回归防护）。
"""

from __future__ import annotations

import asyncio
import json

from backend.app.main import _unhandled_handler
from backend.training.swrd_converter import _flatten_points


def test_unhandled_handler_returns_internal_envelope():
    """任意未捕获异常必须返回统一 INTERNAL 错误包，而非裸 traceback。"""
    resp = asyncio.run(_unhandled_handler(None, RuntimeError("boom")))
    assert resp.status_code == 500
    body = json.loads(resp.body)
    assert body["error"]["code"] == "INTERNAL"
    assert body["error"]["message"] == "服务器内部错误"
    assert "detail" in body["error"]


def test_flatten_paired_points():
    """成对坐标 [[x,y],...] 应展平为 [x1,y1,x2,y2,...]。"""
    pts = [[10, 20], [30, 40], [50, 60]]
    assert _flatten_points(pts) == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]


def test_flatten_flat_points():
    """展平坐标 [x,y,x,y,...] 原样返回 float 列表。"""
    pts = [10, 20, 30, 40]
    assert _flatten_points(pts) == [10.0, 20.0, 30.0, 40.0]


def test_flatten_empty():
    assert _flatten_points([]) == []
