"""前后端契约对账（A1，防契约漂移）。

手写前端类型 src/src/types/api.ts 是本仓库唯一的前端契约真相。后端同名
response_model（Pydantic）新增/改名/删除字段时，若前端未同步，本测试即红，
强制在合并前补齐 api.ts，避免类型静默漂移 / 运行时 undefined。

方向说明：断言「后端响应模型字段 ⊆ 前端接口字段」——后端加了字段而前端没跟，
视为前端落后，必须补；前端允许保留额外字段（与后端非 1:1 的本地视图字段）。
不在本轮强制反向（前端多字段可能是展示派生值，合法）。
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_API_TS = _ROOT / "src" / "src" / "types" / "api.ts"

# 前端接口名 -> "module.ClassName"（后端 Pydantic 响应模型，与 api.ts 同名映射）
CONTRACTS: dict[str, str] = {
    "ReportOut": "backend.app.routers.report:ReportOut",
    "ReportDetectionsOut": "backend.app.routers.report:ReportDetectionsOut",
    "ReviewOut": "backend.app.routers.review:ReviewOut",
    "BatchSubmitOut": "backend.app.routers.batch:BatchSubmitOut",
    "BatchTaskOut": "backend.app.routers.batch:BatchTaskOut",
    "BatchStatusOut": "backend.app.routers.batch:BatchStatusOut",
    "BatchRetryOut": "backend.app.routers.batch:BatchRetryOut",
    "BatchSummaryOut": "backend.app.routers.batch:BatchSummaryOut",
    "CalibrationOut": "backend.app.routers.devices:CalibrationOut",
    "DeviceOut": "backend.app.routers.devices:DeviceOut",
    "ActiveSampleOut": "backend.app.routers.active:SampleOut",
    "ActiveExportOut": "backend.app.routers.active:ExportOut",
    "ActivePoolOut": "backend.app.routers.active:PoolOut",
    "VerifyOut": "backend.app.routers.verify:VerifyOut",
    "RecordsResponse": "backend.app.routers.records:RecordsResponse",
}

_FIELD_RE = re.compile(r"(?:readonly\s+)?([A-Za-z_][\w]*)\??\s*:")


def _ts_interface_fields(text: str, name: str) -> set[str] | None:
    """提取 api.ts 中 ``export interface <name>`` 的顶层字段名集合；不存在返回 None。"""
    match = re.search(rf"export interface {name}(?!\w)(?:\s+extends\s+\w+)?\s*\{{", text)
    if not match:
        return None
    # 从接口体的 { 开始，用大括号配对找到闭合 }，取整块
    i = match.end()  # 指向 '{' 的下一位；开括号已消耗，故 depth 从 1 起
    depth = 1
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    block = text[match.end() - 1 : i + 1]
    fields: set[str] = set()
    for raw_line in block.splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line or line.startswith(("*", "/*")):
            continue
        m = _FIELD_RE.match(line)
        if m:
            fields.add(m.group(1))
    return fields


def _backend_model_fields(ref: str) -> set[str]:
    module, _, cls = ref.rpartition(":")
    model = getattr(importlib.import_module(module), cls)
    return set(model.model_fields)


def test_api_ts_contract_file_exists() -> None:
    assert _API_TS.is_file(), f"找不到前端类型文件: {_API_TS}"


def test_frontend_fields_cover_backend_response_models() -> None:
    text = _API_TS.read_text(encoding="utf-8")
    failures: list[str] = []
    for iface, ref in CONTRACTS.items():
        front = _ts_interface_fields(text, iface)
        if front is None:
            failures.append(f"api.ts 缺少 export interface {iface}")
            continue
        missing = sorted(_backend_model_fields(ref) - front)
        if missing:
            failures.append(f"{iface} 落后于后端模型（缺字段: {missing}）")
    assert not failures, "前后端契约漂移:\n  " + "\n  ".join(failures)


def test_contract_mapping_names_are_resolvable() -> None:
    """CONTRACTS 里的 module:Class 都能导入，防止映射写错名而静默失效。"""
    for ref in CONTRACTS.values():
        module, _, cls = ref.rpartition(":")
        obj = getattr(importlib.import_module(module), cls, None)
        assert obj is not None, f"后端映射不存在: {ref}"
