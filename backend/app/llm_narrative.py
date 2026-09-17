"""本地大模型评片结论（应用层编排）。

把「已落库的算法事实」→「提示词」→「本地大模型」→「可读结论」串起来。
分层：提示词/清理在 ``domain/narrative.py``（纯函数），HTTP 在
``infra/llm_client.py``，本模块只做编排与事实装配（app 层可依赖两者）。

设计要点：
- **不阻断**：任何失败（未启用/端点不可达/超时/空回复）都以 ``status`` 表达，
  评片主链路从不因大模型不可用而失败。
- **不落库**：结论按需实时生成（模型输出带随机性，落库会把一次性输出固化成
  "记录"）。审计留痕由调用端点负责。
- **不越权**：级别/合格判定由算法与持证人负责，提示词与 ``clean_output``
  双重禁止模型输出级别。
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass

from backend.domain.narrative import (
    ZERO_TOLERANCE_IDS,
    NarrativeFacts,
    build_messages,
    class_name_zh,
    clean_output,
)
from backend.infra.llm_client import LlmClient

# 结论状态（前端据此决定展示文本 / 原因 / 隐藏）
STATUS_OK = "ok"
STATUS_DISABLED = "disabled"  # 配置关闭：llm.enabled=false
STATUS_UNAVAILABLE = "unavailable"  # 未启用或端点不可达
STATUS_FAILED = "failed"  # 端点可达但调用失败
STATUS_EMPTY_MATERIAL = "empty"  # 无报告可依据（report/image 不存在）

NARRATIVE_DISCLAIMER = (
    "注意：本段结论由本机部署的本地大模型基于算法检出事实自动撰述，仅供辅助参考与"
    "记录整理；不构成缺陷等级判定，也不构成合格/不合格结论。最终评定须由持证人/"
    "责任工程师依授权标准原文复核并签核。"
)


@dataclass(frozen=True)
class NarrativeOutcome:
    """一次评片结论生成的结果（永不抛错）。"""

    status: str
    text: str = ""
    model: str = ""
    reason: str = ""
    elapsed_ms: int = 0
    disclaimer: str = NARRATIVE_DISCLAIMER


def _llm_settings(reg) -> tuple[bool, str, str | None]:
    """从注册表/配置推导 ``(enabled, endpoint, api_key)``。

    端点优先取已装配的 ``llm_manager``（它已消化"用户选中的模型"：
    托管实例的 host:port 或 external 端点），无 manager 时回落配置。
    密钥只从环境变量名读取（Key 不落配置/不落盘）。
    """
    cfg = reg.config.llm
    if not bool(getattr(cfg, "enabled", False)):
        return False, "", None
    endpoint = ""
    mgr = getattr(reg, "llm_manager", None)
    if mgr is not None:
        endpoint = str(getattr(mgr, "endpoint", "") or "")
    if not endpoint:
        endpoint = str(getattr(cfg, "external_endpoint", "") or "")
    api_key: str | None = None
    env_name = str(getattr(cfg, "api_key_env", "") or "").strip()
    if env_name:
        api_key = os.environ.get(env_name) or None
    return True, endpoint, api_key


def facts_from_report(reg, report_id: str) -> NarrativeFacts | None:
    """按 report_id 装配事实；报告或影像不存在返回 None。"""
    rep = reg.repository.get_report(report_id)
    if rep is None:
        return None
    image_id = rep.get("image_id")
    if not image_id:
        return None
    img = reg.repository.get_image(image_id)
    if img is None:
        return None

    defects = list(img.get("defects") or [])
    counter: Counter[int] = Counter()
    best_by_class: dict[int, tuple[float, float]] = {}
    for d in defects:
        try:
            cid = int(d.get("class_id", -1))
        except (TypeError, ValueError):
            continue
        counter[cid] += 1
        length = float(d.get("length_mm") or 0.0)
        width = float(d.get("width_mm") or 0.0)
        prev = best_by_class.get(cid)
        if prev is None or length > prev[0]:
            best_by_class[cid] = (length, width)

    calibrated = bool(img.get("pixel_spacing_mm"))
    largest = tuple(
        (class_name_zh(cid), round(dim[0], 2), round(dim[1], 2))
        for cid, dim in sorted(best_by_class.items(), key=lambda kv: -kv[1][0])[:3]
        if dim[0] > 0
    )

    warnings: list[str] = []
    if img.get("evaluable") is False:
        warnings.append("底片判定为不可评（黑度/IQI/质量门禁未通过）")
    if img.get("density_ok") is False:
        density = img.get("density")
        warnings.append(f"黑度未达标（实测 {density}）" if density is not None else "黑度未达标")
    if img.get("iqi_pass") is False:
        warnings.append("像质计（IQI）未达要求")
    if img.get("quality_pass") is False:
        warnings.append("底片质量门禁未通过")
    if img.get("stamp_need_review"):
        warnings.append("底片印字需人工确认")
    if not calibrated:
        warnings.append("缺少像素标定，无物理尺寸输出")

    t = img.get("base_metal_thickness_mm")
    return NarrativeFacts(
        defect_count=len(defects),
        class_stats=tuple((class_name_zh(cid), n) for cid, n in counter.most_common()),
        largest=largest,
        zero_tolerance=tuple(
            class_name_zh(cid) for cid in counter if int(cid) in ZERO_TOLERANCE_IDS
        ),
        joint_level=rep.get("joint_level") or img.get("joint_level"),
        grade_basis=tuple(rep.get("basis") or ()),
        quality_warnings=tuple(warnings),
        need_review=bool(img.get("need_review")),
        calibrated=calibrated,
        base_metal_thickness_mm=float(t) if t else None,
        standard_id=str(img.get("standard_id") or "NB/T47013.2-2015"),
    )


def generate(reg, facts: NarrativeFacts) -> NarrativeOutcome:
    """调用本地大模型生成评片结论（失败降级，不抛错）。"""
    enabled, endpoint, api_key = _llm_settings(reg)
    if not enabled:
        return NarrativeOutcome(
            status=STATUS_DISABLED, reason="本地大模型未启用（config.llm.enabled=false）"
        )
    if not endpoint:
        return NarrativeOutcome(status=STATUS_UNAVAILABLE, reason="未配置本地大模型端点")

    client = LlmClient(endpoint, api_key=api_key)
    if not client.available():
        return NarrativeOutcome(
            status=STATUS_UNAVAILABLE,
            reason=f"本地大模型端点不可用：{endpoint}（启动本机服务后重试）",
        )
    system, user = build_messages(facts)
    result = client.chat(system, user)
    if not result.ok:
        return NarrativeOutcome(
            status=STATUS_FAILED,
            reason=result.reason,
            model=result.model,
            elapsed_ms=result.elapsed_ms,
        )
    text = clean_output(result.text)
    if not text:
        # 模型输出被清理后为空：提示词越权被硬拦截，如实说明而非展示空内容
        return NarrativeOutcome(
            status=STATUS_FAILED,
            reason="模型输出未通过合规清理（疑似越权给出等级/合格判定，已拦截）",
            model=result.model,
            elapsed_ms=result.elapsed_ms,
        )
    return NarrativeOutcome(
        status=STATUS_OK,
        text=text,
        model=result.model,
        elapsed_ms=result.elapsed_ms,
    )


__all__ = [
    "NARRATIVE_DISCLAIMER",
    "STATUS_DISABLED",
    "STATUS_EMPTY_MATERIAL",
    "STATUS_FAILED",
    "STATUS_OK",
    "STATUS_UNAVAILABLE",
    "NarrativeOutcome",
    "facts_from_report",
    "generate",
]
