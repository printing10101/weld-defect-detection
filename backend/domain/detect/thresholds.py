"""检测置信度阈值的工作模式解析（纯函数、离线可测）。

行业实践（对标商业评片软件的"高检出/高准确"双档）：同一模型在不同工序
关注点不同——初筛/复核前兜底宁可多报（漏检代价高），终审定级宁可少报
（误报浪费复核人力）。本模块把"一份 class_conf"扩展为"按模式缩放的三档"：

- ``balanced``       ：恒等缩放（与历史行为一致，scale=1.0）；
- ``recall_first``   ：逐类阈值 × recall_conf_scale（<1，整体放宽，优先召回）；
- ``precision_first``：逐类阈值 × precision_conf_scale（>1，整体收紧，压误报）。

缩放而非另维护两份逐类表：阈值的类间相对关系（裂纹最低、气孔最高）由
ADR-010 标定，两份手写表极易漂移失步（历史已发生 class_conf 缺 6 类的
静默收紧事故）；缩放系数是唯一的自由度，语义清晰、易审计。

分层约定：只做数学与输入校验，不接触配置/文件/模型。
"""

from __future__ import annotations

# 阈值裁剪边界：缩放后低于 lo 视为本底噪声闸门失效，高于 hi 等于关闭该类。
CONF_LO = 0.01
CONF_HI = 0.95

BALANCED = "balanced"
RECALL_FIRST = "recall_first"
PRECISION_FIRST = "precision_first"
MODES = (BALANCED, RECALL_FIRST, PRECISION_FIRST)


def mode_scale(mode: str, *, recall_scale: float, precision_scale: float) -> float:
    """模式 → class_conf 缩放系数；未知模式显式抛错（不静默回落 balanced）。"""
    if mode == BALANCED:
        return 1.0
    if mode == RECALL_FIRST:
        return recall_scale
    if mode == PRECISION_FIRST:
        return precision_scale
    raise ValueError(f"未知检测模式: {mode!r}（可选 {'/'.join(MODES)}）")


def resolve_class_conf(
    infer_conf: float,
    class_conf: dict[int, float],
    mode: str,
    *,
    recall_scale: float,
    precision_scale: float,
    lo: float = CONF_LO,
    hi: float = CONF_HI,
) -> tuple[float, dict[int, float]]:
    """按模式解析生效阈值：返回 (infer_conf_eff, class_conf_eff)。

    infer_conf 与逐类表同系数缩放（逐类缺项回落 infer_conf 的语义保持），
    缩放后统一裁剪到 [lo, hi]。输入映射不被修改（防御拷贝）。
    """
    scale = mode_scale(mode, recall_scale=recall_scale, precision_scale=precision_scale)
    conf_eff = _clip(infer_conf * scale, lo, hi)
    if not class_conf:
        return conf_eff, {}
    return conf_eff, {int(cid): _clip(v * scale, lo, hi) for cid, v in class_conf.items()}


def _clip(v: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, v)))
