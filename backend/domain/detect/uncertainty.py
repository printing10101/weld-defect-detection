"""检测不确定性估计：模型无关、校准感知的两级估计。

两级设计（按可用信息递进，均为纯函数、可离线单测）：

1. **单视角启发式**（``estimate_uncertainty``）：综合置信度余量 / 缺陷尺寸 /
   类别安全关键度三类信号。为何是代理而非 MC Dropout：部署用 ONNX 推理路径
   无 dropout，本环境 CPU-only 无法跑深度集成；完整的 MC Dropout/Deep
   Ensemble 需要模型侧支持（重新导出带 dropout 的权重 / 多权重集成），属
   后续工作。
2. **多视角增广集成**（``estimate_ensemble_uncertainty``）：测试时增广（TTA
   多尺度）近似 MC 集成——真缺陷在扰动视角下应被稳定检出且得分稳定；仅在
   个别视角出现或得分剧烈波动的候选更可能是噪声。``infer_tta`` 开启时，
   检出会携带该信号并与单视角信号 max 融合，``need_review`` 触发随之更
   贴近真实模型不确定性。

融合均采用 **max 取大**（任一红 flags 触发即高不确定），比加权求和更直观且
不会出现「刚压线却只有中等不确定」的反直觉结果。输出 ∈ [0,1]。
下游（Nb47013Grader）以 ``detect.review_conf`` 阈值触发复核。
"""

from __future__ import annotations

import numpy as np

from backend.domain.dto import BBox, DefectClass

# 安全关键类别（漏检代价远高于误检）→ 不确定性基线抬升
_SAFETY_CRITICAL = {
    DefectClass.CRACK,
    DefectClass.LACK_OF_FUSION,
    DefectClass.INCOMPLETE_PENETRATION,
}

# 安全关键类别的恒定不确定性基线（确保进入人工复核）
_SAFETY_BASELINE = 0.6

# 边界尺寸（px）：小于此尺寸判定困难度陡增
_TINY_AREA_PX = 50.0
_SMALL_AREA_PX = 400.0


def _is_safety_critical(class_id: int) -> bool:
    try:
        return DefectClass(class_id) in _SAFETY_CRITICAL
    except ValueError:
        return False


def estimate_uncertainty(
    score: float,
    eff_conf: float,
    class_id: int,
    area_px: float,
) -> float:
    """估计单条检测的不确定性（0=确定，1=高度不确定）。

    score    : 检测器输出置信度（∈[0,1]）
    eff_conf : 该类的有效置信度阈值（全局 conf 或逐类阈值）
    class_id : DefectClass.value
    area_px  : 检测框面积（像素）
    """
    score = float(np.clip(score, 0.0, 1.0))
    eff_conf = float(eff_conf)

    # 1) 置信度余量：score=eff_conf → 1；score=1 → 0
    span = max(1e-3, 1.0 - eff_conf)
    u_score = float(np.clip(1.0 - (score - eff_conf) / span, 0.0, 1.0))

    # 2) 尺寸：过小 → 高
    if area_px <= 0 or area_px < _TINY_AREA_PX:
        u_size = 1.0
    elif area_px < _SMALL_AREA_PX:
        u_size = float((_SMALL_AREA_PX - area_px) / (_SMALL_AREA_PX - _TINY_AREA_PX))
    else:
        u_size = 0.0
    u_size = float(np.clip(u_size, 0.0, 1.0))

    # 3) 类别安全关键度（恒定基线）
    u_class = float(_SAFETY_BASELINE) if _is_safety_critical(class_id) else 0.0

    u = max(u_score, u_size, u_class)
    return round(float(np.clip(u, 0.0, 1.0)), 4)


# ---------------------------------------------------------------------------
# 多视角增广集成不确定性（infer_tta 路径）
# ---------------------------------------------------------------------------

# 跨视角得分标准差归一常数：std 达到该值即视为满分不确定。检测置信度在稳定
# 视角间的波动通常 <0.05，0.15 相当于"显著波动"的保守上界（宁可少报不虚报）。
_SPREAD_NORM = 0.15


def estimate_ensemble_uncertainty(
    vote_frac: float,
    score_std: float,
    *,
    spread_norm: float = _SPREAD_NORM,
) -> float:
    """增广集成不确定性：多视角下检测行为的分歧度（0=各视角一致，1=严重分歧）。

    vote_frac : 该缺陷被检出的视角比例 ∈ [0,1]（IoU 匹配 + 同类才算检出）；
    score_std : 各视角置信度的标准差（缺失视角不计入 std）。

    真缺陷在不同尺度下应被稳定检出且得分稳定；"只在单个视角冒出"或"得分
    随尺度剧烈波动"的候选更可能是噪声或边界样本——这是 Deep Ensemble 认知
    不确定性的测试时增广近似，不需要 dropout/多权重即可获得模型侧信号。
    """
    u_votes = 1.0 - float(np.clip(vote_frac, 0.0, 1.0))
    u_spread = float(np.clip(score_std / max(spread_norm, 1e-6), 0.0, 1.0))
    return round(max(u_votes, u_spread), 4)


def _box_iou(a: BBox, b: BBox) -> float:
    ix1 = max(a.x, b.x)
    iy1 = max(a.y, b.y)
    ix2 = min(a.x + a.w, b.x + b.w)
    iy2 = min(a.y + a.h, b.y + b.h)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 1e-6 else 0.0


def ensemble_uncertainty_stats(
    kept: list[tuple[BBox, int]],
    views: list[list[tuple[BBox, int, float]]],
    match_iou: float,
) -> list[tuple[float, float]]:
    """对每个保留检出统计跨视角 (检出比例, 得分标准差)。

    kept  : 跨视角 NMS 后保留的 [(bbox, class_id)]；
    views : 各视角的检出 [(bbox, class_id, score)]（坐标已还原到原图系）；
    match_iou : 跨视角判定"同一缺陷"的 IoU 阈值（同类 + IoU 达标）。

    返回与 kept 等长的 [(vote_frac, score_std)]。视角内多条匹配取最高分
    （同一视角的重复检出已被 NMS 消除，此处仅防御性兜底）。
    """
    n_views = max(1, len(views))
    out: list[tuple[float, float]] = []
    for box, cid in kept:
        matched_scores: list[float] = []
        for view in views:
            best = -1.0
            for vb, vcid, vscore in view:
                if vcid != cid:
                    continue
                i = _box_iou(box, vb)
                if i >= match_iou and vscore > best:
                    best = vscore
            if best >= 0.0:
                matched_scores.append(best)
        vote_frac = len(matched_scores) / n_views
        std = float(np.std(matched_scores)) if len(matched_scores) > 1 else 0.0
        out.append((vote_frac, std))
    return out
