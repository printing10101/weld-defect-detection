"""检测不确定性估计（§5.5，M4）：模型无关、校准感知的代理估计。

为何是代理而非 MC Dropout：部署用 ONNX 推理路径无 dropout，本环境 CPU-only 无法跑集成；
规格书 §5.5 的 MC Dropout/集成需要模型侧支持（重新导出带 dropout 的权重），属后续工作。
此处用**可解释的启发式代理**综合三类信号，作为 ``Detection.uncertainty`` 与
``need_review`` 触发依据，且完全可单测、不依赖具体检测器：

1. 置信度余量（u_score）：score 越接近有效阈值越不可信（刚过线的候选最不可靠）。
2. 缺陷尺寸（u_size）：过小目标边界难定、易漏易误，不确定性高。
3. 类别安全关键度（u_class）：裂纹/未熔合/未焊透漏检代价高，置较高基线（0.6），
   确保重大缺陷即便高置信也进入人工复核（§5.5 安全优先）。

融合采用 **max 取大**（任一红 flags 触发即高不确定），比加权求和更直观且
不会出现「刚压线却只有中等不确定」的反直觉结果。输出 ∈ [0,1]。
下游（Nb47013Grader）以 ``detect.review_conf`` 阈值触发复核。
"""

from __future__ import annotations

import numpy as np

from backend.domain.dto import DefectClass

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
