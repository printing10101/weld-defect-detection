"""Sinkhorn 软匹配损失（SinkLoss）——面向含噪标签的稠密回归训练损失。

来源与动机
----------
方法迁移自 Marigold V2（arXiv 2609.08084，SIGGRAPH Asia 2026，华为 Bayer Lab +
EPFL + 博洛尼亚大学）的 **SinkLoss**：把预测图与真值图切成互不重叠的小块，
块内用 Sinkhorn-Knopp 最优传输做**软的一对一指派**——只要求块内"预测值集合"
与"真值集合"匹配（允许置换），**不强制逐像素对位**。

为什么本系统需要它
------------------
本项目的监督信号存在结构性噪声，逐像素 L1/L2 会被噪声"绑架"：
1. X 光缺陷边界半透明/模糊，两名评片员对同一缺陷轮廓常不重合；
2. 稀有类（裂纹/未熔合/未焊透）样本极少，预标注（prelabel）定位精度有限；
3. 逐像素监督会强迫模型复现上述噪声，产生"飞点"并损伤边缘锐度。

SinkLoss 只约束块内分布，对块内位置的随机抖动不敏感，因而：
- 抗预标注边界噪声（本系统 §5.3 掩膜精修 / §5.5 伪标签闭环的直接痛点）；
- 天然适配"密集气孔群"这类需按集合而非逐像素计数的场景（§16 场景 4）。

接口
----
- ``sinkhorn_log``               : log 域数值稳定的 Sinkhorn-Knopp，返回传输计划 P；
- ``sinkhorn_transport_cost``    : 单块的传输代价（标量，加权平均代价）；
- ``tile_sinkhorn_loss``         : 把稠密图切块后取平均，训练损失主干；
- ``torch_sinkhorn_cost``        : torch 张量版（可反传），torch 懒加载。

约定：输入为稠密图（如缺陷密度图 / 掩膜置信图），单通道 shape (H, W)。
无效像素经 ``invalid_gt`` 标记后，从该块的 pred/gt 两向量中**一并剔除**
（无效 = 无监督目标，不参与匹配）。

**输入量纲契约（实测，务必遵守）**
--------------------------------------------------------------------------------
传输代价的代价矩阵是 ``C=(pred_i - gt_j)^2``。若输入被压到 [0, 1]，平方项被
压到 1 以下，而熵正则又给出一份**与量纲无关的软匹配下限**，结果是损失几乎平坦、
几乎不给梯度。本机实测（64×64 密度图，tile=8，eps=0.05）：

    ===========  ============  =============  ==============
    输入尺度      正确预测代价   全零预测代价    对比度
    ===========  ============  =============  ==============
    1.0           0.0015        0.0100           6.4 ×
    10.0          0.0021        0.9971         466.5 ×
    100.0         0.0014        99.7077      71945.8 ×
    ===========  ============  =============  ==============

即：**先把目标缩放到量级 10~100（或按物理量纲，如"密度 × 预计最大缺陷数"），
再送进本损失**；等价手段是同步减小 ``eps``。用
:func:`backend.training.density_aux.to_sinkhorn_scale` 完成缩放。

本模块**只依赖 numpy**（torch 为可选懒加载），可在 CI 无 ML 环境单测。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sinkhorn_log",
    "sinkhorn_transport_cost",
    "tile_sinkhorn_loss",
    "torch_sinkhorn_cost",
]


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    """数值稳定的 log-sum-exp（numpy，沿指定轴归约，结果降维）。"""
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)  # 全 -inf 行兜底，避免 nan
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))


def sinkhorn_log(
    cost: np.ndarray,
    *,
    eps: float = 0.05,
    n_iter: int = 20,
) -> np.ndarray:
    """log 域 Sinkhorn-Knopp：求熵正则最优传输计划 P。

    参数
    ----
    cost   : (n, m) 代价矩阵，代价越大越不倾向匹配；
    eps    : 熵正则系数（越小越接近硬指派，越大越平滑；须 > 0）；
    n_iter : Sinkhorn 迭代次数（Marigold V2 实际用 5 次；本默认 20 更精确）。

    返回
    ----
    (n, m) 传输计划 P：行和 = 1/n、列和 = 1/m（均匀边缘，总质量 1）。
    全程在 log 域计算，避免 exp(-C/eps) 在小 eps 下下溢。
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError(f"cost 须为二维 (n, m)，实际 shape={cost.shape}")
    if eps <= 0:
        raise ValueError(f"eps 须为正，实际 {eps}")
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.zeros_like(cost)

    log_k = -cost / float(eps)
    log_mu = np.full(n, -np.log(n))
    log_nu = np.full(m, -np.log(m))
    log_u = np.zeros(n)
    log_v = np.zeros(m)
    for _ in range(int(n_iter)):
        log_u = log_mu - _logsumexp(log_k + log_v[None, :], axis=1)
        log_v = log_nu - _logsumexp(log_k + log_u[:, None], axis=0)
    log_p = log_k + log_u[:, None] + log_v[None, :]
    return np.exp(log_p)


def sinkhorn_transport_cost(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    eps: float = 0.05,
    n_iter: int = 20,
) -> float:
    """单组（单块）向量的 Sinkhorn 传输代价：sum(P * C)，C=(pred_i - gt_j)^2。

    对"预测值集合 vs 真值集合"做熵正则最优传输，返回加权平均代价（0 = 集合
    完全一致）。这是 :func:`tile_sinkhorn_loss` 的单块版，便于单测与手工调参。
    """
    cost = _transport_cost(pred, gt, eps, n_iter)
    return 0.0 if cost is None else cost


def _transport_cost(
    pred: np.ndarray,
    gt: np.ndarray,
    eps: float,
    n_iter: int,
) -> float | None:
    """一组（单块）向量的 Sinkhorn 传输代价：sum(P * C)，C=(pred_i - gt_j)^2。

    ``pred`` / ``gt`` 须等长且已剔除无效位置；长度 < 1 时返回 None（该块跳过）。
    """
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    gt = np.asarray(gt, dtype=np.float64).reshape(-1)
    if pred.size != gt.size:
        raise ValueError(f"pred/gt 长度不一致：{pred.size} vs {gt.size}")
    if pred.size == 0:
        return None
    cost = (pred[:, None] - gt[None, :]) ** 2
    plan = sinkhorn_log(cost, eps=eps, n_iter=n_iter)
    return float(np.sum(plan * cost))


def tile_sinkhorn_loss(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    tile: int = 5,
    eps: float = 0.05,
    n_iter: int = 20,
    invalid_gt: np.ndarray | None = None,
) -> float:
    """分块 Sinkhorn 软匹配损失：把稠密图切块，块内做软匹配，取块平均。

    参数
    ----
    pred, gt   : (H, W) 归一化稠密图（预测 / 真值）；
    tile       : 块边长（Marigold V2 用 5，即 5×5 块）；
    eps, n_iter: 透传给 :func:`sinkhorn_log`；
    invalid_gt : 可选 (H, W) 布尔图，True 表示该真值像素无效。无效位置
                 从该块的 pred/gt 两向量中**一并剔除**（无效 = 无监督目标），
                 避免 Sinkhorn 的边缘约束把质量强行塞进无意义的列。

    说明：为整除切块，会裁掉不满足 ``tile`` 倍数的右/下边缘像素（不足一块的
    余数被忽略——对损失统计量影响可忽略，且避免半块引入边缘偏差）。
    返回所有"含有效像素"块的传输代价均值；若无此类块则返回 0.0。
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape 不一致：{pred.shape} vs {gt.shape}")
    if pred.ndim != 2:
        raise ValueError(f"pred/gt 须为二维 (H, W)，实际 shape={pred.shape}")
    tile = int(tile)
    if tile <= 0:
        raise ValueError(f"tile 须为正整数，实际 {tile}")

    h, w = pred.shape
    n_h, n_w = h // tile, w // tile
    if n_h == 0 or n_w == 0:
        return 0.0
    hh, ww = n_h * tile, n_w * tile
    p = pred[:hh, :ww]
    g = gt[:hh, :ww]
    inv = np.asarray(invalid_gt, dtype=bool)[:hh, :ww] if invalid_gt is not None else None

    costs: list[float] = []
    for i in range(n_h):
        for j in range(n_w):
            sl = (slice(i * tile, (i + 1) * tile), slice(j * tile, (j + 1) * tile))
            bp = p[sl].reshape(-1)
            bg = g[sl].reshape(-1)
            if inv is not None:
                keep = ~inv[sl].reshape(-1)
                bp, bg = bp[keep], bg[keep]
            c = _transport_cost(bp, bg, eps, n_iter)
            if c is not None:
                costs.append(c)
    return float(np.mean(costs)) if costs else 0.0


def torch_sinkhorn_cost(
    pred,
    gt,
    *,
    eps: float = 0.05,
    n_iter: int = 20,
):
    """torch 张量版 Sinkhorn 传输代价（可反传），供训练损失使用。

    ``pred`` / ``gt`` 形状 ``(L,)`` 或 ``(..., L)``：**任意前导维度会被展平为 T**，
    即 ``(B, T, L)``（批量 × 分块）与 ``(T, L)`` 等价，最终对所有块统一取平均。
    返回标量。torch 懒加载——无 torch 环境调用时才 ImportError，
    不影响本模块在 CI 无 ML 环境下的 numpy 单测。
    """
    import torch

    p = torch.as_tensor(pred, dtype=torch.float32)
    g = torch.as_tensor(gt, dtype=torch.float32)
    if p.dim() == 1:
        p = p.unsqueeze(0)
        g = g.unsqueeze(0)
    elif p.dim() > 2:
        # (B, T, L) / (..., L) -> (B*T, L)：分块维度统一展平，语义不变
        p = p.reshape(-1, p.shape[-1])
        g = g.reshape(-1, g.shape[-1])
    if p.shape != g.shape:
        raise ValueError(f"pred/gt shape 不一致：{tuple(p.shape)} vs {tuple(g.shape)}")

    # (T, L, 1) - (T, 1, L) -> (T, L, L)
    cost = (p.unsqueeze(-1) - g.unsqueeze(-2)) ** 2
    log_k = -cost / float(eps)
    t, n, m = log_k.shape
    log_u = torch.zeros(t, n, dtype=log_k.dtype, device=log_k.device)
    log_v = torch.zeros(t, m, dtype=log_k.dtype, device=log_k.device)
    log_mu = torch.full((t, n), float(-np.log(n)), dtype=log_k.dtype, device=log_k.device)
    log_nu = torch.full((t, m), float(-np.log(m)), dtype=log_k.dtype, device=log_k.device)
    for _ in range(int(n_iter)):
        log_u = log_mu - torch.logsumexp(log_k + log_v.unsqueeze(-2), dim=-1)
        log_v = log_nu - torch.logsumexp(log_k + log_u.unsqueeze(-1), dim=-2)
    log_p = log_k + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)
    plan = torch.exp(log_p)
    return (plan * cost).sum(dim=(-1, -2)).mean()
