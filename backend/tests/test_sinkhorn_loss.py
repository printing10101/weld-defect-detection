"""SinkLoss 单测（迁移自 Marigold V2 的 Sinkhorn 软匹配损失）。

验证要点（按重要性）：
1. **置换不变性**——块内打乱像素不影响损失，这是 SinkLoss 与逐像素 L1/L2 的
   唯一本质区别，也是它抗标注边界噪声的根源；
2. 同集合自匹配代价最小（趋近 0），异集合代价更大；
3. eps 越小越接近硬指派（自匹配代价越小）；
4. 掩膜：无效真值位置从该块两向量中剔除；全无效 → 0；
5. 数值稳定性（大数值不溢出）与入参校验；
6. torch 版与 numpy 版一致（torch 缺失时跳过，CI 无 ML 环境不受影响）。

纯 numpy 运行，不依赖 torch / onnxruntime。
"""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

from backend.training.losses import (
    sinkhorn_log,
    sinkhorn_transport_cost,
    tile_sinkhorn_loss,
    torch_sinkhorn_cost,
)


def _rng() -> np.random.Generator:
    return np.random.default_rng(20260914)


def test_sinkhorn_plan_is_doubly_stochastic() -> None:
    """传输计划须满足均匀边缘约束（行和=1/n，列和=1/m）。"""
    rng = _rng()
    cost = rng.random((6, 6))
    plan = sinkhorn_log(cost, eps=0.1, n_iter=200)
    assert plan.shape == (6, 6)
    assert np.all(plan >= 0.0)
    np.testing.assert_allclose(plan.sum(axis=1), np.full(6, 1 / 6), atol=1e-6)
    np.testing.assert_allclose(plan.sum(axis=0), np.full(6, 1 / 6), atol=1e-6)


def test_permutation_invariance_core_property() -> None:
    """块内置换不改变损失——软匹配只看值的集合，不看位置。"""
    rng = _rng()
    gt = rng.random((5, 5))
    pred = np.clip(gt + rng.normal(0, 0.15, size=(5, 5)), 0.0, 1.0)
    perm = rng.permutation(25)
    pred_perm = pred.reshape(-1)[perm].reshape(5, 5)

    base = sinkhorn_transport_cost(pred.reshape(-1), gt.reshape(-1), eps=0.02, n_iter=50)
    shuffled = sinkhorn_transport_cost(pred_perm.reshape(-1), gt.reshape(-1), eps=0.02, n_iter=50)
    assert shuffled == approx(base, abs=1e-6)


def test_pixelwise_mse_breaks_but_sinkhorn_does_not() -> None:
    """对照实验：同一置换，逐像素 MSE 变了，SinkLoss 不变。"""
    rng = _rng()
    gt = rng.random((5, 5))
    pred = rng.random((5, 5))
    perm = rng.permutation(25)
    pred_perm = pred.reshape(-1)[perm].reshape(5, 5)

    mse_base = float(np.mean((pred - gt) ** 2))
    mse_perm = float(np.mean((pred_perm - gt) ** 2))
    assert mse_perm != approx(mse_base, abs=1e-6)  # 对位置敏感

    sk_base = sinkhorn_transport_cost(pred.reshape(-1), gt.reshape(-1))
    sk_perm = sinkhorn_transport_cost(pred_perm.reshape(-1), gt.reshape(-1))
    assert sk_perm == approx(sk_base, abs=1e-6)  # 只看集合


def test_self_matching_is_minimal_and_small() -> None:
    """同一集合自匹配代价最小且很小；整体偏移后代价更大。"""
    rng = _rng()
    gt = rng.random(36)
    self_cost = sinkhorn_transport_cost(gt, gt, eps=0.01, n_iter=50)
    shifted_cost = sinkhorn_transport_cost(np.clip(gt + 0.3, 0.0, 1.0), gt, eps=0.01, n_iter=50)
    assert self_cost < shifted_cost
    assert self_cost < 0.5 * shifted_cost


def test_smaller_eps_shrinks_self_cost() -> None:
    """eps 越小越接近硬指派：自匹配代价随 eps 减小而下降。"""
    rng = _rng()
    gt = rng.random(36)
    loose = sinkhorn_transport_cost(gt, gt, eps=0.2, n_iter=80)
    tight = sinkhorn_transport_cost(gt, gt, eps=0.01, n_iter=80)
    assert tight < loose


def test_masked_positions_are_excluded_from_block() -> None:
    """无效真值位置须从该块的 pred/gt 两向量中剔除，等价于只对有效子集算损失。"""
    rng = _rng()
    p = rng.random((5, 5))
    g = rng.random((5, 5))
    inv = np.zeros((5, 5), dtype=bool)
    inv[0, :] = True  # 首行全无效

    got = tile_sinkhorn_loss(p, g, tile=5, invalid_gt=inv)
    keep = ~inv.reshape(-1)
    expected = sinkhorn_transport_cost(p.reshape(-1)[keep], g.reshape(-1)[keep])
    assert got == approx(expected, rel=1e-9, abs=1e-12)


def test_all_invalid_returns_zero() -> None:
    """整块无效 → 该块跳过；无有效块时损失为 0（不污染均值）。"""
    rng = _rng()
    p = rng.random((5, 5))
    g = rng.random((5, 5))
    inv = np.ones((5, 5), dtype=bool)
    assert tile_sinkhorn_loss(p, g, tile=5, invalid_gt=inv) == 0.0


def test_partial_tiles_are_cropped() -> None:
    """非整除尺寸只使用完整块（7×7 + tile 5 → 仅左上 5×5 区参与）。"""
    rng = _rng()
    p = rng.random((7, 7))
    g = rng.random((7, 7))
    got = tile_sinkhorn_loss(p, g, tile=5)
    expected = sinkhorn_transport_cost(p[:5, :5].reshape(-1), g[:5, :5].reshape(-1))
    assert got == approx(expected, rel=1e-9, abs=1e-12)


def test_numerical_stability_large_magnitudes() -> None:
    """大数值（代价 ~1e5）下 log 域实现仍应有限、非负、无 nan。"""
    rng = _rng()
    p = rng.random((5, 5)) * 1000.0
    g = rng.random((5, 5)) * 1000.0
    v = tile_sinkhorn_loss(p, g, tile=5, eps=0.5, n_iter=80)
    assert np.isfinite(v)
    assert v >= 0.0


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        sinkhorn_log(np.zeros((3,)), eps=0.1)  # 非二维
    with pytest.raises(ValueError):
        sinkhorn_log(np.zeros((3, 3)), eps=0.0)  # eps 非正
    with pytest.raises(ValueError):
        tile_sinkhorn_loss(np.zeros((5, 5)), np.zeros((4, 4)))  # shape 不一致
    with pytest.raises(ValueError):
        tile_sinkhorn_loss(np.zeros((5, 5)), np.zeros((5, 5)), tile=0)  # tile 非正
    with pytest.raises(ValueError):
        sinkhorn_transport_cost(np.zeros(4), np.zeros(3))  # 长度不一致


def test_torch_sinkhorn_matches_numpy_when_available() -> None:
    """torch 版（float32）与 numpy 版（float64）结果一致；缺 torch 则跳过。"""
    torch = pytest.importorskip("torch")
    rng = _rng()
    p = rng.random(25)
    g = rng.random(25)
    np_cost = sinkhorn_transport_cost(p, g, eps=0.05, n_iter=100)
    t_cost = float(
        torch_sinkhorn_cost(
            torch.tensor(p, dtype=torch.float32),
            torch.tensor(g, dtype=torch.float32),
            eps=0.05,
            n_iter=100,
        )
    )
    assert t_cost == approx(np_cost, rel=1e-3, abs=1e-5)


def test_torch_sinkhorn_supports_leading_batch_dims() -> None:
    """(B, T, L) 前导维度展平后语义不变：批量复制不得改变结果。

    训练时张量形状是 (batch, 分块数, 块内像素)，此前只支持 (T, L) 会直接抛
    "too many values to unpack"——本测试把这个回归钉住。
    """
    torch = pytest.importorskip("torch")
    rng = _rng()
    p = torch.tensor(rng.random((4, 25)), dtype=torch.float32)
    g = torch.tensor(rng.random((4, 25)), dtype=torch.float32)
    base = float(torch_sinkhorn_cost(p, g, eps=0.05, n_iter=50))
    batched = float(
        torch_sinkhorn_cost(
            p.unsqueeze(0).repeat(3, 1, 1), g.unsqueeze(0).repeat(3, 1, 1), eps=0.05, n_iter=50
        )
    )
    assert batched == approx(base, rel=1e-5, abs=1e-6)


def test_torch_sinkhorn_gradient_flows_through_batch() -> None:
    """批量路径必须可反传（否则训练时静默不更新）。"""
    torch = pytest.importorskip("torch")
    p = torch.rand(2, 4, 25, requires_grad=True)
    g = torch.rand(2, 4, 25)
    loss = torch_sinkhorn_cost(p, g, eps=0.05, n_iter=10)
    loss.backward()
    assert p.grad is not None
    assert bool(torch.isfinite(p.grad).all())
    assert float(p.grad.abs().sum()) > 0.0
