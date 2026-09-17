"""受控实验：SinkLoss vs 逐像素 MSE 在**含噪标注**下的鲁棒性。

问题
----
本系统的监督信号有结构性噪声：X 光缺陷边界半透明、评片员轮廓不一致、稀有类靠
预标注（prelabel）且定位精度有限。逐像素 L1/L2 会强迫模型复现这些噪声。Marigold
V2 的 SinkLoss 只约束"块内值的集合"（置换不变），理论上应更抗噪。

本脚本做的是**受控对比**，不是"跑一遍真实训练看看"：
- 输入（模拟底片）始终由**干净**几何渲染 —— 噪声只加在**训练标签**上，
  这正是"标注噪声"的定义，与"输入噪声"是两回事；
- 测试集用**干净**标签评估 —— 测的是对真实几何的泛化，而非对噪声的拟合；
- 标签几何取自项目自有 Golden Set（data/eval/golden_v3 的 YOLO 标签），
  不是凭空造的分布；
- 三种损失在同一份数据、同一随机种子、同一初始化下对比，只换损失。

三种配置
--------
1. ``mse``      ：逐像素 MSE（当前 YOLO 路线的等价物）；
2. ``sinkhorn`` ：纯 tile SinkLoss；
3. ``combo``    ：``mse + w * sinkhorn``。w 在初始化时自动标定，使两项量级相当
   （否则 sinkhorn 会因量纲差异压倒 mse，组合实验就失去意义）。

警告：已知陷阱（本脚本已被迫处理）：SinkLoss 的代价矩阵是平方项，若输入压到 [0,1]，
对比度会塌缩到个位数倍（见 losses 模块"输入量纲契约"）。故送入前统一乘
``--magnitude``（默认 10）。

用法
----
    python scripts/bench_density_loss_noise.py --epochs 120 --seeds 2
    python scripts/bench_density_loss_noise.py --quick       # 冒烟（2 epoch）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.training.density_aux import (  # noqa: E402
    boxes_from_yolo_lines,
    count_peaks,
    gaussian_density_map,
    to_sinkhorn_scale,
)

H = W = 64  # 渲染分辨率（标签是归一化的，固定画布即可）
TILE = 8  # SinkLoss 分块边长
SINKHORN_ITERS = 10  # Sinkhorn 迭代次数（Marigold V2 用 5；10 兼顾精度与 CPU 开销）
LABEL_DIR = _ROOT / "data" / "eval" / "golden_v3" / "labels"

# 密度核参数。注意本 Golden Set 的框**极小**（sqrt(w·h) 中位数仅 1.6px），
# 默认 ratio=0.25/min=1.0 会让每个缺陷只覆盖 ~0.7% 像素、全图前景仅 2%，
# 目标过度稀疏 → 模型塌陷到"全零输出"。放大核宽到前景约 10% 量级，
# 训练才具备可学性（这是**让实验公平可跑的必要条件**，不是调参刷分）。
SIG_RATIO = 1.0
SIG_MIN = 2.0


def _kernel(boxes: list[list[float]]) -> np.ndarray:
    """按当前核参数渲染密度图（全流程统一口径）。"""
    return gaussian_density_map((H, W), boxes, ratio=SIG_RATIO, min_sigma=SIG_MIN)


# ---------------------------------------------------------------------------
# 数据：项目自有标签几何 + 确定性渲染
# ---------------------------------------------------------------------------


def load_geometries(label_dir: Path = LABEL_DIR) -> list[list[list[float]]]:
    """读出项目 Golden Set 的框几何（归一化标签 → 当前画布下的绝对框）。"""
    files = sorted(label_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"未找到 YOLO 标签：{label_dir}")
    geoms: list[list[list[float]]] = []
    for f in files:
        boxes, _ = boxes_from_yolo_lines(f.read_text(encoding="utf-8").splitlines(), H, W)
        if boxes:  # 只保留有缺陷的图（本实验只考察缺陷定位）
            geoms.append(boxes)
    return geoms


def render_input(boxes: list[list[float]], rng: np.random.Generator) -> np.ndarray:
    """由**干净**几何渲染"类底片"输入：暗斑对应缺陷 + 低频纹理 + 噪声。

    这不是在造假数据冒充真实底片——它是受控实验的**输入侧**。本实验要隔离的
    变量是"标签噪声"，因此必须让输入干净、且几何完全可复原（模型理论上可学到）。
    """
    img = np.full((H, W), 0.85, dtype=np.float64)
    # 低频亮度起伏（焊接余高/厚度变化的粗糙代理）
    yy, xx = np.mgrid[0:H, 0:W]
    img += 0.04 * np.sin(2 * np.pi * xx / 23.0) * np.cos(2 * np.pi * yy / 17.0)
    # 缺陷：暗斑（衰减更大），强度随尺寸变化
    for bx, by, bw, bh in boxes:
        cx, cy = bx + bw / 2, by + bh / 2
        # 与密度目标同口径：输入里的暗斑尺度必须与标签密度核一致，
        # 否则"输入可见的缺陷"与"要被预测的密度"对不上，任务本身不自洽。
        sig = max(SIG_MIN, SIG_RATIO * float(np.sqrt(bw * bh)))
        r = int(np.ceil(3 * sig))
        x0, x1 = max(0, int(cx) - r), min(W, int(cx) + r + 1)
        y0, y1 = max(0, int(cy) - r), min(H, int(cy) + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        ys = np.arange(y0, y1)[:, None] - cy
        xs = np.arange(x0, x1)[None, :] - cx
        blob = np.exp(-(xs**2 + ys**2) / (2 * sig * sig))
        img[y0:y1, x0:x1] -= 0.45 * blob
    img += rng.normal(0.0, 0.015, size=(H, W))
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def jitter_boxes(
    boxes: list[list[float]],
    rng: np.random.Generator,
    *,
    pos_sigma: float,
    size_sigma: float,
) -> list[list[float]]:
    """把"评片员标注误差"注入几何：中心抖动 + 尺寸对数正态扰动。

    只扰动中心与尺寸，不改变缺陷个数——本实验考察定位/形状噪声，不掺入
    "漏标/多标"这类另一维度的噪声（那会让结论不可归因）。
    """
    out: list[list[float]] = []
    for bx, by, bw, bh in boxes:
        cx = bx + bw / 2 + rng.normal(0.0, pos_sigma)
        cy = by + bh / 2 + rng.normal(0.0, pos_sigma)
        sw = bw * float(np.exp(rng.normal(0.0, size_sigma)))
        sh = bh * float(np.exp(rng.normal(0.0, size_sigma)))
        out.append([cx - sw / 2, cy - sh / 2, max(1.0, sw), max(1.0, sh)])
    return out


@dataclass
class Split:
    x_train: np.ndarray  # (N, 1, H, W) 干净输入
    y_train: np.ndarray  # (N, 1, H, W) 训练标签（可能含噪）
    x_test: np.ndarray
    y_test: np.ndarray  # 干净标签
    test_boxes: list[list[list[float]]]


def build_split(
    geoms: list[list[list[float]]],
    *,
    pos_sigma: float,
    size_sigma: float,
    seed: int,
    test_frac: float = 0.25,
) -> Split:
    rng = np.random.default_rng(seed)
    n_test = max(1, int(len(geoms) * test_frac))
    idx = rng.permutation(len(geoms))
    test_idx, train_idx = set(idx[:n_test].tolist()), idx[n_test:].tolist()

    def stack(indices, noisy: bool):
        xs, ys, boxes_used = [], [], []
        for i in indices:
            g = geoms[int(i)]
            xs.append(render_input(g, rng))
            g_lab = jitter_boxes(g, rng, pos_sigma=pos_sigma, size_sigma=size_sigma) if noisy else g
            ys.append(_kernel(g_lab))
            boxes_used.append(g)
        return (
            np.stack(xs)[:, None, :, :],
            np.stack(ys)[:, None, :, :],
            boxes_used,
        )

    x_tr, y_tr, _ = stack(train_idx, noisy=True)
    x_te, y_te, tb = stack(sorted(test_idx), noisy=False)
    return Split(x_tr, y_tr, x_te, y_te, tb)


# ---------------------------------------------------------------------------
# 模型：小型 U-Net（密度回归头，sigmoid 输出 [0,1]）
# ---------------------------------------------------------------------------


def _torch():
    import torch

    return torch


def build_model(seed: int):
    torch = _torch()
    torch.manual_seed(seed)
    nn = torch.nn

    class UNetLite(nn.Module):
        """2 级 U-Net 精简版：(64,64) -> pool -> (32,32) -> up -> (64,64)。

        注意解码侧只做**一次**上采样：编码侧 pool 了一次，解码侧就必须只升一次，
        否则输出 128×128 与标签形状不匹配（曾踩）。
        """

        def __init__(self) -> None:
            super().__init__()
            self.enc1 = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU()
            )
            self.enc2 = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
            )
            self.pool = nn.MaxPool2d(2)
            self.up = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.dec = nn.Sequential(
                nn.Conv2d(16 + 16, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, 3, padding=1),
            )

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            d = self.up(e2)
            logits = self.dec(torch.cat([d, e1], dim=1))
            # 用 softplus 而非 sigmoid：稀疏目标（前景仅数个百分点）下，sigmoid 会被
            # 优化器推进饱和区——输出恒 0、**梯度恒为 0**，训练不可恢复地死掉
            # （实测：step 40 起 gradnorm=0.000e+00，损失纹丝不动在 5 位小数上）。
            # softplus 无上界饱和、梯度处处非零，值域下界 0 符合密度的物理含义。
            return torch.nn.functional.softplus(logits)

    return UNetLite()


# ---------------------------------------------------------------------------
# 损失
# ---------------------------------------------------------------------------


def sinkhorn_term(pred, gt, *, magnitude: float):
    """tile SinkLoss（torch 版，可反传）。输入先按 magnitude 放大量纲。"""
    torch = _torch()
    from backend.training.losses import torch_sinkhorn_cost

    n, _, h, w = pred.shape
    nh, nw = h // TILE, w // TILE
    p = pred[:, 0, : nh * TILE, : nw * TILE].reshape(n, nh, TILE, nw, TILE)
    g = gt[:, 0, : nh * TILE, : nw * TILE].reshape(n, nh, TILE, nw, TILE)
    p = p.permute(0, 1, 3, 2, 4).reshape(n, nh * nw, TILE * TILE) * magnitude
    g = g.permute(0, 1, 3, 2, 4).reshape(n, nh * nw, TILE * TILE) * magnitude
    return torch_sinkhorn_cost(p, g, eps=0.05, n_iter=SINKHORN_ITERS)


def train_one(
    split: Split,
    *,
    loss_name: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    magnitude: float,
    fg_weight: float = 20.0,
    log_every: int = 0,
) -> dict:
    """训练一个模型并评估。

    ``fg_weight`` 是**前景加权**系数：``w = 1 + fg_weight * (gt > 0.01)``。
    这是必需的，不是可选优化——实测**裸 MSE 在稀疏密度目标上根本不可训**：
    全图前景仅数个百分点，MSE 的最优解就是"输出≈0 的常数图"，优化器 15 轮内就把
    网络推到这个平凡解（loss 从第 15 轮起 300 轮纹丝不动，梯度衰到 8e-21），
    对**干净**测试标签的 MSE 恒为 0.077642，与训练轮数无关。
    因此把裸 MSE 记为 ``mse``（留作塌陷对照），可训版本记为 ``mse_fg``。
    """
    torch = _torch()
    torch.manual_seed(seed)
    model = build_model(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    xt = torch.from_numpy(split.x_train)
    yt = torch.from_numpy(split.y_train)
    n = xt.shape[0]
    use_fg = loss_name in ("mse_fg", "combo")

    def pixel_loss(pred, gt):
        if use_fg and fg_weight > 1.0:
            w = 1.0 + fg_weight * (gt > 0.01).to(gt.dtype)
            return ((pred - gt) ** 2 * w).sum() / w.sum()
        return ((pred - gt) ** 2).mean()

    def compute_loss(pred, gt):
        px = pixel_loss(pred, gt)
        if loss_name in ("mse", "mse_fg"):
            return px, px.detach(), torch.zeros(())
        sk = sinkhorn_term(pred, gt, magnitude=magnitude)
        if loss_name == "sinkhorn":
            return sk, torch.zeros(()), sk.detach()
        return px + combo_w * sk, px.detach(), sk.detach()

    # combo 权重自动标定：让步两项在初始化时量级相当（否则 sinkhorn 会因量纲压倒 mse）
    combo_w = 1.0
    if loss_name == "combo":
        with torch.no_grad():
            nb0 = min(batch_size, n)
            p0 = model(xt[:nb0])
            m0 = float(pixel_loss(p0, yt[:nb0]).item())
            s0 = float(sinkhorn_term(p0, yt[:nb0], magnitude=magnitude).item())
        combo_w = m0 / s0 if s0 > 0 else 1.0

    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch_size):
            b = perm[i : i + batch_size]
            opt.zero_grad()
            loss, _, _ = compute_loss(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(b)
        if log_every and (ep + 1) % log_every == 0:
            print(f"      ep {ep + 1:>3}/{epochs}  loss={tot / n:.5f}", flush=True)

    return evaluate(
        model, split, loss_name=loss_name, magnitude=magnitude, combo_w=combo_w
    )


def evaluate(model, split: Split, *, loss_name: str, magnitude: float, combo_w: float) -> dict:
    torch = _torch()
    with torch.no_grad():
        pred = model(torch.from_numpy(split.x_test)).numpy()[:, 0]

    gt = split.y_test[:, 0]
    mse = float(np.mean((pred - gt) ** 2))

    # 塌陷探针：稀疏目标下网络可能被推向饱和而死（梯度恒 0），此时所有指标都是
    # 无意义的"死网络读数"。这里独立于训练损失，用 MSE 在一个训练批次上测一次
    # 梯度范数——严格为 0 即意味网络已死，结果不可解读。
    nb = min(8, len(split.x_train))
    xb = torch.from_numpy(split.x_train[:nb])
    yb = torch.from_numpy(split.y_train[:nb])
    model.zero_grad()
    ((model(xb) - yb) ** 2).mean().backward()
    grad_norm = float(
        sum(p.grad.detach().norm().item() ** 2 for p in model.parameters() if p.grad is not None)
        ** 0.5
    )
    model.zero_grad()

    out_max = float(pred.max())
    # 判据只看"输出是否恒零"（真的死了），不用 grad_norm 精确等零：
    # round(grad_norm, 6) 曾把 1e-9 量级的健康梯度抹成 0.0，造出假塌陷警报。
    # grad_norm 仍原样保留供诊断（不做 round）。
    collapsed = bool(out_max < 1e-3 or grad_norm < 1e-12)

    # 峰值计数与中心命中（检测相关的读数，比纯 MSE 更贴近评片需求）
    count_err, hits, total_defects, n_pred_peaks = [], 0, 0, []
    for p, boxes in zip(pred, split.test_boxes, strict=True):
        peaks = count_peaks(p, min_distance=3, threshold_abs=0.3)
        n_pred_peaks.append(len(peaks))
        total_defects += len(boxes)
        count_err.append(abs(len(peaks) - len(boxes)))
        for bx, by, bw, bh in boxes:
            cx, cy = bx + bw / 2, by + bh / 2
            if any((py - cy) ** 2 + (px - cx) ** 2 <= 9.0 for py, px in peaks):  # 3px 容差
                hits += 1
    return {
        "mse_clean": round(mse, 6),
        "peak_count_mae": round(float(np.mean(count_err)), 4),
        "defect_hit_rate": round(hits / total_defects, 4) if total_defects else 0.0,
        "mean_pred_peaks": round(float(np.mean(n_pred_peaks)), 3),
        "output_max": round(out_max, 6),
        "grad_norm": grad_norm,  # 不 round（见上：round(...,6) 会造出假塌陷）
        "collapsed": collapsed,
        "combo_w": round(combo_w, 4),
    }


# ---------------------------------------------------------------------------
# 确定性探针：固定预测下，损失对标签噪声的敏感度（无需训练）
# ---------------------------------------------------------------------------


def loss_noise_probe(geoms: list[list[list[float]]], *, magnitude: float, seed: int) -> dict:
    """同一批"预测"下，标签加噪造成的**假惩罚**，以"完全漏检"的量级归一化。

    这是一个**无需训练**的确定性测量。关键在于分母的选择：不能拿"干净标签下的
    损失"作分母（那在最优附近趋近 0，比值会虚高到几十上百倍，毫无可比性）。
    这里用"预测全零 vs 干净真值"的损失作分母，回答一个具体且可解释的问题：

        **标注抖动带来的假惩罚，相当于"一个缺陷都没检出"的百分之几？**

    该比值越小，说明损失越不把标注抖动当回事（越抗噪）。
    """
    rng = np.random.default_rng(seed)
    mse_delta: list[float] = []
    sk_delta: list[float] = []
    mse_empty: list[float] = []
    sk_empty: list[float] = []
    for boxes in geoms[:40]:
        clean = _kernel(boxes)
        noisy = _kernel(jitter_boxes(boxes, rng, pos_sigma=2.0, size_sigma=0.25))
        pred = clean * 0.9  # 模拟一个"学得差不多"的模型输出
        zeros = np.zeros_like(clean)

        mse_delta.append(max(0.0, _mse(pred, noisy) - _mse(pred, clean)))
        mse_empty.append(_mse(pred, zeros))
        sk_delta.append(max(0.0, _sk(pred, noisy, magnitude) - _sk(pred, clean, magnitude)))
        sk_empty.append(_sk(pred, zeros, magnitude))

    mse_d, sk_d = float(np.mean(mse_delta)), float(np.mean(sk_delta))
    mse_e, sk_e = float(np.mean(mse_empty)), float(np.mean(sk_empty))
    return {
        "mse_abs_delta": round(mse_d, 6),
        "sinkhorn_abs_delta": round(sk_d, 6),
        "mse_noise_penalty_pct": round(mse_d / max(mse_e, 1e-12) * 100, 3),
        "sinkhorn_noise_penalty_pct": round(sk_d / max(sk_e, 1e-12) * 100, 3),
        "improvement_x": round(
            (mse_d / max(mse_e, 1e-12)) / max(sk_d / max(sk_e, 1e-12), 1e-12), 3
        ),
    }


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2))


def _sk(pred: np.ndarray, gt: np.ndarray, magnitude: float) -> float:
    from backend.training.losses import tile_sinkhorn_loss

    return tile_sinkhorn_loss(
        to_sinkhorn_scale(pred, magnitude=magnitude),
        to_sinkhorn_scale(gt, magnitude=magnitude),
        tile=TILE,
        n_iter=SINKHORN_ITERS,
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

NOISE_LEVELS = {"clean": 0.0, "mild": 1.5, "heavy": 3.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--magnitude", type=float, default=10.0)
    ap.add_argument("--out", type=str, default="data/reports/density_loss_noise.json")
    ap.add_argument("--noises", type=str, default="clean,mild,heavy", help="噪声档位，逗号分隔")
    ap.add_argument(
        "--losses",
        type=str,
        default="mse,mse_fg,sinkhorn,combo",
        help="损失：mse(裸MSE,会塌陷) / mse_fg(前景加权MSE) / sinkhorn / combo",
    )
    ap.add_argument("--fg-weight", type=float, default=20.0, help="前景加权系数（见 train_one）")
    ap.add_argument("--quick", action="store_true", help="冒烟模式：2 epoch / 1 seed")
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.seeds = 2, 1

    noises = [n.strip() for n in args.noises.split(",") if n.strip()]
    losses = [x.strip() for x in args.losses.split(",") if x.strip()]
    valid_losses = {"mse", "mse_fg", "sinkhorn", "combo"}
    bad = (set(noises) - set(NOISE_LEVELS)) | (set(losses) - valid_losses)
    if bad:
        raise SystemExit(
            f"未知取值：{sorted(bad)}（噪声档 {list(NOISE_LEVELS)}，损失 {sorted(valid_losses)}）"
        )

    geoms = load_geometries()
    print(f"标签几何：{len(geoms)} 张图，共 {sum(len(g) for g in geoms)} 个缺陷", flush=True)

    print("\n[1/2] 确定性探针：标签噪声造成的假惩罚（无需训练）", flush=True)
    probe = loss_noise_probe(geoms, magnitude=args.magnitude, seed=999)
    print(
        f"  逐像素 MSE     : 假惩罚 = 漏检量级的 {probe['mse_noise_penalty_pct']:.2f}%"
        f"  (|ΔL|={probe['mse_abs_delta']})",
        flush=True,
    )
    print(
        f"  tile SinkLoss  : 假惩罚 = 漏检量级的 {probe['sinkhorn_noise_penalty_pct']:.2f}%"
        f"  (|ΔL|={probe['sinkhorn_abs_delta']})",
        flush=True,
    )
    print(f"  → MSE 对噪声的假惩罚是 SinkLoss 的 {probe['improvement_x']:.2f} 倍", flush=True)

    print("\n[2/2] 受控训练对比（同数据/同种子/同初始化，仅换损失）", flush=True)
    results: dict[str, dict] = {}
    t0 = time.time()
    for noise_name in noises:
        pos_sigma = NOISE_LEVELS[noise_name]
        size_sigma = 0.0 if pos_sigma == 0.0 else 0.20
        for seed in range(args.seeds):
            split = build_split(geoms, pos_sigma=pos_sigma, size_sigma=size_sigma, seed=100 + seed)
            for loss_name in losses:
                tag = f"{noise_name}|{loss_name}|s{seed}"
                ts = time.time()
                m = train_one(
                    split,
                    loss_name=loss_name,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    magnitude=args.magnitude,
                    fg_weight=args.fg_weight,
                    log_every=max(1, args.epochs // 4),
                )
                m["noise"] = noise_name
                m["loss"] = loss_name
                m["seed"] = seed
                m["seconds"] = round(time.time() - ts, 1)
                results[tag] = m
                flag = "  [塌陷]" if m["collapsed"] else ""
                print(
                    f"  {tag:<22} mse={m['mse_clean']:.5f} "
                    f"hit={m['defect_hit_rate']:.3f} cnt_mae={m['peak_count_mae']:.2f} "
                    f"grad={m['grad_norm']:.2e} ({m['seconds']}s){flag}",
                    flush=True,
                )

    n_collapsed = sum(1 for r in results.values() if r["collapsed"])
    if n_collapsed:
        print(
            f"\n  警告：{n_collapsed}/{len(results)} 次运行**网络塌陷**（梯度恒 0）——"
            f"这些读数不可解读，需先修可学性再比较损失。",
            flush=True,
        )

    summary = _summarize(results, args.seeds)
    payload = {
        "config": {
            "resolution": H,
            "tile": TILE,
            "epochs": args.epochs,
            "seeds": args.seeds,
            "magnitude": args.magnitude,
            "fg_weight": args.fg_weight,
            "sigma_ratio": SIG_RATIO,
            "sigma_min": SIG_MIN,
            "noise_levels": {k: NOISE_LEVELS[k] for k in noises},
            "losses": losses,
            "labels": str(LABEL_DIR),
            "device": "cpu",
        },
        "probe": probe,
        "summary": summary,
        "runs": results,
        "total_seconds": round(time.time() - t0, 1),
    }
    out_path = _ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_table(summary)
    print(f"\n结果已写入：{out_path}")
    return 0


def _summarize(results: dict[str, dict], n_seeds: int) -> dict:
    all_losses = sorted({r["loss"] for r in results.values()})
    agg: dict[str, dict] = {}
    for noise in NOISE_LEVELS:
        for loss_name in all_losses:
            vals = [r for r in results.values() if r["noise"] == noise and r["loss"] == loss_name]
            if not vals:
                continue
            agg[f"{noise}|{loss_name}"] = {
                "mse_clean_mean": round(float(np.mean([v["mse_clean"] for v in vals])), 6),
                "mse_clean_std": round(float(np.std([v["mse_clean"] for v in vals])), 6),
                "defect_hit_rate_mean": round(float(np.mean([v["defect_hit_rate"] for v in vals])), 4),
                "peak_count_mae_mean": round(float(np.mean([v["peak_count_mae"] for v in vals])), 4),
                "output_max_mean": round(float(np.mean([v["output_max"] for v in vals])), 6),
                "collapsed_runs": sum(1 for v in vals if v["collapsed"]),
                "n_seeds": len(vals),
                "seconds_per_run": round(float(np.mean([v["seconds"] for v in vals])), 1),
            }
    # 相对基线的改善。基线优先取 mse_fg（可训版本）；裸 mse 在稀疏目标上会塌陷，
    # 拿它当基线会得到"改善无穷大"之类的无意义数字，只作塌陷对照展示。
    for noise in NOISE_LEVELS:
        base = agg.get(f"{noise}|mse_fg") or agg.get(f"{noise}|mse")
        if not base:
            continue
        for loss_name in all_losses:
            if loss_name in ("mse_fg", "mse"):
                continue
            cur = agg.get(f"{noise}|{loss_name}")
            if not cur:
                continue
            cur["baseline"] = "mse_fg" if f"{noise}|mse_fg" in agg else "mse"
            cur["mse_clean_improve_pct"] = round(
                (base["mse_clean_mean"] - cur["mse_clean_mean"]) / base["mse_clean_mean"] * 100, 2
            )
            cur["hit_rate_delta"] = round(
                cur["defect_hit_rate_mean"] - base["defect_hit_rate_mean"], 4
            )
    return agg


def _print_table(summary: dict) -> None:
    print("\n" + "=" * 100)
    print(
        f"{'噪声档':<8}{'损失':<12}{'干净MSE':>12}{'±std':>10}"
        f"{'命中率':>10}{'计数MAE':>10}{'塌陷':>6}{'vs基线':>10}"
    )
    print("-" * 100)
    for key, v in summary.items():
        noise, loss_name = key.split("|")
        imp = v.get("mse_clean_improve_pct")
        imp_s = f"{imp:+.2f}%" if imp is not None else "—"
        col = f"{v['collapsed_runs']}/{v['n_seeds']}" if v["collapsed_runs"] else "—"
        print(
            f"{noise:<8}{loss_name:<12}{v['mse_clean_mean']:>12.6f}"
            f"{v['mse_clean_std']:>10.6f}{v['defect_hit_rate_mean']:>10.3f}"
            f"{v['peak_count_mae_mean']:>10.2f}{col:>6}{imp_s:>10}"
        )
    print("=" * 100)
    print("干净MSE = 对**干净**测试标签的均方误差（越低越好）")
    print("vs基线 = 相对同噪声档 mse_fg 基线的改善；塌陷列为'网络塌陷运行数/总运行数'")
    if any(v["collapsed_runs"] for v in summary.values()):
        print("警告：存在塌陷运行：这些行的读数不可解读（网络输出恒零，未在学习）")


if __name__ == "__main__":
    raise SystemExit(main())
