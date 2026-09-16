"""论文插图：从资格评定基准产物绘制出版级图表（含流程架构图）。

风格约定（延续 Origin 系出版样式，面板编号改用 Nature 惯例）：
- 全框坐标轴（四边 spine 齐全）+ 四边刻度**朝内**，主/次刻度分级；
- 无网格线；白底；图例带细边框；
- 点（填充 marker + 黑色描边）线组合，误差棒带帽；
- 面板编号用 Nature 惯例：粗体 **a**/**b**/**c**，无括号，置于面板左上外侧；
- 字体微软雅黑（覆盖拉丁与中日韩字形），磅值与线宽全图统一；
- 600 dpi 输出，图宽按 Nature 单栏 88 mm / 双栏 180 mm 规格。

数据来源：``scripts/bench_qualification_protocol.py`` 落盘的 JSON（只做可视化，
不重算任何指标），保证图与数字同源、可复现。真实域画像取自运行库 ``data/scan.db``。

用法::

    backend/.venv/Scripts/python.exe scripts/make_qualification_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.ticker import LogLocator, MultipleLocator  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "data/reports/figures"

# ---- 配色：高对比深色系，保持色盲安全（双系列另用底纹做第二区分） ----
C_BLUE = "#1F4E9C"
C_RED = "#C0392B"
C_GREEN = "#1E7A4C"
C_ORANGE = "#D68910"
C_VIOLET = "#6C3483"
C_GREY = "#7F8C8D"
C_BLACK = "#000000"

# 示意图用的浅色填充（保证黑字可读）
F_BLUE = "#DCE7F7"
F_GREEN = "#DBF0E4"
F_ORANGE = "#FBEAD4"
F_VIOLET = "#E9DCF2"
F_GREY = "#EDEFF1"

CLASS_NAMES = ["气孔", "夹渣", "未焊透", "未熔合", "裂纹", "咬边", "内凹"]
STD_CLASS_ORDER = [1, 2, 3, 4, 5, 6, 7]
STD_CLASS_NAMES = ["圆形缺陷", "条形缺陷", "裂纹", "未熔合", "未焊透", "内凹", "咬边"]

FS_TICK = 8.0
FS_LABEL = 9.0
FS_LEGEND = 8.0
FS_PANEL = 11.0
LW_AXIS = 1.0
LW_PLOT = 1.3
MS = 5.0

# 层级框配色（L1–L4）
LAYERS = [
    ("L1 标准分级层", F_BLUE, C_BLUE),
    ("L2 尺寸可靠性层", F_GREEN, C_GREEN),
    ("L3 置信度校准层", F_ORANGE, C_ORANGE),
    ("L4 扰动稳定性层", F_VIOLET, C_VIOLET),
]


def _setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            # 以雅黑为主字体：它同时覆盖拉丁与中日韩字形，避免 Arial 主字体时
            # 中文回退失效（实测 matplotlib 的 fallback 对部分汉字不生效，会出方框）。
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": FS_TICK,
            "axes.labelsize": FS_LABEL,
            "axes.titlesize": FS_LABEL,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_LEGEND,
            "axes.linewidth": LW_AXIS,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.major.width": LW_AXIS,
            "ytick.major.width": LW_AXIS,
            "xtick.minor.width": LW_AXIS * 0.8,
            "ytick.minor.width": LW_AXIS * 0.8,
            "axes.grid": False,
            "legend.frameon": True,
            "legend.edgecolor": C_BLACK,
            "legend.framealpha": 1.0,
            "legend.fancybox": False,
            "legend.borderpad": 0.4,
            "legend.handlelength": 2.2,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
        }
    )


def _load(name: str) -> dict:
    return json.loads((_ROOT / f"data/reports/{name}").read_text(encoding="utf-8"))


def _save(fig, name: str) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    p = _OUT / name
    fig.savefig(p, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  已输出 {p.relative_to(_ROOT)}")


def _panel(ax, letter: str) -> None:
    """面板编号：Nature 惯例的粗体小写字母（无括号），置于坐标框左上外侧。

    放在框外而非框内，是为了不与框内图例、数值标注重叠（实测框内左上会被
    图例压住）。
    """
    ax.text(
        0.0,
        1.035,
        letter,
        transform=ax.transAxes,
        fontsize=FS_PANEL,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _origin_box(ax) -> None:
    """全框 + 四边刻度（四边都有朝内刻度线）。"""
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(LW_AXIS)
        ax.spines[side].set_color(C_BLACK)


# ---------------------------------------------------------------------------
# 示意图基元（流程架构图用；坐标系为 0–100 的归一化画布）
# ---------------------------------------------------------------------------
def _canvas(w: float, h: float):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def _dbox(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str] | tuple[str, ...] = (),
    face: str = "white",
    edge: str = C_BLACK,
    title_fs: float = 7.8,
    line_fs: float = 6.8,
    rounding: float = 0.9,
) -> None:
    """带标题与若干行的圆角框。

    标题与正文行按**点值**自动分布：画布的 x/y 单位不等比（100 单位对应不同
    的物理尺寸），若把行距写成固定 y 单位，文字必然在某些框里贴底或与标题重叠。
    这里先把框高换算为磅，再按字号排布；空间不足时等比压缩行距（不缩字号，
    保持可读），有余量时整体垂直居中。
    """
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            linewidth=0.9,
            edgecolor=edge,
            facecolor=face,
            zorder=2,
        )
    )
    cx = x + w / 2
    n = len(lines)
    if n == 0:
        ax.text(cx, y + h / 2, title, ha="center", va="center", fontsize=title_fs,
                fontweight="bold", zorder=3)
        return
    u = ax.figure.get_size_inches()[1] * 72.0 / 100.0  # 每 y 单位对应的磅值
    h_pt = h * u
    title_d = 1.05 * title_fs
    # 中文字形视觉高度大，行距系数须比西文宽（0.85/1.60 实测不重叠）
    gap = 0.85 * title_fs + 0.85 * line_fs
    step = 1.60 * line_fs
    bottom_margin = 0.75 * line_fs
    need = title_d + gap + (n - 1) * step + bottom_margin
    if need > h_pt:
        slack = h_pt - title_d - bottom_margin
        span = gap + (n - 1) * step
        scale = slack / span if span > 0 else 1.0
        gap *= scale
        step *= scale
        need = h_pt
    offset = (h_pt - need) / 2.0
    title_d += offset
    ax.text(cx, y + h - title_d / u, title, ha="center", va="center",
            fontsize=title_fs, fontweight="bold", zorder=3)
    for k, ln in enumerate(lines):
        d_pt = title_d + gap + k * step
        ax.text(cx, y + h - d_pt / u, ln, ha="center", va="center",
                fontsize=line_fs, zorder=3)


def _band(ax, x: float, y: float, w: float, h: float, text: str,
          face: str = F_GREY, edge: str = C_GREY, fs: float = 6.8) -> None:
    """横贯约束条。"""
    ax.add_patch(
        Rectangle((x, y), w, h, linewidth=0.8, edgecolor=edge,
                  facecolor=face, zorder=2, linestyle=(0, (3, 2)))
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3)


def _arrow(ax, p, q, color: str = C_BLACK, lw: float = 1.0, rad: float = 0.0,
           ms: float = 6.5, ls: str = "-") -> None:
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            linestyle=ls,
            zorder=4,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
        )
    )


# ---------------------------------------------------------------------------
def fig_protocol_architecture() -> None:
    """图 1：资格评定协议总体架构与数据流（流程架构图）。"""
    fig, ax = _canvas(7.2, 4.9)

    # ---- 输入层 ----
    ax.text(1.5, 98.6, "输入：被评定的对象（三者同时锁定）", ha="left", va="center",
            fontsize=7.4, fontweight="bold")
    for x, title, lines in (
        (1.5, "部署权重",
         ["best.onnx", "sha256 指纹", "指纹 best::8370aeeef0b1"]),
        (34.5, "锁定评估集",
         ["图像 + 标注", "划分指纹", "跨集合互斥校验"]),
        (67.5, "固定推理配置",
         ["逐类置信度阈值", "分类别 NMS / IoU 0.5", "letterbox 640 灰 114"]),
    ):
        _dbox(ax, x, 82.5, 31.0, 14.0, title, lines, face=F_GREY, edge=C_BLACK)

    _arrow(ax, (50, 82.5), (50, 79.0))
    ax.plot([13.0, 87.0], [78.0, 78.0], color=C_BLACK, linewidth=1.0, zorder=3)
    for cx in (13.0, 38.0, 63.0, 88.0):
        _arrow(ax, (cx, 78.0), (cx, 76.3), ms=5.5)
    # ---- 四层证据 ----
    layer_lines = [
        ["TDRn / FDRn / MDRn 逐类", "KDR / WDR / TDR / FRR 综合",
         "双口径取较差者", "（标准 0.1 / 严格 0.3）", "→ 等级 L1–L4 或未定级"],
        ["特征长度 √A（像素）", "分位数等频分箱", "hit/miss 工作点 POD",
         "Wilson 95% 置信区间", "→ 尺寸条件化能力边界"],
        ["逐类温度缩放拟合", "ECE（15 分箱）/ MCE", "分箱可靠性图",
         "工作点保持不变式", "→ 复核分流阈值依据"],
        ["7 种灰度扰动", "检出保持率（绝对 / 相对）", "量化平均相对偏差",
         "每图新增误检", "→ 相对衰减量判定"],
    ]
    xs = [1.5, 26.5, 51.5, 76.5]
    for x, (title, face, edge), lines in zip(xs, LAYERS, layer_lines):
        _dbox(ax, x, 50.0, 23.0, 25.0, title, lines, face=face, edge=edge)
        _arrow(ax, (x + 11.5, 50.0), (50, 45.0), color=edge, lw=0.9, rad=0.0, ms=5.5)
    _arrow(ax, (50, 45.0), (50, 40.6), ms=6.5)

    # ---- 输出层 ----
    for x, title, lines in (
        (1.5, "标准评价记录表",
         ["逐类 td/fd/md 与 TDRn", "综合指标与误检方向"]),
        (34.5, "等级与风险判定",
         ["等级：L1–L4 或未定级", "漏检 / 误检 / 误报风险"]),
        (67.5, "能力边界陈述",
         ["尺寸-POD 与校准质量", "获批模式：AssistDR / AutoDR"]),
    ):
        _dbox(ax, x, 26.0, 31.0, 14.0, title, lines, face=F_GREY, edge=C_BLACK)

    # ---- 贯穿约束 ----
    _band(ax, 1.5, 13.5, 97.0, 8.0,
          "贯穿前提 1｜口径分离：标准口径 IoU ≥ 0.1 · 严格口径 IoU ≥ 0.3 · 检测口径 IoU ≥ 0.5，三线并行呈现、互不换算")
    _band(ax, 1.5, 2.5, 97.0, 8.0,
          "贯穿前提 2｜管线锁定与指纹绑定：权重 sha256 · 数据划分指纹 · 阈值与 NMS 参数全落盘；校准表与权重指纹绑定")

    _save(fig, "fig1_protocol_architecture.png")


def fig_data_assets() -> None:
    """图 2：数据资产构成、来源与互斥校验（三列式数据流图）。"""
    fig, ax = _canvas(7.2, 4.0)

    for x, label in ((1.5, "来源"), (34.5, "集合与规模"), (67.5, "用途与可用性")):
        ax.text(x + 15.5, 97.0, label, ha="center", va="center", fontsize=8.0,
                fontweight="bold")
        ax.plot([x, x + 31.0], [94.0, 94.0], color=C_BLACK, linewidth=1.0)
    _arrow(ax, (16.0, 93.0), (16.0, 88.0), ms=5.5)

    rows = [
        ("程序化合成生成器", ["planB_run.py::generate", "种子 42（可复现）"],
         "合成全集 600 张", ["训练 425 · 验证 50", "测试 125"],
         "训练 / 校准拟合 / 主评估",
         ["测试集 713 处缺陷", "标注齐备，指标可算"], F_BLUE, C_BLUE),
        ("程序化合成生成器", ["make_golden_set_v3.py", "种子 20260945"],
         "独立回归集 120 张", ["与全集零重叠", "407 处缺陷"],
         "权重版本回归基准",
         ["非真实域精度证书", "用于复现性交叉验证"], F_GREEN, C_GREEN),
        ("现场射线底片", ["2448 × 2048 像素", "扫描数字化件"],
         "真实底片 165 张", ["标注目录为空", "运行库 75 张"],
         "仅输出画像，不含精度",
         ["不报告任何真实域精度", "涵盖 2026-09-07 至 09-14"], F_ORANGE, C_ORANGE),
    ]
    ys = [66.0, 40.0, 14.0]
    for (t1, l1, t2, l2, t3, l3, face, edge), y in zip(rows, ys):
        _dbox(ax, 1.5, y, 31.0, 20.0, t1, l1, face="white", edge=edge)
        _dbox(ax, 34.5, y, 31.0, 20.0, t2, l2, face=face, edge=edge)
        _dbox(ax, 67.5, y, 31.0, 20.0, t3, l3, face="white", edge=edge)
        _arrow(ax, (32.5, y + 10.0), (34.5, y + 10.0), ms=5.5)
        _arrow(ax, (65.5, y + 10.0), (67.5, y + 10.0), ms=5.5)

    # 互斥校验说明：置于第二、三行之间的空档（该处无框体，不会压线）
    ax.text(50.0, 37.0,
            "合成来源之间执行跨集合互斥校验：dHash 汉明距离 ≥ 5（阈值 > 4），实测零重叠",
            ha="center", va="center", fontsize=6.8, color=C_GREEN)

    _save(fig, "fig2_data_assets.png")


def fig_standard_grading(a: dict, b: dict) -> None:
    """图 3：标准分级结果。

    (a) 各综合指标与 L1 级门槛的差距（哑铃图）；
    (b) 标准口径逐类正检率 TDRn；
    (c) 检测口径逐类 AP@0.5。
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.85))

    # ---- (a) 门槛差距哑铃图 ----
    ax = axes[0]
    metrics = [("KDR", 0.95), ("WDR", 0.92), ("TDR", 0.85)]
    rows = []
    for name, thr in metrics:
        for data, color, lab in ((a, C_BLUE, "测试集"), (b, C_RED, "验证集")):
            s = data["std501807"]["standard"]
            rows.append((f"{name}·{'测试集' if data is a else '验证集'}",
                         s[name.lower()], thr, color))
    ys = np.arange(1, 7)[::-1]  # 底部空出一行放图例，避免压住数值标注
    for y, (lab, val, thr, color) in zip(ys, rows):
        ax.plot([min(val, thr), max(val, thr)], [y, y], color=C_GREY,
                linewidth=1.0, zorder=1)
        ax.plot([thr], [y], marker="s", markersize=4.6, markerfacecolor="white",
                markeredgecolor=C_BLACK, markeredgewidth=0.7, zorder=3)
        ax.plot([val], [y], marker="o", markersize=MS, markerfacecolor=color,
                markeredgecolor=C_BLACK, markeredgewidth=0.6, zorder=4)
        ax.text(val - 0.006, y, f"{val * 100:.2f}%", ha="right", va="center",
                fontsize=5.9, color=color)
        # 门槛标号写在空心方块右侧（0.012 单位 ≈ 8 pt），写在方块中心会与其重叠
        ax.text(thr + 0.012, y, f"{thr * 100:.0f}%", ha="left", va="center",
                fontsize=5.6, color=C_BLACK)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.2)
    ax.set_xlabel("指标值")
    # 横轴下限须低于最低实测值（TDR 为 0.77/0.78），否则数值标注会被挤到
    # 坐标区之外并压住刻度标签。取 0.72 以容纳 TDR 与 0.85 门槛的全部点。
    ax.set_xlim(0.72, 1.01)
    ax.set_ylim(-0.9, 6.8)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.025))
    ax.plot([], [], marker="o", markersize=MS, markerfacecolor=C_BLUE,
            markeredgecolor=C_BLACK, markeredgewidth=0.6, linestyle="none",
            label="实测值")
    ax.plot([], [], marker="s", markersize=4.6, markerfacecolor="white",
            markeredgecolor=C_BLACK, markeredgewidth=0.7, linestyle="none",
            label="L1 级门槛")
    ax.legend(loc="lower left", handletextpad=0.4, borderaxespad=0.3,
              handlelength=1.2)
    _panel(ax, "a")

    # ---- (b) 标准口径逐类 TDRn ----
    ax = axes[1]
    x = np.arange(7)
    w = 0.36
    for k, (data, color, hatch, lab) in enumerate(
        ((a, C_BLUE, "", "测试集"), (b, C_RED, "///", "验证集"))
    ):
        pc = data["std501807"]["standard"]["per_class"]
        vals, ns = [], []
        for n in STD_CLASS_ORDER:
            c = pc[str(n)]
            tot = c["td"] + c["fd"] + c["md"]
            vals.append(c["tdr"] if tot else np.nan)
            ns.append(tot)
        ax.bar(x + (k - 0.5) * w, vals, w, color=color, edgecolor=C_BLACK,
               linewidth=0.7, hatch=hatch, label=lab, zorder=2)
        for xi, v, n in zip(x, vals, ns):
            if not np.isfinite(v):
                continue
            # 近零柱（条形缺陷）的标注向外侧错开，否则相邻两根的 "0.00" 会粘连
            dx = 0.0 if v > 0.005 else (-0.045 if k == 0 else 0.045)
            ax.text(xi + (k - 0.5) * w + dx, v + 0.022, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=5.4,
                    color=C_BLACK if n >= 10 else C_GREY)
    ax.set_xticks(x)
    ax.set_xticklabels(STD_CLASS_NAMES, rotation=32, ha="right", fontsize=6.5)
    ax.set_ylabel("$TDR_n$")
    ax.set_ylim(0, 1.42)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(MultipleLocator(0.125))
    ax.annotate("$n$ = 2", xy=(1.0, 0.02), xytext=(1.0, 0.42), fontsize=6.5,
                color=C_GREY, ha="center",
                arrowprops={"arrowstyle": "->", "color": C_GREY, "linewidth": 0.7})
    ax.legend(loc="upper right", handletextpad=0.4, borderaxespad=0.3,
              handlelength=1.5)
    _panel(ax, "b")

    # ---- (c) 检测口径逐类 AP@0.5 ----
    ax = axes[2]
    for k, (data, color, hatch, lab) in enumerate(
        ((a, C_BLUE, "", "测试集"), (b, C_RED, "///", "验证集"))
    ):
        bc = data["detection"]["metrics"]["by_class"]
        vals = [bc.get(str(i), {}).get("ap50", 0.0) for i in range(7)]
        ax.bar(x + (k - 0.5) * w, vals, w, color=color, edgecolor=C_BLACK,
               linewidth=0.7, hatch=hatch, label=lab, zorder=2)
        for xi, v in zip(x, vals):
            ax.text(xi + (k - 0.5) * w, v + 0.022, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=5.4)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=32, ha="right", fontsize=6.5)
    ax.set_ylabel("$AP@0.5$")
    ax.set_ylim(0, 1.42)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(MultipleLocator(0.125))
    ax.legend(loc="upper right", handletextpad=0.4, borderaxespad=0.3,
              handlelength=1.5)
    _panel(ax, "c")

    for ax in axes:
        _origin_box(ax)
    fig.tight_layout(w_pad=1.4)
    _save(fig, "fig3_standard_grading.png")


def fig_pod(a: dict, b: dict) -> None:
    """图 4：尺寸条件化检出概率。

    (a) 两数据集 POD 曲线与 Wilson 95% 置信区间；(b) 各尺寸箱的缺陷数与检出数,
    用于说明小尺寸箱的低 POD 并非抽样噪声。
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.90))

    ax = axes[0]
    for data, color, marker, label in (
        (a, C_BLUE, "o", f"测试集（{a['dataset']['n_gt']} 处缺陷）"),
        (b, C_RED, "s", f"验证集（{b['dataset']['n_gt']} 处缺陷）"),
    ):
        bins = data["pod"]["bins"]
        x = [(r["size_min_px"] + r["size_max_px"]) / 2 for r in bins]
        y = [r["pod"] for r in bins]
        lo = [r["pod"] - r["ci95"][0] for r in bins]
        hi = [r["ci95"][1] - r["pod"] for r in bins]
        ax.errorbar(x, y, yerr=[lo, hi], color=color, marker=marker,
                    markersize=MS, markerfacecolor=color,
                    markeredgecolor=C_BLACK, markeredgewidth=0.6,
                    linewidth=LW_PLOT, capsize=2.5, capthick=LW_AXIS * 0.8,
                    elinewidth=LW_AXIS * 0.8, label=label, zorder=3)
    ax.axhline(0.9, color=C_GREY, linestyle=(0, (4, 2.5)), linewidth=0.9, zorder=1)
    ax.text(5.6, 0.918, "POD = 0.90", color=C_GREY, fontsize=6.8, ha="left",
            va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("缺陷特征长度 $\\sqrt{A}$ / px")
    ax.set_ylabel("工作点检出概率 $POD$")
    ax.set_ylim(0.0, 1.06)
    ax.set_xlim(5.3, 155)
    ax.set_xticks([6, 10, 20, 40, 80, 140])
    ax.set_xticklabels(["6", "10", "20", "40", "80", "140"])
    ax.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1), numticks=100)
    )
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    _origin_box(ax)
    ax.legend(loc="lower right", handletextpad=0.5, borderaxespad=0.3)
    _panel(ax, "a")

    # (b) 各尺寸箱的缺陷数与检出数（堆叠：检出 / 漏检）
    ax = axes[1]
    bins = a["pod"]["bins"]
    labels = [f"{r['size_min_px']:.0f}–{r['size_max_px']:.0f}" for r in bins]
    x = np.arange(len(bins))
    w = 0.36
    for k, (data, color, hatch) in enumerate(((a, C_BLUE, ""), (b, C_RED, "///"))):
        det = [r["detected"] for r in data["pod"]["bins"]]
        tot = [r["n"] for r in data["pod"]["bins"]]
        miss = [t - d for t, d in zip(tot, det)]
        ax.bar(x + (k - 0.5) * w, det, w, color=color, edgecolor=C_BLACK,
               linewidth=0.7, hatch=hatch, label="测试集" if k == 0 else "验证集",
               zorder=2)
        ax.bar(x + (k - 0.5) * w, miss, w, bottom=det, color="white",
               edgecolor=C_BLACK, linewidth=0.7, hatch="xx", zorder=2)
        for xi, (d, t) in zip(x + (k - 0.5) * w, zip(det, tot)):
            ax.text(xi, t + 6, f"{d}/{t}", ha="center", va="bottom", fontsize=5.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_xlabel("尺寸箱 $\\sqrt{A}$ / px")
    ax.set_ylabel("缺陷数")
    ax.set_ylim(0, 250)
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(25))
    _origin_box(ax)
    ax.legend(loc="upper left", handletextpad=0.4, borderaxespad=0.3,
              handlelength=1.5)
    _panel(ax, "b")

    fig.tight_layout(w_pad=1.2)
    _save(fig, "fig4_pod_curve.png")


def fig_calibration(a: dict) -> None:
    """图 5：置信度校准。

    (a) 校准前、(b) 逐类温度校准后的分箱可靠性图；(c) 两集上 ECE 与 MCE 的
    反向变化——ECE 下降而 MCE 上升。
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75))
    lim = (0.0, 1.02)

    for ax, key, letter, title in (
        (axes[0], "uncalibrated", "a", "校准前"),
        (axes[1], "calibrated", "b", "逐类温度校准后"),
    ):
        e = a["ece"][key]
        bins = [r for r in e["bins"] if r["n"] > 0]
        conf = [r["conf_mean"] for r in bins]
        acc = [r["accuracy"] for r in bins]
        ax.plot(lim, lim, color=C_GREY, linestyle=(0, (4, 2.5)), linewidth=0.9,
                zorder=1)
        ax.plot(conf, acc, color=C_RED, marker="o", markersize=MS,
                markerfacecolor=C_RED, markeredgecolor=C_BLACK,
                markeredgewidth=0.6, linewidth=LW_PLOT, zorder=3)
        ax.set_title(title, fontsize=8.2, pad=4)
        ax.set_xlabel("平均置信度")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.xaxis.set_major_locator(MultipleLocator(0.25))
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.xaxis.set_minor_locator(MultipleLocator(0.125))
        ax.yaxis.set_minor_locator(MultipleLocator(0.125))
        _origin_box(ax)
        _panel(ax, letter)
        ax.text(0.97, 0.05,
                f"ECE = {e['ece']:.4f}\nMCE = {e['mce']:.4f}",
                transform=ax.transAxes, fontsize=6.2, ha="right", va="bottom")
    axes[0].set_ylabel("实测准确率")
    axes[0].plot([], [], color=C_GREY, linestyle=(0, (4, 2.5)), linewidth=0.9,
                 label="理想校准线 $y=x$")
    axes[0].plot([], [], color=C_RED, marker="o", markersize=MS,
                 markeredgecolor=C_BLACK, markeredgewidth=0.6,
                 linewidth=LW_PLOT, label="实测（分箱）")
    axes[0].legend(loc="upper left", handletextpad=0.5, borderaxespad=0.35)

    # (c) ECE 与 MCE 的反向变化
    ax = axes[2]
    b = _load("qualification_protocol_golden_v3.json")
    groups = [
        ("ECE\n测试集", a["ece"]["uncalibrated"]["ece"], a["ece"]["calibrated"]["ece"]),
        ("ECE\n验证集", b["ece"]["uncalibrated"]["ece"], b["ece"]["calibrated"]["ece"]),
        ("MCE\n测试集", a["ece"]["uncalibrated"]["mce"], a["ece"]["calibrated"]["mce"]),
        ("MCE\n验证集", b["ece"]["uncalibrated"]["mce"], b["ece"]["calibrated"]["mce"]),
    ]
    x = np.arange(len(groups))
    w = 0.34
    before = [g[1] for g in groups]
    after = [g[2] for g in groups]
    ax.bar(x - w / 2, before, w, color="white", edgecolor=C_BLACK, linewidth=0.8,
           hatch="//", label="校准前", zorder=2)
    ax.bar(x + w / 2, after, w, color=C_RED, edgecolor=C_BLACK, linewidth=0.8,
           label="校准后", zorder=2)
    for xi, (v0, v1) in zip(x, zip(before, after)):
        ax.text(xi - w / 2, v0 + 0.012, f"{v0:.3f}", ha="center", va="bottom",
                fontsize=5.6)
        ax.text(xi + w / 2, v1 + 0.012, f"{v1:.3f}", ha="center", va="bottom",
                fontsize=5.6)
        # 方向箭头跟随实际变化方向：ECE 组向下（改善），MCE 组向上（恶化）
        up = v1 > v0
        top = max(v0, v1) + 0.036
        tip = max(v0, v1) + 0.082
        ax.annotate(
            "",
            xy=(xi, tip if up else top),
            xytext=(xi, top if up else tip),
            arrowprops={"arrowstyle": "-|>", "color": C_BLACK, "linewidth": 0.9},
        )
        ax.text(xi, tip + 0.008, "恶化" if up else "改善", ha="center",
                va="bottom", fontsize=5.8, color=C_RED if up else C_GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=6.2)
    ax.set_ylabel("校准误差")
    ax.set_ylim(0, 0.86)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.axvline(1.5, color=C_GREY, linewidth=0.9, linestyle=(0, (4, 2.5)),
               zorder=1)
    ax.text(0.5, 0.80, "ECE", ha="center", va="center", fontsize=7.6,
            color=C_BLACK, fontweight="bold", transform=ax.get_xaxis_transform())
    ax.text(2.5, 0.80, "MCE", ha="center", va="center", fontsize=7.6,
            color=C_BLACK, fontweight="bold", transform=ax.get_xaxis_transform())
    ax.legend(loc="upper left", handletextpad=0.4, borderaxespad=0.3,
              handlelength=1.5)
    _origin_box(ax)
    _panel(ax, "c")

    fig.tight_layout(w_pad=1.4)
    _save(fig, "fig5_calibration.png")


def fig_robustness(a: dict, b: dict) -> None:
    """图 6：灰度扰动下的相对保持率与量化偏差（两数据集）。"""
    cond = a["robustness"]["conditions"]
    names = list(cond)
    base = cond["brightness_gain@1.0"]
    rel_a = [cond[n]["retention"] / base["retention"] for n in names]
    qdev_a = [cond[n]["quant_dev_mean"] for n in names]
    cond_b = b["robustness"]["conditions"]
    base_b = cond_b["brightness_gain@1.0"]
    rel_b = [cond_b[n]["retention"] / base_b["retention"] for n in names]
    qdev_b = [cond_b[n]["quant_dev_mean"] for n in names]
    labels = ["恒等", "增益\n0.7", "增益\n1.3", "偏移\n-0.15", "偏移\n+0.15",
              "γ = 0.6", "γ = 1.6", "对比度\n0.7"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.90))
    x = np.arange(len(names))
    w = 0.36

    for ax, (sa, sb), ylab, ylim, tol, toltxt, letter in (
        (axes[0], (rel_a, rel_b), "相对保持率", (0.85, 1.06), 1.0, "基线 1.00", "a"),
        (axes[1], (qdev_a, qdev_b), "量化平均相对偏差", (0.0, 0.135), 0.10,
         "容差 0.10", "b"),
    ):
        ax.bar(x - w / 2, sa, w, color=C_BLUE, edgecolor=C_BLACK, linewidth=0.7,
               label="测试集", zorder=2)
        ax.bar(x + w / 2, sb, w, color=C_RED, edgecolor=C_BLACK, linewidth=0.7,
               hatch="///", label="验证集", zorder=2)
        ax.axhline(tol, color=C_GREY, linestyle=(0, (4, 2.5)), linewidth=0.9,
                   zorder=1)
        ax.text(len(names) - 0.45, tol + (0.006 if tol > 1 else 0.004), toltxt,
                color=C_GREY, fontsize=6.8, ha="right", va="bottom")
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_xlim(-0.6, len(names) - 0.4)
        if tol > 1:
            ax.yaxis.set_major_locator(MultipleLocator(0.05))
            ax.yaxis.set_minor_locator(MultipleLocator(0.025))
        else:
            ax.yaxis.set_major_locator(MultipleLocator(0.025))
            ax.yaxis.set_minor_locator(MultipleLocator(0.0125))
        ax.legend(loc="lower left", handletextpad=0.4, borderaxespad=0.3,
                  handlelength=1.5)
        _panel(ax, letter)

    for ax in axes:
        _origin_box(ax)
    fig.tight_layout(w_pad=1.2)
    _save(fig, "fig6_robustness.png")


def fig_error_structure(a: dict, b: dict) -> None:
    """图 7：可复现的误检方向结构（两组数据集的流向图）。

    左侧为真值类别、右侧为预测类别，连线粗细正比于误检数。红 = 关注级缺陷被
    误检为一般关注缺陷（触发误检风险 Ⅰ 类）；蓝 = 关注级内部相互误检；
    橙 = 一般关注缺陷被误检为关注级；灰 = 一般关注级内部误检。
    """
    FOCUS = {"裂纹", "未熔合", "未焊透", "内凹", "咬边"}

    def color_of(gt: str, pred: str) -> str:
        if gt in FOCUS and pred not in FOCUS:
            return C_RED
        if gt not in FOCUS and pred in FOCUS:
            return C_ORANGE
        if gt in FOCUS and pred in FOCUS:
            return C_BLUE
        return C_GREY

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.10))

    for ax, data, letter, title in (
        (axes[0], a, "a", f"测试集（{a['std501807']['standard']['fp_extra']} 个未匹配预测）"),
        (axes[1], b, "b", f"验证集（{b['std501807']['standard']['fp_extra']} 个未匹配预测）"),
    ):
        pairs = data["std501807"]["standard"]["fd_pairs"]
        gt_nodes = sorted({k.split("->")[0] for k in pairs})
        pr_nodes = sorted({k.split("->")[1] for k in pairs})

        def ypos(nodes, name):
            if name not in nodes:
                return None
            i = nodes.index(name)
            return 12 + (len(nodes) - 1 - i) * (76 / max(len(nodes) - 1, 1))

        for name in gt_nodes:
            y = ypos(gt_nodes, name)
            face = F_BLUE if name in FOCUS else F_GREY
            edge = C_BLUE if name in FOCUS else C_GREY
            _dbox(ax, 1.0, y - 4.2, 24.0, 8.4, name, [], face=face, edge=edge,
                  title_fs=7.0)
        for name in pr_nodes:
            y = ypos(pr_nodes, name)
            face = F_BLUE if name in FOCUS else F_GREY
            edge = C_BLUE if name in FOCUS else C_GREY
            _dbox(ax, 75.0, y - 4.2, 24.0, 8.4, name, [], face=face, edge=edge,
                  title_fs=7.0)

        for key, cnt in sorted(pairs.items(), key=lambda kv: -kv[1]):
            gt, pred = key.split("->")
            y0, y1 = ypos(gt_nodes, gt), ypos(pr_nodes, pred)
            col = color_of(gt, pred)
            down = y0 > y1
            _arrow(ax, (25.0, y0), (75.0, y1), color=col,
                   lw=0.7 + cnt * 0.42, rad=0.16 if down else -0.16, ms=7.0)
            # 标注沿 x 向错开：一对反向的误检对（如 咬边↔圆形缺陷）中点重合，
            # 不错开则两个数字叠在一起。
            ax.text(58.0 if down else 42.0, (y0 + y1) / 2 + (1.8 if down else -1.8),
                    str(cnt), ha="center", va="center", fontsize=6.4, color=col,
                    zorder=6,
                    bbox={"boxstyle": "round,pad=0.12", "facecolor": "white",
                          "edgecolor": "none", "alpha": 0.85})

        ax.text(13.0, 94.5, "真值类别", ha="center", va="center", fontsize=7.0,
                fontweight="bold")
        ax.text(87.0, 94.5, "预测类别", ha="center", va="center", fontsize=7.0,
                fontweight="bold")
        ax.set_title(title, fontsize=7.4, pad=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        _panel(ax, letter)

    # 图例：置于两面板下方居中，避免与流向节点重叠
    handles = [
        plt.Line2D([], [], color=c, linewidth=1.6, label=lab)
        for c, lab in (
            (C_RED, "关注级 → 一般关注级（触发误检风险 Ⅰ 类）"),
            (C_BLUE, "关注级内部相互误检"),
            (C_ORANGE, "一般关注级 → 关注级"),
            (C_GREY, "一般关注级内部误检"),
        )
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               handletextpad=0.4, columnspacing=1.2, bbox_to_anchor=(0.5, -0.035))

    fig.tight_layout(w_pad=1.0)
    _save(fig, "fig7_error_structure.png")


def fig_real_profile() -> None:
    """图 8：真实域运行画像（75 张真实底片、340 条检出记录）。"""
    import sqlite3

    con = sqlite3.connect(_ROOT / "data/scan.db")
    cls = dict(con.execute("select class_id, count(*) from defects group by class_id"))
    gates = dict(
        con.execute("select quality_pass || '-' || evaluable, count(*) from images group by 1")
    )
    lvl = dict(con.execute("select ifnull(joint_level,'未定级'), count(*) from images group by 1"))
    con.close()

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.60))

    ax = axes[0]
    names = [CLASS_NAMES[i] for i in sorted(cls)]
    vals = [cls[i] for i in sorted(cls)]
    ax.bar(range(len(vals)), vals, 0.6, color=C_BLUE, edgecolor=C_BLACK,
           linewidth=0.7, zorder=2)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.04, str(v), ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, fontsize=7.0)
    ax.set_ylabel("检出数")
    ax.set_ylim(0, 400)
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(50))
    _panel(ax, "a")

    ax = axes[1]
    gnames = ["不合格\n不可评", "不合格\n可评", "合格\n不可评", "合格\n可评"]
    gvals = [gates.get("0-0", 0), gates.get("0-1", 0), gates.get("1-0", 0), gates.get("1-1", 0)]
    gcols = [C_RED, C_ORANGE, C_GREY, C_GREEN]
    ax.bar(range(4), gvals, 0.6, color=gcols, edgecolor=C_BLACK, linewidth=0.7,
           zorder=2)
    for i, v in enumerate(gvals):
        ax.text(i, v + 1.5, str(v), ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(gnames, fontsize=6.8)
    ax.set_ylabel("底片数")
    ax.set_ylim(0, 70)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    _panel(ax, "b")

    ax = axes[2]
    lnames = list(lvl)
    lvals = [lvl[k] for k in lnames]
    ax.bar(range(len(lvals)), lvals, 0.5, color=C_VIOLET, edgecolor=C_BLACK,
           linewidth=0.7, zorder=2)
    for i, v in enumerate(lvals):
        ax.text(i, v + 1.5, str(v), ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(range(len(lvals)))
    ax.set_xticklabels(["未定级", "IV 级"], fontsize=7.0)
    ax.set_ylabel("底片数")
    ax.set_ylim(0, 88)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    _panel(ax, "c")

    for ax in axes:
        _origin_box(ax)
    fig.tight_layout(w_pad=1.1)
    _save(fig, "fig8_real_profile.png")


def main() -> None:
    _setup()
    a = _load("qualification_protocol_synthetic.json")
    b = _load("qualification_protocol_golden_v3.json")
    print("生成插图：")
    fig_protocol_architecture()
    fig_data_assets()
    fig_standard_grading(a, b)
    fig_pod(a, b)
    fig_calibration(a)
    fig_robustness(a, b)
    fig_error_structure(a, b)
    fig_real_profile()
    print("完成。")


if __name__ == "__main__":
    main()
