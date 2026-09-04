"""72h 稳定性长跑（S-07）：循环批量评片 + RSS 采样 + 泄漏趋势判定。

可配置时长（默认 72 小时，支持 ``--hours 0.01`` 冒烟），每轮对一批合成底片
跑检测推理（复用检测器装配链路，不落库、不生成报告——长跑关注的是推理进程
的资源稳定性，而非业务数据沉淀），周期采样进程 RSS 与每轮耗时，产出 soak
报告 JSON（起止时间、轮次、成功率、RSS 曲线摘要、泄漏趋势=线性回归斜率）。

泄漏判定（诚实口径）：对 RSS 曲线做最小二乘线性回归，斜率 > ``leak_slope_mb_per_round``
（默认 0.5 MB/轮，可用 ``--leak-slope`` 调整）判为"疑似泄漏"——斜率仅是趋势
证据，非结论；报告同时给出首尾差与 RSS 曲线（限幅落盘），供人工复核。

检测器：默认装配 config 配置的检测器（kind=trained_yolo 时需权重存在；
开发/CI 可 ``--baseline`` 强制基线连通域检测器，不依赖权重）。

用法::

    python -m backend.scripts.soak_72h [--hours 72] [--rounds N]
        [--batch-size 8] [--interval 0] [--out data/compliance] [--baseline]

退出码：全部轮次成功 0；任一轮异常或成功率低于 100% → 1（异常退出非零）。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from backend.scripts.recovery_drill import _INSTALL_ROOT  # 复用安装根锚点

_LEAK_DEFAULT_MB_PER_ROUND = 0.5

# 长跑采样向量的保留上限（S-07 加固）：72h 高频轮次下，round_elapsed / rss_curve
# 若无界 append 会由 Python float 对象自身撑到数 GB，造成"工具先内存泄漏、RSS 斜率
# 虚高、误报 FAIL"的假信号。超上限时按"取偶数序号"成对压缩（长度折半，趋势形状
# 近似保留），保证泡水测量的只是检测器的真实资源稳定性，而非脚本自身记账膨胀。
_SOAK_SAMPLE_CAP = 20000


def _bounded_append(vec: list[float], val: float, cap: int = _SOAK_SAMPLE_CAP) -> None:
    """追加一个采样点；超过上限时压缩为最长折半（丢弃奇数下标，保持整体趋势）。"""
    vec.append(val)
    if len(vec) > cap:
        del vec[1::2]


def synthesize_film(w: int = 512, h: int = 512, seed: int = 0) -> np.ndarray:
    """合成一张含类缺陷暗斑的 8bit 灰度底片（确定性，长跑输入稳定）。"""
    rng = np.random.default_rng(seed)
    img = rng.normal(128.0, 8.0, size=(h, w)).clip(0, 255).astype(np.uint8)
    for _ in range(6):  # 暗斑模拟气孔/夹渣
        cy, cx = int(rng.integers(40, h - 40)), int(rng.integers(40, w - 40))
        r = int(rng.integers(3, 9))
        yy, xx = np.ogrid[:h, :w]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        img[mask] = np.clip(img[mask].astype(int) - int(rng.integers(60, 110)), 0, 255).astype(
            np.uint8
        )
    return img


def _linear_regression_slope(ys: list[float]) -> float:
    """对 RSS 序列（x=轮次）做最小二乘斜率（MB/轮）。样本 <2 返回 0.0。"""
    if len(ys) < 2:
        return 0.0
    x = np.arange(len(ys), dtype=float)
    y = np.asarray(ys, dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    return slope


def run_soak(
    *,
    hours: float = 72.0,
    rounds: int | None = None,
    batch_size: int = 8,
    interval_sec: float = 0.0,
    leak_slope: float = _LEAK_DEFAULT_MB_PER_ROUND,
    force_baseline: bool = False,
    sampler=None,
    detector=None,
) -> dict:
    """执行长跑循环，返回报告 dict（不落盘，由 main / 测试落盘）。

    rounds 优先于 hours（给冒烟/测试用）；两者都给时以先到者为准。
    sampler/detector 可注入（测试确定性）。
    """
    started_wall = datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()

    if detector is None:
        from backend.domain.detect import get_detector
        from backend.infra.config import load_config

        cfg = load_config()
        if force_baseline or cfg.detect.kind == "baseline_blob" or cfg.detect.baseline_enabled:
            detector = get_detector("baseline_blob")
        else:
            from backend.app.dependencies import _resolve_model_uri

            detector = get_detector(
                "trained_yolo",
                model_uri=_resolve_model_uri(cfg.model.default_uri),
                backend=cfg.model.backend,
                providers=cfg.model.providers,
            )

    sample = sampler or _default_sampler
    rounds_done = 0
    failures = 0
    errors: list[str] = []
    rss_curve: list[float] = []
    round_elapsed: list[float] = []
    deadline = t0 + max(0.0, hours) * 3600.0

    while True:
        if rounds is not None and rounds_done >= rounds:
            break
        if rounds is None and time.perf_counter() >= deadline:
            break
        rt0 = time.perf_counter()
        try:
            for i in range(batch_size):
                frame = synthesize_film(seed=rounds_done * batch_size + i)
                detector.infer(frame, conf=0.3, iou=0.5)
        except Exception as exc:  # noqa: BLE001 - 轮次失败计入成功率，不中断长跑
            failures += 1
            errors.append(f"round {rounds_done}: {type(exc).__name__}: {exc}"[:200])
        rounds_done += 1
        _bounded_append(round_elapsed, round(time.perf_counter() - rt0, 4))
        rss_mb, source = sample()
        if rss_mb >= 0:
            _bounded_append(rss_curve, round(rss_mb, 1))
        if interval_sec > 0:
            time.sleep(interval_sec)

    slope = _linear_regression_slope(rss_curve)
    leak_suspected = bool(rss_curve) and slope > leak_slope
    success_rate = (rounds_done - failures) / rounds_done if rounds_done else 0.0
    report: dict = {
        "drill": "soak",
        "started_at": started_wall,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "planned_hours": hours,
        "rounds": rounds_done,
        "batch_size": batch_size,
        "success_rate": round(success_rate, 4),
        "failures": failures,
        "errors": errors[:50],
        "rss_sample_source": getattr(sample, "source_name", "psutil-or-fallback"),
        "rss_curve_mb": rss_curve[-2000:],  # 限幅落盘，防超长跑报告膨胀
        "rss_first_mb": rss_curve[0] if rss_curve else None,
        "rss_last_mb": rss_curve[-1] if rss_curve else None,
        "rss_max_mb": max(rss_curve) if rss_curve else None,
        "round_elapsed_sec_avg": (
            round(sum(round_elapsed) / len(round_elapsed), 4) if round_elapsed else None
        ),
        "rss_slope_mb_per_round": round(slope, 5),
        "leak_slope_threshold": leak_slope,
        "leak_suspected": leak_suspected,
        "conclusion": "PASS" if failures == 0 and not leak_suspected else "FAIL",
    }
    return report


def _default_sampler():
    """默认采样：infra.watchdog.sample_rss_mb（psutil → ctypes/resource 尽力回退）。"""
    from backend.infra.watchdog import sample_rss_mb

    return sample_rss_mb()


def main() -> int:
    ap = argparse.ArgumentParser(description="S-07 稳定性长跑（默认 72h，支持冒烟）")
    ap.add_argument("--hours", type=float, default=72.0, help="长跑时长（小时），0.01 可冒烟")
    ap.add_argument("--rounds", type=int, default=None, help="固定轮次（优先于 hours，测试用）")
    ap.add_argument("--batch-size", type=int, default=8, help="每轮评片张数")
    ap.add_argument("--interval", type=float, default=0.0, help="轮间休息秒数（降负载）")
    ap.add_argument("--leak-slope", type=float, default=_LEAK_DEFAULT_MB_PER_ROUND,
                    help="泄漏判定斜率阈值（MB/轮）")
    ap.add_argument("--baseline", action="store_true", help="强制基线检测器（不依赖训练权重）")
    ap.add_argument("--out", default="data/compliance", help="soak 报告输出目录")
    args = ap.parse_args()

    report = run_soak(
        hours=args.hours,
        rounds=args.rounds,
        batch_size=args.batch_size,
        interval_sec=args.interval,
        leak_slope=args.leak_slope,
        force_baseline=args.baseline,
    )
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = _INSTALL_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"soak_{ts}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[soak] {report['conclusion']} rounds={report['rounds']} "
        f"success={report['success_rate']:.2%} rss_slope={report['rss_slope_mb_per_round']}MB/轮 "
        f"report={out_path}"
    )
    return 0 if report["conclusion"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
