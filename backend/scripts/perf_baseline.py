#!/usr/bin/env python3
"""性能基线实测（GB/T 25000.51 性能效率：以可复现实测数据支撑 NFR 声明）。

口径与诚实边界：
- 输入为合成 8bit 底片（512×512，含类缺陷暗斑，确定性种子）；
- 测量覆盖完整链路：上传 → 质量门禁 → 检测 → 量化 → 评级 → 落库 →
  影像静态加密（SM4）→ PDF 报告生成；
- 检测器为当前配置实际装载者：无训练权重时即基线连通域检测器（结论中
  如实标注）。YOLO 权重路径（GPU/CPU ≤150/500ms NFR）须在有权重与 GPU
  的机器上重跑本脚本留档；
- 鉴权以依赖注入方式旁路（与测试同口径）：认证开销不进入测量，且鉴权
  真实链路另有 test_auth 覆盖，不影响本脚本结论的有效范围。

用法（仓库根目录）：
    python -m backend.scripts.perf_baseline [--n 12] [--warmup 2]

输出：
    data/compliance/perf_baseline_<时间戳>.json / .md
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

# Request 必须模块级可解析：脚本启用 PEP 563（延迟注解），FastAPI 解析
# override 函数注解时在模块全局查找名字——函数内导入会解析失败，把 request
# 误判为必需 query 参数（症状：POST /report 422）。
from fastapi import Request

_INSTALL_ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = _INSTALL_ROOT / "data" / "compliance"


def _prepare_env(tmp: Path) -> None:
    """隔离数据目录 + 测试口径开关（须在导入 backend.app.main 之前生效）。"""
    os.environ["SCAN_PATHS__DB_PATH"] = str(tmp / "perf.db")
    os.environ["SCAN_PATHS__IMAGES_DIR"] = str(tmp / "images")
    os.environ["SCAN_PATHS__REPORTS_DIR"] = str(tmp / "reports")
    os.environ.setdefault("SCAN_IPC__ENFORCE", "false")
    os.environ.setdefault("SCAN_RATE_LIMIT", "0")
    # 检测器确定性：与 conftest 同口径（基线检测器），保证结果可复现
    os.environ.setdefault("SCAN_DETECT__BASELINE_ENABLED", "true")
    os.environ.setdefault("SCAN_GATE__ALLOW_8BIT", "true")
    for sub in ("images", "reports"):
        (tmp / sub).mkdir(parents=True, exist_ok=True)


def _encode_png(img) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("合成底片 PNG 编码失败")
    return buf.tobytes()


def main() -> int:
    ap = argparse.ArgumentParser(description="性能基线实测（完整评片链路）")
    ap.add_argument("--n", type=int, default=12, help="正式测量张数（默认 12）")
    ap.add_argument("--warmup", type=int, default=2, help="预热张数（不计入统计，默认 2）")
    args = ap.parse_args()

    # 数据目录隔离须先于 backend 导入生效（create_app 在导入期读取配置）
    _prepare_env(Path(tempfile.mkdtemp(prefix="perf_baseline_")))

    # 环境就绪后再导入
    from fastapi.testclient import TestClient

    from backend.app import auth as auth_mod
    from backend.app.main import app
    from backend.infra.watchdog import sample_rss_mb
    from backend.scripts.soak_72h import synthesize_film

    def _fake_principal(request: Request):
        principal = auth_mod.Principal(
            account_id="perf-baseline", username="性能实测", role="sysadmin"
        )
        request.state.principal = principal
        return principal

    app.dependency_overrides[auth_mod.get_principal] = _fake_principal

    films = [
        (f"perf_{i:03d}.png", _encode_png(synthesize_film(seed=100 + i)))
        for i in range(args.warmup + args.n)
    ]

    result: dict = {
        "generated_at": datetime.now(UTC).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"),
        "n": args.n,
        "warmup": args.warmup,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "detector": "baseline(blob)（无训练权重；YOLO 路径待有权重机器重测）",
    }

    with TestClient(app) as client:
        # 1. 服务可达：/health 即返回（registry 后台线程装配，见 main.py 设计）
        t0 = time.perf_counter()
        resp = client.get("/api/v1/health")
        result["health_ready_sec"] = round(time.perf_counter() - t0, 3)
        result["health_ok"] = resp.status_code == 200
        from backend.app.dependencies import get_registry

        result["encrypt"] = bool(get_registry().config.security.encrypt)

        # 2. 首张出片（含后台装配完成等待 + 模型/加密器首次初始化）+ 热身
        for i, (name, png) in enumerate(films[: args.warmup]):
            t = time.perf_counter()
            client.post(
                "/api/v1/report",
                files={"image": (name, png, "image/png")},
                data={"pixel_spacing_mm": "0.1", "base_metal_thickness_mm": "20", "force": "true"},
            )
            if i == 0:
                result["first_report_sec"] = round(time.perf_counter() - t, 3)

        # 3. 正式测量：完整评片链路逐张计时
        rss_before, _src = sample_rss_mb()
        per_image_ms: list[float] = []
        for name, png in films[args.warmup :]:
            t = time.perf_counter()
            resp = client.post(
                "/api/v1/report",
                files={"image": (name, png, "image/png")},
                data={
                    "pixel_spacing_mm": "0.1",
                    "base_metal_thickness_mm": "20",
                    "force": "true",
                },
            )
            elapsed_ms = (time.perf_counter() - t) * 1000.0
            if resp.status_code != 200:
                result["error"] = f"评片请求失败 {resp.status_code}: {resp.text[:200]}"
                break
            per_image_ms.append(round(elapsed_ms, 1))
        rss_after, rss_src = sample_rss_mb()

    if per_image_ms:
        per_image_ms.sort()
        result["per_image_ms"] = per_image_ms
        result["avg_ms"] = round(statistics.fmean(per_image_ms), 1)
        result["p50_ms"] = per_image_ms[len(per_image_ms) // 2]
        result["p95_ms"] = per_image_ms[max(0, int(len(per_image_ms) * 0.95) - 1)]
        result["throughput_per_min"] = round(60000.0 / result["avg_ms"], 1)
    result["rss_before_mb"] = rss_before
    result["rss_after_mb"] = rss_after
    result["rss_source"] = rss_src

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).replace(tzinfo=None).strftime("%Y%m%d_%H%M%S")
    json_path = _OUT_DIR / f"perf_baseline_{stamp}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if "error" in result or not per_image_ms:
        print(f"FAIL: {result.get('error', '无有效样本')}")
        print(f"partial: {json_path}")
        return 1

    md = [
        "# 性能基线实测（完整评片链路）",
        "",
        f"- 时间：{result['generated_at']}；样本：{result['n']} 张（预热 {result['warmup']} 张）",
        f"- 口径：合成 8bit 底片 + {result['detector']}；含上传/门禁/检测/评级/加密落库/PDF 全链路",
        f"- 环境：Python {result['python']} / {result['platform']}；静态加密={'开' if result['encrypt'] else '关'}",
        "",
        "| 指标 | 数值 |",
        "|---|---|",
        f"| /health 就绪 | {result['health_ready_sec']} s（registry 后台装配，不阻塞） |",
        f"| 首张出片（含装配等待+首次初始化） | {result['first_report_sec']} s |",
        f"| 单张平均（稳态） | {result['avg_ms']} ms |",
        f"| 单张 P50 / P95 | {result['p50_ms']} / {result['p95_ms']} ms |",
        f"| 吞吐 | {result['throughput_per_min']} 张/分钟 |",
        f"| RSS（测前 → 测后） | {rss_before} → {rss_after} MB（{rss_src}） |",
        "",
        "> 注：本表为基线检测器口径的链路实测；规格书 NFR 的 GPU/CPU 推理",
        "> 指标（≤150/500ms）须在装载 YOLO 权重的机器上重跑本脚本后补充。",
        "",
    ]
    md_path = _OUT_DIR / f"perf_baseline_{stamp}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(
        f"OK: avg={result['avg_ms']}ms p95={result['p95_ms']}ms first={result['first_report_sec']}s"
    )
    print(f"report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
