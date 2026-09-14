"""端到端验证：本地大模型参与评片（真实底片 + 真实 llama-server）。

用途
----
证明"软件真的能用本地模型做评片"这一交付口径，且**不依赖运行中的开发版后端**：
脚本自建一次性用户数据目录（DB/报告/日志全隔离，绝不污染 %LOCALAPPDATA% 既有数据），
以**安装版口径**（llm.mode=external，只连接本机已有服务）装配真实应用，然后：

  1. POST /api/v1/report  真实底片 → 全链路评片（门禁→检测→评级→PDF）
  2. GET  /api/v1/report/{id}/narrative → 本地大模型生成评片结论

判定口径
--------
- 缺陷级别始终由算法（StandardGrader，NB/T47013.2）给出；`grade_preliminary=true`
  表示底片质量门禁未过（8bit 翻拍）而走**预筛通道**，结论为"非正式级别"。
- 大模型只做结论撰述，**不得输出级别/合格判定**（compliance 红线）。

用法
----
    # 先确保本机 llama-server 在跑（或 Ollama 等 OpenAI 兼容端点）
    backend/.venv/Scripts/python.exe -B scripts/verify_local_llm_report.py
    # 指定底片与端点：
    ... --image 定检/定检/PG101-1-2.jpg --endpoint http://127.0.0.1:18780

退出码：0 = 两项均通过；1 = 有环节未通过（原因打印在报告里）。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# 环境必须在导入 backend.app.main 之前落地：create_app() 在导入期构造，
# 配置对象随之固化（含 paths/db 锚点、llm 模式）。
# ---------------------------------------------------------------------------


def _prepare_env(user_data_dir: Path, endpoint: str) -> None:
    os.environ["SCANDETECTION_USER_DATA_DIR"] = str(user_data_dir)
    # 安全面：本地验证不启用 IPC 令牌与限流（与 conftest 同口径）
    os.environ["SCAN_IPC__ENFORCE"] = "false"
    os.environ["SCAN_RATE_LIMIT"] = "0"
    os.environ["SCAN_AUTH__GUEST_MODE"] = "true"
    # 评片门禁：真实底片是 8bit 翻拍 JPG，按配置降级放行（告警留档）
    os.environ["SCAN_GATE__ALLOW_8BIT"] = "true"
    # 印字识别关掉：与"本地大模型评片"验证无关，且会拖慢单次评片
    os.environ["SCAN_STAMP__ENABLED"] = "false"
    # 检测器：必须用训练权重，禁止回落到连通域基线（否则证明不了模型在跑）
    os.environ["SCAN_DETECT__BASELINE_ENABLED"] = "false"
    # 本地大模型：安装版口径 —— 不自拉起，只连接本机已有服务
    os.environ["SCAN_LLM__ENABLED"] = "true"
    os.environ["SCAN_LLM__MODE"] = "external"
    os.environ["SCAN_LLM__EXTERNAL_ENDPOINT"] = endpoint


def _h(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def main() -> int:
    # 行缓冲：输出被块缓冲时（重定向到文件）看不到进度，故障排查像"静默挂死"。
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="本地大模型评片端到端验证")
    ap.add_argument("--image", default="定检/定检/PG101-1-2.jpg", help="底片路径（相对仓库根）")
    ap.add_argument("--endpoint", default="http://127.0.0.1:18780", help="本地 OpenAI 兼容端点")
    ap.add_argument(
        "--thickness-mm", type=float, default=12.0, help="母材厚度（评级必需，NB/T47013.2）"
    )
    ap.add_argument(
        "--pixel-spacing-mm",
        type=float,
        default=0.1,
        help="像素标定 mm/px（验证用给定值）。不给则评级因无量纲依据而熔断，"
        "只会得到『需人工复核』而拿不到级别——这是标准要求，不是缺陷。",
    )
    ap.add_argument("--keep", action="store_true", help="保留一次性用户数据目录（排查用）")
    ap.add_argument(
        "--dump-after",
        type=float,
        default=0.0,
        metavar="SEC",
        help="超过 SEC 秒未结束则打印所有线程的调用栈并退出（排查挂死用，0=关闭）",
    )
    args = ap.parse_args()

    # 挂死自证：以阻塞形态出现的问题（等 DB 锁/等 HTTP 槽位/等线程池）光看日志
    # 无从定位，必须把当时的线程栈打出来。
    if args.dump_after > 0:
        import faulthandler

        faulthandler.dump_traceback_later(args.dump_after, exit=True)

    film = (ROOT / args.image).resolve()
    if not film.is_file():
        print(f"[FATAL] 底片不存在: {film}")
        return 1

    # 一次性数据目录必须落在**仓库同卷**并作为 CWD：
    # crypto 的历史密钥迁移以 CWD 为源（<cwd>/data/.crypto_key），跨卷 os.replace
    # 会失败；更危险的是——若 CWD 恰好是仓库根，源就是**仓库真实的开发密钥**，
    # 迁移成功即把它搬走、随后清理临时目录会连它一起删掉，导致既有加密影像
    # 永久不可解。把 CWD 切进临时目录后，源与目标同路径，迁移分支自然跳过。
    run_dir = Path(tempfile.mkdtemp(prefix=".verify-run-", dir=ROOT))
    os.chdir(run_dir)
    _prepare_env(run_dir, args.endpoint)
    assert not (run_dir / "data" / ".crypto_key").exists()
    assert Path.cwd() == run_dir, "必须切到一次性目录，避免迁移/删除仓库真实密钥"

    from fastapi.testclient import TestClient

    from backend.app.main import app

    def _wait_ready(client, deadline_sec: float = 240.0) -> dict:
        """等待应用初始化完成（检测器装载 + 大模型管理器启动）。

        检测器权重是**异步装载**的（lifespan 里不阻塞启动），刚进 lifespan 时
        /health 会报 ``detector=loading``、``uri=``，此时做判定必然误判；大模型
        管理器同样是 start_async。必须轮询到"加载完成"再核验，否则得到的是
        启动瞬间的假象而不是结论。
        """
        import time

        t0 = time.monotonic()
        health: dict = {}
        # disabled 也视为"未就绪"：本次验证显式开启 llm.enabled，若此刻报 disabled
        # 说明管理器尚未装配（start_async 还没跑），不是真实结论。
        pending = {"starting", "disabled"}
        while time.monotonic() - t0 < deadline_sec:
            health = client.get("/api/v1/health").json()
            engine = client.get("/api/v1/llm/status").json().get("engine", {})
            if health.get("detector") != "loading" and engine.get("state") not in pending:
                return health
            time.sleep(1.0)
        print(f"[WARN] 等待就绪超时（{deadline_sec:.0f}s），以下为超时瞬间状态")
        return health

    failures: list[str] = []
    try:
        with TestClient(app) as client:
            _h("① 引擎核验：本地大模型与检测器")
            health = _wait_ready(client)
            detector = health.get("detector")
            degraded = health.get("detector_degraded")
            uri = health.get("uri")
            print(f"detector        = {detector}")
            print(f"detector_degraded = {degraded}")
            print(f"model uri       = {uri}")
            if detector != "trained_yolo" or degraded is not False:
                failures.append("检测器未加载训练权重（回落基线/降级）")

            llm = client.get("/api/v1/llm/status").json()
            engine = llm.get("engine", {})
            print(f"llm.mode        = {engine.get('mode')}")
            print(f"llm.state       = {engine.get('state')}")
            print(f"llm.endpoint    = {engine.get('endpoint')}")
            if engine.get("state") != "ready":
                failures.append(
                    f"本地大模型未就绪: state={engine.get('state')} err={engine.get('error')}"
                )

            _h("② 评片全链路（真实底片 + 预筛级别通道）")
            print(f"底片            = {film.name}  ({film.stat().st_size / 1024:.0f} KB)")
            print(
                f"验证用输入      = 母材厚度 {args.thickness_mm} mm、像素标定 "
                f"{args.pixel_spacing_mm} mm/px（真实底片无标定，此处为验证给定）"
            )
            with film.open("rb") as fh:
                resp = client.post(
                    "/api/v1/report",
                    headers={"X-Operator-Name": "verify-local-llm"},
                    files={"image": (film.name, fh, "image/jpeg")},
                    data={
                        "base_metal_thickness_mm": str(args.thickness_mm),
                        "pixel_spacing_mm": str(args.pixel_spacing_mm),
                        "allow_preliminary_grade": "true",
                        "force": "true",
                    },
                )
            print(f"HTTP            = {resp.status_code}")
            if resp.status_code != 200:
                print(resp.text[:800])
                failures.append(f"评片失败 HTTP {resp.status_code}")
                rep = {}
            else:
                rep = resp.json()
            report_id = rep["report_id"]
            for key in (
                "report_id",
                "joint_level",
                "grade_preliminary",
                "need_review",
                "evaluable",
                "defect_count",
                "disposition",
                "disposition_label",
                "density",
                "density_ok",
                "iqi_pass",
                "photo_mode",
            ):
                print(f"{key:<18}= {rep.get(key)!r}")
            print("warnings        =")
            for w in rep.get("warnings") or []:
                print(f"  - {w}")
            print("basis           =")
            for b in rep.get("basis") or []:
                print(f"  - {b}")
            if rep.get("defect_count", 0) <= 0:
                failures.append("真实底片未检出任何缺陷（检测链路可疑）")
            if not rep.get("grade_preliminary"):
                failures.append("预筛级别未生效（grade_preliminary 非 true）")

            _h("③ 本地大模型评片结论")
            report_id = rep.get("report_id")
            n: dict = {}
            if report_id:
                nar = client.get(f"/api/v1/report/{report_id}/narrative")
                print(f"HTTP            = {nar.status_code}")
                if nar.status_code != 200:
                    print(nar.text[:800])
                    failures.append(f"评片结论接口失败 HTTP {nar.status_code}")
                else:
                    n = nar.json()
                    print(f"status          = {n.get('status')}")
                    print(f"model           = {n.get('model')}")
                    print(f"elapsed_ms      = {n.get('elapsed_ms')}")
                    if n.get("reason"):
                        print(f"reason          = {n.get('reason')}")
                    print()
                    print("---- 结论正文 ----")
                    print(n.get("text") or "(空)")
                    print("---- 免责声明 ----")
                    print(n.get("disclaimer") or "(空)")
                    if n.get("status") != "ok":
                        failures.append(
                            f"大模型结论未生成: status={n.get('status')} reason={n.get('reason')}"
                        )
                    if not n.get("disclaimer"):
                        failures.append("结论缺少免责声明")
            else:
                print("跳过：评片未产出 report_id")

            _h("④ 合规红线核验：结论不得给出级别/合格判定")
            # 允许出现"级别由持证人判定""不得据此判定合格"这类**否定式/转交式**表述，
            # 因此只检查是否出现"结论性级别"形态（如 "Ⅳ级" / "评定为不合格"）。
            text = (n.get("text") or "").replace(" ", "")
            level_hits = [
                kw
                for kw in ("Ⅰ级", "Ⅱ级", "Ⅲ级", "Ⅳ级", "I级", "II级", "III级", "IV级")
                if kw in text
            ]
            verdict_hits = [
                kw
                for kw in (
                    "评定为合格",
                    "评定为不合格",
                    "结论为合格",
                    "结论为不合格",
                    "判定为合格",
                    "判定为不合格",
                )
                if kw in text
            ]
            print(f"结论性级别命中  = {level_hits or '(无)'}")
            print(f"合格判定命中    = {verdict_hits or '(无)'}")
            if level_hits or verdict_hits:
                failures.append(
                    f"合规红线被突破：大模型输出了级别/合格判定 {level_hits + verdict_hits}"
                )
            print("判定            = 见下方汇总")

        _report(failures, run_dir, args.keep)
        return 1 if failures else 0
    except Exception:  # noqa: BLE001 - 验证脚本兜底：任何异常都必须转成可读报告
        # 异常也要产出可读报告，避免"静默失败"
        import traceback

        failures.append("脚本异常，见下方 traceback")
        traceback.print_exc()
        _report(failures, run_dir, args.keep)
        return 1


def _report(failures: list[str], run_dir: Path, keep: bool) -> int:
    _h("汇总")
    if failures:
        print(f"结论: FAIL（{len(failures)} 项未通过）")
        for f in failures:
            print(f"  ✗ {f}")
    else:
        print("结论: PASS —— 本地模型确实参与评片，全链路可用")
    if not keep:
        shutil.rmtree(run_dir, ignore_errors=True)
        print(f"临时数据已清理: {run_dir}")
    else:
        print(f"临时数据保留: {run_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
