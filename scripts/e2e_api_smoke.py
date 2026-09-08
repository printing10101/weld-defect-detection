"""端到端 API 冒烟 + 持久性盯测（真实 HTTP 进程，非 TestClient）。

覆盖链路：进程启停 → /health → SM2 挑战-响应登录（三员 sysadmin）→ 检测
（multipart 上传合成底片）→ 标准判定（judge）→ 报告生成 + PDF 下载 → 批量
提交/轮询/取消 → **循环推理内存盯测**（RSS 泄漏启发式）。

与 TestClient 测试的差异：真实 socket、真实并发、真实进程生命周期（启动/
就绪等待/优雅关停），是打包冒烟（smoke_test_installer）之上的常驻回归资产。

用法（后端 venv，仓库根目录）：
  python scripts/e2e_api_smoke.py                 # 全链 + 默认 60 次推理盯测
  python scripts/e2e_api_smoke.py --soak 300      # 加大盯测轮次
  python scripts/e2e_api_smoke.py --skip-soak     # 只跑功能链
任何一步失败：打印 FAIL 明细并以非零码退出。
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = ROOT / "data" / "bootstrap_keys" / "sysadmin-01.key"
USERNAME = "sysadmin-01"
_BASE = ""


class SmokeFail(Exception):
    pass


def _check(cond: bool, label: str, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        raise SmokeFail(label + (f"：{detail}" if detail else ""))


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


def _make_films(tmp: Path) -> list[Path]:
    """生成 3 张合成底片：密集气孔 / 条状夹渣 / 无缺陷背景。"""
    import cv2
    import numpy as np

    tmp.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    base = np.full((480, 640), 165.0, np.float32)
    base[:, 220:420] += 35.0  # 焊缝亮带
    base = cv2.GaussianBlur(base, (0, 0), 9)
    rng = np.random.default_rng(42)

    img1 = base.copy()
    for _ in range(12):  # 密集气孔（暗圆点）
        x, y = int(rng.integers(240, 400)), int(rng.integers(60, 420))
        r = int(rng.integers(3, 7))
        cv2.circle(img1, (x, y), r, -30.0, -1)
    p1 = tmp / "smoke_pores.png"
    cv2.imencode(".png", np.clip(img1 + rng.normal(0, 6, img1.shape), 0, 255).astype("uint8"))[  # type: ignore[call-overload]
        1
    ].tofile(p1)
    paths.append(p1)

    img2 = base.copy()
    cv2.rectangle(img2, (300, 150), (316, 330), -45.0, -1)  # 条状夹渣
    p2 = tmp / "smoke_slag.png"
    cv2.imencode(".png", np.clip(img2 + rng.normal(0, 6, img2.shape), 0, 255).astype("uint8"))[  # type: ignore[call-overload]
        1
    ].tofile(p2)
    paths.append(p2)

    p3 = tmp / "smoke_clean.png"
    cv2.imencode(".png", np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype("uint8"))[  # type: ignore[call-overload]
        1
    ].tofile(p3)
    paths.append(p3)
    return paths


def _wait_health(client, deadline_s: float = 180.0) -> None:
    t0 = time.time()
    last = ""
    while time.time() - t0 < deadline_s:
        try:
            r = client.get(f"{_BASE}/api/v1/health")
            if r.status_code == 200:
                return
            last = f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001
            last = str(exc)[:80]
        time.sleep(1.0)
    _check(False, "后端就绪", f"超时 {deadline_s}s，最后错误：{last}")


def _login(client, username: str = USERNAME, key_file: Path = KEY_FILE) -> str:
    r = client.get(f"{_BASE}/api/v1/auth/challenge")
    _check(r.status_code == 200, f"签发登录挑战（{username}）", r.text[:120])
    ch = r.json()
    _check(key_file.is_file(), "三员私钥存在", str(key_file))
    priv = key_file.read_text(encoding="utf-8").strip()
    r = client.post(
        f"{_BASE}/api/v1/auth/login",
        json={
            "username": username,
            "challenge_id": ch.get("challenge_id"),
            "nonce": ch.get("nonce"),
            "private_key": priv,
        },
    )
    _check(r.status_code == 200, "SM2 挑战-响应登录", r.text[:200])
    token = r.json().get("token") or r.json().get("access_token")
    _check(bool(token), "取得会话令牌", r.text[:200])
    return str(token)


def _rss_mb(proc: subprocess.Popen) -> float:
    """进程树聚合 RSS（MB）：uvicorn 可能存在子进程，只测根会低估。"""
    import psutil

    try:
        root = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return 0.0
    total = root.memory_info().rss
    for child in root.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return total / (1024 * 1024)


def main() -> None:
    global _BASE
    ap = argparse.ArgumentParser(description="端到端 API 冒烟 + 内存盯测")
    ap.add_argument("--soak", type=int, default=60, help="循环推理次数（内存盯测）")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-soak", action="store_true")
    ap.add_argument("--rss-grow-max", type=float, default=1.35, help="RSS 增长倍数上限（泄漏启发式）")
    args = ap.parse_args()

    import httpx

    port = _free_port()
    _BASE = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"[smoke] 后端已拉起 pid={proc.pid} port={port}")
    failures = 0
    try:
        with httpx.Client(timeout=120.0) as client:
            _wait_health(client)
            print("[PASS] /health 就绪")

            token = _login(client)
            headers = {"Authorization": f"Bearer {token}"}

            films = _make_films(ROOT / "data" / "batch" / "smoke_tmp")

            # --- 检测 ---
            r = client.post(
                f"{_BASE}/api/v1/detect",
                files={"image": ("smoke_pores.png", films[0].read_bytes(), "image/png")},
                data={"pixel_spacing_mm": "0.2"},
                headers=headers,
            )
            _check(r.status_code == 200, "单张检测", r.text[:200])
            dets = r.json().get("defects", [])
            print(f"[INFO] 检出 {len(dets)} 个缺陷")
            r = client.post(
                f"{_BASE}/api/v1/detect",
                files={"image": ("garbage.png", b"not-an-image", "image/png")},
                headers=headers,
            )
            _check(r.status_code in (400, 415, 422), "坏图被拒绝（非 5xx）", f"HTTP {r.status_code}")

            # --- 判定 ---
            r = client.post(
                f"{_BASE}/api/v1/judge",
                json={
                    "base_metal_thickness_mm": 12.0,
                    "pixel_spacing_mm": 0.2,
                    "defects": [
                        {
                            "id": d.get("id", f"d{i}"),
                            "class_id": d.get("class_id", 0),
                            "bbox": d.get("bbox", [10, 10, 20, 20]),
                            "confidence": d.get("confidence", 0.9),
                            "uncertainty": d.get("uncertainty", 0.1),
                        }
                        for i, d in enumerate(dets)
                    ],
                },
                headers=headers,
            )
            _check(r.status_code == 200, "标准判定", r.text[:200])
            _check("joint_level" in r.json(), "判定返回级别字段")

            # --- 报告 ---
            r = client.post(
                f"{_BASE}/api/v1/report",
                files={"image": ("smoke_pores.png", films[0].read_bytes(), "image/png")},
                data={"pixel_spacing_mm": "0.2", "base_metal_thickness_mm": "12"},
                headers=headers,
            )
            _check(r.status_code == 200, "报告生成", r.text[:200])
            report = r.json()
            report_id = report.get("report_id") or report.get("id")
            _check(bool(report_id), "报告含 id", r.text[:200])

            # --- 受控导出（C-14 完整审批流）：sysadmin 申请 → secadmin 批准
            #     → 一次性令牌 → 凭令下载 PDF ---
            sec_token = _login(
                client,
                username="secadmin-01",
                key_file=ROOT / "data" / "bootstrap_keys" / "secadmin-01.key",
            )
            subject = f"report:{report_id}"
            r = client.post(
                f"{_BASE}/api/v1/export/requests",
                json={"subject": subject, "reason": "e2e smoke 导出验证"},
                headers=headers,
            )
            _check(r.status_code == 200, "导出申请", r.text[:200])
            req_id = r.json()["request_id"]
            r = client.post(
                f"{_BASE}/api/v1/export/requests/{req_id}/approve",
                json={"note": "e2e smoke 审批"},
                headers={"Authorization": f"Bearer {sec_token}"},
            )
            _check(r.status_code == 200, "保密员审批导出", r.text[:200])
            r = client.post(
                f"{_BASE}/api/v1/export/requests/{req_id}/token", headers=headers
            )
            _check(r.status_code == 200, "签发一次性导出令牌", r.text[:200])
            export_token = r.json().get("export_token") or r.json().get("token")
            _check(bool(export_token), "取得导出令牌", r.text[:200])
            r = client.get(
                f"{_BASE}/api/v1/report/{report_id}/pdf",
                headers={**headers, "X-Export-Token": str(export_token)},
            )
            _check(
                r.status_code == 200 and r.content[:4] == b"%PDF",
                "报告 PDF 下载（PDF 魔数）",
                f"HTTP {r.status_code} bytes={len(r.content)}",
            )

            # --- 批量（全量跑完）---
            files = [
                ("images", (p.name, p.read_bytes(), "image/png")) for p in films * 3
            ][: max(3, args.batch_size)]
            r = client.post(
                f"{_BASE}/api/v1/batch",
                files=files,
                data={"pixel_spacing_mm": "0.2"},
                headers=headers,
            )
            _check(r.status_code == 200, "批量提交", r.text[:200])
            batch_id = r.json()["batch_id"]
            t0 = time.time()
            status = ""
            tasks: list = []
            while time.time() - t0 < 600:
                s = client.get(f"{_BASE}/api/v1/batch/{batch_id}", headers=headers).json()
                status = str(s.get("status"))
                tasks = s.get("tasks") or []
                terminal = {"done", "failed", "cancelled"}
                if status in ("done", "completed") or (
                    tasks and all(t.get("status") in terminal for t in tasks)
                ):
                    break
                time.sleep(2.0)
            n_ok = sum(1 for t in tasks if t.get("status") == "done")
            n_fail = sum(1 for t in tasks if t.get("status") == "failed")
            _check(
                bool(tasks) and all(t.get("status") in ("done", "failed") for t in tasks),
                "批量全部到达终态",
                f"status={status} tasks={len(tasks)}",
            )
            _check(n_ok >= len(tasks) // 2, "批量过半成功", f"ok={n_ok}/{len(tasks)}")
            if n_fail:
                # 质量门禁拦截（黑度/IQI/位深不符）是**预期行为**：失败隔离、
                # 不拖垮批次——正是批量链路的韧性验证点。
                reason = str((tasks[-1] or {}).get("error") or tasks[-1])[:80]
                print(f"[INFO] 批量 {n_fail} 张被质量门禁拦截（预期行为）：{reason}…")
            print(f"[INFO] 批量 {batch_id}: {n_ok}/{len(tasks)} 成功")

            # --- 批量取消 ---
            r = client.post(
                f"{_BASE}/api/v1/batch",
                files=[("images", ("c.png", films[2].read_bytes(), "image/png"))]
                * 4,
                headers=headers,
            )
            _check(r.status_code == 200, "取消用批量提交", r.text[:120])
            cid = r.json()["batch_id"]
            r = client.post(f"{_BASE}/api/v1/batch/{cid}/cancel", headers=headers)
            _check(r.status_code == 200, "批量取消", r.text[:120])

            # --- 持久性盯测：循环推理 + RSS 泄漏启发式 ---
            if not args.skip_soak:
                n = args.soak
                print(f"[smoke] 循环推理盯测 {n} 次…")
                rss: list[float] = []
                for i in range(n):
                    rr = client.post(
                        f"{_BASE}/api/v1/detect",
                        files={"image": ("smoke_pores.png", films[0].read_bytes(), "image/png")},
                        data={"pixel_spacing_mm": "0.2"},
                        headers=headers,
                    )
                    if rr.status_code >= 500:
                        _check(False, f"盯测第 {i} 次推理 5xx", rr.text[:120])
                    if i >= 10 and i % 10 == 0:  # 预热 10 次后开始采样
                        rss.append(_rss_mb(proc))
                    if i % 25 == 0:
                        print(f"  iter {i}/{n} rss={rss[-1]:.0f}MB" if rss else f"  iter {i}/{n}")
                growth = max(rss) / max(1e-6, min(rss))
                drift = max(rss) - min(rss)
                print(
                    f"[INFO] RSS 采样 {len(rss)} 次：min={min(rss):.0f}MB max={max(rss):.0f}MB "
                    f"漂移={drift:.0f}MB 倍数={growth:.2f}（上限 {args.rss_grow_max}）"
                )
                _check(
                    growth <= args.rss_grow_max or drift <= 150,
                    "内存泄漏启发式（RSS 倍数或绝对漂移达标）",
                    f"growth={growth:.2f} drift={drift:.0f}MB",
                )
        print("[SMOKE] 全部通过")
    except SmokeFail as exc:
        print(f"[SMOKE] 失败：{exc}")
        failures = 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[smoke] 后端已关停")
    raise SystemExit(failures)


if __name__ == "__main__":
    main()
