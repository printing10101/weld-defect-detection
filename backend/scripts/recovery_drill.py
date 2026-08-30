"""恢复演练（S-13）：备份 → 注入损坏 → 校验检出 → 恢复 → 校验 → 记录 RTO。

对 infra.backup 全链路做端到端演练，产出演练记录 JSON（时间戳、各步耗时、
结论）到 data/compliance/，供稳定性测试报告引用。演练内容：

1. 造演练工作集（小文本/JSON 模拟 DB 与注册表状态）并 create_backup（SM3）；
2. 复制归档副本并篡改其中一条目字节（模拟介质损坏/篡改）；
3. verify_backup 对损坏副本必须抛 ValueError（检出即演练通过项）；
4. 对完好归档 restore_backup 回写到恢复目录；
5. 逐条目比对恢复内容与源内容一致；
6. 记录各步耗时与总 RTO（restore 起点至校验完成）。

用法::

    python -m backend.scripts.recovery_drill [--out data/compliance] [--keep-workdir]

退出码：演练全部通过 0；任一步骤失败 1（异常退出非零）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

from backend.infra.backup import create_backup, restore_backup, verify_backup

_INSTALL_ROOT = Path(__file__).resolve().parents[2]


def _make_workload(work: Path) -> dict[str, Path]:
    """造演练工作集（模拟 DB/注册表/基线三个关键状态文件）。"""
    db = work / "scan.db"
    db.write_bytes(b"SQLITE-HEADER" + bytes(range(256)) * 8)
    registry = work / "model_registry.json"
    registry.write_text(json.dumps({"active_id": "best::deadbeefdead"}), encoding="utf-8")
    baseline = work / "drift_baseline.json"
    baseline.write_text(json.dumps({"size_mean": 12.5, "conf_mean": 0.71}), encoding="utf-8")
    return {"scan.db": db, "model_registry.json": registry, "drift_baseline.json": baseline}


def _tamper_copy(archive: Path, tampered: Path) -> str:
    """复制归档并篡改首个非 manifest 条目的字节，返回被篡改的条目名。"""
    shutil.copy(archive, tampered)
    with zipfile.ZipFile(tampered, "a") as zf:
        names = [n for n in zf.namelist() if n != "manifest.json"]
        target = names[0]
        original = zf.read(target)
        zf.writestr(target, bytes([original[0] ^ 0xFF]) + original[1:])
    return target


def run_drill(out_dir: str | Path, *, work_root: str | Path | None = None) -> dict:
    """执行完整演练，返回记录 dict 并落盘 JSON。任一关键断言失败抛 AssertionError。"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    started = time.perf_counter()
    record: dict = {
        "drill": "recovery_drill",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "steps": [],
        "conclusion": "",
    }

    def _step(name: str, t0: float, ok: bool, detail: str = "") -> None:
        record["steps"].append(
            {
                "step": name,
                "ok": ok,
                "elapsed_sec": round(time.perf_counter() - t0, 4),
                "detail": detail,
            }
        )

    tmp_ctx = None
    work = Path(work_root) if work_root else None
    if work is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="recovery_drill_")
        work = Path(tmp_ctx.__enter__())
    failure: str | None = None
    try:
        work.mkdir(parents=True, exist_ok=True)
        src_dir = work / "state"
        src_dir.mkdir()

        # 1. 备份
        t0 = time.perf_counter()
        sources = _make_workload(src_dir)
        archive = work / "backup.zip"
        result = create_backup(sources, archive, app_version="0.1.0", hash_algo="sm3")
        assert archive.is_file(), "归档未生成"
        _step("create_backup(sm3)", t0, True, f"entries={len(result['manifest']['entries'])}")

        # 2. 注入损坏（篡改副本字节，不动原件）
        t0 = time.perf_counter()
        tampered = work / "backup_tampered.zip"
        target = _tamper_copy(archive, tampered)
        _step("inject_corruption", t0, True, f"tampered_entry={target}")

        # 3. 校验检出（损坏副本必须被拒）
        t0 = time.perf_counter()
        detected = False
        try:
            verify_backup(tampered)
        except ValueError:
            detected = True
        assert detected, "损坏副本未被校验检出（演练失败）"
        _step("detect_corruption", t0, True, "verify_backup raised ValueError")

        # 4. 恢复（完好归档）+ 5. 校验一致；RTO 从恢复起点计至校验完成
        t0 = time.perf_counter()
        rto_start = time.perf_counter()
        sink = work / "restored"
        sink.mkdir()
        restore_backup(archive, {k: sink / k for k in sources})
        for key, src in sources.items():
            restored_bytes = (sink / key).read_bytes()
            assert restored_bytes == src.read_bytes(), f"恢复内容不一致: {key}"
        record["rto_sec"] = round(time.perf_counter() - rto_start, 4)
        _step("restore_and_verify", t0, True, "restored bytes identical to source")

    except AssertionError as exc:
        failure = str(exc)
        raise
    finally:
        if tmp_ctx is not None:
            tmp_ctx.__exit__(None, None, None)
        record["total_elapsed_sec"] = round(time.perf_counter() - started, 4)
        record["finished_at"] = datetime.now().isoformat(timespec="seconds")
        record["conclusion"] = "FAIL" if failure else "PASS"
        if failure:
            record["error"] = failure

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / f"recovery_drill_{ts}.json"
        out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        record["report_path"] = str(out_path)
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description="S-13 恢复演练（备份→损坏→检出→恢复→RTO）")
    ap.add_argument("--out", default="data/compliance", help="演练记录 JSON 输出目录")
    ap.add_argument("--workdir", default=None, help="演练工作目录（缺省用临时目录）")
    args = ap.parse_args()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = _INSTALL_ROOT / args.out
    try:
        record = run_drill(out_dir, work_root=args.workdir)
    except AssertionError as exc:
        print(f"[recovery-drill] FAIL: {exc}")
        return 1
    print(
        f"[recovery-drill] {record['conclusion']} RTO={record.get('rto_sec')}s "
        f"report={record.get('report_path')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
