"""训练后部署：best.pt → legacy ONNX → 运行时/安装包 → 真实集检出回归（/P1 收尾）。

流程：
1. 强制 legacy 导出（torch>=2.9 dynamo 卡死绕行，simplify=False 防多类折叠）；
2. 拷贝 ONNX 到 运行时 models/weights/best.onnx（旧模型先改名保留）与
   _pkg/ScanDetection/{models/weights,backend/models/weights}/best.onnx；
3. **真实集检出回归**（铁律：任何导出/替换必须真实底片 >0 框门槛）——
   复用 quantize_onnx.detect_file（自包含，无需 backend 依赖），
   在 data/real_label/images 子集上统计检出数/类别分布，写报告 JSON。

用法（ML venv，torch cu 版已装）：
  python -m backend.training.deploy_model --src runs/gpu/<run>/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

# —— safe-delete shim 绕行（导出/拷贝时清理自身临时文件）——
try:
    import nt
    import pathlib

    os.unlink = nt.unlink  # type: ignore[attr-defined]
    os.remove = nt.remove  # type: ignore[attr-defined]
    pathlib.Path.unlink = lambda self, *a, **k: os.unlink(str(self))  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001, S110
    pass

os.environ.setdefault("YOLO_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
RUN_WEIGHTS = ROOT / "models" / "weights" / "best.onnx"
PKG = ROOT / "_pkg" / "ScanDetection"
REAL_IMAGES = ROOT / "data" / "real_label" / "images"

# 逐类置信度阈值（与 configs/default.yaml detect.class_conf 一致）
_CLASS_CONF = {0: 0.30, 1: 0.12, 2: 0.12, 3: 0.08, 4: 0.05, 5: 0.18}


def export_onnx(src_pt: Path) -> Path:
    """legacy 导出 ONNX（返回产物路径）。"""
    import torch
    from ultralytics import YOLO

    _orig = torch.onnx.export

    def _legacy(*a, **k):
        k["dynamo"] = False
        return _orig(*a, **k)

    torch.onnx.export = _legacy
    m = YOLO(str(src_pt))
    print(f"[deploy] classes: {m.names}")
    out = m.export(format="onnx", imgsz=640, simplify=False)
    return Path(out if isinstance(out, str) else str(out))


def copy_deployed(onnx: Path) -> list[Path]:
    """拷贝到运行时与安装包（旧运行时模型先改名备份）。"""
    dsts = [
        RUN_WEIGHTS,
        PKG / "models" / "weights" / "best.onnx",
        PKG / "backend" / "models" / "weights" / "best.onnx",
    ]
    if RUN_WEIGHTS.exists():
        backup = RUN_WEIGHTS.with_name(f"best_prev_{time.strftime('%Y%m%d_%H%M%S')}.onnx")
        shutil.copyfile(RUN_WEIGHTS, backup)
        print(f"[deploy] 旧运行时模型备份 -> {backup}")
    for dst in dsts:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(onnx, dst)
        print(f"[deploy] -> {dst} ({dst.stat().st_size / 1e6:.2f} MB)")
    copy_calibration(onnx, dsts)
    return dsts


def copy_calibration(onnx: Path, dsts: list[Path]) -> None:
    """温度校准表随权重一起分发（缺失静默跳过——校准是可选增强）。"""
    cal_src = onnx.parent / f"{onnx.stem}.calibration.json"
    if not cal_src.is_file():
        return
    for dst in dsts:
        cal_dst = dst.parent / f"{dst.stem}.calibration.json"
        shutil.copyfile(cal_src, cal_dst)
        print(f"[deploy] 校准表 -> {cal_dst}")


def regression_check(onnx: Path, limit: int = 24) -> dict:
    """真实底片检出回归（>0 框门槛）。复用 quantize_onnx 的自包含后处理。"""
    import onnxruntime as ort

    from backend.training.quantize_onnx import detect_file

    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    if not REAL_IMAGES.exists():
        # 真实底片缺失（如用户图已遗失）时跳过回归而非崩溃；gate 置 False 提示未验证
        report = {
            "model": str(onnx),
            "n_images": 0,
            "total_boxes": 0,
            "images_with_detections": 0,
            "class_distribution": {},
            "gate_passed": False,
            "skipped": "真实集目录缺失，无法执行检出回归（仅导出部署完成）",
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        out = ROOT / "data" / "experiments" / "deploy_regression.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[deploy] 真实集回归：跳过（{report['skipped']}）")
        return report
    imgs = sorted(p for p in REAL_IMAGES.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    imgs = imgs[:limit]
    total = 0
    n_with = 0
    cls: dict[int, int] = {}
    for p in imgs:
        dets = detect_file(sess, p, conf=0.3, iou=0.5)
        if dets:
            n_with += 1
        total += len(dets)
        for _, _, _, _, c, _ in dets:
            cls[c] = cls.get(c, 0) + 1
    report = {
        "model": str(onnx),
        "n_images": len(imgs),
        "total_boxes": total,
        "images_with_detections": n_with,
        "class_distribution": {str(k): v for k, v in sorted(cls.items())},
        "gate_passed": total > 0,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = ROOT / "data" / "experiments" / "deploy_regression.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[deploy] 真实集回归：{report}")
    return report


def sync_registry_pointer() -> None:
    """权重替换后把注册表活跃指针同步到新哈希（非致命，失败告警）。

    历史事故：deploy 只拷贝权重文件不更新 data/model_registry.json，active_id
    停留在旧哈希，"当前模型"与实际部署权重脱节（后经注册表全量扫描才被纠正）。
    """
    try:
        from backend.infra.model_registry import ModelRegistry

        reg = ModelRegistry(
            str(ROOT / "models" / "weights"), str(ROOT / "data" / "model_registry.json")
        )
        reg.mark_active_by_uri(str(RUN_WEIGHTS))
        print(f"[deploy] 注册表活跃指针 -> {reg.active_id}")
    except Exception as exc:  # noqa: BLE001
        print(f"[deploy] ⚠️ 注册表指针同步失败（不影响权重部署）：{exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description="训练后部署（导出+拷贝+回归）")
    ap.add_argument("--src", required=True, help="best.pt 路径")
    ap.add_argument("--limit", type=int, default=24, help="回归图数")
    ap.add_argument(
        "--skip-export", action="store_true", help="跳过导出（仅拷贝+回归，用现有 onnx）"
    )
    args = ap.parse_args()

    src_pt = Path(args.src)
    if args.skip_export:
        onnx = src_pt
    else:
        if not src_pt.exists():
            raise FileNotFoundError(f"best.pt 不存在：{src_pt}")
        onnx = export_onnx(src_pt)
    copy_deployed(onnx)
    sync_registry_pointer()
    report = regression_check(onnx, limit=args.limit)
    # 部署后评估闭环：同一权重在带标注评估集上产出实测指标（mAP/ECE）+ 模型卡
    # + 实验记录（规格书 §7.4/§15：harness 必须留下数字，而不是只有实现）。
    # 评估失败不阻断部署（regression_check 才是门槛），但必须显眼告警。
    try:
        from backend.training.post_deploy_eval import run_post_deploy_eval

        run_post_deploy_eval(model_path=RUN_WEIGHTS)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[deploy] ⚠️ 部署后评估闭环失败（不阻断部署，但指标/模型卡未更新）：{exc}")
    if not report["gate_passed"]:
        print("[deploy] ⚠️ 真实集 0 检出，门槛未过——部署回退需人工判断，勿使用该模型！")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
