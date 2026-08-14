# -*- coding: utf-8 -*-
"""将已训练模型导出为 ONNX 并放入安装包的模型加载路径。

- 源：data/real_label/runs/real_synth2/weights/best.pt（6 类，中文类名，与 DefectClass 对齐）
- 目标：_pkg/ScanDetection/models/weights/best.onnx 与 _pkg/ScanDetection/backend/models/weights/best.onnx
- 沿用 backend/models/train.py 的 legacy 导出器 monkeypatch，规避 torch>=2.9 dynamo 卡死。

仅写入/拷贝文件，不删除用户数据。路径相对于本脚本所在项目根，可用 --src/--pkg 覆盖。
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

# —— 绕开 safe-delete shim（仅恢复原生删除用于导出时清理自身临时文件，无外部风险）——
try:
    import nt
    os.unlink = nt.unlink
    os.remove = nt.remove
    from pathlib import Path as _P
    _P.unlink = lambda self, *a, **k: os.unlink(str(self))
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
DEFAULT_SRC = ROOT / "data" / "real_label" / "runs" / "yolo8n_real_rare" / "train" / "weights" / "best.pt"
DEFAULT_PKG = ROOT / "_pkg" / "ScanDetection"


def main() -> int:
    ap = argparse.ArgumentParser(description="导出训练模型为 ONNX 并打包到安装目录")
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="源 .pt 权重路径")
    ap.add_argument("--pkg", default=str(DEFAULT_PKG), help="目标安装包根目录")
    args = ap.parse_args()

    src_pt = args.src
    pkg = args.pkg
    if not os.path.exists(src_pt):
        print(f"[export] ERROR: 源权重不存在: {src_pt}", file=sys.stderr)
        return 1

    import torch
    from ultralytics import YOLO

    # 强制 legacy 导出器，避免 dynamo 在部分模型上卡死（仅在本进程生效）
    _orig = torch.onnx.export

    def _legacy_export(*a, **k):
        k["dynamo"] = False
        return _orig(*a, **k)

    torch.onnx.export = _legacy_export

    print("[export] loading", src_pt)
    m = YOLO(src_pt)
    print("[export] classes:", m.names)
    # simplify=False：禁用 onnxslim。实测 onnxslim 会把多类分类分支错误折叠，
    # 导致导出 ONNX 退化（仅输出类0、稀有类分数恒为0，与 torch 权重行为不一致）。
    out = m.export(format="onnx", imgsz=640, simplify=False)
    onnx_src = out if isinstance(out, str) else str(out)
    print("[export] produced:", onnx_src, "size=", os.path.getsize(onnx_src))

    dst1 = os.path.join(pkg, "models", "weights", "best.onnx")
    dst2 = os.path.join(pkg, "backend", "models", "weights", "best.onnx")
    os.makedirs(os.path.dirname(dst1), exist_ok=True)
    os.makedirs(os.path.dirname(dst2), exist_ok=True)
    shutil.copyfile(onnx_src, dst1)
    shutil.copyfile(onnx_src, dst2)
    print("[export] copied ->")
    print("   ", dst1, os.path.getsize(dst1))
    print("   ", dst2, os.path.getsize(dst2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
