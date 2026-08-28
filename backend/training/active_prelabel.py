"""主动预标注 + 不确定性采样闭环。

把 ai_prelabel.py 的**硬编码字典**升级为真实推理闭环：
1. 用已部署训练模型（YoloDetector，config 驱动）对 data/real_label 未标注图推理；
2. 写出 YOLO 预标注到 prelabels/（不覆盖用户已保存的 labels/，与 promote_seeds 一致）；
3. 按  不确定性 + 主动学习价值聚合排序（复用 domain/active_learning.high_value_score），
   输出 data/real_label/active_queue.json —— 人工核对优先级队列（高价值/高不确定优先送标）；
4. --export-top N：把前 N 个高价值样本的预标注导出到训练池（复用
   active_learning.export_training_labels），供下一轮训练合并。

用法：
  python -m backend.training.active_prelabel --real data/real_label
  python -m backend.training.active_prelabel --real data/real_label --export-top 20 --pool data/active/training_pool
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from backend.domain.active_learning import export_training_labels, high_value_score
from backend.domain.dto import Detection
from backend.infra.config import load_config, resolve_config_path
from backend.infra.pool_store import FilePoolStore

_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _read_gray(path: Path) -> np.ndarray | None:
    """中文路径安全灰度读取（cv2.imread 对中文绝对路径静默失败）。"""
    arr = np.fromfile(str(path), dtype=np.uint8)
    if arr.size == 0:
        return None
    im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    elif im.ndim == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
    return im


def _family_of(stem: str) -> str:
    """从文件名推断底片族（PG103-1-1 → PG103；无前缀返回 unknown）。"""
    head = stem.split("-")[0]
    if head.startswith(("PG", "PL")):
        return head
    return "unknown"


def _unlabeled_images(real_root: Path) -> list[Path]:
    """未标注图：images/ 下存在、但 labels/{stem}.txt 不存在的图。"""
    img_dir = real_root / "images"
    lbl_dir = real_root / "labels"
    if not img_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in _EXTS:
            continue
        if not (lbl_dir / f"{p.stem}.txt").exists():
            out.append(p)
    return out


def active_prelabel(
    real_root: Path,
    detector,
    *,
    conf: float = 0.3,
    iou: float = 0.5,
    class_conf: dict[int, float] | None = None,
) -> dict:
    """对未标注图推理 → 写预标注 → 生成优先级队列。

    detector: 任何实现 ``infer(image, conf, iou, class_conf) -> list[Detection]`` 的对象
    （YoloDetector 或测试桩）。返回队列 dict：
    {
      "generated_at": ...,
      "total": n,
      "queue": [{stem, family, n_detections, max_uncertainty, max_value, classes, prelabel_written, no_detection}]
    }
    队列按 max_value 降序（高价值优先人工核对）。
    """
    real_root = Path(real_root)
    pre_dir = real_root / "prelabels"
    lbl_dir = real_root / "labels"
    pre_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    queue: list[dict] = []
    for img_p in _unlabeled_images(real_root):
        gray = _read_gray(img_p)
        if gray is None:
            continue
        h, w = gray.shape
        dets = detector.infer(gray, conf=conf, iou=iou, class_conf=class_conf)
        # 写预标注（仅当正式标注不存在；promote 流程不会用预标注覆盖用户保存）
        lbl_path = lbl_dir / f"{img_p.stem}.txt"
        pre_path = pre_dir / f"{img_p.stem}.txt"
        if not lbl_path.exists():
            lines = [_to_yolo_line(d, w, h) for d in dets]
            pre_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            wrote = True
        else:
            wrote = False  # 用户已标注，跳过预标注
        values = [high_value_score(d) for d in dets]
        queue.append(
            {
                "stem": img_p.stem,
                "family": _family_of(img_p.stem),
                "n_detections": len(dets),
                "max_uncertainty": round(max((d.uncertainty for d in dets), default=0.0), 4),
                "max_value": round(max(values, default=0.0), 4),
                "classes": sorted({d.class_id.value for d in dets}),
                "prelabel_written": wrote,
                "no_detection": len(dets) == 0,
            }
        )
    queue.sort(
        key=lambda e: (e["max_value"], e["max_uncertainty"], e["n_detections"]), reverse=True
    )
    result = {
        "generated_at": _now(),
        "total": len(queue),
        "queue": queue,
    }
    (real_root / "active_queue.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _to_yolo_line(d: Detection, w: int, h: int) -> str:
    """Detection → YOLO normalized 标注行（坐标裁剪到 [0,1]）。"""
    bw = max(0.0, min(d.bbox.w / max(w, 1e-9), 1.0))
    bh = max(0.0, min(d.bbox.h / max(h, 1e-9), 1.0))
    cx = max(0.0, min((d.bbox.x + d.bbox.w / 2) / max(w, 1e-9), 1.0))
    cy = max(0.0, min((d.bbox.y + d.bbox.h / 2) / max(h, 1e-9), 1.0))
    return f"{int(d.class_id.value)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")


def export_top(
    real_root: Path,
    queue: dict,
    top_k: int,
    pool_dir: Path,
    *,
    conf: float = 0.3,
    iou: float = 0.5,
    class_conf: dict[int, float] | None = None,
    detector=None,
) -> int:
    """把队列前 top_k 张的预标注导出到训练池（供重训合并）。

    仅导出 prelabel_written=True 且 n_detections>0 的样本；标注内容=已写好的
    预标注文件（人工核对前先入池，训练侧以"伪标签"处理，人工确认后以
    active/export 覆盖正式标签）。返回导出数。
    """
    pool_dir = Path(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    exported = 0
    for entry in queue["queue"][:top_k]:
        if not entry["prelabel_written"] or entry["n_detections"] == 0:
            continue
        stem = entry["stem"]
        img_p = real_root / "images" / f"{stem}.jpg"
        if not img_p.exists():
            # 扩展名不确定：扫描匹配
            candidates = list((real_root / "images").glob(f"{stem}.*"))
            img_p = candidates[0] if candidates else None
        if img_p is None:
            continue
        gray = _read_gray(img_p)
        if gray is None:
            continue
        h, w = gray.shape
        dets = detector.infer(gray, conf=conf, iou=iou, class_conf=class_conf)
        if not dets:
            continue
        export_training_labels(stem, dets, float(w), float(h), store=FilePoolStore(pool_dir))
        exported += 1
    return exported


def _build_detector():
    """按 config 加载已部署检测器；训练模型缺失时回退 BlobDetector 并告警。"""
    cfg = load_config()
    dcfg = cfg.detect
    try:
        from backend.domain.detect.yolo_detector import YoloDetector

        det = YoloDetector()
        uri = resolve_config_path(cfg.model.default_uri)
        det.load(str(uri), cfg.model.backend)
        return det
    except Exception as exc:  # noqa: BLE001 - 回退基线不应阻断预标注
        import logging

        logging.getLogger("scandetection.active_prelabel").warning(
            "训练模型加载失败，回退 BlobDetector 基线：%s", exc
        )
        from backend.domain.detect.blob_detector import BlobConfig, BlobDetector

        return BlobDetector(
            BlobConfig(
                min_area_px=dcfg.min_area_px,
                max_area_px=dcfg.max_area_px,
                min_size_px=dcfg.min_size_px,
                noise_sigma_ratio=dcfg.noise_sigma_ratio,
                abs_threshold=dcfg.abs_threshold,
                dark_only=dcfg.dark_only,
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="主动预标注 + 不确定性采样（P0-G）")
    ap.add_argument("--real", default="data/real_label", help="真实标注根目录")
    ap.add_argument("--export-top", type=int, default=0, help="导出前 N 张到训练池（0=不导出）")
    ap.add_argument("--pool", default="data/active/training_pool", help="训练池目录")
    args = ap.parse_args()

    real_root = resolve_config_path(args.real)
    pool_dir = resolve_config_path(args.pool)
    cfg = load_config()
    dcfg = cfg.detect
    det = _build_detector()
    queue = active_prelabel(
        real_root,
        det,
        conf=dcfg.infer_conf,
        iou=dcfg.infer_iou,
        class_conf=dcfg.class_conf,
    )
    print(
        f"[active_prelabel] 未标注 {queue['total']} 张，队列已写 {real_root / 'active_queue.json'}"
    )
    n_high = sum(1 for e in queue["queue"] if e["max_value"] >= dcfg.review_conf)
    print(f"[active_prelabel] 高价值（>=review_conf={dcfg.review_conf}）{n_high} 张，优先人工核对")
    if args.export_top > 0:
        n = export_top(
            real_root,
            queue,
            args.export_top,
            pool_dir,
            detector=det,
            conf=dcfg.infer_conf,
            iou=dcfg.infer_iou,
            class_conf=dcfg.class_conf,
        )
        print(f"[active_prelabel] 已导出 {n} 张到训练池 {pool_dir}")


if __name__ == "__main__":
    main()
