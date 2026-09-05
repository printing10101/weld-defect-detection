"""规格专项指标评估 CLI（技术规格书 §15.2/§15.3/§15.4 承诺的三个验证项）。

补齐 std501807（DB50/T 1807 全套）之外、规格书单列的三项评估：
- §15.2 量化一致性：量化相对误差 ≤5% + Bland–Altman（--img-dir/--label-dir）；
- §15.3 评级一致率：一致率 ≥95% 且 Cohen's κ ≥0.8（--pairs，人工复核沉淀的配对数据）；
- §15.4 置信度校准：ECE ≤0.05（--model，部署 ONNX 推理取置信度）。

各节按所给输入独立运行；输出统一落盘 JSON，verdict 逐项对照规格阈值。

用法示例：
  python -m backend.evaluation.run_spec_eval \
      --img-dir data/real_label/images --label-dir data/real_label/labels \
      --model models/weights/best.onnx \
      --pairs data/eval/grade_pairs.jsonl \
      [--out data/eval/spec_metrics.json]

grade_pairs.jsonl 每行一对：{"auto_grade": "II", "human_grade": "III"}
（人工评级来源：评片复核工作流的逐缺陷级别记录）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from backend.evaluation.agreement import (
    DEFAULT_MIN_AGREEMENT,
    grading_agreement,
)
from backend.evaluation.calibration import (
    DEFAULT_MAX_ECE,
    expected_calibration_error,
    match_confidences,
)
from backend.evaluation.quant_agreement import (
    DEFAULT_REL_THRESHOLD,
    geometry_pairs_for_image,
    quantification_summary,
)

# 复用 run_std_eval 的同源锚点与加载/推理链路，避免第三份口径
from backend.evaluation.run_std_eval import (
    _ROOT as _STD_ROOT,
)
from backend.evaluation.run_std_eval import (
    _image_size,
    _list_images,
    _load_yolo_labels,
    _predict_with_onnx,
)
from backend.infra.timeutil import fmt_naive_utc


def _load_production_gray(p: Path, app_cfg) -> np.ndarray | None:
    """按生产口径解码单张底片（uint8 灰度）。

    与生产链路同源：经 image_loader.load_image 解码（unicode 安全 + 16bit
    min-max 拉伸 + GIF/HEIC 回退），preprocess.enabled 时再套与生产一致的
    增强（denoise+gamma）——规格考核的是生产量化环节，位深/增强口径须与
    线上一致（此前 IMREAD_GRAYSCALE 直读把 16bit 底片截断成 8bit）。
    解码失败返回 None（调用方跳过并告警）。
    """
    import numpy as np

    from backend.infra.image_loader import load_image

    try:
        gray, _meta = load_image(str(p))
    except Exception:  # noqa: BLE001 - 单图解码失败不阻断整批（含 ImageUnreadableError）
        return None
    if gray is None or gray.size == 0:
        return None
    if app_cfg is not None and app_cfg.preprocess.enabled:
        from backend.domain.preprocess.pipeline import OpencvPreprocessor

        pp = OpencvPreprocessor(
            bilateral_d=app_cfg.preprocess.bilateral_d,
            bilateral_sigma_color=app_cfg.preprocess.bilateral_sigma_color,
            bilateral_sigma_space=app_cfg.preprocess.bilateral_sigma_space,
            median_k=app_cfg.preprocess.median_k,
            clahe_clip=app_cfg.preprocess.clahe_clip,
            clahe_grid=app_cfg.preprocess.clahe_grid,
            canny_kernel=app_cfg.preprocess.canny_kernel,
            morph_k_open=app_cfg.preprocess.morph_k_open,
            morph_k_close=app_cfg.preprocess.morph_k_close,
        )
        gray = pp.enhance(pp.denoise(np.asarray(gray)), app_cfg.preprocess.gamma)
    return np.asarray(gray)


def _run_quant(
    img_dir: Path,
    lbl_dir: Path,
    spacing: float,
    rel_thr: float,
    app_cfg,
    mask_cfg,
) -> dict:
    pairs: list[tuple[dict, dict]] = []
    n_img = 0
    for p in _list_images(img_dir):
        gray = _load_production_gray(p, app_cfg)
        if gray is None:
            print(f"警告：解码失败跳过 {p.name}")
            continue
        # 尺寸取自已解码灰度图（此前 _image_size 再解码一次全图，8K 底片 I/O 翻倍）
        wh = (int(gray.shape[1]), int(gray.shape[0]))
        gts = _load_yolo_labels(lbl_dir / f"{p.stem}.txt", wh)
        if not gts:
            continue
        pairs.extend(
            geometry_pairs_for_image(gray, [g["bbox"] for g in gts], spacing, cfg=mask_cfg)
        )
        n_img += 1
    summary = quantification_summary(pairs, rel_threshold=rel_thr)
    summary["n_images"] = n_img
    summary["pixel_spacing_mm"] = spacing
    return summary


def _run_calibration(
    img_dir: Path, lbl_dir: Path, model_path: Path, iou_thr: float, max_ece: float
) -> dict:
    img_paths = _list_images(img_dir)
    preds_by_stem = _predict_with_onnx(img_paths, model_path)
    confs: list[float] = []
    corrects: list[bool] = []
    for p in img_paths:
        gts = _load_yolo_labels(lbl_dir / f"{p.stem}.txt", _image_size(p))
        for c, ok in match_confidences(preds_by_stem.get(p.stem, []), gts, iou_thr=iou_thr):
            confs.append(c)
            corrects.append(ok)
    return expected_calibration_error(confs, corrects, max_ece=max_ece)


def _run_agreement(pairs_path: Path, min_agreement: float, min_kappa: float) -> dict:
    auto: list[str | None] = []
    human: list[str | None] = []
    skipped = 0
    for line in pairs_path.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(row, dict):
            # 合法 JSON 但非对象（如 [1,2] / "II" / null）：计跳过而非崩溃。
            skipped += 1
            continue
        auto.append(row.get("auto_grade"))
        human.append(row.get("human_grade"))
    result = grading_agreement(auto, human, min_agreement=min_agreement, min_kappa=min_kappa)
    result["n_skipped_lines"] = skipped
    return result


def main() -> None:
    from backend.infra.config import MaskRefineCfg, load_config

    app_cfg = load_config()
    ap = argparse.ArgumentParser(description="规格专项指标评估（§15.2/§15.3/§15.4）")
    ap.add_argument("--img-dir", help="评估图像目录（量化一致性 / 校准需要）")
    ap.add_argument("--label-dir", help="真值标签目录（YOLO 归一化，与 --img-dir 配套）")
    ap.add_argument("--model", help="部署 ONNX 路径（校准评估需要）")
    ap.add_argument("--pairs", help="评级配对 JSONL（评级一致率需要）")
    ap.add_argument("--spacing-mm", type=float, default=0.1, help="像素标定 mm/px（量化用）")
    ap.add_argument("--match-iou", type=float, default=0.5, help="校准正确性匹配 IoU 阈值")
    ap.add_argument("--max-rel-err", type=float, default=DEFAULT_REL_THRESHOLD)
    ap.add_argument("--min-agreement", type=float, default=DEFAULT_MIN_AGREEMENT)
    # κ 阈值默认取部署配置（review.kappa_threshold 为权威来源），不再第三处硬编码。
    ap.add_argument("--min-kappa", type=float, default=app_cfg.review.kappa_threshold)
    ap.add_argument("--max-ece", type=float, default=DEFAULT_MAX_ECE)
    ap.add_argument("--out", default="data/eval/spec_metrics.json")
    args = ap.parse_args()

    if not (args.img_dir or args.pairs):
        ap.error("--img-dir（量化/校准）与 --pairs（评级一致率）至少提供一个")
    if args.pairs and not (_STD_ROOT / args.pairs).is_file():
        ap.error(f"评级配对文件不存在: {args.pairs}（每行一対 auto_grade/human_grade 的 JSONL）")
    if args.spacing_mm <= 0:
        ap.error("--spacing-mm 必须为正数（≤0 会使相对误差全部无意义）")
    if args.img_dir:
        if not args.label_dir:
            ap.error("--img-dir 需要配套 --label-dir")
        if not (_STD_ROOT / args.img_dir).is_dir():
            ap.error(f"图像目录不存在: {args.img_dir}")
        if not (_STD_ROOT / args.label_dir).is_dir():
            ap.error(f"标签目录不存在: {args.label_dir}")
    if args.model and not (_STD_ROOT / args.model).is_file():
        ap.error(f"模型文件不存在: {args.model}")

    payload: dict = {
        "generated_at": fmt_naive_utc(),
        "spec_refs": {
            "quant_agreement": "§15.2 量化相对误差≤5% + Bland–Altman",
            "grading_agreement": "§15.3 一致率≥95% 且 κ≥0.8",
            "calibration": "§15.4 ECE≤0.05",
        },
    }
    sections: list[str] = []

    if args.img_dir:
        img_dir = _STD_ROOT / args.img_dir
        lbl_dir = _STD_ROOT / args.label_dir
        payload["quant_agreement"] = _run_quant(
            img_dir,
            lbl_dir,
            args.spacing_mm,
            args.max_rel_err,
            app_cfg,
            # 掩膜精修与生产同参（configs.mask_refine），运维调参后评估口径随动。
            MaskRefineCfg(**app_cfg.mask_refine.model_dump()),
        )
        sections.append("quant_agreement")
        if args.model:
            payload["calibration"] = _run_calibration(
                img_dir, lbl_dir, _STD_ROOT / args.model, args.match_iou, args.max_ece
            )
            sections.append("calibration")

    if args.pairs:
        payload["grading_agreement"] = _run_agreement(
            _STD_ROOT / args.pairs, args.min_agreement, args.min_kappa
        )
        sections.append("grading_agreement")

    passed = all(payload[s]["verdict"]["passed"] for s in sections)
    payload["overall_passed"] = passed

    out = _STD_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for s in sections:
        v = payload[s]["verdict"]
        tag = "通过" if v["passed"] else "未通过"
        if s == "quant_agreement":
            q = payload[s]
            print(
                f"[量化一致性] {'通过' if q['verdict']['passed'] else '未通过'} "
                f"n={q['n_defects']} 长误差={q['length_mm']['mean_abs_rel_err']:.2%} "
                f"宽误差={q['width_mm']['mean_abs_rel_err']:.2%}（阈值 {args.max_rel_err:.0%}）"
            )
        elif s == "calibration":
            print(
                f"[置信度校准] {tag} ECE={payload[s]['ece']:.4f}（阈值 {args.max_ece}，"
                f"n={payload[s]['n_samples']}）"
            )
        else:
            g = payload[s]
            print(
                f"[评级一致率] {tag} 一致率={g['agreement_rate']:.2%} κ={g['cohens_kappa']:.3f}"
                f"（阈值 {args.min_agreement:.0%} / {args.min_kappa}，n={g['n_pairs']}）"
            )
    print(f"总体：{'通过' if passed else '未通过'} → {out}")


if __name__ == "__main__":
    main()
