"""规格专项指标 harness 单测（合成数据，无需权重）。

覆盖 §15.2 Bland–Altman/量化误差、§15.3 一致率/κ、§15.4 ECE 的主计算、
阈值判定与配对匹配逻辑；量化配对走一张合成底片的真实掩膜精修链路。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.evaluation.agreement import grading_agreement
from backend.evaluation.calibration import (
    expected_calibration_error,
    match_confidences,
)
from backend.evaluation.quant_agreement import (
    bland_altman,
    geometry_pairs_for_image,
    quantification_summary,
)
from backend.evaluation.run_spec_eval import _run_agreement, _run_quant
from backend.evaluation.run_std_eval import _load_yolo_labels


# ---------------------------------------------------------------------------
# §15.2 Bland–Altman / 量化相对误差
# ---------------------------------------------------------------------------
class TestBlandAltman:
    def test_perfect_agreement(self) -> None:
        r = bland_altman([5.0] * 8, [5.0] * 8)
        assert r["n"] == 8
        assert r["mean_bias"] == 0.0
        assert r["loa_upper"] == 0.0 and r["loa_lower"] == 0.0
        assert r["mean_abs_rel_err"] == 0.0
        assert r["pct_within_threshold"] == 1.0
        assert r["verdict"]["passed"] is True

    def test_known_values(self) -> None:
        # 手算：diffs=[0.4,-0.4,0.2,-0.2,0] → bias=0, sd=sqrt(0.1)≈0.3162,
        # LoA=±1.96×0.3162≈±0.6196；rel=[.04,.04,.02,.02,0] → mean=.024
        r = bland_altman([10.4, 9.6, 10.2, 9.8, 10.0], [10.0] * 5)
        assert r["mean_bias"] == 0.0
        assert abs(r["sd_diff"] - 0.3162) < 1e-3
        assert abs(r["loa_upper"] - 0.6196) < 1e-3
        assert abs(r["mean_abs_rel_err"] - 0.024) < 1e-6
        assert r["verdict"]["passed"] is True

    def test_over_threshold_fails(self) -> None:
        r = bland_altman([11.0] * 5, [10.0] * 5)  # 恒偏 +10%
        assert r["mean_bias"] == 1.0
        assert r["mean_abs_rel_err"] == pytest.approx(0.10)
        assert r["verdict"]["passed"] is False

    def test_empty_not_passed(self) -> None:
        r = bland_altman([], [])
        assert r["n"] == 0
        assert r["verdict"]["passed"] is False  # 无数据不构成达标证据

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            bland_altman([1.0], [1.0, 2.0])

    def test_zero_ref_zero_auto_is_perfect(self) -> None:
        """ref=0 且 auto=0 是完美一致：相对误差记 0 而非 inf（不污染均值/占比）。"""
        r = bland_altman([0.0, 5.0], [0.0, 5.0])
        assert r["mean_abs_rel_err"] == 0.0
        assert r["pct_within_threshold"] == 1.0
        assert r["verdict"]["passed"] is True


# ---------------------------------------------------------------------------
# §15.3 评级一致率 / Cohen's κ
# ---------------------------------------------------------------------------
class TestGradingAgreement:
    def test_perfect(self) -> None:
        r = grading_agreement(["II", "III", "I"], ["II", "III", "I"])
        assert r["agreement_rate"] == 1.0
        assert r["cohens_kappa"] == 1.0
        assert r["verdict"]["passed"] is True

    def test_known_kappa(self) -> None:
        # 手算：observed=.75；pa={II:.5,III:.5} pb={II:.25,III:.75} → expected=.5
        # κ=(.75-.5)/(.5)=0.5
        r = grading_agreement(["II", "II", "III", "III"], ["II", "III", "III", "III"])
        assert r["agreement_rate"] == pytest.approx(0.75)
        assert r["cohens_kappa"] == pytest.approx(0.5)
        assert r["verdict"]["passed"] is False

    def test_none_is_a_category(self) -> None:
        # 熔断（None）配对不得剔除——剔除会虚高一致率
        r = grading_agreement([None, "II"], ["II", "II"])
        assert r["n_pairs"] == 2
        assert r["agreement_rate"] == 0.5
        assert r["confusion"]["<无级别>"]["II"] == 1

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            grading_agreement(["II"], [])

    def test_empty_not_passed(self) -> None:
        """空输入（pairs 文件为空/全部解析失败）不得判"通过"——无数据不是达标证据。"""
        r = grading_agreement([], [])
        assert r["n_pairs"] == 0
        assert r["agreement_rate"] == 0.0
        assert r["verdict"]["passed"] is False


# ---------------------------------------------------------------------------
# §15.4 ECE 置信度校准
# ---------------------------------------------------------------------------
class TestECE:
    def test_perfectly_calibrated(self) -> None:
        confs = [0.5] * 10
        correct = [True] * 5 + [False] * 5
        r = expected_calibration_error(confs, correct)
        assert r["ece"] == pytest.approx(0.0, abs=1e-6)
        assert r["verdict"]["passed"] is True

    def test_overconfident(self) -> None:
        # 单桶：conf 均值 .9、实测准确率 .5 → ECE=.4，超 0.05 阈值
        r = expected_calibration_error([0.9] * 4, [True, True, False, False])
        assert r["ece"] == pytest.approx(0.4)
        assert r["verdict"]["passed"] is False
        assert r["mce"] == pytest.approx(0.4)

    def test_empty_not_passed(self) -> None:
        assert expected_calibration_error([], [])["verdict"]["passed"] is False

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            expected_calibration_error([0.5], [])


class TestMatchConfidences:
    def test_match_and_unmatched(self) -> None:
        gts = [{"class_id": 0, "bbox": [0.0, 0.0, 10.0, 10.0]}]
        preds = [
            {"class_id": 0, "score": 0.8, "bbox": [1.0, 0.0, 10.0, 10.0]},  # IoU≈.82
            {"class_id": 0, "score": 0.6, "bbox": [50.0, 50.0, 10.0, 10.0]},  # 无可配
        ]
        out = match_confidences(preds, gts)
        assert out == [(0.8, True), (0.6, False)]

    def test_gt_used_once(self) -> None:
        gts = [{"class_id": 1, "bbox": [0.0, 0.0, 10.0, 10.0]}]
        preds = [
            {"class_id": 1, "score": 0.9, "bbox": [0.0, 0.0, 10.0, 10.0]},
            {"class_id": 1, "score": 0.7, "bbox": [0.5, 0.0, 10.0, 10.0]},  # 同一真值被占
        ]
        out = match_confidences(preds, gts)
        assert out == [(0.9, True), (0.7, False)]

    def test_class_mismatch_no_match(self) -> None:
        gts = [{"class_id": 0, "bbox": [0.0, 0.0, 10.0, 10.0]}]
        preds = [{"class_id": 4, "score": 0.95, "bbox": [0.0, 0.0, 10.0, 10.0]}]
        assert match_confidences(preds, gts) == [(0.95, False)]


# ---------------------------------------------------------------------------
# 量化配对：合成底片走真实掩膜精修链路
# ---------------------------------------------------------------------------
def _synthetic_film(rect: tuple[int, int, int, int], size=(300, 300)) -> np.ndarray:
    """亮背景 + 暗缺陷矩形（X 射线底片的典型对比形态）。"""
    img = np.full(size, 200, dtype=np.uint8)
    x, y, w, h = rect
    img[y : y + h, x : x + w] = 40
    return img


class TestQuantPairs:
    def test_geometry_pairs_fallback_path(self) -> None:
        """均匀图（无缺陷轮廓）→ MaskQuantifier 退化为包围盒 → auto == ref。

        验证配对机制与退化语义：精修失效时配对仍有效，误差为 0。
        """
        img = np.full((300, 300), 150, dtype=np.uint8)
        box = [100.0, 100.0, 60.0, 40.0]
        pairs = geometry_pairs_for_image(img, [box], spacing=0.1)
        assert len(pairs) == 1
        auto, ref = pairs[0]
        assert ref["length_mm"] == pytest.approx(6.0)
        assert ref["width_mm"] == pytest.approx(4.0)
        assert auto == ref  # 逐字段相等（回退到同一包围盒量化）

    def test_geometry_pairs_mask_path(self) -> None:
        """真实掩膜路径：暗缺陷矩形被轮廓法量化，光晕带内有界（非全 ROI 膨胀）。"""
        img = np.full((300, 300), 200, dtype=np.uint8)
        img[60:160, 60:210] = 40  # 暗缺陷 150×100
        box = [60.0, 60.0, 150.0, 100.0]
        pairs = geometry_pairs_for_image(img, [box], spacing=0.1)
        auto, ref = pairs[0]
        # 修复亮通道符号后：掩膜 ≈ 缺陷轮廓 + 自适应阈值光晕带（硬边合成图上
        # 实测 ~10%），而非修复前的恒全 ROI（+40%）；留 25% 容差作回归界。
        assert abs(auto["length_mm"] - ref["length_mm"]) / ref["length_mm"] < 0.25
        assert abs(auto["width_mm"] - ref["width_mm"]) / ref["width_mm"] < 0.25
        assert ref["area_mm2"] * 0.5 < auto["area_mm2"] < ref["area_mm2"] * 1.6

    def test_summary_structure(self) -> None:
        img = np.full((300, 300), 150, dtype=np.uint8)  # 均匀图 → 全部走回退路径
        pairs = geometry_pairs_for_image(img, [[100, 100, 60, 40], [10, 10, 20, 20]], spacing=0.1)
        s = quantification_summary(pairs)
        assert s["n_defects"] == 2
        for dim in ("length_mm", "width_mm", "area_mm2"):
            assert s[dim]["n"] == 2
            assert s[dim]["mean_abs_rel_err"] == 0.0  # 回退路径 auto==ref
            assert "loa_upper" in s[dim] and "pct_within_threshold" in s[dim]
        assert s["verdict"]["passed"] is True


# ---------------------------------------------------------------------------
# CLI 组装函数（量化 / 一致率节）
# ---------------------------------------------------------------------------
class TestCliSections:
    def test_run_quant_from_files(self, tmp_path: Path) -> None:
        import cv2

        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()
        rect = (100, 100, 60, 40)
        ok, buf = cv2.imencode(".png", _synthetic_film(rect))
        assert ok
        (img_dir / "a.png").write_bytes(buf.tobytes())
        # YOLO 归一化标签：class cx cy w h（300×300 图）
        (lbl_dir / "a.txt").write_text(
            f"0 {(rect[0] + 30) / 300} {(rect[1] + 20) / 300} {60 / 300} {40 / 300}\n",
            encoding="utf-8",
        )
        summary = _run_quant(
            img_dir, lbl_dir, spacing=0.1, rel_thr=0.05, app_cfg=None, mask_cfg=None
        )
        assert summary["n_images"] == 1
        assert summary["n_defects"] == 1

    def test_run_agreement_from_jsonl(self, tmp_path: Path) -> None:
        p = tmp_path / "pairs.jsonl"
        rows = [
            {"auto_grade": "II", "human_grade": "II"},
            {"auto_grade": "III", "human_grade": "III"},
            {"auto_grade": None, "human_grade": "II"},
        ]
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        r = _run_agreement(p, min_agreement=0.95, min_kappa=0.8)
        assert r["n_pairs"] == 3
        assert r["agreement_rate"] == pytest.approx(0.6667)  # 2/3，harness 保留 4 位小数
        assert r["verdict"]["passed"] is False

    def test_load_yolo_labels_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "a.txt"
        p.write_text("0 0.5 0.5 0.2 0.1\n4 0.25 0.25 0.1 0.1\n", encoding="utf-8")
        gts = _load_yolo_labels(p, (200, 100))
        assert len(gts) == 2
        assert gts[0]["class_id"] == 0
        assert gts[0]["bbox"] == pytest.approx([80.0, 45.0, 40.0, 10.0])


# ---------------------------------------------------------------------------
# 配置一致性：default.yaml 与 DetectCfg 默认值不得漂移
# ---------------------------------------------------------------------------
class TestConfigConsistency:
    def test_class_conf_yaml_matches_code_default(self) -> None:
        """运行时实际生效的 default.yaml 逐类阈值必须与代码默认一致。

        此前 default.yaml 缺 6: 0.10（内凹），生产回落 infer_conf=0.30，
        召回被静默收紧 3 倍且校准评估口径与线上漂移——此测试防复发。
        """
        from backend.infra.config import DetectCfg, load_config

        loaded = dict(load_config().detect.class_conf)
        default = dict(DetectCfg().class_conf)
        assert loaded == default, (
            f"default.yaml 与 DetectCfg 的 class_conf 漂移: {loaded} != {default}"
        )
