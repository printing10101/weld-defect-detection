"""P0-B GLASS 式缺陷合成生成器测试（backend/training/defect_synth）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from backend.training.defect_synth import (
    _DEFAULT_CLASS_WEIGHTS,
    CLASS_NAMES,
    generate_synthetic_set,
    synthesize_image,
)


class TestSynthesizeImage:
    def test_shape_and_dtype(self):
        img, labels = synthesize_image(0, 480, 640, 3)
        assert img.ndim == 2
        assert img.shape == (480, 640)
        assert img.dtype == np.uint8
        assert len(labels) == 3

    def test_labels_normalized_and_valid(self):
        _, labels = synthesize_image(1, 480, 640, 4)
        for cls, cx, cy, bw, bh in labels:
            assert 0 <= cls <= 5
            assert 0.0 < cx <= 1.0
            assert 0.0 < cy <= 1.0
            assert 0.0 < bw <= 1.0
            assert 0.0 < bh <= 1.0

    def test_defects_are_darker_than_background(self):
        """合成缺陷应暗于同带邻域（X 光底片缺陷语义）。

        细线缺陷（未焊透/裂纹）bbox 内背景占比高、均值被稀释，跳过；
        块状缺陷（气孔/夹渣/条带/咬边）与左右同带邻域比较。
        """
        img, labels = synthesize_image(2, 480, 640, 3)
        h, w = img.shape
        checked = 0
        for cls, cx, cy, bw, bh in labels:
            x0 = max(0, int((cx - bw / 2) * w))
            y0 = max(0, int((cy - bh / 2) * h))
            x1 = min(w, int((cx + bw / 2) * w))
            y1 = min(h, int((cy + bh / 2) * h))
            if (x1 - x0) * (y1 - y0) < 60:
                continue
            region = img[y0:y1, x0:x1].astype(np.float32)
            win = 40
            refs: list[np.ndarray] = []
            if x0 - win >= 0:
                refs.append(img[y0:y1, x0 - win : x0].ravel())
            if x1 + win <= w:
                refs.append(img[y0:y1, x1 : x1 + win].ravel())
            if not refs:
                continue
            ref = float(np.concatenate(refs).mean())
            assert float(region.mean()) < ref, f"cls={cls} region={region.mean():.1f} ref={ref:.1f}"
            checked += 1
        assert checked >= 1

    def test_rare_class_forced(self):
        """定向生成指定类别（罕见类可强制出现）。"""
        weights = {4: 1.0}  # 只出裂纹
        _, labels = synthesize_image(3, 480, 640, 5, class_weights=weights)
        assert all(c == 4 for c, *_ in labels)

    def test_default_weights_cover_all_classes(self):
        assert set(_DEFAULT_CLASS_WEIGHTS) == set(range(6))
        assert len(CLASS_NAMES) == 6


class TestGenerateSyntheticSet:
    def test_generates_valid_yolo_set(self, tmp_path: Path):
        n = generate_synthetic_set(tmp_path, n_images=10, size=(320, 240), seed=7)
        assert n == 10
        imgs = sorted((tmp_path / "images").glob("*.png"))
        lbls = sorted((tmp_path / "labels").glob("*.txt"))
        assert len(imgs) == 10
        assert len(lbls) == 10
        for img_p, lbl_p in zip(imgs, lbls):
            assert img_p.stem == lbl_p.stem
            # 图像可读且为灰度
            arr = np.fromfile(str(img_p), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            assert img is not None and img.shape == (240, 320)
            # 标签格式合法
            for ln in lbl_p.read_text(encoding="utf-8").splitlines():
                p = ln.split()
                assert len(p) == 5
                assert 0 <= int(p[0]) <= 5
                assert all(0.0 <= float(v) <= 1.0 for v in p[1:])

    def test_deterministic_with_seed(self, tmp_path: Path):
        d1 = tmp_path / "a"
        d2 = tmp_path / "b"
        generate_synthetic_set(d1, n_images=3, size=(160, 120), seed=9)
        generate_synthetic_set(d2, n_images=3, size=(160, 120), seed=9)
        a = sorted(p.read_bytes() for p in (d1 / "images").glob("*.png"))
        b = sorted(p.read_bytes() for p in (d2 / "images").glob("*.png"))
        assert a == b

    def test_zero_images_no_crash(self, tmp_path: Path):
        assert generate_synthetic_set(tmp_path, n_images=0) == 0
