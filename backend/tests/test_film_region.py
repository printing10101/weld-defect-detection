"""底片区域检测（翻拍照片分割）测试。

覆盖：翻拍照片的胶片区分割与 is_photo 判定、满幅扫描件不误判翻拍、
黑度掩膜估计排除亮背景、两级 Otsu 防塌缩护栏。
"""

from __future__ import annotations

import numpy as np

from backend.domain.density import estimate_density
from backend.domain.film_region import FilmRegionCfg, detect_film_region


def _photo_film(size: int = 640) -> np.ndarray:
    """合成翻拍照片：灯箱亮背景 + 暗胶片 + 更暗焊缝带与缺陷。"""
    arr = np.full((size, size), 245, dtype=np.uint8)  # 灯箱过曝背景
    arr[size // 8 : size - size // 8, size // 5 : size - size // 5] = 130  # 胶片
    y0, y1 = size // 2 - 20, size // 2 + 20
    arr[y0:y1, size // 5 : size - size // 5] = 100  # 焊缝带
    ys, xs = np.ogrid[:size, :size]
    cy, cx, r = int(size * 0.65), size // 2, size // 32
    arr[(xs - cx) ** 2 + (ys - cy) ** 2 <= r**2] = 55  # 圆形缺陷
    return arr


def test_detect_photo_film_region() -> None:
    img = _photo_film(640)
    fr = detect_film_region(img)
    assert fr is not None
    assert fr.is_photo is True, "灯箱亮背景 + 局部胶片应判为翻拍"
    assert 0.3 < fr.area_frac < 0.6
    # 外接框应落在胶片实际范围内（允许自适应形态学的边界余量）
    assert fr.x > 50 and fr.y > 30
    assert fr.x + fr.w < 640 - 50 and fr.y + fr.h <= 640
    # 掩膜应基本覆盖胶片区、排除灯箱背景
    assert fr.mask[int(640 * 0.65), 320], "缺陷所在胶片中心应在掩膜内"
    assert not fr.mask[10, 10], "灯箱角点不应在掩膜内"


def test_full_frame_scan_is_not_photo() -> None:
    """满幅胶片（焊缝带 + 缺陷）不应判翻拍：分割失败返回 None（走整图路径）。"""
    img = np.full((512, 512), 160, dtype=np.uint8)
    img[:, 230:282] = 120  # 焊缝带
    ys, xs = np.ogrid[:512, :512]
    img[(xs - 256) ** 2 + (ys - 256) ** 2 <= 18**2] = 55  # 缺陷
    fr = detect_film_region(img)
    assert fr is None or fr.is_photo is False


def test_thin_margin_scan_is_not_photo() -> None:
    """扫描件细白边（~1% 线性边距）不触发翻拍判定。"""
    img = np.full((512, 512), 255, dtype=np.uint8)
    img[6:506, 6:506] = 60  # 深色胶片几乎满幅
    fr = detect_film_region(img)
    assert fr is not None
    assert fr.is_photo is False


def test_uniform_image_no_crash() -> None:
    fr = detect_film_region(np.full((256, 256), 128, dtype=np.uint8))
    assert fr is None or fr.is_photo is False


def test_two_stage_otsu_no_collapse() -> None:
    """两级 Otsu 护栏：二次分割锁到孤立小特征时应保留单级结果。"""
    img = _photo_film(512)
    # 在亮背景上放一个孤立暗斑：单级 Otsu 后暗类=胶片+暗斑，
    # 二次分割可能锁到暗斑；护栏应拒绝塌缩到极小子类。
    img[40:60, 40:60] = 30
    fr = detect_film_region(img)
    assert fr is not None
    assert fr.area_frac >= FilmRegionCfg().min_area_frac


def test_estimate_density_with_mask_excludes_background() -> None:
    img = np.full((100, 100), 250, dtype=np.uint8)  # 亮背景
    img[20:80, 20:80] = 26  # 胶片区（D≈1）
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 20:80] = True
    d_masked = estimate_density(img, bit_depth=8, mask=mask)
    d_whole = estimate_density(img, bit_depth=8)
    assert d_masked > 0.9, "掩膜黑度应接近胶片区真实黑度"
    assert d_whole < 0.5, "整图黑度被亮背景严重稀释"
    # 掩膜形状不匹配 / 空掩膜 → 回退整图，不抛错
    assert estimate_density(img, 8, mask=np.ones((5, 5), dtype=bool)) == d_whole
    empty = np.zeros((100, 100), dtype=bool)
    assert estimate_density(img, 8, mask=empty) == d_whole


def test_detect_empty_and_none_inputs() -> None:
    assert detect_film_region(None) is None  # type: ignore[arg-type]
    assert detect_film_region(np.array([])) is None
