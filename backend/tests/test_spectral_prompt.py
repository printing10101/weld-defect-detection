"""频域分解与频带提示（backend.domain.preprocess.spectral）单元测试。

覆盖：DCT 往返/Parseval 能量守恒、径向频带的互斥与完备、逐带重建的可加性、
能量占比的语义（常数图 vs 高频棋盘）、提示张量的值域与逐带归一化、
以及全部参数校验分支。
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.domain.preprocess.spectral import (
    band_edges,
    band_energy_fractions,
    band_pass_images,
    dct2,
    idct2,
    normalized_radius,
    radial_band_masks,
    spectral_prompt,
)


def _rng_image(h: int = 32, w: int = 48) -> np.ndarray:
    rng = np.random.default_rng(20260914)
    return rng.normal(128.0, 20.0, size=(h, w))


# ---------------------------------------------------------------------------
# DCT / 逆变换
# ---------------------------------------------------------------------------


def test_dct_idct_roundtrip_is_exact() -> None:
    img = _rng_image()
    back = idct2(dct2(img))
    assert back.shape == img.shape
    assert np.allclose(back, img, atol=1e-9, rtol=0.0)


def test_dct_preserves_energy_parseval() -> None:
    """正交归一化 DCT 必须保能量——频带能量占比的可解释性依赖于此。"""
    img = _rng_image()
    assert float(np.sum(img**2)) == pytest.approx(float(np.sum(dct2(img) ** 2)), rel=1e-10)


def test_constant_image_energy_concentrates_at_dc() -> None:
    const = np.full((16, 24), 7.5, dtype=np.float64)
    spec = dct2(const)
    # 只有 DC 非零，其余频点数值上为零
    assert spec[0, 0] != pytest.approx(0.0)
    off_dc = spec.copy()
    off_dc[0, 0] = 0.0
    assert float(np.max(np.abs(off_dc))) < 1e-10


def test_dct_accepts_uint8_input() -> None:
    img = (np.arange(64, dtype=np.uint8).reshape(8, 8) * 3).astype(np.uint8)
    assert dct2(img).shape == (8, 8)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((4, 4, 1)),  # 三维
        np.zeros((0, 4)),  # 空
    ],
)
def test_dct_rejects_invalid_input(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        dct2(bad)


def test_idct_rejects_non_array() -> None:
    with pytest.raises(ValueError):
        idct2([[1.0, 2.0], [3.0, 4.0]])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 径向频率与频带边界
# ---------------------------------------------------------------------------


def test_normalized_radius_shape_dc_and_range() -> None:
    r = normalized_radius((32, 64))
    assert r.shape == (32, 64)
    assert r[0, 0] == pytest.approx(0.0)  # DC
    assert float(r.max()) <= 1.0 + 1e-12
    # 单调性：频率索引越大，径向频率越大
    assert r[1, 0] > r[0, 0]
    assert r[4, 8] > r[2, 4]


def test_normalized_radius_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        normalized_radius((0, 8))


def test_band_edges_count_and_spacing() -> None:
    e = band_edges(4)
    assert e.size == 5
    assert e[0] == pytest.approx(0.0)
    assert e[-1] == pytest.approx(1.0)
    assert np.all(np.diff(e) > 0)
    # 等间隔
    assert np.allclose(np.diff(e), np.diff(e)[0])


@pytest.mark.parametrize("bad_n", [0, -1])
def test_band_edges_rejects_non_positive_bands(bad_n: int) -> None:
    with pytest.raises(ValueError):
        band_edges(bad_n)


@pytest.mark.parametrize("bad_radius", [0.0, -0.1, 1.5])
def test_band_edges_rejects_bad_radius(bad_radius: float) -> None:
    with pytest.raises(ValueError):
        band_edges(3, max_radius=bad_radius)


def test_band_edges_log_scheme_widens_towards_high_frequency() -> None:
    """log 划分：低频窄、高频宽（真实底片能量高度集中于低频，线性等分失效）。"""
    lin = band_edges(4, scheme="linear")
    log = band_edges(4, scheme="log")
    assert log.size == 5 and lin.size == 5
    assert log[0] == pytest.approx(0.0)
    assert log[-1] == pytest.approx(1.0)
    assert np.all(np.diff(log) > 0)
    widths_lin = np.diff(lin)
    widths_log = np.diff(log)
    # 低频段更窄、末段更宽
    assert widths_log[0] < widths_lin[0]
    assert widths_log[-1] > widths_lin[-1]


def test_band_edges_log_scheme_single_band() -> None:
    e = band_edges(1, scheme="log")
    assert e.size == 2
    assert e[0] == pytest.approx(0.0)
    assert e[-1] == pytest.approx(1.0)


def test_energy_fractions_sum_to_one_for_log_single_band() -> None:
    """回归锁：``n_bands=1`` + ``log`` 时上边界必须到顶。

    曾经的实现用 ``geomspace(x, top, 1)``，它只返回起点，边界停在 x，导致
    (x, top] 的频点无频带覆盖，能量占比之和 < 1（完备性被破坏）。
    """
    fracs = band_energy_fractions(_rng_image(), n_bands=1, scheme="log")
    assert fracs.shape == (1,)
    assert float(fracs.sum()) == pytest.approx(1.0, rel=1e-10)


def test_band_edges_rejects_unknown_scheme() -> None:
    with pytest.raises(ValueError, match="scheme"):
        band_edges(4, scheme="quadratic")


def test_energy_fractions_work_with_log_scheme() -> None:
    fracs = band_energy_fractions(_rng_image(), n_bands=4, scheme="log")
    assert fracs.shape == (4,)
    assert float(fracs.sum()) == pytest.approx(1.0, rel=1e-10)


def test_spectral_prompt_propagates_scheme() -> None:
    """scheme 必须一路透传到频带划分；log 与 linear 的提示张量不应相同。"""
    img = _rng_image()
    p_lin = spectral_prompt(img, n_bands=4, scheme="linear")
    p_log = spectral_prompt(img, n_bands=4, scheme="log")
    assert p_lin.shape == p_log.shape == (4, *img.shape)
    assert not np.allclose(p_lin, p_log)


# ---------------------------------------------------------------------------
# 频带掩膜：互斥 + 完备
# ---------------------------------------------------------------------------


def test_band_masks_are_mutually_exclusive_and_complete() -> None:
    masks = radial_band_masks((21, 33), n_bands=5)
    assert len(masks) == 5
    stacked = np.stack(masks)
    # 互斥：任一点至多属于一个频带
    assert int(stacked.sum(axis=0).max()) == 1
    # 完备：并集覆盖全部频点
    assert bool(stacked.any(axis=0).all())


def test_band_masks_dc_belongs_to_first_band() -> None:
    masks = radial_band_masks((16, 16), n_bands=4)
    assert bool(masks[0][0, 0])
    assert not bool(masks[-1][0, 0])


def test_band_masks_accept_custom_edges() -> None:
    edges = np.array([0.0, 0.25, 0.5, 1.0])
    masks = radial_band_masks((16, 16), edges=edges)
    assert len(masks) == 3
    stacked = np.stack(masks)
    assert int(stacked.sum(axis=0).max()) == 1


@pytest.mark.parametrize(
    "bad_edges",
    [
        np.array([0.5]),  # 边界不足 2 个
        np.array([[0.0, 0.5], [0.5, 1.0]]),  # 非一维
        np.array([0.0, 0.6, 0.4, 1.0]),  # 非递增
    ],
)
def test_band_masks_reject_bad_edges(bad_edges: np.ndarray) -> None:
    with pytest.raises(ValueError):
        radial_band_masks((16, 16), edges=bad_edges)


# ---------------------------------------------------------------------------
# 逐带重建：可加性（本模块最重要的数学性质）
# ---------------------------------------------------------------------------


def test_band_pass_images_sum_equals_original() -> None:
    img = _rng_image()
    bps = band_pass_images(img, n_bands=4)
    assert bps.shape == (4, *img.shape)
    assert np.allclose(bps.sum(axis=0), img, atol=1e-9, rtol=0.0)


def test_band_pass_images_reject_zero_bands() -> None:
    with pytest.raises(ValueError):
        band_pass_images(_rng_image(), n_bands=0)


def test_band_pass_single_band_equals_original() -> None:
    """只有一个频带时，该带就是全图。"""
    img = _rng_image(16, 16)
    bps = band_pass_images(img, n_bands=1)
    assert bps.shape == (1, 16, 16)
    assert np.allclose(bps[0], img, atol=1e-9, rtol=0.0)


# ---------------------------------------------------------------------------
# 能量占比：语义正确性
# ---------------------------------------------------------------------------


def test_energy_fractions_sum_to_one() -> None:
    fracs = band_energy_fractions(_rng_image(), n_bands=4)
    assert fracs.shape == (4,)
    assert float(fracs.sum()) == pytest.approx(1.0, rel=1e-10)
    assert np.all(fracs >= 0.0)


def test_energy_fractions_low_dominant_for_smooth_gradient() -> None:
    """平缓灰度渐变（模拟母材/焊缝带的厚度起伏）应以低频能量为主。"""
    ramp = np.tile(np.linspace(20.0, 200.0, 64), (64, 1))
    fracs = band_energy_fractions(ramp, n_bands=4)
    assert fracs[0] > 0.9


def test_energy_fractions_high_dominant_for_checkerboard() -> None:
    """棋盘（最高空间频率）能量应落在高频带，与平滑图形成对照。

    用零均值棋盘（±1）——含直流偏置的 0/255 棋盘会把一半能量放进 DC，
    那是 DC 分量的功劳而非"高频"，会掩盖本测试要验证的结论。
    """
    yy, xx = np.mgrid[0:64, 0:64]
    checker = ((xx + yy) % 2).astype(np.float64) * 2.0 - 1.0
    fracs = band_energy_fractions(checker, n_bands=4)
    assert int(np.argmax(fracs)) == 3
    assert fracs[3] > 0.9


def test_energy_fractions_flat_image_returns_uniform() -> None:
    """全零图无能量可分，退化为等分占比而非除零。"""
    fracs = band_energy_fractions(np.zeros((16, 16)), n_bands=4)
    assert np.allclose(fracs, 0.25)


# ---------------------------------------------------------------------------
# 频域提示张量
# ---------------------------------------------------------------------------


def test_spectral_prompt_shape_and_range() -> None:
    p = spectral_prompt(_rng_image(), n_bands=4)
    assert p.shape == (4, 32, 48)
    assert float(p.min()) >= 0.0
    assert float(p.max()) <= 1.0


def test_spectral_prompt_is_centered_and_reaches_full_scale() -> None:
    """逐带归一化的契约：每个通道以 0.5 居中，且至少一端触达 0 / 1。

    该断言锁定一个真实缺陷的修复——低频带含 DC，若不去均值，通道会整体漂到
    1.0 附近（实测值域曾为 [0.84, 1.0]），动态范围被压扁、提示失效。
    """
    p = spectral_prompt(_rng_image(), n_bands=4)
    for k in range(p.shape[0]):
        assert float(p[k].mean()) == pytest.approx(0.5, abs=1e-9)
        reach = max(float(p[k].max()) - 0.5, 0.5 - float(p[k].min()))
        assert reach == pytest.approx(0.5, abs=1e-9)


def test_spectral_prompt_constant_image_degenerates_to_midpoint() -> None:
    """常数图各频带无响应 → 全 0.5（无信息），且不产生 NaN。"""
    p = spectral_prompt(np.full((16, 16), 42.0), n_bands=3)
    assert np.allclose(p, 0.5)
    assert bool(np.isfinite(p).all())


def test_spectral_prompt_channels_are_not_degenerate() -> None:
    """各通道方差别塌成 0——否则提示通道形同虚设（全局归一化的失败模式）。"""
    p = spectral_prompt(_rng_image(), n_bands=4)
    for k in range(p.shape[0]):
        assert float(np.std(p[k])) > 1e-3


def test_spectral_prompt_rejects_bad_eps() -> None:
    with pytest.raises(ValueError):
        spectral_prompt(_rng_image(), eps=0.0)
