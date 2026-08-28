"""黑度估计与门限测试。"""

from __future__ import annotations

import numpy as np

from backend.domain.density import check_density, estimate_density


def test_white_near_zero_density() -> None:
    d = estimate_density(np.full((8, 8), 255, np.uint8))
    assert d < 0.05


def test_black_8bit_matches_formula() -> None:
    d = estimate_density(np.zeros((8, 8), np.uint8))
    assert abs(d - 2.408) < 0.01  # log10(256/1)


def test_uint16_reaches_ab_range() -> None:
    d = estimate_density(np.zeros((8, 8), np.uint16))
    assert d > 4.5  # log10(65536) ≈ 4.816


def test_density_gate() -> None:
    assert check_density(2.0) is True
    assert check_density(4.5) is True
    assert check_density(1.9) is False
    assert check_density(4.6) is False
    assert check_density(4.6, low=2.0, high=5.0) is True
