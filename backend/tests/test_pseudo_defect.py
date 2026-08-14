"""伪缺陷筛查测试（§4.2 底片质量校验）。

验证启发式筛查的阻断/告警逻辑：长直划痕默认阻断；尘点/显影不均默认仅告警；
纯底本噪声不误报。注意：当 IQI 自身占满整帧时，其长直丝会被划痕启发式命中，
属预期（真实底片 IQI 仅占角落，见 test_locate/endpoint 用大图场景）。
"""

from __future__ import annotations

import cv2
import numpy as np

from backend.domain.pseudo_defect import PseudoDefectCfg, screen_pseudo_defects

_W, _H = 400, 400


def _clean() -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.normal(128.0, 3.0, (_H, _W)).astype(np.uint8)


def _with_scratch() -> np.ndarray:
    img = _clean()
    # 一条横贯全图的亮划痕（≈100% 对角线） → 应触发阻断
    cv2.line(img, (0, _H // 2), (_W - 1, _H // 2), 230, 2)
    return img


def test_clean_passes() -> None:
    rep = screen_pseudo_defects(_clean(), PseudoDefectCfg())
    assert rep.passed is True
    # 无阻断类伪缺陷（尘点/显影不均默认仅告警，不进 notes 阻断）
    assert "划痕" not in "".join(rep.notes)


def test_long_scratch_blocks() -> None:
    rep = screen_pseudo_defects(_with_scratch(), PseudoDefectCfg())
    assert rep.passed is False
    assert any("划痕" in n for n in rep.notes)


def test_metrics_populated() -> None:
    rep = screen_pseudo_defects(_with_scratch(), PseudoDefectCfg())
    assert "scratch_max_ratio" in rep.metrics
    assert "uniformity_ratio" in rep.metrics
    assert "dust_count" in rep.metrics
    assert isinstance(rep.notes, tuple)


def _with_grating() -> np.ndarray:
    """成排平行的长亮线（模拟像质计丝）→ 周期性光栅，不应误判为划痕。"""
    img = _clean()
    n = 12
    for i in range(n):
        y = 20 + i * 14
        cv2.line(img, (0, y), (_W - 1, y), 220, 2)
    return img


def test_iqi_grating_not_flagged_as_scratch() -> None:
    """像质计丝（周期性光栅）须与孤立长划痕区分：不阻断评片。"""
    rep = screen_pseudo_defects(_with_grating(), PseudoDefectCfg())
    assert rep.passed is True
    assert not any("划痕" in n for n in rep.notes)


def test_single_scratch_still_blocked() -> None:
    """孤立长划痕（非光栅）仍须阻断。"""
    rep = screen_pseudo_defects(_with_scratch(), PseudoDefectCfg())
    assert rep.passed is False
