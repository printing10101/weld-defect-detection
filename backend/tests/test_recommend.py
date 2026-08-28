"""合规处置建议引擎测试（domain/recommend + POST /recommend）。"""

from __future__ import annotations

from backend.app.main import app
from backend.domain.dto import BBox, DefectClass, Detection
from backend.domain.recommend import (
    DISPOSITION_ACCEPT,
    DISPOSITION_CONDITIONAL,
    DISPOSITION_RECHECK,
    DISPOSITION_REWORK,
    recommend,
)


def _det(class_id: DefectClass, score: float = 0.9, uncertainty: float = 0.1) -> Detection:
    return Detection(
        id=f"d-{class_id.value}",
        bbox=BBox(10, 10, 20, 20),
        class_id=class_id,
        score=score,
        uncertainty=uncertainty,
    )


class TestRecommendDomain:
    def test_zero_tolerance_always_rework(self):
        """裂纹/未熔合/未焊透存在 → 无论级别一律不合格。"""
        for cls in (
            DefectClass.CRACK,
            DefectClass.LACK_OF_FUSION,
            DefectClass.INCOMPLETE_PENETRATION,
        ):
            rec = recommend("I", [_det(cls)])
            assert rec.disposition == DISPOSITION_REWORK

    def test_deep_hole_rework(self):
        rec = recommend("II", [_det(DefectClass.POROSITY, uncertainty=0.9)], need_review=False)
        # 深孔字段在 recommend 中由缺陷 deep_hole 判定，这里构造 deep_hole 缺陷
        d = Detection(
            id="dh",
            bbox=BBox(0, 0, 10, 10),
            class_id=DefectClass.POROSITY,
            score=0.95,
            uncertainty=0.1,
            deep_hole=True,
        )
        rec = recommend("II", [d])
        assert rec.disposition == DISPOSITION_REWORK

    def test_iv_rework(self):
        rec = recommend("IV", [], need_review=False)
        assert rec.disposition == DISPOSITION_REWORK

    def test_need_review_recheck_even_accept_level(self):
        """I/II 级但触发人工复核兜底 → 暂缓处置。"""
        rec = recommend("II", [_det(DefectClass.POROSITY)], need_review=True)
        assert rec.disposition == DISPOSITION_RECHECK

    def test_iii_conditional(self):
        rec = recommend("III", [_det(DefectClass.SLAG)], need_review=False)
        assert rec.disposition == DISPOSITION_CONDITIONAL

    def test_i_ii_accept(self):
        for lv in ("I", "II"):
            rec = recommend(lv, [_det(DefectClass.POROSITY)], need_review=False)
            assert rec.disposition == DISPOSITION_ACCEPT

    def test_none_level_recheck(self):
        """方法标准/未授权 → 级别不可得，交人工。"""
        rec = recommend(None, [], standard_id="ASME-V")
        assert rec.disposition == DISPOSITION_RECHECK
        assert "ASME-V" in rec.standard_id

    def test_unknown_level_recheck(self):
        rec = recommend("IX", [])
        assert rec.disposition == DISPOSITION_RECHECK

    def test_never_raises_on_empty(self):
        """引擎永不抛错（容错承诺）。"""
        assert recommend(None, []).disposition == DISPOSITION_RECHECK

    def test_default_disclaimer_present(self):
        rec = recommend("I", [])
        assert rec.disclaimer is not None
        assert "不构成" in rec.disclaimer


class TestRecommendApi:
    def _call(self, payload: dict):
        from fastapi.testclient import TestClient

        with TestClient(app) as c:
            return c.post("/api/v1/recommend", json=payload)

    def test_zero_tolerance_endpoint(self):
        r = self._call(
            {
                "base_metal_thickness_mm": 10.0,
                "pixel_spacing_mm": 0.1,
                "standard_id": "NB/T47013.2-2015",
                "defects": [
                    {"id": "c1", "class_id": 4, "bbox": [10, 10, 20, 20], "confidence": 0.9}
                ],
            }
        )
        assert r.status_code == 200
        body = r.json()
        assert body["disposition"] == DISPOSITION_REWORK
        assert body["joint_level"] == "IV"  # 零容忍 → IV
        assert body["disclaimer"]

    def test_accept_endpoint(self):
        r = self._call(
            {
                "base_metal_thickness_mm": 10.0,
                "pixel_spacing_mm": 0.1,
                "standard_id": "NB/T47013.2-2015",
                "defects": [],
            }
        )
        assert r.status_code == 200
        body = r.json()
        assert body["disposition"] == DISPOSITION_ACCEPT

    def test_unauthorized_degrades_to_recheck(self):
        """评级熔断（如 GB/T3323 数值表未转录）→ 建议降级为复核而非 422。"""
        from fastapi.testclient import TestClient

        from backend.app.main import app as app_mod

        with TestClient(app_mod) as c:
            r = c.post(
                "/api/v1/recommend",
                json={
                    "base_metal_thickness_mm": 10.0,
                    "pixel_spacing_mm": 0.1,
                    # GB/T3323 数值表未转录/未授权 → 判定器熔断
                    "standard_id": "GB/T3323-2019",
                    "defects": [
                        {"id": "c1", "class_id": 1, "bbox": [10, 10, 20, 20], "confidence": 0.6}
                    ],
                },
            )
            assert r.status_code == 200
            body = r.json()
            assert body["disposition"] == DISPOSITION_RECHECK
            assert body["joint_level"] is None
            assert body["disclaimer"] and "熔断" in body["disclaimer"]
