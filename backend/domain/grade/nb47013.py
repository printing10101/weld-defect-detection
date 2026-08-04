"""NB/T47013.2-2015 标准判定引擎（§6，M5 实现）。

实现冻结的 StandardGrader 契约。规则框架基于公开解读（§6.2）：
- 零容忍：裂纹/未熔合/未焊透在 I-III 级均不允许 → 存在即 IV 级；
- 圆形缺陷：按 T 选评定区 → 区内点数求和 → 对照点数上限分级；
  长径 > T/2 直判 IV；小于不计点数阈值的不计；
- 条形缺陷：单个最大长度对照限值（II ≤T/3(min4)且≤20；III ≤2T/3(min6)且≤30）；
- 综合评级：圆形级别 + 条形级别 − 1（≤IV）。

熔断（§T8）：tables.authorized=false 时**不输出任何级别**，抛
GradingAmbiguousError（应用层转 422，前端视为"需人工复核"）。
⚠️ 数值全部来自表（YAML，公开参考占位）；正式使用须以授权原文复核并置 authorized=true。
"""
from __future__ import annotations

from backend.domain.dto import (
    DefectClass,
    Detection,
    GradeResult,
    ImageMeta,
    JointLevel,
)
from backend.domain.errors import GradingAmbiguousError
from backend.domain.standards.tables.loader import StandardTables

_ZERO_TOLERANCE = {
    DefectClass.CRACK,
    DefectClass.LACK_OF_FUSION,
    DefectClass.INCOMPLETE_PENETRATION,
}
_ORDER = {JointLevel.I: 1, JointLevel.II: 2, JointLevel.III: 3, JointLevel.IV: 4}


class Nb47013Grader:
    """NB/T47013.2-2015 评级实现。"""

    def __init__(self, tables: StandardTables) -> None:
        self.tables = tables

    def grade(self, defects: list[Detection], context: ImageMeta) -> GradeResult:
        if not self.tables.authorized:
            raise GradingAmbiguousError("标准数值未授权：禁止输出级别，需人工复核")
        t = context.base_metal_thickness_mm
        if t is None or t <= 0:
            raise GradingAmbiguousError("缺少有效母材厚度 T，无法判定")
        spacing = context.pixel_spacing_mm or 1.0

        if any(d.class_id in _ZERO_TOLERANCE for d in defects):
            return self._result(
                JointLevel.IV,
                defects,
                ("NB/T47013.2-2015：裂纹/未熔合/未焊透在 I-III 级不允许（零容忍）",),
                need_review=True,
            )

        round_defs = [d for d in defects if self._aspect(d, spacing) <= 3.0]
        linear_defs = [d for d in defects if self._aspect(d, spacing) > 3.0]
        round_level = self._grade_round(round_defs, t, spacing)
        linear_level = self._grade_linear(linear_defs, t, spacing)

        if round_defs and linear_defs:
            combined = min(4, _ORDER[round_level] + _ORDER[linear_level] - 1)
            joint = _from_order(combined)
            basis = ("综合评级：圆形级别 + 条形级别 − 1（≤IV）",)
        else:
            joint = round_level if _ORDER[round_level] >= _ORDER[linear_level] else linear_level
            basis = ("单一类型缺陷取最差级别",)
        return self._result(joint, defects, basis, need_review=False)

    def _grade_round(self, round_defs: list[Detection], t: float, spacing: float) -> JointLevel:
        if not round_defs:
            return JointLevel.I
        total = 0
        for d in round_defs:
            dia = self._diameter_mm(d, spacing)
            if dia > t / 2:
                return JointLevel.IV  # 长径 > T/2 直判
            if dia <= self._ignore_diameter(t):
                continue
            total += self._to_points(dia)
        return self._limits_level(t, total)

    def _grade_linear(self, linear_defs: list[Detection], t: float, spacing: float) -> JointLevel:
        if not linear_defs:
            return JointLevel.I
        worst = max(self._diameter_mm(d, spacing) for d in linear_defs)
        l2 = self.tables.data["linear_limits"]["level2"]
        l3 = self.tables.data["linear_limits"]["level3"]
        lim2 = min(max(t * l2["t_factor"], l2["min_mm"]), l2["max_mm"])
        lim3 = min(max(t * l3["t_factor"], l3["min_mm"]), l3["max_mm"])
        if worst <= lim2:
            return JointLevel.II
        if worst <= lim3:
            return JointLevel.III
        return JointLevel.IV

    def _zone_width(self, t: float) -> float:
        for row in self.tables.data["round_rating_zone_mm"]:
            if t <= row["max_t"]:
                return float(row["width"])
        return 30.0

    def _ignore_diameter(self, t: float) -> float:
        for row in self.tables.data["round_ignore_size_mm"]:
            if t <= row["max_t"]:
                return float(row["max_d"])
        return 1.5

    def _to_points(self, dia: float) -> int:
        for row in self.tables.data["round_points"]:
            if dia <= row["max_d_mm"]:
                return int(row["points"])
        return 40

    def _limits_level(self, t: float, points: int) -> JointLevel:
        for row in self.tables.data["round_grade_limits"]:
            if t <= row["max_t"]:
                if points <= row["I"]:
                    return JointLevel.I
                if points <= row["II"]:
                    return JointLevel.II
                if points <= row["III"]:
                    return JointLevel.III
                return JointLevel.IV
        return JointLevel.IV

    def _aspect(self, d: Detection, spacing: float) -> float:
        w_mm = d.bbox.w * spacing
        h_mm = d.bbox.h * spacing
        return max(w_mm, h_mm) / max(min(w_mm, h_mm), 1e-6)

    def _diameter_mm(self, d: Detection, spacing: float) -> float:
        return max(d.bbox.w, d.bbox.h) * spacing

    def _result(
        self,
        level: JointLevel,
        defects: list[Detection],
        basis: tuple[str, ...],
        need_review: bool,
    ) -> GradeResult:
        return GradeResult(
            joint_level=level,
            per_defect_grade=tuple([level] * len(defects)),
            basis=basis,
            need_review=need_review,
            standard_id=self.tables.standard_id,
            standard_version=self.tables.version,
        )


def _from_order(order: int) -> JointLevel:
    for level, o in _ORDER.items():
        if o == order:
            return level
    return JointLevel.IV
