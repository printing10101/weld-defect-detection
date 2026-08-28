"""NB/T47013.2-2015 标准判定引擎。

实现冻结的 StandardGrader 契约。规则框架基于公开解读：
- 零容忍：裂纹/未熔合/未焊透在 I-III 级均不允许 → 存在即 IV 级；
- 圆形缺陷：按 T 选评定区 → 区内点数求和 → 对照点数上限分级；
  长径 > T/2 直判 IV；小于不计点数阈值的不计；
- 条形缺陷：单个最大长度对照限值（II ≤T/3(min4)且≤20；III ≤2T/3(min6)且≤30）；
- 综合评级：圆形级别 + 条形级别 − 1（≤IV）。

熔断：tables.authorized=false 时**不输出任何级别**，抛
GradingAmbiguousError（应用层转 422，前端视为"需人工复核"）。
 数值全部来自表（YAML，公开参考占位）；正式使用须以授权原文复核并置 authorized=true。
"""

from __future__ import annotations

from backend.domain.dto import (
    DefectClass,
    DefectShape,
    Detection,
    GradeResult,
    ImageMeta,
    JointLevel,
)
from backend.domain.errors import GradingAmbiguousError
from backend.domain.standards.tables.loader import StandardTables, disclaimer_for

_ZERO_TOLERANCE = {
    DefectClass.CRACK,
    DefectClass.LACK_OF_FUSION,
    DefectClass.INCOMPLETE_PENETRATION,
}
_ORDER = {JointLevel.I: 1, JointLevel.II: 2, JointLevel.III: 3, JointLevel.IV: 4}


class Nb47013Grader:
    """NB/T47013.2-2015 评级实现。"""

    def __init__(self, tables: StandardTables, review_uncertainty: float = 0.5) -> None:
        self.tables = tables
        self.review_uncertainty = review_uncertainty

    def grade(self, defects: list[Detection], context: ImageMeta) -> GradeResult:
        if not self.tables.authorized:
            raise GradingAmbiguousError("标准数值未授权：禁止输出级别，需人工复核")
        t = context.base_metal_thickness_mm
        if t is None or t <= 0:
            raise GradingAmbiguousError("缺少有效母材厚度 T，无法判定")
        spacing = context.pixel_spacing_mm
        if spacing is None or spacing <= 0:
            raise GradingAmbiguousError("缺少有效像素标定 pixel_spacing_mm，无法换算物理尺寸")
        # 人工兜底：任一检测不确定性超阈值则升级人工复核（ + ）
        needs_human = any(d.uncertainty > self.review_uncertainty for d in defects)

        if any(d.class_id in _ZERO_TOLERANCE for d in defects):
            per_all = tuple(JointLevel.IV for _ in defects)
            return self._result(
                JointLevel.IV,
                per_all,
                ("NB/T47013.2-2015：裂纹/未熔合/未焊透在 I-III 级不允许（零容忍）",),
                need_review=True,
            )

        # 深孔（黑度>母材）直判 IV：检测器/管线标注的深孔缺陷一律 IV。
        # 特征来源（deep_hole 字段）由检测侧提供；本规则保证"标注即直判"，
        # 不依赖尺寸/点数（深孔属重大缺陷，与零容忍同级对待）。
        deep_holes = [d for d in defects if d.deep_hole]
        if deep_holes:
            per_all = tuple(JointLevel.IV for _ in defects)
            return self._result(
                JointLevel.IV,
                per_all,
                ("NB/T47013.2-2015：深孔（黑度>母材）直判 IV（§6.2）",),
                need_review=True,
            )

        # 内凹（DB50/T 1807 重点关注缺陷）：按深度评定，深度未测无法自动定级 →
        # 暂按Ⅲ级保守处理并强制人工复核，不参与圆形/条形点数聚合（防污染其口径）。
        concav_defs = [d for d in defects if d.class_id is DefectClass.CONCAVITY]
        graded_defs = [d for d in defects if d.class_id is not DefectClass.CONCAVITY]
        round_defs = [d for d in graded_defs if self._is_round(d, spacing)]
        linear_defs = [d for d in graded_defs if not self._is_round(d, spacing)]
        round_level = self._grade_round(round_defs, t, spacing)
        linear_level = self._grade_linear(linear_defs, t, spacing)

        if round_defs and linear_defs:
            ro, lo = _ORDER[round_level], _ORDER[linear_level]
            # 综合评级：round_level + linear_level − 1（≤IV）
            combined = min(4, ro + lo - 1)
            joint = _from_order(combined)
            combined_basis = (
                f"综合评级：圆形{round_level.value}级 + 条形{linear_level.value}级 → "
                f"{round_level.value}+{linear_level.value}−1 = {joint.value}级"
            )
            basis = (combined_basis,)
        else:
            joint = round_level if _ORDER[round_level] >= _ORDER[linear_level] else linear_level
            basis = ("单一类型缺陷取最差级别",)

        if concav_defs:
            # 保守暂定Ⅲ级：宁可偏严进入人工复核，也不因深度缺失而放行
            joint = max((joint, JointLevel.III), key=lambda L: _ORDER[L])
            basis = basis + (
                "内凹为重点关注缺陷，按深度评定需深度数据，暂按Ⅲ级保守处理并强制人工复核",
            )

        # 逐缺陷级别（供报告缺陷清单与 κ 一致性计算，杜绝统一复制）
        per: list[JointLevel] = []
        for d in defects:
            if d.class_id is DefectClass.CONCAVITY:
                per.append(JointLevel.III)  # 保守暂定，强制人工复核后覆盖
            elif d.class_id in _ZERO_TOLERANCE or d.deep_hole:
                per.append(JointLevel.IV)
            elif self._is_round(d, spacing):
                per.append(self._grade_round([d], t, spacing))
            else:
                per.append(self._grade_linear([d], t, spacing))

        # 尺寸临界（长径≈T/2、点数压线、条形长度压线）→ 需人工复核
        near_critical = self._near_critical(defects, round_defs, t, spacing)
        return self._result(
            joint, tuple(per), basis, need_review=bool(needs_human or near_critical or concav_defs)
        )

    def _grade_round(self, round_defs: list[Detection], t: float, spacing: float) -> JointLevel:
        if not round_defs:
            return JointLevel.I
        zone_w = self._zone_width(t)  # 评定区长边 mm
        zone_h = min(zone_w, 10.0)  # 评定区短边固定 10mm（§6.2）
        ignore_d = self._ignore_diameter(t)  # 循环外求值，避免重复扫描
        # (cx_mm, cy_mm, points, is_ignored)：is_ignored=长径≤不计点数阈值（points=0）
        scored: list[tuple[float, float, int, bool]] = []
        for d in round_defs:
            dia = self._diameter_mm(d, spacing)
            if dia > t / 2:
                return JointLevel.IV  # 长径 > T/2 直判
            cx = (d.bbox.x + d.bbox.w / 2) * spacing
            cy = (d.bbox.y + d.bbox.h / 2) * spacing
            if dia <= ignore_d:
                scored.append((cx, cy, 0, True))  # 不计点数缺陷（仅计数，不累计点数）
                continue
            scored.append((cx, cy, self._to_points(dia), False))
        if not scored:
            return self._limits_level(t, 0)
        # 以每个缺陷为评定区左上锚点滑窗，取区内点数最大者定为该区域级别；
        # 同步统计窗口内不计点数缺陷数量。
        worst_points = 0
        worst_ignored = 0
        for ax, ay, _, _ in scored:
            pts = sum(
                p
                for x, y, p, ign in scored
                if not ign and ax <= x < ax + zone_w and ay <= y < ay + zone_h
            )
            ignored = sum(
                1
                for x, y, _, ign in scored
                if ign and ax <= x < ax + zone_w and ay <= y < ay + zone_h
            )
            worst_points = max(worst_points, pts)
            worst_ignored = max(worst_ignored, ignored)
        level = self._limits_level(t, worst_points)
        # I 级及 T≤5mm 的 II 级评定区内不计点缺陷>10 个 → 降一级
        if (level is JointLevel.I or (t <= 5 and level is JointLevel.II)) and worst_ignored > 10:
            return _from_order(min(4, _ORDER[level] + 1))
        return level

    def _grade_linear(self, linear_defs: list[Detection], t: float, spacing: float) -> JointLevel:
        if not linear_defs:
            return JointLevel.I
        # 同线间距≤小缺陷长度 → 合并（先合并再判单条/组累计，防止割裂计数）
        merged = self._merge_collinear(linear_defs, spacing)
        worst = max(e - s for s, e in merged)
        lim2, lim3 = self._linear_limits(t)
        if worst <= lim2:
            single = JointLevel.II
        elif worst <= lim3:
            single = JointLevel.III
        else:
            return JointLevel.IV
        # 组内(12T区)累计：II 级累计 ≤2T/3(最小6)且≤30（组表缺失时按单条判定）
        group = self.tables.data["linear_limits"].get("group")
        if group is None:
            return single
        zone_len = group["zone_t_factor"] * t
        cumulative = self._max_12t_cumulative(merged, zone_len)
        g2 = group["level2"]
        g3 = group["level3"]
        glim2 = min(max(t * g2["t_factor"], g2["min_mm"]), g2["max_mm"])
        glim3 = min(max(t * g3["t_factor"], g3["min_mm"]), g3["max_mm"])
        if cumulative <= glim2:
            group_level = JointLevel.II
        elif cumulative <= glim3:
            group_level = JointLevel.III
        else:
            return JointLevel.IV
        return max(single, group_level, key=lambda L: _ORDER[L])

    def _linear_limits(self, t: float) -> tuple[float, float]:
        """条形单条限值 (lim2, lim3)。"""
        l2 = self.tables.data["linear_limits"]["level2"]
        l3 = self.tables.data["linear_limits"]["level3"]
        lim2 = min(max(t * l2["t_factor"], l2["min_mm"]), l2["max_mm"])
        lim3 = min(max(t * l3["t_factor"], l3["min_mm"]), l3["max_mm"])
        return lim2, lim3

    def _near_critical(
        self,
        defects: list[Detection],
        round_defs: list[Detection],
        t: float,
        spacing: float,
    ) -> bool:
        """尺寸临界判定：长径≈T/2、条形长度压线、点数压线 → True。

        与 uncertainty 高触发同义（任一命中即 need_review），保证临界缺陷
        即便高置信也进入人工复核。
        """
        half_t = t / 2
        for d in defects:
            if d.class_id in _ZERO_TOLERANCE or d.class_id is DefectClass.CONCAVITY:
                continue  # 零容忍已强制 need_review；内凹深度未测亦已强制 need_review
            dia = self._diameter_mm(d, spacing)
            if self._is_round(d, spacing):
                if 0.9 * half_t <= dia <= half_t:  # 长径接近 T/2 直判线
                    return True
            else:
                lim2, lim3 = self._linear_limits(t)
                if 0.9 * lim2 <= dia <= lim2 or 0.9 * lim3 <= dia <= lim3:  # 条形压线
                    return True
        # 点数临界：评定区内点数恰为该级上限（压线即临界，防"差一点降级"误判）
        if round_defs:
            for row in self.tables.data["round_grade_limits"]:
                if t <= row["max_t"]:
                    zone_w = self._zone_width(t)
                    zone_h = min(zone_w, 10.0)
                    points_in_zone = self._round_zone_points(round_defs, t, spacing, zone_w, zone_h)
                    for limit in (row["I"], row["II"], row["III"]):
                        if points_in_zone == limit:
                            return True
                    break
        return False

    def _round_zone_points(
        self,
        round_defs: list[Detection],
        t: float,
        spacing: float,
        zone_w: float,
        zone_h: float,
    ) -> int:
        """评定区内点数累计最大值（供临界判定复用滑窗口径，与 _grade_round 一致）。"""
        ignore_d = self._ignore_diameter(t)
        scored: list[tuple[float, float, int]] = []
        for d in round_defs:
            dia = self._diameter_mm(d, spacing)
            if dia > t / 2:
                continue  # 直判 IV，不进滑窗
            if dia <= ignore_d:
                continue
            scored.append(
                (
                    (d.bbox.x + d.bbox.w / 2) * spacing,
                    (d.bbox.y + d.bbox.h / 2) * spacing,
                    self._to_points(dia),
                )
            )
        if not scored:
            return 0
        return max(
            sum(p for x, y, p in scored if ax <= x < ax + zone_w and ay <= y < ay + zone_h)
            for ax, ay, _ in scored
        )

    def _merge_collinear(self, defs: list[Detection], spacing: float) -> list[tuple[float, float]]:
        """同线间距≤小缺陷长度 → 合并。

        简化：缺陷长轴方向即投影轴（水平 bbox 投 x，垂直投 y）——假设焊缝方向
        与检测框长轴一致，故所有条形缺陷沿各自长轴一维投影；排序后贪心合并，
        相邻间距 ≤ 较小缺陷长度时并为一个区间（跨度 = 两端包围盒）。
        返回合并后的 (start_mm, end_mm) 区间列表。
        """
        segments: list[list[float]] = []
        for d in defs:
            if d.bbox.w >= d.bbox.h:  # 水平长轴 → x 投影
                s, e = d.bbox.x * spacing, (d.bbox.x + d.bbox.w) * spacing
            else:  # 垂直长轴 → y 投影
                s, e = d.bbox.y * spacing, (d.bbox.y + d.bbox.h) * spacing
            segments.append([s, e])
        if not segments:
            return []
        segments.sort(key=lambda seg: seg[0])
        merged: list[list[float]] = [segments[0]]
        for s, e in segments[1:]:
            cs, ce = merged[-1]
            gap = s - ce
            if gap <= min(ce - cs, e - s):  # 间距≤小缺陷长度 → 合并
                merged[-1][1] = max(ce, e)
            else:
                merged.append([s, e])
        return [(s, e) for s, e in merged]

    def _max_12t_cumulative(self, intervals: list[tuple[float, float]], zone_len: float) -> float:
        """12T 评定区滑窗：窗口内区间长度累计的最大值（mm）。

        以每个区间起点为窗口左端滑窗（覆盖最密区段），窗口内各区间按
        与窗口的交叠长度累计（截断到窗口边界），取累计最大者。
        """
        if not intervals:
            return 0.0
        starts = sorted(s for s, _ in intervals)
        worst = 0.0
        for w0 in starts:
            w1 = w0 + zone_len
            cum = sum(min(e, w1) - max(s, w0) for s, e in intervals if s < w1 and e > w0)
            worst = max(worst, cum)
        return worst

    def _zone_width(self, t: float) -> float:
        for row in self.tables.data["round_rating_zone_mm"]:
            if t <= row["max_t"]:
                return float(row["width"])
        return 30.0

    def _ignore_diameter(self, t: float) -> float:
        for row in self.tables.data["round_ignore_size_mm"]:
            if t <= row["max_t"]:
                return float(row["max_d"])
        raise GradingAmbiguousError(f"不计点数阈值表未覆盖母材厚度 T={t:.2f}mm，需人工复核")

    def _to_points(self, dia: float) -> int:
        for row in self.tables.data["round_points"]:
            if dia <= row["max_d_mm"]:
                return int(row["points"])
        raise GradingAmbiguousError(f"点数表未覆盖长径 {dia:.2f}mm，需人工复核")

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

    def _is_round(self, d: Detection, spacing: float) -> bool:
        """优先采信检测器给出的形状，否则按长宽比估计（NB/T47013：L/W<=3 为圆形）。"""
        if d.shape is not None:
            return d.shape is DefectShape.ROUND
        return self._aspect(d, spacing) <= 3.0

    def _diameter_mm(self, d: Detection, spacing: float) -> float:
        return max(d.bbox.w, d.bbox.h) * spacing

    def _result(
        self,
        level: JointLevel,
        per_defect_grade: tuple[JointLevel, ...],
        basis: tuple[str, ...],
        need_review: bool,
    ) -> GradeResult:
        return GradeResult(
            joint_level=level,
            per_defect_grade=per_defect_grade,
            basis=basis,
            need_review=need_review,
            standard_id=self.tables.standard_id,
            standard_version=self.tables.version,
            disclaimer=disclaimer_for(self.tables),
        )


def _from_order(order: int) -> JointLevel:
    for level, o in _ORDER.items():
        if o == order:
            return level
    return JointLevel.IV
