"""GJB 1187A 军标评级适配器（S-21/E-16）。

军标评级骨架：按 registry.register_standard 模式注册（standard_id="GJB 1187A"），
结构对齐既有适配器样板（gb3323.py）。

诚实边界（务必读）：
- 评级数值表（tables/gjb1187.yaml）**全部为占位值**，不来自军标原文——军标属
  受控文件，本系统未获得授权正本，未做任何转录核对。表值待军标原文校核后启用。
- 因此表 yaml 置 ``authorized: false``，走既有熔断语义：``grade`` 一律抛
  ``GradingAmbiguousError``（应用层 422，需人工复核），禁止静默错判——
  宁可"需人工复核"也不输出一个无依据的级别。
- 本模块提供的是**分级框架**（缺陷圆/条形分组、零容忍缺陷识别、表值启用前置
  校验），评级引擎本体在表值经军标原文校核并签核之前不实现、不启用。
  双重熔断：表 ``authorized=false`` 或版本号仍为 draft（``0-draft*``）时均拒绝。

启用路径：获得授权正本 → 逐条转录并签核 → 替换 yaml 占位值 → authorized=true
且版本号升级（非 draft）→ 实现引擎并补全真值单测。
"""

from __future__ import annotations

from backend.domain.dto import Detection, GradeResult, ImageMeta
from backend.domain.errors import GradingAmbiguousError
from backend.domain.interfaces import StandardGrader

# 零容忍缺陷（对齐 NB/T47013.2 语义：任一出现即不评级别，直接拒收）。
# GJB 1187A 的零容忍集合待军标原文校核后修订，此处为骨架占位。
ZERO_TOLERANCE_CLASSES = ("CRACK", "LACK_OF_FUSION", "INCOMPLETE_PENETRATION")


class Gjb1187Grader(StandardGrader):
    """GJB 1187A 军标评级（表值占位、authorized=false，一律熔断）。"""

    standard_id = "GJB 1187A"

    def __init__(self, tables) -> None:
        self.tables = tables

    # ---- 分级框架（纯结构，不读数值、不输出级别） --------------------------
    @staticmethod
    def partition(defects: list[Detection]) -> dict[str, list[Detection]]:
        """缺陷分组框架：圆形/条形（按 Detection.shape）+ 零容忍缺陷单列。

        圆/条形分界沿用检测几何结论（L/W<=3 为圆形，见 detect.round_aspect_max）；
        零容忍（裂纹/未熔合/未焊透）无论形状单列，命中即拒收框架的一部分。
        """
        out: dict[str, list[Detection]] = {"round": [], "linear": [], "zero_tolerance": []}
        for d in defects or []:
            if d.class_id.name in ZERO_TOLERANCE_CLASSES:
                out["zero_tolerance"].append(d)
            elif d.shape is not None and d.shape.value == "round":
                out["round"].append(d)
            else:
                out["linear"].append(d)
        return out

    # ---- 评级（熔断） --------------------------------------------------------
    def _fusible(self) -> str | None:
        """返回熔断原因；None 表示（理论上）允许评级。双重校验防误启用。"""
        tables = self.tables
        if tables is None:
            return "评级数值表缺失"
        if not getattr(tables, "authorized", False):
            return "评级数值表未授权（表值待军标原文校核后启用）"
        version = str(getattr(tables, "version", ""))
        if version.startswith("0-draft"):
            return f"评级数值表仍为占位草稿版本（version={version}）"
        return None

    def grade(self, defects: list[Detection], context: ImageMeta) -> GradeResult:
        reason = self._fusible()
        if reason is not None:
            raise GradingAmbiguousError(
                f"标准 {self.standard_id} {reason}，禁止输出级别，需人工复核"
            )
        # 表值经军标原文校核、authorized=true 且版本非 draft 后，
        # 在此接入真实评级引擎（圆/条形等效点数分级）并补全真值单测。
        raise GradingAmbiguousError(
            f"标准 {self.standard_id} 评级引擎待军标原文校核后实现，需人工复核"
        )
