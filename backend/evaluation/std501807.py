"""DB50/T 1807-2025 评价体系核心计算（§9 测试 / §10 指标 / §11 结果）。

与现有 mAP@0.5 体系（harness.py）**口径不同、并行共存**：
- 本模块按地方标准口径：IOU≥0.1 即视为位置匹配（非 0.5）；误检=位置对但
  类型错（FD）；漏检=GT 未被匹配（MD）；误报=无缺陷底片整图级误报（FRR）。
- 比标准更严格：
  1) 双阈值并行：标准口径 0.1 之外同时按严格口径（默认 0.3）评估，记录
     系统分级取两者较差者；
  2) FRR 分级线默认取收紧值（自动 3% / 手工 4%，即标准 L2 线）而非 L1 线；
  3) 未匹配预测（FP_extra）单独暴露计数，不静默吞掉；
  4) 漏检风险在无人工评级信息时保守按"≥Ⅱ级"处理（归Ⅰ类风险）。

纯 numpy / 标准库，可离线单测。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 标准缺陷类别（§10.1 n=1..7）与模型类别映射
# ---------------------------------------------------------------------------

STD_CLASS_NAMES: dict[int, str] = {
    1: "圆形缺陷",
    2: "条形缺陷",
    3: "裂纹",
    4: "未熔合",
    5: "未焊透",
    6: "内凹",
    7: "咬边",
}

# 重点关注缺陷（§8.2）：双面焊 = 裂纹/未熔合/未焊透；单面焊另含内凹/咬边
FOCUS_DEFECTS: dict[str, set[int]] = {
    "double": {3, 4, 5},
    "single": {3, 4, 5, 6, 7},
}

# 模型类别（DefectClass，dto.py）→ 标准类别 n。
# 夹渣按形状分流：长宽比 >3 归条形缺陷，否则圆形缺陷（在 _to_std_class 处理）。
_MODEL_TO_STD: dict[int, int] = {
    0: 1,  # POROSITY 气孔 → 圆形缺陷
    1: -1,  # SLAG 夹渣 → 按形状分流（-1 标记）
    2: 5,  # INCOMPLETE_PENETRATION 未焊透
    3: 4,  # LACK_OF_FUSION 未熔合
    4: 3,  # CRACK 裂纹
    5: 7,  # UNDERCUT 咬边
    6: 6,  # CONCAVITY 内凹
}

_ROUND, _LINEAR = 1, 2


def _to_std_class(
    class_id: int,
    bbox: Iterable[float] | None,
    aspect_round_max: float = 3.0,
    mapping: dict[int, int] | None = None,
) -> int:
    """模型类别 → 标准类别 n；夹渣按长宽比分流圆形/条形（L/W≤3 为圆形）。"""
    mp = mapping or _MODEL_TO_STD
    n = mp[class_id]
    if n != -1:
        return n
    if bbox is None:
        return _ROUND
    w, h = bbox[2], bbox[3]
    aspect = max(w, h) / max(min(w, h), 1e-6)
    return _ROUND if aspect <= aspect_round_max else _LINEAR


# ---------------------------------------------------------------------------
# 配置（configs/default.yaml 的 std_eval 节）
# ---------------------------------------------------------------------------


@dataclass
class StdEvalConfig:
    """标准评价配置；默认值即"比标准更严格"口径。"""

    iou_standard: float = 0.1  # §9.2 标准口径
    iou_strict: float = 0.3  # 严格口径（取严分级用）
    weld_form: str = "single"  # single/double（§8.2 单/双面焊，决定重点关注集）
    weld_method: str = "manual"  # manual/auto（§11.1 FRR 分级线不同）
    frr_l1: dict[str, float] = field(default_factory=lambda: {"auto": 0.08, "manual": 0.10})
    # 收紧 FRR 线（默认启用，等值于标准 L2 线，严于 L1 线）
    strict_frr: bool = True
    frr_strict: dict[str, float] = field(default_factory=lambda: {"auto": 0.03, "manual": 0.04})
    aspect_round_max: float = 3.0
    class_to_std: dict[int, int] = field(default_factory=lambda: dict(_MODEL_TO_STD))

    @property
    def focus(self) -> set[int]:
        return FOCUS_DEFECTS[self.weld_form]

    def frr_limit(self, *, strict: bool) -> float:
        """FRR 判定线：strict=True 时取收紧线，否则取标准 L1 线。"""
        if strict or self.strict_frr:
            table = self.frr_strict if self.strict_frr else self.frr_l1
        else:
            table = self.frr_l1
        return table[self.weld_method]


# ---------------------------------------------------------------------------
# §9.2 匹配：GT×预测全局贪心（IOU 降序），每预测至多配一个 GT
# ---------------------------------------------------------------------------

# 逐 GT 判定结果：td / fd / md；fd 附带被误检成的预测类别
Verdict = tuple[str, int | None]  # ("td"| "fd"| "md", pred_std_class or None)


def match_image(
    gts: list[dict[str, Any]],
    preds: list[dict[str, Any]],
    iou_threshold: float,
    cfg: StdEvalConfig | None = None,
) -> tuple[list[Verdict], list[int]]:
    """单张底片匹配。返回 (每个 GT 的判定, 未匹配预测的标准类别列表 FP_extra)。

    gts: [{bbox:[x,y,w,h] 绝对像素, class_id:int}]；preds 另含 score。
    全局贪心：所有 (GT,pred) 对按 IOU 降序消费，IOU≥阈值即配对；
    配对后类型一致=td，不一致=fd；GT 无配对=md。
    """
    cfg = cfg or StdEvalConfig()
    gt_std = [
        _to_std_class(g["class_id"], g["bbox"], cfg.aspect_round_max, cfg.class_to_std)
        for g in gts
    ]
    pred_std = [
        _to_std_class(p["class_id"], p["bbox"], cfg.aspect_round_max, cfg.class_to_std)
        for p in preds
    ]
    pairs: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gts):
        for pi, p in enumerate(preds):
            v = _iou(g["bbox"], p["bbox"])
            if v >= iou_threshold:
                pairs.append((v, gi, pi))
    pairs.sort(key=lambda t: -t[0])
    gt_used: list[Verdict | None] = [None] * len(gts)
    pred_used: set[int] = set()
    gt_matched: set[int] = set()
    for _, gi, pi in pairs:
        if gi in gt_matched or pi in pred_used:
            continue
        gt_used[gi] = ("td", None) if gt_std[gi] == pred_std[pi] else ("fd", pred_std[pi])
        gt_matched.add(gi)
        pred_used.add(pi)
    verdicts: list[Verdict] = []
    for gi, v in enumerate(gt_used):
        verdicts.append(v if v else ("md", None))
    fp_extra = [pred_std[pi] for pi in range(len(preds)) if pi not in pred_used]
    return verdicts, fp_extra


def _iou(a: Iterable[float], b: Iterable[float]) -> float:
    a = list(a)
    b = list(b)
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# §10 指标 + §11.1 分级 + §11.2 风险
# ---------------------------------------------------------------------------


@dataclass
class ClassCounts:
    """单标准类别的 TD/FD/MD 计数（图3 混淆矩阵行）。"""

    td: int = 0
    fd: int = 0
    md: int = 0

    @property
    def total(self) -> int:
        return self.td + self.fd + self.md

    def rates(self) -> dict[str, float]:
        d = self.total
        f = lambda x: round(x / d, 4) if d else 0.0  # noqa: E731
        return {"tdr": f(self.td), "fdr": f(self.fd), "mdr": f(self.md)}


@dataclass
class StdEvalResult:
    """一次标准评价的完整结果（可 JSON 化，附录A 记录表数据源）。"""

    iou_threshold: float
    per_class: dict[int, ClassCounts]
    fr_by_class: dict[int, int]  # 无缺陷底片误报按预测类别分布（图4 行）
    n_no_defect_reported: int  # a
    n_no_defect_clean: int  # b
    fp_extra: int  # 未匹配预测数（比标准多暴露）
    confusion: list[dict[str, Any]]  # 图3/图4 混淆矩阵行
    kdr: float
    wdr: float
    tdr: float
    frr: float
    level: str | None  # "L1".."L4" 或 None=未定级
    risks: dict[str, str]  # {"miss": "Ⅰ类"|..., "false_detect": ..., "false_report": ...}
    fd_pairs: dict[str, int]  # "gt->pred" 误检方向计数（误检风险证据）
    md_focus: int  # 重点关注漏检数（漏检风险证据）

    def to_dict(self) -> dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "per_class": {
                str(n): {"name": STD_CLASS_NAMES[n], **c.__dict__ | c.rates()}
                for n, c in self.per_class.items()
            },
            "fr_by_class": self.fr_by_class,
            "frr": self.frr,
            "no_defect": {"reported_a": self.n_no_defect_reported, "clean_b": self.n_no_defect_clean},
            "fp_extra": self.fp_extra,
            "confusion": self.confusion,
            "kdr": self.kdr,
            "wdr": self.wdr,
            "tdr": self.tdr,
            "level": self.level,
            "risks": self.risks,
            "fd_pairs": self.fd_pairs,
            "md_focus": self.md_focus,
        }


# 系统分级表（§11.1 表1）：KDR/WDR/TDR 下限 + FRR 上限按 weld_method
_LEVEL_TABLE: list[dict[str, Any]] = [
    {"level": "L4", "kdr": 1.00, "kdr_exact": True, "wdr": 1.00, "wdr_exact": True, "tdr": 0.98},
    {"level": "L3", "kdr": 1.00, "kdr_exact": True, "wdr": 0.98, "wdr_exact": False, "tdr": 0.96},
    {"level": "L2", "kdr": 0.98, "kdr_exact": False, "wdr": 0.95, "wdr_exact": False, "tdr": 0.92},
    {"level": "L1", "kdr": 0.95, "kdr_exact": False, "wdr": 0.92, "wdr_exact": False, "tdr": 0.85},
]
# FRR 上限：L1 用宽线（自动 8%/手工 10%），L2~L4 用严线（3%/4%）
_FRR_L2 = {"auto": 0.03, "manual": 0.04}


def grade_level(
    kdr: float, wdr: float, tdr: float, frr: float, cfg: StdEvalConfig
) -> str | None:
    """§11.1 表1 分级：四项指标全部达标的最优级别；全不达标 → None（未定级）。

    strict_frr=True（默认）时 FRR 一律按收紧线（3%/4%）判定，严于标准 L1 线。
    """
    frr_l1 = cfg.frr_limit(strict=False)  # strict_frr=True 时它已是收紧线
    for row in _LEVEL_TABLE:
        frr_limit = frr_l1 if row["level"] == "L1" else _FRR_L2[cfg.weld_method]
        kdr_ok = (kdr == row["kdr"]) if row["kdr_exact"] else kdr >= row["kdr"]
        wdr_ok = (wdr == row["wdr"]) if row["wdr_exact"] else wdr >= row["wdr"]
        if kdr_ok and wdr_ok and tdr >= row["tdr"] and frr <= frr_limit:
            return row["level"]
    return None


def _worse(a: str | None, b: str | None) -> str | None:
    """两个分级取较差者（None 最差；L1 < L2 < L3 < L4）。"""
    if a is None or b is None:
        return None
    return min(a, b)


def evaluate(
    defect_set: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]],
    no_defect_set: list[tuple[str, list[dict[str, Any]]]],
    cfg: StdEvalConfig | None = None,
) -> dict[str, Any]:
    """完整评价（§9~§11）。

    defect_set: [(image_id, gts, preds)]；no_defect_set: [(image_id, preds)]。
    返回 {"standard": StdEvalResult.to_dict(), "strict": ..., "level_recorded": ...}：
    standard=IOU 0.1 口径，strict=严格口径；level_recorded 取两者较差。
    """
    cfg = cfg or StdEvalConfig()
    std_res = _evaluate_at(defect_set, no_defect_set, cfg.iou_standard, cfg, strict=False)
    strict_res = _evaluate_at(defect_set, no_defect_set, cfg.iou_strict, cfg, strict=True)
    # 比标准严：无缺陷测试集缺失 → FRR 未测，不得按 0% 通过分级，整体不定级
    if not no_defect_set:
        std_res["level"] = None
        strict_res["level"] = None
        std_res["frr_measured"] = strict_res["frr_measured"] = False
    else:
        std_res["frr_measured"] = strict_res["frr_measured"] = True
    level_recorded = _worse(std_res["level"], strict_res["level"])
    return {
        "standard": std_res,
        "strict": strict_res,
        "level_recorded": level_recorded,
        "weld_form": cfg.weld_form,
        "weld_method": cfg.weld_method,
    }


def _evaluate_at(
    defect_set: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]],
    no_defect_set: list[tuple[str, list[dict[str, Any]]]],
    iou_thr: float,
    cfg: StdEvalConfig,
    *,
    strict: bool,
) -> dict[str, Any]:
    per: dict[int, ClassCounts] = {n: ClassCounts() for n in STD_CLASS_NAMES}
    fd_pairs: dict[str, int] = {}
    fp_extra_total = 0
    md_focus = 0
    for _, gts, preds in defect_set:
        verdicts, fp_extra = match_image(gts, preds, iou_thr, cfg)
        fp_extra_total += len(fp_extra)
        gt_std = [
            _to_std_class(g["class_id"], g["bbox"], cfg.aspect_round_max, cfg.class_to_std)
            for g in gts
        ]
        for (kind, pred_cls), gt_cls in zip(verdicts, gt_std):
            c = per[gt_cls]
            if kind == "td":
                c.td += 1
            elif kind == "fd":
                c.fd += 1
                key = f"{STD_CLASS_NAMES[gt_cls]}->{STD_CLASS_NAMES[pred_cls]}"
                fd_pairs[key] = fd_pairs.get(key, 0) + 1
            else:
                c.md += 1
                if gt_cls in cfg.focus:
                    md_focus += 1

    fr_by_class: dict[int, int] = {n: 0 for n in STD_CLASS_NAMES}
    a = 0
    for _, preds in no_defect_set:
        if preds:
            a += 1
            for p in preds:
                n = _to_std_class(p["class_id"], p["bbox"], cfg.aspect_round_max, cfg.class_to_std)
                fr_by_class[n] += 1
    b = len(no_defect_set) - a
    fr_total = sum(fr_by_class.values())

    denom = sum(c.total for c in per.values())
    tdr = round(sum(c.td for c in per.values()) / denom, 4) if denom else 0.0
    # WDR/KDR：(TD+FD)/全量 —— 与 TDR 同分母（TD+FD+MD），仅分子不同
    wdr = tdr
    focus_num = focus_den = 0
    for n, c in per.items():
        if n in cfg.focus:
            focus_num += c.td + c.fd
            focus_den += c.total
    kdr = round(focus_num / focus_den, 4) if focus_den else 0.0
    frr = round(a / (a + b), 4) if (a + b) else 0.0

    confusion = [
        {
            "class": n,
            "name": STD_CLASS_NAMES[n],
            **c.__dict__,
            "focus": n in cfg.focus,
            "fr": fr_by_class[n],
        }
        for n, c in per.items()
    ]

    level = grade_level(kdr, wdr, tdr, frr, cfg)
    risks = {
        "miss": _miss_risk(per, cfg),
        "false_detect": _false_detect_risk(fd_pairs, cfg),
        "false_report": "Ⅰ类" if frr > cfg.frr_limit(strict=strict) else "Ⅱ类",
    }
    res = StdEvalResult(
        iou_threshold=iou_thr,
        per_class=per,
        fr_by_class=fr_by_class,
        n_no_defect_reported=a,
        n_no_defect_clean=b,
        fp_extra=fp_extra_total,
        confusion=confusion,
        kdr=kdr,
        wdr=wdr,
        tdr=tdr,
        frr=frr,
        level=level,
        risks=risks,
        fd_pairs=fd_pairs,
        md_focus=md_focus,
    )
    return res.to_dict()


# ---------------------------------------------------------------------------
# §11.2 风险分析
# ---------------------------------------------------------------------------


def _miss_risk(per: dict[int, ClassCounts], cfg: StdEvalConfig) -> str:
    """漏检风险（表2）：Ⅰ类=重点关注漏检，或一般关注漏检（保守按评定≥Ⅱ级）；
    Ⅱ类=仅评定为Ⅰ级的圆形缺陷漏检。无逐缺陷评级数据时保守归Ⅰ类（取严）。"""
    if any(per[n].md for n in cfg.focus):
        return "Ⅰ类"
    if any(per[n].md for n in (1, 2)):
        # 一般关注漏检：默认保守按"评定Ⅱ级及以上"→Ⅰ类；如确认全为Ⅰ级
        # （需逐缺陷评级证据传入），可降为Ⅱ类。
        return "Ⅰ类"
    return "Ⅱ类"


def _false_detect_risk(fd_pairs: dict[str, int], cfg: StdEvalConfig) -> str:
    """误检风险（表3）：Ⅰ类=重点关注误检为一般关注；Ⅱ类=一般关注误检为重点关注。

    fd_pairs key 形如 "裂纹->圆形缺陷"（GT 类别->被误检成的预测类别）。
    """
    name_to_n = {v: k for k, v in STD_CLASS_NAMES.items()}
    heavy_to_light = False
    light_to_heavy = False
    for key, cnt in fd_pairs.items():
        if not cnt:
            continue
        g_name, p_name = key.split("->")
        g, p = name_to_n[g_name], name_to_n[p_name]
        if _is_focus(g, cfg) and not _is_focus(p, cfg):
            heavy_to_light = True
        elif not _is_focus(g, cfg) and _is_focus(p, cfg):
            light_to_heavy = True
    if heavy_to_light:
        return "Ⅰ类"
    if light_to_heavy:
        return "Ⅱ类"
    return "Ⅱ类"


def _is_focus(n: int, cfg: StdEvalConfig) -> bool:
    return n in cfg.focus
