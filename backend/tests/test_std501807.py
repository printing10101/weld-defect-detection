"""DB50/T 1807-2025 评价体系单测：公式精确性 / 分级 / 风险 / 双阈值取严。"""

from __future__ import annotations

from backend.evaluation.std501807 import (
    STD_CLASS_NAMES,
    StdEvalConfig,
    evaluate,
    grade_level,
    match_image,
    _to_std_class,
)


def _gt(class_id: int, bbox: list[float]) -> dict:
    return {"bbox": bbox, "class_id": class_id}


def _pred(class_id: int, bbox: list[float], score: float = 0.9) -> dict:
    return {"bbox": bbox, "class_id": class_id, "score": score}


CRACK, LOF, IP, POR, SLAG, UNDERCUT, CONCAV = 4, 3, 2, 0, 1, 5, 6


# ---------------------------------------------------------------------------
# 匹配：正检 / 误检 / 漏检
# ---------------------------------------------------------------------------


def test_td_same_type():
    verdicts, fp = match_image([_gt(CRACK, [10, 10, 20, 20])], [_pred(CRACK, [10, 10, 20, 20])], 0.1)
    assert verdicts == [("td", None)]
    assert fp == []


def test_fd_type_mismatch():
    # 位置对（IOU=1）但类型错：裂纹 GT 被检成未熔合 → 误检
    verdicts, fp = match_image([_gt(CRACK, [10, 10, 20, 20])], [_pred(LOF, [10, 10, 20, 20])], 0.1)
    assert verdicts == [("fd", 4)]  # 预测被映射为标准类 4（未熔合）


def test_md_iou_below_threshold():
    # IOU=0 → 漏检
    verdicts, fp = match_image([_gt(CRACK, [0, 0, 10, 10])], [_pred(CRACK, [100, 100, 10, 10])], 0.1)
    assert verdicts == [("md", None)]
    assert fp == [3]


def test_unmatched_pred_is_fp_extra():
    # GT 被匹配，多余预测单列为 FP_extra（不混入标准指标）
    gts = [_gt(CRACK, [0, 0, 10, 10])]
    preds = [_pred(CRACK, [0, 0, 10, 10]), _pred(CRACK, [50, 50, 5, 5])]
    verdicts, fp = match_image(gts, preds, 0.1)
    assert verdicts == [("td", None)]
    assert fp == [3]


def test_greedy_one_pred_one_gt():
    # 两个预测都覆盖同一 GT：最高 IOU 者配对，另一个为 FP_extra
    gts = [_gt(CRACK, [0, 0, 10, 10])]
    preds = [_pred(CRACK, [0, 0, 10, 10], 0.8), _pred(CRACK, [1, 1, 10, 10], 0.9)]
    verdicts, fp = match_image(gts, preds, 0.1)
    assert verdicts == [("td", None)]
    assert len(fp) == 1


def test_slag_shape_split_round_vs_linear():
    # 夹渣按形状分流：L/W>3 → 条形缺陷(2)，否则圆形缺陷(1)
    assert _to_std_class(SLAG, [0, 0, 30, 5]) == 2
    assert _to_std_class(SLAG, [0, 0, 6, 6]) == 1
    assert _to_std_class(POR, [0, 0, 8, 8]) == 1
    assert _to_std_class(CONCAV, [0, 0, 10, 4]) == 6


# ---------------------------------------------------------------------------
# 指标公式（手算对照）
# ---------------------------------------------------------------------------


def _fixture_eval(cfg: StdEvalConfig) -> dict:
    """固定场景：裂纹 3 个(2正检+1误检)，未熔合 2 个(1正检+1漏检)，气孔 1 个(漏检)，
    无缺陷底片 3 张其中 1 张误报一条未焊通预测。"""
    defect_set = [
        (
            "img1",
            [
                _gt(CRACK, [10, 10, 20, 20]),
                _gt(CRACK, [40, 10, 20, 20]),
                _gt(CRACK, [70, 10, 20, 20]),
            ],
            [
                _pred(CRACK, [10, 10, 20, 20]),
                _pred(CRACK, [40, 10, 20, 20]),
                _pred(LOF, [70, 10, 20, 20]),
            ],
        ),
        (
            "img2",
            [_gt(LOF, [10, 10, 20, 20]), _gt(LOF, [40, 10, 20, 20])],
            [_pred(LOF, [10, 10, 20, 20])],
        ),
        ("img3", [_gt(POR, [10, 10, 10, 10])], []),
    ]
    no_defect_set = [("clean1", []), ("clean2", []), ("clean3", [_pred(IP, [5, 5, 8, 8])])]
    return evaluate(defect_set, no_defect_set, cfg)


def test_per_class_formulas():
    cfg = StdEvalConfig()
    res = _fixture_eval(cfg)
    pc = res["standard"]["per_class"]
    # 裂纹 n=3：TD=2, FD=1, MD=0 → TDR3=2/3, FDR3=1/3, MDR3=0
    assert pc["3"]["td"] == 2 and pc["3"]["fd"] == 1 and pc["3"]["md"] == 0
    assert pc["3"]["tdr"] == 0.6667 and pc["3"]["fdr"] == 0.3333 and pc["3"]["mdr"] == 0.0
    # 未熔合 n=4：TD=1, FD=0, MD=1 → TDR=MDR=0.5
    assert pc["4"]["td"] == 1 and pc["4"]["md"] == 1
    assert pc["4"]["tdr"] == 0.5 and pc["4"]["mdr"] == 0.5
    # 气孔 n=1：全漏检
    assert pc["1"]["md"] == 1 and pc["1"]["mdr"] == 1.0


def test_composite_metrics():
    res = _fixture_eval(StdEvalConfig())
    std = res["standard"]
    # TDR = 3/6 = 0.5；WDR 与 TDR 同分母同分子结构 → 0.5
    assert std["tdr"] == 0.5 and std["wdr"] == 0.5
    # KDR（单面焊重点关注 3..7）：(TD3+FD3+TD4)/(3+2) = 4/5
    assert std["kdr"] == 0.8
    # FRR = 1/3
    assert std["frr"] == 0.3333


def test_kdr_double_form_excludes_undercut():
    # 双面焊重点关注只有 3/4/5：咬边全漏检不影响 KDR（仍影响漏检风险）
    cfg = StdEvalConfig(weld_form="double")
    defect_set = [("i", [_gt(UNDERCUT, [0, 0, 10, 4]), _gt(CRACK, [20, 0, 10, 10])], [_pred(CRACK, [20, 0, 10, 10])])]
    res = evaluate(defect_set, [], cfg)
    std = res["standard"]
    assert std["kdr"] == 1.0  # 重点关注全正检
    assert std["per_class"]["7"]["md"] == 1


def test_frr_by_class_distribution():
    res = _fixture_eval(StdEvalConfig())
    fr = res["standard"]["fr_by_class"]
    assert fr == {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0, 7: 0}


def test_fp_extra_exposed():
    res = _fixture_eval(StdEvalConfig())
    # img1/img2 的预测都被匹配，无多余；无额外预测 → 0
    assert res["standard"]["fp_extra"] == 0


# ---------------------------------------------------------------------------
# 分级
# ---------------------------------------------------------------------------


def test_grade_l4():
    # L4：KDR=100%、WDR=100%、TDR≥98%、FRR≤3%(自动焊)
    assert grade_level(1.0, 1.0, 0.99, 0.02, StdEvalConfig(weld_method="auto")) == "L4"
    # 手工焊 FRR 线 4%
    assert grade_level(1.0, 1.0, 0.99, 0.035, StdEvalConfig(weld_method="manual")) == "L4"


def test_grade_kdr_must_be_exact_100_for_l3():
    # KDR=99.9% 差一点点：L3/L4 要求精确 100% → 落到 L2（KDR≥98%）
    assert grade_level(0.999, 1.0, 0.99, 0.0, StdEvalConfig()) == "L2"
    assert grade_level(1.0, 0.98, 0.96, 0.0, StdEvalConfig()) == "L3"


def test_grade_frr_stricter_line_blocks_l1():
    # FRR=5%：标准 L1 线(手工10%)可通过，但收紧线(4%)不通过 → 未定级
    cfg = StdEvalConfig(weld_method="manual", strict_frr=True)
    assert grade_level(0.96, 1.0, 0.97, 0.05, cfg) is None
    cfg_lenient = StdEvalConfig(weld_method="manual", strict_frr=False)
    assert grade_level(0.96, 1.0, 0.97, 0.05, cfg_lenient) == "L1"


def test_level_recorded_takes_stricter():
    # 严格口径（IOU 0.3）下部分框退化为漏检 → 分级变差，记录取较差者
    cfg = StdEvalConfig(iou_strict=0.9)
    defect_set = [("i", [_gt(CRACK, [0, 0, 10, 10])], [_pred(CRACK, [2, 0, 10, 10])])]
    res = evaluate(defect_set, [("c", [])], cfg)
    assert res["standard"]["level"] in ("L4", "L3", "L2", "L1")
    assert res["strict"]["per_class"]["3"]["md"] == 1
    assert res["level_recorded"] is None or res["level_recorded"] != res["standard"]["level"]


def test_no_clean_set_means_frr_unmeasured_no_level():
    # 无缺陷测试集缺失 → FRR 未测，即便缺陷集全对也不给分级
    defect_set = [("i", [_gt(CRACK, [0, 0, 10, 10])], [_pred(CRACK, [0, 0, 10, 10])])]
    res = evaluate(defect_set, [], StdEvalConfig())
    assert res["standard"]["frr"] == 0.0
    assert res["standard"]["frr_measured"] is False
    assert res["level_recorded"] is None


# ---------------------------------------------------------------------------
# 风险
# ---------------------------------------------------------------------------


def test_miss_risk_focus_is_class1():
    cfg = StdEvalConfig()
    defect_set = [("i", [_gt(CRACK, [0, 0, 10, 10])], [])]
    res = evaluate(defect_set, [], cfg)
    assert res["standard"]["risks"]["miss"] == "Ⅰ类"
    assert res["standard"]["md_focus"] == 1


def test_miss_risk_general_conservative_class1():
    # 一般关注（气孔）漏检：无逐缺陷评级证据 → 保守归Ⅰ类
    cfg = StdEvalConfig()
    res = evaluate([("i", [_gt(POR, [0, 0, 5, 5])], [])], [], cfg)
    assert res["standard"]["risks"]["miss"] == "Ⅰ类"


def test_false_detect_risk_heavy_to_light():
    # 重点关注（裂纹）被误检为一般关注（圆形）→ Ⅰ类
    defect_set = [("i", [_gt(CRACK, [0, 0, 10, 10])], [_pred(POR, [0, 0, 10, 10])])]
    res = evaluate(defect_set, [], StdEvalConfig())
    assert res["standard"]["risks"]["false_detect"] == "Ⅰ类"


def test_false_detect_risk_light_to_heavy():
    # 一般关注（气孔）被误检为重点关注（裂纹）→ Ⅱ类
    defect_set = [("i", [_gt(POR, [0, 0, 10, 10])], [_pred(CRACK, [0, 0, 10, 10])])]
    res = evaluate(defect_set, [], StdEvalConfig())
    assert res["standard"]["risks"]["false_detect"] == "Ⅱ类"


def test_false_report_risk_with_strict_line():
    # FRR=5%：收紧线(手工4%)下为Ⅰ类；关闭收紧后为Ⅱ类
    no_defect = [("c1", [_pred(IP, [0, 0, 5, 5])]), ("c2", [])] + [(f"c{i}", []) for i in range(18)]
    res = evaluate([], no_defect, StdEvalConfig(weld_method="manual", strict_frr=True))
    assert res["standard"]["frr"] == 0.05
    assert res["standard"]["risks"]["false_report"] == "Ⅰ类"
    res2 = evaluate([], no_defect, StdEvalConfig(weld_method="manual", strict_frr=False))
    assert res2["standard"]["risks"]["false_report"] == "Ⅱ类"


def test_confusion_matrix_structure():
    res = _fixture_eval(StdEvalConfig())
    rows = res["standard"]["confusion"]
    assert len(rows) == 7
    row3 = next(r for r in rows if r["class"] == 3)
    assert row3["name"] == "裂纹" and row3["td"] == 2 and row3["fd"] == 1
    assert row3["focus"] is True
    row5 = next(r for r in rows if r["class"] == 5)
    assert row5["fr"] == 1  # 无缺陷集误报行（图4）


def test_std_class_names_complete():
    assert len(STD_CLASS_NAMES) == 7
    assert STD_CLASS_NAMES[6] == "内凹"
