"""评片结论单测：合规红线（不得输出级别/合格判定）、提示词、降级路径。

本项目最关键的一条合规约束是"缺陷等级只能由持证人给出"。本测试把这条约束
钉死在两处：提示词必须显式禁止，且 ``clean_output`` 必须能**删除**模型越权
生成的级别/合格判定语句（软约束 + 硬约束）。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.app.llm_narrative import (
    STATUS_DISABLED,
    STATUS_UNAVAILABLE,
    NarrativeOutcome,
    facts_from_report,
    generate,
)
from backend.domain.narrative import (
    NarrativeFacts,
    build_facts_block,
    build_messages,
    class_name_zh,
    clean_output,
)


class TestClassNames:
    def test_known_and_unknown(self) -> None:
        assert class_name_zh(0) == "气孔"
        assert class_name_zh(4) == "裂纹"
        assert class_name_zh(99) == "类别99"  # 未知不抛错


class TestPrompt:
    def test_system_prompt_forbids_level_output(self) -> None:
        system, user = build_messages(NarrativeFacts(defect_count=3))
        assert "禁止给出缺陷等级" in system
        assert "禁止给出" in system and "合格/不合格" in system
        # 用户消息里也要重申（长上下文下系统提示易被稀释）
        assert "不要给出缺陷等级" in user

    def test_facts_block_carries_algorithm_outputs(self) -> None:
        facts = NarrativeFacts(
            defect_count=2,
            class_stats=(("气孔", 2),),
            zero_tolerance=("裂纹",),
            grade_basis=("零容忍直判",),
            quality_warnings=("黑度未达标",),
            calibrated=True,
            largest=(("气孔", 3.5, 2.1),),
            base_metal_thickness_mm=12.0,
            standard_id="NB/T47013.2-2015",
        )
        block = build_facts_block(facts)
        assert "气孔 2 个" in block
        assert "裂纹" in block  # 零容忍命中如实透出
        assert "长 3.50mm 宽 2.10mm" in block
        assert "黑度未达标" in block

    def test_facts_block_marks_uncalibrated(self) -> None:
        block = build_facts_block(NarrativeFacts(defect_count=1, calibrated=False))
        assert "像素标定不可信" in block


class TestCleanOutput:
    def test_strips_code_fence(self) -> None:
        assert clean_output("```text\n结论\n```") == "结论"

    @pytest.mark.parametrize(
        "line",
        [
            "该焊缝评为III级",
            "综合判定为不合格",
            "级别：IV",
            "结论：合格",
            "评定该接头为不合格",
        ],
    )
    def test_removes_out_of_scope_conclusions(self, line: str) -> None:
        """模型越权给出级别/合格判定 → 整行删除（合规红线不依赖模型自觉）。"""
        text = f"检测概况：检出气孔 3 个。\n{line}\n复核要点：需人工确认。"
        out = clean_output(text)
        assert line not in out
        assert "检测概况" in out
        assert "复核要点" in out

    def test_keeps_legitimate_text(self) -> None:
        text = (
            "检测概况：检出气孔 12 个，未检出零容忍类缺陷。\n"
            "复核要点：底片黑度超出标准范围，需人工复核。\n"
            "建议动作：重新扫描底片。"
        )
        assert clean_output(text) == text

    def test_empty_and_truncation(self) -> None:
        assert clean_output("") == ""
        assert clean_output("   ") == ""
        long_text = "字" * 2000
        out = clean_output(long_text)
        assert len(out) <= 1201
        assert out.endswith("…")


def _reg_with(report: dict | None, image: dict | None, *, enabled: bool = True):
    """构造仅含 repository 与 config 的假 Registry（facts 装配用）。"""
    repo = SimpleNamespace(
        get_report=lambda _rid: report,
        get_image=lambda _iid: image,
    )
    llm_cfg = SimpleNamespace(enabled=enabled, external_endpoint="", api_key_env="")
    return SimpleNamespace(
        repository=repo,
        config=SimpleNamespace(llm=llm_cfg),
        llm_manager=None,
    )


class TestFactsFromReport:
    def test_missing_report_returns_none(self) -> None:
        assert facts_from_report(_reg_with(None, None), "r1") is None

    def test_missing_image_returns_none(self) -> None:
        assert facts_from_report(_reg_with({"image_id": "i1"}, None), "r1") is None

    def test_assembles_defect_stats_and_warnings(self) -> None:
        report = {"image_id": "i1", "joint_level": None, "basis": ["熔断原因"]}
        image = {
            "defects": [
                {"class_id": 0, "length_mm": 2.0, "width_mm": 1.0},
                {"class_id": 0, "length_mm": 5.0, "width_mm": 3.0},
                {"class_id": 4, "length_mm": 1.0, "width_mm": 0.5},
            ],
            "pixel_spacing_mm": 0.1,
            "base_metal_thickness_mm": 12.0,
            "evaluable": False,
            "density_ok": False,
            "density": 1.05,
            "iqi_pass": False,
            "quality_pass": True,
            "need_review": True,
            "standard_id": "NB/T47013.2-2015",
        }
        facts = facts_from_report(_reg_with(report, image), "r1")
        assert facts is not None
        assert facts.defect_count == 3
        assert ("气孔", 2) in facts.class_stats
        assert "裂纹" in facts.zero_tolerance  # class 4 属零容忍
        assert any("黑度" in w for w in facts.quality_warnings)
        assert any("像质计" in w for w in facts.quality_warnings)
        # 最大缺陷按长度取（气孔 5.0 而非 2.0）
        assert facts.largest[0] == ("气孔", 5.0, 3.0)
        assert facts.calibrated is True


class TestGenerate:
    def test_disabled_when_config_off(self) -> None:
        outcome = generate(_reg_with({}, {}, enabled=False), NarrativeFacts())
        assert outcome.status == STATUS_DISABLED
        assert outcome.text == ""

    def test_unavailable_when_no_endpoint(self) -> None:
        outcome = generate(_reg_with({}, {}, enabled=True), NarrativeFacts())
        assert outcome.status == STATUS_UNAVAILABLE
        assert outcome.reason

    def test_unavailable_when_endpoint_dead(self) -> None:
        """端点配了但没服务 → unavailable + 可读原因（不抛错）。"""
        reg = _reg_with({}, {}, enabled=True)
        reg.config.llm.external_endpoint = "http://127.0.0.1:9"
        outcome = generate(reg, NarrativeFacts())
        assert outcome.status == STATUS_UNAVAILABLE
        assert "不可用" in outcome.reason

    def test_disclaimer_never_grants_authority(self) -> None:
        """无论何种状态，附带的免责声明都必须否定合规效力。"""
        for outcome in (
            generate(_reg_with({}, {}, enabled=False), NarrativeFacts()),
            NarrativeOutcome(status="ok", text="x"),
        ):
            assert "不构成" in outcome.disclaimer
