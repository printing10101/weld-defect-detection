"""评片结论（AI 撰述）的事实装配、提示词构造与输出清理。

**合规红线（本模块的存在理由）**：缺陷等级与合格/不合格判定是持证人/责任工程师
的法定职责，不得由大模型给出。因此提示词与输出清理共同保证：

1. 送进模型的是**已由算法确定的**事实（检出统计、几何量、质量门禁结论、评级
   basis、处置建议），大模型只做"把事实写成评片员可读的结论"；
2. 提示词显式禁止模型输出级别（I/II/III/IV）、禁止下合格/不合格结论、禁止
   编造未提供的数值；
3. ``clean_output`` 兜底：清理 markdown 围栏/多余空白，并**删除模型越权生成的
   级别判定语句行**——提示词是软约束，清理是硬约束（防提示注入与模型失控）。

本模块为纯函数（无 I/O、无第三方依赖），便于单测；HTTP 调用在
``infra/llm_client.py``，编排在 ``app/llm_narrative.py``。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 输出长度上限（字符）：本地 4B 模型偶发复读，硬截断防前端渲染失控。
_MAX_OUTPUT_CHARS = 1200

# 越权语句识别：出现"级别 + 结论"字样的整行一律剔除。
# 覆盖 "评为II级" / "判定为不合格" / "级别：III" / "结论：合格" 等常见形态。
_FORBIDDEN_LINE_PATTERNS = (
    re.compile(
        r"(?:评为|定为|判定为|判定|评定为|评定|评级为|评级|级别为|级别[:：]|等级[:：])"
        r"[^。；\n]{0,12}(?:I{1,3}V?|IV|合格|不合格)"
    ),
    re.compile(r"(?:结论|综合结论|验收结论)[:：][^。；\n]{0,20}(?:合格|不合格)"),
    re.compile(r"(?:合格|不合格)\s*(?:的)?\s*(?:结论|判定|评定)"),
)

# 缺陷类别中文名（键=DefectClass.value）。与 domain/grade/nb47013.py 的展示
# 口径一致；此处独立一份避免在 infra/app 侧 import 判定实现（import-linter
# 禁止 infra 依赖 domain.grade）。
CLASS_ZH: dict[int, str] = {
    0: "气孔",
    1: "夹渣",
    2: "未焊透",
    3: "未熔合",
    4: "裂纹",
    5: "咬边",
    6: "内凹",
}

# 零容忍类（NB/T47013.2-2015：I-III 级均不允许）
ZERO_TOLERANCE_IDS: frozenset[int] = frozenset({4, 3, 2})  # 裂纹 / 未熔合 / 未焊透


def class_name_zh(class_id: int) -> str:
    """缺陷类别中文名（未知 id 回退 ``类别<id>``，不抛错）。"""
    return CLASS_ZH.get(int(class_id), f"类别{int(class_id)}")


_SYSTEM_PROMPT = """你是承压设备射线检测（RT）的评片辅助助手，服务于持证评片员。

你的唯一职责：把**已经由检测算法与标准判定引擎确定的事实**整理成一段简洁、
专业、可读的评片结论说明，帮助评片员快速掌握本张底片的情况。

必须严格遵守（违反即为不合格输出）：
1. 禁止给出缺陷等级（I 级 / II 级 / III 级 / IV 级），禁止给出"合格/不合格"结论，
   禁止给出验收判定。等级与合格判定只能由持证人依标准原文人工作出。
2. 禁止编造任何未在"事实"中给出的数值、缺陷、部位或结论。缺信息就说不确定。
3. 只使用"事实"中的内容；不确定的一律表述为"需人工确认"。
4. 不输出 markdown 标题、表格、代码块或 emoji；用简洁的中文短段落或分号句。

请按以下结构输出（不加小标题编号之外的修饰）：
1. 检测概况：检出缺陷的类别与数量、是否存在零容忍类缺陷、几何量是否已标定。
2. 复核要点：需要人工重点确认的事项（含底片质量门禁告警、复核灰区、印字区屏蔽等）。
3. 建议动作：接下来该做什么（如人工复核、重新扫描、补测厚度等），仅限流程性建议。
"""


@dataclass(frozen=True)
class NarrativeFacts:
    """评片结论的输入事实（全部来自算法输出，不含模型推断）。"""

    defect_count: int = 0
    class_stats: tuple[tuple[str, int], ...] = ()  # (中文缺陷名, 数量)
    largest: tuple[tuple[str, float, float], ...] = ()  # (中文缺陷名, 长mm, 宽mm)
    zero_tolerance: tuple[str, ...] = ()  # 命中的零容忍缺陷名
    joint_level: str | None = None
    grade_basis: tuple[str, ...] = ()  # 评级依据/熔断原因（算法给出）
    quality_warnings: tuple[str, ...] = ()  # 质量门禁告警（含翻拍/位深降级）
    disposition_label: str | None = None  # 处置建议标签（算法给出）
    disposition_actions: tuple[str, ...] = ()
    need_review: bool = False
    calibrated: bool = False  # 像素标定是否可信（未标定则无物理尺寸）
    base_metal_thickness_mm: float | None = None
    standard_id: str = "NB/T47013.2-2015"
    extra: dict[str, str] = field(default_factory=dict)


def _join(items: tuple[str, ...], empty: str = "（无）") -> str:
    return "；".join(items) if items else empty


def build_facts_block(facts: NarrativeFacts) -> str:
    """把事实装配为给模型看的纯文本（只列事实，不下结论）。"""
    lines: list[str] = []
    lines.append(f"标准体系：{facts.standard_id}")
    lines.append(f"检出缺陷总数：{facts.defect_count}")
    if facts.class_stats:
        lines.append(
            "缺陷类别与数量：" + "；".join(f"{name} {n} 个" for name, n in facts.class_stats)
        )
    else:
        lines.append("缺陷类别与数量：未检出任何缺陷")
    lines.append("零容忍类缺陷（裂纹/未熔合/未焊透）命中：" + (_join(facts.zero_tolerance, "无")))
    if facts.zero_tolerance:
        lines.append("说明：零容忍类缺陷存在时，按标准 I-III 级均不允许。")
    if facts.calibrated and facts.largest:
        lines.append(
            "最大缺陷尺寸（已标定）："
            + "；".join(
                f"{name} 长 {length:.2f}mm 宽 {width:.2f}mm"
                for name, length, width in facts.largest
            )
        )
    else:
        lines.append("最大缺陷尺寸：像素标定不可信，无物理尺寸（仅形状比可用）")
    lines.append(
        f"母材厚度 T：{facts.base_metal_thickness_mm if facts.base_metal_thickness_mm else '未提供'}"
    )
    lines.append(f"标准判定引擎输出级别：{facts.joint_level or '未输出级别（熔断/不可评）'}")
    lines.append("判定依据/熔断原因：" + _join(facts.grade_basis))
    lines.append("底片质量与门禁告警：" + _join(facts.quality_warnings))
    lines.append(f"系统处置建议：{facts.disposition_label or '未给出'}")
    if facts.disposition_actions:
        lines.append("建议动作（算法给出）：" + "；".join(facts.disposition_actions))
    lines.append(f"是否需要人工复核：{'是' if facts.need_review else '否'}")
    for key, value in facts.extra.items():
        lines.append(f"{key}：{value}")
    return "\n".join(lines)


def build_messages(facts: NarrativeFacts) -> tuple[str, str]:
    """返回 ``(system, user)`` 两条消息。"""
    user = (
        "以下是一张射线底片的**算法事实**（全部由检测与判定程序产出，未经人工确认）：\n\n"
        f"{build_facts_block(facts)}\n\n"
        "请按职责与结构要求撰写评片结论说明。再次强调：不要给出缺陷等级，"
        "不要给出合格/不合格结论，不要编造未提供的数值。"
    )
    return _SYSTEM_PROMPT, user


def clean_output(text: str) -> str:
    """清理模型输出：去 markdown 围栏、剔除越权级别/合格判定行、限长。

    提示词是软约束，本函数是硬约束——即便模型无视指令输出"评为 II 级"，
    也会在落到报告/API 之前被删除，保证合规红线不依赖模型自觉。
    """
    if not text:
        return ""
    cleaned = text.strip()
    # 去 ```...``` 围栏
    cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    kept: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip().lstrip("-*•").strip()
        if not line:
            continue
        if any(p.search(line) for p in _FORBIDDEN_LINE_PATTERNS):
            # 越权行整体丢弃（不保留残句，避免半句话被误读为结论）
            continue
        kept.append(line)
    out = "\n".join(kept).strip()
    if len(out) > _MAX_OUTPUT_CHARS:
        out = out[:_MAX_OUTPUT_CHARS].rstrip() + "…"
    return out


__all__ = [
    "CLASS_ZH",
    "ZERO_TOLERANCE_IDS",
    "NarrativeFacts",
    "build_facts_block",
    "build_messages",
    "class_name_zh",
    "clean_output",
]
