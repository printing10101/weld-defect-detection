<script setup lang="ts">
/**
 * 报告结论横幅（DESIGN.md：结论先行）。
 * 数据诚实性：级别/待复核/可评片/缺陷数全部来自后端 ReportOut 真实字段；
 * 文案为基于真实状态的规则化解读，不构造任何数值。
 */
import { computed } from "vue";
import type { ReportOut } from "../types/api";

const props = defineProps<{ result: ReportOut }>();

type Tone = "ok" | "review" | "fail";
const tone = computed<Tone>(() => {
  if (!props.result.evaluable) return "fail";
  if (props.result.need_review) return "review";
  return "ok";
});

const levelText = computed<string>(() => {
  if (!props.result.evaluable) return "不可评片";
  if (props.result.need_review && !props.result.joint_level) return "待复核";
  return props.result.joint_level ?? "待复核";
});

const conclusion = computed<string>(() => {
  const count = props.result.defect_count;
  if (tone.value === "fail") {
    return "影像质量未达标准（IQI/黑度校验未通过），系统按保守原则未输出评级结论。";
  }
  const head =
    props.result.need_review && props.result.joint_level
      ? `综合评定 ${props.result.joint_level} 级，但本报告需人工复核。`
      : props.result.joint_level
        ? `综合评定 ${props.result.joint_level} 级。`
        : "本报告需人工复核，暂无自动评级结论。";
  const defects = count > 0 ? `检出 ${count} 处缺陷，明细见 PDF 报告。` : "未检出缺陷。";
  return head + defects;
});

const reason = computed<string>(() => {
  if (tone.value === "fail") return "可能原因：像质计丝号不达标或黑度超出 AB 级范围。可调整曝光/增补像质计后重拍；或转人工评片。";
  if (tone.value === "review") return "可能原因：标准数值未授权（tables.authorized=false）或双人评片出现分歧，按 §12.2 升级仲裁。这并非误报，而是「宁保守不误放行」。";
  return "已通过 IQI/黑度校验；如需归档请导出 PDF/A 报告。";
});
</script>

<template>
  <div
    class="banner"
    :class="tone"
  >
    <div class="lv">
      {{ levelText }}
    </div>
    <div class="con">
      <b>{{ conclusion }}</b>
      <span class="why">{{ reason }}</span>
    </div>
  </div>
</template>
