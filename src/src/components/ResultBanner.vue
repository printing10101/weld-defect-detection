<script setup lang="ts">
/**
 * 报告结论横幅（设计稿：结论先行）。
 * 数据诚实性：级别/待复核/可评片/缺陷数全部来自后端 ReportOut 真实字段；
 * 文案为基于真实状态的规则化解读，不构造任何数值。
 */
import { computed } from "vue";
import { stampBadge } from "../utils/filmStamp";
import type { ReportOut } from "../types/api";

const props = defineProps<{ result: ReportOut }>();

type Tone = "ok" | "review" | "fail";
const tone = computed<Tone>(() => {
  if (!props.result.evaluable) return "fail";
  if (props.result.need_review) return "review";
  return "ok";
});

/** 底片印字（扫描日期/编号）结论行；无识别数据（重新生成模式/旧版本）不显示。 */
const stamp = computed(() => stampBadge(props.result.stamp ?? null));

const levelText = computed<string>(() => {
  if (!props.result.evaluable) return "不可评片";
  if (props.result.need_review && !props.result.joint_level) return "待复核";
  return props.result.joint_level ?? "待复核";
});

const conclusion = computed<string>(() => {
  const count = props.result.defect_count;
  if (tone.value === "fail") {
    return "底片质量未达标准要求（像质计灵敏度/黑度 D 校验未通过），系统按保守原则未输出级别结论。";
  }
  const head =
    props.result.need_review && props.result.joint_level
      ? `综合评定 ${props.result.joint_level} 级，本报告须经人工复核确认。`
      : props.result.joint_level
        ? `综合评定 ${props.result.joint_level} 级。`
        : "本报告待人工复核，暂无自动评级结论。";
  const defects = count > 0 ? `共检出 ${count} 处缺陷，明细见 PDF/A 报告。` : "未检出缺陷。";
  return head + defects;
});

const reason = computed<string>(() => {
  if (tone.value === "fail") return "可能原因：像质计（IQI）丝号未达要求，或黑度 D 超出 AB 级规定范围。可调整曝光参数或补加像质计后重新透照；必要时转人工评片。";
  if (tone.value === "review") return "可能原因：标准限值未获授权（tables.authorized=false），或双人对评结论存在分歧，按 §12.2 升级至仲裁流程。此为「宁保守、不误放行」的保守判定策略，并非系统误报。";
  return "像质计灵敏度与黑度 D 校验均通过；如需归档，请导出 PDF/A 检测报告。";
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
      <span
        v-if="stamp"
        class="stamp-line"
        :class="`stamp-${stamp.cls}`"
        :title="stamp.title"
      >底片印字：{{ stamp.label }}</span>
      <span class="why">{{ reason }}</span>
    </div>
  </div>
</template>

<style scoped>
.stamp-line {
  display: inline-block;
  font-size: 11px;
  padding: 1px 8px;
  border-radius: 10px;
  white-space: nowrap;
}
.stamp-line.stamp-ok {
  background: rgba(255, 255, 255, 0.85);
  color: #1e7a3d;
}
.stamp-line.stamp-warn {
  background: rgba(255, 235, 235, 0.92);
  color: #b03030;
  font-weight: 600;
}
.stamp-line.stamp-muted {
  background: rgba(255, 255, 255, 0.75);
  color: #6a7b99;
}
</style>
