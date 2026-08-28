<script setup lang="ts">
/**
 * 合规处置建议面板：展示后端 ReportOut 的处置徽标 / 动作清单 / 免责声明。
 * 数据诚实性：徽标与动作全部来自后端 recommend 引擎真实输出，前端不做规则推导。
 */
import { computed } from "vue";
import type { ReportOut } from "../types/api";

const props = defineProps<{ result: ReportOut }>();

const tone = computed<string>(() => {
  switch (props.result.disposition) {
    case "accept":
      return "accept";
    case "conditional":
      return "conditional";
    case "rework":
      return "rework";
    default:
      return "recheck";
  }
});
</script>

<template>
  <div
    v-if="result.disposition"
    class="disp"
    :class="tone"
  >
    <div class="badge">
      {{ result.disposition_label ?? result.disposition }}
    </div>
    <ul class="actions">
      <li
        v-for="(a, i) in result.disposition_actions"
        :key="i"
      >
        {{ a }}
      </li>
    </ul>
    <div
      v-if="result.disclaimer"
      class="disc"
    >
      {{ result.disclaimer }}
    </div>
  </div>
</template>

<style scoped>
.disp {
  margin: 14px 0;
  border: 1px solid var(--border, #333);
  border-radius: 8px;
  padding: 12px 14px;
}
.badge {
  display: inline-block;
  font-weight: 700;
  font-size: 13px;
  padding: 3px 10px;
  border-radius: 999px;
  color: #fff;
}
.actions {
  margin: 10px 0 0;
  padding-left: 18px;
  font-size: 12px;
  line-height: 1.7;
}
.disc {
  margin-top: 10px;
  font-size: 11px;
  opacity: 0.75;
  border-top: 1px dashed var(--border, #333);
  padding-top: 8px;
  white-space: pre-line;
}
.accept .badge {
  background: #1a7f4e;
}
.conditional .badge {
  background: #b58a16;
}
.rework .badge {
  background: #c0392b;
}
.recheck .badge {
  background: #7d7d8a;
}
</style>
