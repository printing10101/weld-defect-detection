<script setup lang="ts">
/**
 * 评片告警面板：门禁降级原因、印字区误检屏蔽、单张查重命中等（后端统一
 * 并入 ReportOut.warnings 文本）。展示口径：仅原样展示后端真实告警，
 * 前端不做规则推导；无告警时整块不渲染。
 */
import { computed } from "vue";
import type { ReportOut } from "../types/api";

const props = defineProps<{ result: ReportOut }>();

const notices = computed<string[]>(() => [...(props.result.warnings ?? [])]);
</script>

<template>
  <div
    v-if="notices.length > 0"
    class="notices"
    role="alert"
  >
    <div class="n-h">
      需注意
    </div>
    <ul>
      <li
        v-for="(w, i) in notices"
        :key="i"
      >
        {{ w }}
      </li>
    </ul>
  </div>
</template>

<style scoped>
.notices {
  margin: 14px 0;
  border: 1px solid var(--warn-border, #b58a16);
  background: rgba(181, 138, 22, 0.08);
  border-radius: 8px;
  padding: 10px 14px;
}
.n-h {
  font-weight: 700;
  font-size: 12px;
  color: var(--warn-text, #8a6a10);
}
.notices ul {
  margin: 6px 0 0;
  padding-left: 18px;
  font-size: 12px;
  line-height: 1.7;
}
</style>
