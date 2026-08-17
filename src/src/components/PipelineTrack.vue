<script setup lang="ts">
/**
 * 处理中视图（DESIGN.md：真实流水线阶段 + 计时 + 状态）。
 * 数据诚实性：阶段清单是后端真实流水线顺序（PIPELINE_STAGES，见 types/api.ts），
 * 仅作流程说明；"进行中/失败"与耗时来自真实请求状态，不模拟任何阶段完成。
 */
import { PIPELINE_STAGES } from "../types/api";

const props = defineProps<{
  status: "running" | "error";
  elapsedMs: number;
  errorMessage: string | null;
}>();
const emit = defineEmits<{ retry: [] }>();

function fmt(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

const running = () => props.status === "running";
</script>

<template>
  <div>
    <div class="proc-head">
      <div>
        <span style="font-size: 11px; color: var(--ink-faint)">已耗时</span>
        <span class="clock">{{ fmt(elapsedMs) }}</span>
      </div>
      <div class="expect">
        预计 15–30 秒 · 请求进行中，请勿关闭
      </div>
    </div>

    <div class="track">
      <div
        v-for="(name, i) in PIPELINE_STAGES"
        :key="name"
        class="stage-row"
        :class="{ run: running(), fail: !running() }"
      >
        <span class="idx">{{ String(i + 1).padStart(2, "0") }}</span>
        <span class="nm">{{ name }}</span>
        <span class="st">{{ running() ? "请求中" : "失败" }}</span>
      </div>
    </div>

    <p
      v-if="running()"
      class="tip"
    >
      <span class="spin" />正在向后端提交影像并等待真实处理结果…
    </p>
    <p
      v-else
      class="tip"
    >
      <span style="color: var(--signal)">✕ 处理失败：{{ errorMessage }}</span>
    </p>
    <div class="leave">
      你可以在等待时切换到「档案检索」查看历史；本次结果会自动归档到本地。
    </div>

    <button
      v-if="!running()"
      class="btn ghost"
      type="button"
      @click="emit('retry')"
    >
      ← 返回重试
    </button>
  </div>
</template>
