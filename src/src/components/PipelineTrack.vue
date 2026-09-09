<script setup lang="ts">
/**
 * 处理中视图（设计稿：真实流水线阶段 + 计时 + 状态）。
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
        <span style="font-size: 11px; color: var(--ink-faint)">已运行</span>
        <span class="clock">{{ fmt(elapsedMs) }}</span>
      </div>
      <div class="expect">
        预计耗时 15–30 秒 · 评定进行中，请勿关闭窗口
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
        <span class="st">{{ running() ? "执行中" : "失败" }}</span>
      </div>
    </div>

    <p
      v-if="running()"
      class="tip"
    >
      <span class="spin" />正在向本地推理服务提交底片并等待评定结果…
    </p>
    <p
      v-else
      class="tip"
    >
      <span style="color: var(--signal)">✕ 评定失败：{{ errorMessage }}</span>
    </p>
    <div class="leave">
      等待期间可切换至「检测档案」查阅历史记录；评定结果将自动归档至本地数据库。
    </div>

    <button
      v-if="!running()"
      class="btn ghost"
      type="button"
      @click="emit('retry')"
    >
      ← 返回并重试
    </button>
  </div>
</template>
