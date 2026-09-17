<script setup lang="ts">
/**
 * 处理中视图（设计稿：真实流水线阶段 + 计时 + 状态）。
 * 展示口径：阶段清单是后端真实流水线顺序（PIPELINE_STAGES，见 types/api.ts）。
 * 同步请求拿不到逐阶段进度，界面不再把所有阶段统一标成「执行中」误导操作员：
 * 运行中只显示整体进行态 + 已耗时 + 超时提示（§判定卡住可看 elapsed），
 * 阶段清单仅作流程说明。
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
/** 超出常规耗时（大底片/慢盘属正常范围），给出「仍在处理」的判断依据而不是让操作员猜卡死。 */
const slow = () => props.elapsedMs > 60_000;
</script>

<template>
  <div>
    <div class="proc-head">
      <div>
        <span style="font-size: 11px; color: var(--ink-faint)">已运行</span>
        <span class="clock">{{ fmt(elapsedMs) }}</span>
      </div>
      <div class="expect">
        预计耗时 5–15 秒（超大底片略久）· 评定进行中，请勿关闭窗口
      </div>
    </div>

    <div class="track">
      <div
        v-for="(name, i) in PIPELINE_STAGES"
        :key="name"
        class="stage-row"
        :class="running() ? 'run' : 'fail'"
      >
        <span class="idx">{{ String(i + 1).padStart(2, "0") }}</span>
        <span class="nm">{{ name }}</span>
        <span
          v-if="running()"
          class="st"
        >{{ i === 0 ? "提交后依序执行" : "排队" }}</span>
        <span
          v-else
          class="st"
        >已中断</span>
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
    <p
      v-if="running() && slow()"
      class="tip slow-hint"
      role="status"
    >
      已超过常规耗时。大尺寸底片或机械硬盘场景下 1–3 分钟属正常范围，计时仍在走动即说明请求未挂起；若超过 3 分钟无响应，可返回检查服务状态后重新提交。
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
      ← 返回修改（保留已选底片与参数）
    </button>
  </div>
</template>

<style scoped>
.slow-hint {
  color: #7a5900;
}
</style>
