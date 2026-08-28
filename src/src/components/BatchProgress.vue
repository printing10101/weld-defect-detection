<script setup lang="ts">
/** 批量进度面板：进度条 + 计数 + 逐任务状态 + 取消/重试操作。 */
import type { BatchStatusOut } from "../types/api";

defineProps<{ status: BatchStatusOut }>();
const emit = defineEmits<{ cancel: []; retry: []; archive: [] }>();

const TASK_STATUS_LABEL: Record<string, string> = {
  pending: "等待",
  running: "处理中",
  done: "完成",
  failed: "失败",
  cancelled: "已取消",
};

const STATUS_BADGE: Record<string, string> = {
  pending: "badge-muted",
  running: "badge-run",
  done: "badge-ok",
  failed: "badge-err",
  cancelled: "badge-muted",
};
</script>

<template>
  <div class="bp">
    <div class="bp-head">
      <div class="bp-bar">
        <div
          class="bp-fill"
          :style="{ width: `${Math.round(status.progress * 100)}%` }"
        />
      </div>
      <div class="bp-meta">
        <span class="bp-pct">{{ Math.round(status.progress * 100) }}%</span>
        <span class="bp-counts">
          完成 {{ status.done }} / {{ status.total }}
          <em
            v-if="status.failed"
            class="bp-fail"
          >失败 {{ status.failed }}</em>
          <em v-if="status.cancelled">取消 {{ status.cancelled }}</em>
        </span>
        <span
          v-if="status.status === 'running'"
          class="bp-est"
        >
          预计剩余 ≈ {{ status.estimated_sec }}s
        </span>
        <span
          v-else-if="status.status === 'finished'"
          class="bp-fin"
        >已结束</span>
      </div>
    </div>

    <div class="bp-tasks">
      <div
        v-for="t in status.tasks"
        :key="t.task_id"
        class="bp-task"
      >
        <span
          class="bp-name"
          :title="t.error ?? undefined"
        >{{ t.image_name }}</span>
        <span
          v-if="t.joint_level"
          class="bp-level"
        >级别 {{ t.joint_level }}</span>
        <span
          v-else-if="t.need_review"
          class="bp-rev"
        >需复核</span>
        <span
          class="badge"
          :class="STATUS_BADGE[t.status] ?? 'badge-muted'"
        >
          {{ TASK_STATUS_LABEL[t.status] ?? t.status }}
        </span>
        <span
          v-if="t.error"
          class="bp-err"
          :title="t.error"
        >⚠ {{ t.error }}</span>
      </div>
    </div>

    <div class="bp-ops">
      <button
        v-if="status.status === 'running'"
        type="button"
        class="btn ghost"
        @click="emit('cancel')"
      >
        取消批次
      </button>
      <button
        v-if="status.status === 'finished' && status.failed > 0"
        type="button"
        class="btn"
        @click="emit('retry')"
      >
        重试失败 {{ status.failed }} 项 →
      </button>
      <button
        type="button"
        class="btn ghost"
        @click="emit('archive')"
      >
        去档案检索
      </button>
    </div>
  </div>
</template>

<style scoped>
.bp {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.bp-bar {
  height: 10px;
  border-radius: 6px;
  background: rgba(120, 140, 180, 0.18);
  overflow: hidden;
}
.bp-fill {
  height: 100%;
  border-radius: 6px;
  background: linear-gradient(90deg, #2f6bff, #5b8bff);
  transition: width 0.6s ease;
}
.bp-meta {
  display: flex;
  align-items: baseline;
  gap: 14px;
  font-size: 13px;
  color: #5a6b8a;
}
.bp-pct {
  font-size: 22px;
  font-weight: 700;
  color: #22355c;
  font-variant-numeric: tabular-nums;
}
.bp-counts em {
  font-style: normal;
  margin-left: 6px;
}
.bp-fail {
  color: #c33;
}
.bp-est {
  color: #2f6bff;
}
.bp-fin {
  color: #2a8f4a;
}
.bp-tasks {
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 320px;
  overflow-y: auto;
  border: 1px solid rgba(120, 140, 180, 0.25);
  border-radius: 8px;
  padding: 8px;
}
.bp-task {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
}
.bp-name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #22355c;
}
.bp-level {
  color: #2a8f4a;
  font-weight: 600;
}
.bp-rev {
  color: #b08000;
}
.badge {
  flex: none;
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
}
.badge-ok {
  background: rgba(42, 143, 74, 0.15);
  color: #1e7a3d;
}
.badge-err {
  background: rgba(204, 51, 51, 0.13);
  color: #b03030;
}
.badge-run {
  background: rgba(47, 107, 255, 0.16);
  color: #2f6bff;
}
.badge-muted {
  background: rgba(120, 140, 180, 0.15);
  color: #6a7b99;
}
.bp-err {
  color: #b03030;
  font-size: 11px;
  max-width: 40%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.bp-ops {
  display: flex;
  gap: 10px;
}
</style>
