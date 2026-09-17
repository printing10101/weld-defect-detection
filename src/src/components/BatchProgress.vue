<script setup lang="ts">
/** 批量进度面板：进度条 + 计数 + 逐任务状态 + 取消/重试操作。
 *  每个已完成任务提供「报告」入口（此前批量评完无法查看任何一张的完整报告）。 */
import { stampBadge } from "../utils/filmStamp";
import type { BatchStatusOut } from "../types/api";

defineProps<{ status: BatchStatusOut }>();
const emit = defineEmits<{
  cancel: [];
  retry: [];
  archive: [];
  pause: [];
  resume: [];
  /** 打开某任务的 PDF 报告（经受控导出通道） */
  openReport: [reportId: string];
}>();

const TASK_STATUS_LABEL: Record<string, string> = {
  pending: "排队中",
  running: "评定中",
  done: "已完成",
  failed: "已失败",
  cancelled: "已取消",
};

/** 后端任务错误信息中文化（旧快照里持久化的英文错误一并覆盖）。 */
function errorText(raw: string | null): string {
  if (!raw) return "";
  if (raw.includes("interrupted by restart")) return "因程序重启中断，可整批重试";
  return raw;
}

const STATUS_BADGE: Record<string, string> = {
  pending: "badge-muted",
  running: "badge-run",
  done: "badge-ok",
  failed: "badge-err",
  cancelled: "badge-muted",
};

/** 批次级状态标签（paused = 用户主动暂停，待恢复）。 */
const BATCH_STATUS_LABEL: Record<string, string> = {
  running: "评定中",
  paused: "已暂停",
  awaiting_review: "待重复性核查",
  finished: "已完结",
};

/** 逐任务印字徽标（无识别数据的旧快照返回 null，不占位）。 */
function stampOf(t: BatchStatusOut["tasks"][number]) {
  return t.status === "done" ? stampBadge(t) : null;
}

/** 印字占比裁决摘要文案（批次收尾后展示豁免/补标结论）。 */
function stampSummaryLine(status: BatchStatusOut): string {
  const s = status.stamp_summary;
  if (!s) return "";
  const base = `底片印字：${s.present}/${s.evaluated} 张识别到日期/编号印字（${Math.round(s.ratio * 100)}%）`;
  return s.suppressed
    ? `${base} —— 该批普遍无印字，缺印字已按规则豁免人工复核`
    : `${base} —— 缺印字底片已转人工复核 ${s.flagged} 张`;
}
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
          已完成 {{ status.done }} / {{ status.total }}
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
          预计剩余 ≈ {{ status.estimated_sec }} 秒
        </span>
        <span
          v-else-if="status.status === 'paused'"
          class="bp-paused"
        >已暂停 · 点击「继续评定」恢复</span>
        <span
          v-else
          class="bp-fin"
        >{{ BATCH_STATUS_LABEL[status.status] ?? "已完结" }}</span>
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
        >评定级别 {{ t.joint_level }}</span>
        <span
          v-else-if="t.need_review"
          class="bp-rev"
        >待人工复核</span>
        <!-- 有问题的底片显式红标缺陷数；无缺陷底片不标注 -->
        <span
          v-if="t.status === 'done' && (t.defect_count ?? 0) > 0"
          class="bp-defect"
          title="该底片检出缺陷，已在「底片观察」中叠加标注框"
        >缺陷 {{ t.defect_count }} 处</span>
        <span
          v-if="t.dup_kind"
          class="bp-dup"
          :title="t.dup_kind === 'history' ? `与历史影像 ${t.dup_ref ?? ''} 内容指纹重复` : `与批内 ${t.dup_ref ?? ''} 内容指纹重复`"
        >重复</span>
        <!-- 底片印字性质（扫描日期/编号，正/镜像）：识别到展示内容徽标，缺印字按复核状态标色 -->
        <span
          v-if="stampOf(t)"
          class="bp-stamp"
          :class="`stamp-${stampOf(t)!.cls}`"
          :title="stampOf(t)!.title"
        >{{ stampOf(t)!.label }}</span>
        <span
          class="badge"
          :class="STATUS_BADGE[t.status] ?? 'badge-muted'"
        >
          {{ TASK_STATUS_LABEL[t.status] ?? t.status }}
        </span>
        <span
          v-if="t.error"
          class="bp-err"
          :title="errorText(t.error)"
        >注意：{{ errorText(t.error) }}</span>
        <button
          v-if="t.status === 'done' && t.report_id"
          type="button"
          class="bp-report"
          title="打开该底片的 PDF/A 检测报告（含缺陷标注与检出明细）"
          @click="emit('openReport', t.report_id)"
        >
          报告
        </button>
      </div>
    </div>

    <p
      v-if="status.stamp_summary"
      class="bp-stamp-summary"
    >
      {{ stampSummaryLine(status) }}
    </p>

    <div class="bp-ops">
      <button
        v-if="status.status === 'running'"
        type="button"
        class="btn ghost"
        title="未启动的底片暂停派发，正在评定的底片会正常完成；可随时恢复"
        @click="emit('pause')"
      >
        暂停评定
      </button>
      <button
        v-if="status.status === 'paused'"
        type="button"
        class="btn"
        title="把暂停时未启动的底片重新投入评定"
        @click="emit('resume')"
      >
        继续评定 →
      </button>
      <button
        v-if="status.status === 'running' || status.status === 'paused'"
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
        重试失败任务（{{ status.failed }} 项）→
      </button>
      <button
        type="button"
        class="btn ghost"
        @click="emit('archive')"
      >
        查阅检测档案
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
.bp-paused {
  color: #b06a10;
  font-weight: 600;
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
.bp-defect {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  background: rgba(204, 51, 51, 0.12);
  color: #b03030;
  font-weight: 600;
  white-space: nowrap;
}
.bp-dup {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  background: rgba(214, 134, 21, 0.16);
  color: #b06a10;
  white-space: nowrap;
}
.bp-stamp {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  white-space: nowrap;
  max-width: 260px;
  overflow: hidden;
  text-overflow: ellipsis;
}
.stamp-ok {
  background: rgba(42, 143, 74, 0.14);
  color: #1e7a3d;
}
.stamp-warn {
  background: rgba(204, 51, 51, 0.12);
  color: #b03030;
  font-weight: 600;
}
.stamp-muted {
  background: rgba(120, 140, 180, 0.15);
  color: #6a7b99;
}
.bp-stamp-summary {
  margin: 0;
  font-size: 12px;
  color: #5a6b8a;
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
.bp-report {
  flex: none;
  font-size: 11px;
  padding: 2px 10px;
  border-radius: 10px;
  border: 1px solid rgba(47, 107, 255, 0.45);
  background: rgba(47, 107, 255, 0.08);
  color: #2f6bff;
  cursor: pointer;
  white-space: nowrap;
}
.bp-report:hover {
  background: #2f6bff;
  color: #fff;
}
.bp-ops {
  display: flex;
  gap: 10px;
}
</style>
