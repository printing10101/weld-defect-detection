<script setup lang="ts">
/**
 * 批量处理视图：多底片/文件夹导入 → 异步队列 → 进度可视化 → 取消/重试 → 历史。
 * 数据全部来自真实后端 /batch 系列接口；进度经 2s 轮询实时更新。
 */
import { onMounted, onUnmounted, ref } from "vue";
import { IMAGE_ACCEPT, IMAGE_EXTS as EXTS } from "../services/imageFormats";
import { toErrorMessage } from "../utils/errorMessage";
import BatchProgress from "../components/BatchProgress.vue";
import ConfirmDialog from "../components/ConfirmDialog.vue";
import { cancelBatch, getBatchStatus, listBatches, retryBatch, submitBatch } from "../services/api";
import type { BatchStatusOut, BatchSummaryOut } from "../types/api";

const emit = defineEmits<{ archive: [] }>();

const MAX_PER_BATCH = 100;

type Phase = "upload" | "running" | "result";
const phase = ref<Phase>("upload");
const status = ref<BatchStatusOut | null>(null);
const files = ref<File[]>([]);
const activeBatchId = ref<string | null>(null);
const submitError = ref<string | null>(null);
const submitting = ref(false);
const history = ref<BatchSummaryOut[]>([]);

const pixelSpacingMm = ref("0.1000");
const baseMetalThicknessMm = ref("");
const workpieceNo = ref("");
const weldNo = ref("");
const force = ref(true);

let timer: number | null = null;
/** 当前轮询的批次 id（null=未在轮询）。setTimeout 链式调度的归属标记。 */
let pollingId: string | null = null;

// 轮询健壮性：连续失败退避 + 后端离线态，避免后端宕机时无限空转。
const POLL_BASE_MS = 2000;
const MAX_OFFLINE_STRIKES = 3;
const pollErrorCount = ref(0);
const backendDown = ref(false);

function pollIntervalMs(): number {
  // 连续失败指数退避：2s → 4s → 8s（上限），恢复即回 2s
  return Math.min(POLL_BASE_MS * 2 ** pollErrorCount.value, 8000);
}

/* ── 历史批次 ── */
async function refreshHistory(): Promise<void> {
  try {
    history.value = await listBatches();
  } catch {
    /* 列表刷新失败不打扰当前流程 */
  }
}

function openHistory(row: BatchSummaryOut): void {
  // 先停掉旧批次的轮询：否则在途的 tick(旧) 会通过归属守卫，把旧批次
  // 状态覆盖到刚点开的历史批次的界面上（甚至到终态时反向 stopPolling）。
  stopPolling();
  activeBatchId.value = row.batch_id;
  if (row.status === "finished") {
    void fetchStatusOnce(row.batch_id);
  } else {
    phase.value = "running";
    startPolling(row.batch_id);
  }
}

/** 一次性状态拉取的世代标记：迟到的响应不得覆盖用户切换后的视图。 */
let fetchToken = 0;

async function fetchStatusOnce(id: string): Promise<void> {
  const token = ++fetchToken;
  try {
    const s = await getBatchStatus(id);
    if (token !== fetchToken) return; // 已切走（切换/新提交都会使代次失效）
    status.value = s;
    phase.value = "result";
  } catch {
    /* 忽略 */
  }
}

/* ── 文件选择（多文件 / 文件夹，文件夹经 webkitdirectory 递归收集） ── */
function onInputChanged(e: Event): void {
  const input = e.target as HTMLInputElement;
  pickFiles(input.files);
  // 复位 input：不清空的话，再次选择完全相同的文件不触发 change（静默无响应）
  input.value = "";
}

function pickFiles(list: FileList | null): void {
  if (!list || list.length === 0) return;
  const accepted: File[] = [];
  for (const f of Array.from(list)) {
    const ext = f.name.split(".").pop()?.toLowerCase() ?? "";
    if ((EXTS as readonly string[]).includes(ext)) accepted.push(f);
  }
  if (accepted.length === 0) {
    submitError.value = "所选文件/文件夹中没有支持的影像（DICOM .dcm / JPG / PNG / BMP / GIF / WebP / TIFF / HEIC 等）。";
    return;
  }
  if (accepted.length > MAX_PER_BATCH) {
    submitError.value = `单批最多 ${MAX_PER_BATCH} 张，当前 ${accepted.length} 张，请分批。`;
    return;
  }
  files.value = accepted;
  submitError.value = null;
}

const fileSummary = () => {
  const n = files.value.length;
  if (n === 0) return "";
  const mb = files.value.reduce((s, f) => s + f.size, 0) / 1024 / 1024;
  return `${n} 张 · ${mb.toFixed(1)} MB`;
};

function openFilePicker(): void {
  (document.getElementById("pick-files") as HTMLInputElement | null)?.click();
}

function openDirPicker(): void {
  (document.getElementById("pick-dir") as HTMLInputElement | null)?.click();
}

/* ── 提交与轮询 ── */
function onSubmit(): void {
  submitError.value = null;
  if (files.value.length === 0) {
    submitError.value = "请先选择底片文件或文件夹。";
    return;
  }
  if (!baseMetalThicknessMm.value.trim()) {
    submitError.value = "母材厚度 T 必填（评级依据）。";
    return;
  }
  const fd = new FormData();
  for (const f of files.value) fd.append("images", f);
  fd.append("pixel_spacing_mm", pixelSpacingMm.value || "");
  fd.append("base_metal_thickness_mm", baseMetalThicknessMm.value.trim());
  if (workpieceNo.value.trim()) fd.append("workpiece_no", workpieceNo.value.trim());
  if (weldNo.value.trim()) fd.append("weld_no", weldNo.value.trim());
  fd.append("force", force.value ? "true" : "false");
  void doSubmit(fd);
}

async function doSubmit(fd: FormData): Promise<void> {
  if (submitting.value) return; // 防双击重复提交（§D2）
  submitting.value = true;
  try {
    const out = await submitBatch(fd);
    activeBatchId.value = out.batch_id;
    phase.value = "running";
    startPolling(out.batch_id);
  } catch (e) {
    submitError.value = toErrorMessage(e);
  } finally {
    submitting.value = false;
  }
}

function clearTimer(): void {
  if (timer !== null) {
    window.clearTimeout(timer);
    timer = null;
  }
}

function scheduleNext(id: string): void {
  clearTimer();
  timer = window.setTimeout(() => void tick(id), pollIntervalMs());
}

function startPolling(id: string): void {
  stopPolling();
  backendDown.value = false;
  pollErrorCount.value = 0;
  fetchToken++; // 使在途的 fetchStatusOnce 失效，防止一次性拉取覆盖轮询视图
  pollingId = id;
  void tick(id);
}

function stopPolling(): void {
  pollingId = null;
  clearTimer();
}

async function tick(id: string): Promise<void> {
  try {
    const s = await getBatchStatus(id);
    // 归属守卫：请求在途时用户可能已切换/提交了新批次，迟到响应不得覆盖
    // 新批次状态，更不得触发 stopPolling 杀掉新批次的轮询。
    if (pollingId !== id) return;
    status.value = s;
    pollErrorCount.value = 0;
    backendDown.value = false;
    if (s.status === "finished") {
      stopPolling();
      phase.value = "result";
      void refreshHistory();
      return;
    }
  } catch {
    if (pollingId !== id) return; // 同上：迟到失败的响应也不影响新批次
    // 单次轮询失败：累计并退避；超过阈值判定后端离线，停止空转并提示。
    pollErrorCount.value += 1;
    if (pollErrorCount.value >= MAX_OFFLINE_STRIKES) {
      backendDown.value = true;
      stopPolling();
      return;
    }
  }
  // 链式调度：上一次请求完成后再排下一次。此前用 setInterval，请求挂起时
  // 定时器照发，最坏堆叠十余个并发请求打向同一个挂死后端。
  if (pollingId === id) scheduleNext(id);
}

function retryConnection(): void {
  if (activeBatchId.value) startPolling(activeBatchId.value);
}

// 取消批次须二次确认（用户差错防御）：确认后才真正调用取消接口
const cancelConfirmOpen = ref(false);

function onCancel(): void {
  if (!activeBatchId.value) return;
  cancelConfirmOpen.value = true;
}

async function onCancelConfirmed(): Promise<void> {
  cancelConfirmOpen.value = false;
  if (!activeBatchId.value) return;
  try {
    await cancelBatch(activeBatchId.value);
  } catch {
    /* 取消失败忽略（轮询会继续展示真实状态） */
  }
}

async function onRetry(): Promise<void> {
  if (!activeBatchId.value) return;
  try {
    await retryBatch(activeBatchId.value);
    phase.value = "running";
    startPolling(activeBatchId.value);
  } catch (e) {
    submitError.value = toErrorMessage(e);
  }
}

function reset(): void {
  stopPolling();
  fetchToken++; // 失效在途的 fetchStatusOnce：迟到响应不得把用户拽回上一批次的结束视图
  phase.value = "upload";
  status.value = null;
  activeBatchId.value = null;
  files.value = [];
  submitError.value = null;
  void refreshHistory();
}

onMounted(() => {
  void refreshHistory();
});

onUnmounted(() => {
  stopPolling();
});
</script>

<template>
  <div>
    <h1
      class="title-zine"
      data-t="批量检测"
    >
      批量检测
    </h1>
    <div class="lede">
      多底片/文件夹批量检测，支持异步队列、进度查看与历史批次
    </div>

    <!-- 阶段1：选择文件与参数 -->
    <div v-if="phase === 'upload'">
      <div class="guide">
        <div class="g">
          <div class="n">
            一 · 导入底片
          </div>
          <div class="t">
            选择多个文件或整个文件夹（DICOM 及常见图像格式均可），单批 ≤ {{ MAX_PER_BATCH }} 张。
          </div>
        </div>
        <div class="g">
          <div class="n">
            二 · 公共参数
          </div>
          <div class="t">
            母材厚度 T 必填，应用到批内所有底片；不合格底片默认强制出片并标记「需复核」。
          </div>
        </div>
        <div class="g">
          <div class="n">
            三 · 异步执行
          </div>
          <div class="t">
            提交后立即返回批次号，多 worker 并行推理，页面实时显示进度。
          </div>
        </div>
      </div>

      <div class="row">
        <div class="grow">
          <div class="chips">
            <span class="chip on">DICOM .dcm</span>
            <span class="chip on">JPG/PNG/BMP</span>
            <span class="chip on">TIFF/GIF/WebP</span>
            <span class="chip on">HEIC/AVIF</span>
          </div>
          <div class="hint">
            文件夹导入会递归收集子目录影像；非影像文件自动跳过。
          </div>
          <div
            class="drop"
            @click="openFilePicker"
          >
            <div class="big">
              拖入请点此：选择底片文件
            </div>
            <div class="hint">
              支持 Ctrl/Shift 多选；影像只在本机处理
            </div>
          </div>
          <input
            id="pick-files"
            type="file"
            :accept="IMAGE_ACCEPT"
            multiple
            style="display: none"
            @change="onInputChanged($event)"
          >
          <button
            type="button"
            class="btn ghost"
            @click="openDirPicker"
          >
            或选择整个文件夹…
          </button>
          <input
            id="pick-dir"
            type="file"
            webkitdirectory
            multiple
            style="display: none"
            @change="onInputChanged($event)"
          >
          <div
            v-if="files.length"
            class="preview show"
          >
            <div class="meta">
              {{ fileSummary() }}
            </div>
            <div class="meta faint">
              {{ files.slice(0, 8).map((f) => f.name).join("、") }}<span v-if="files.length > 8">…</span>
            </div>
          </div>
          <div
            v-if="submitError"
            class="err show"
          >
            ⚠ {{ submitError }}
          </div>
        </div>

        <div class="grow">
          <div class="field">
            <label for="spacing">像素标定（mm/px）</label>
            <input
              id="spacing"
              v-model="pixelSpacingMm"
            >
            <div class="why">
              默认 0.1000 mm/px；用于把像素尺寸换算为真实当量。
            </div>
          </div>
          <div class="field">
            <label for="thick">母材厚度 T（mm）<span class="req">*</span></label>
            <input
              id="thick"
              v-model="baseMetalThicknessMm"
              placeholder="如 20"
            >
            <div class="why">
              评级必须（NB/T47013.2 按 T 分档评定区与限值），应用到批内所有底片。
            </div>
          </div>
          <div class="field">
            <label for="wp">工件号（可选）</label>
            <input
              id="wp"
              v-model="workpieceNo"
              placeholder="如 WP-7781"
            >
          </div>
          <div class="field">
            <label for="wn">焊口编号（可选）</label>
            <input
              id="wn"
              v-model="weldNo"
              placeholder="如 W-12"
            >
          </div>
          <label class="check">
            <input
              v-model="force"
              type="checkbox"
            >
            强制出片（不合格底片标记「需复核」并继续，不阻断整批）
          </label>
          <button
            class="btn"
            type="button"
            :disabled="files.length === 0 || submitting"
            @click="onSubmit"
          >
            {{ submitting ? "提交中…" : "提交批量检测 →" }}
          </button>
        </div>
      </div>
    </div>

    <!-- 阶段2/3：进度与结果 -->
    <div v-else>
      <div
        v-if="backendDown"
        class="err show"
      >
        ⚠ 后端无响应，已暂停进度轮询。<button
          class="btn link"
          type="button"
          @click="retryConnection"
        >
          重试连接
        </button>
      </div>
      <div
        class="sec-label"
        :data-t="`BATCH ${activeBatchId ? activeBatchId.slice(0, 8) : ''}`"
      >
        批次 {{ activeBatchId ? activeBatchId.slice(0, 8) : "" }}
        <span
          v-if="phase === 'running'"
          class="sec-state run"
        >执行中</span>
        <span
          v-else
          class="sec-state fin"
        >已结束</span>
      </div>
      <BatchProgress
        v-if="status"
        :status="status"
        @cancel="onCancel"
        @retry="onRetry"
        @archive="emit('archive')"
      />
      <div
        v-if="phase === 'result'"
        class="row"
        style="margin-top: 14px"
      >
        <button
          type="button"
          class="btn"
          @click="reset()"
        >
          新批次 →
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

    <!-- 历史批次（断点续跑入口） -->
    <div
      v-if="history.length"
      class="hist"
    >
      <div class="section-h">
        历史批次
      </div>
      <div class="hist-list">
        <button
          v-for="row in history"
          :key="row.batch_id"
          type="button"
          class="hist-row"
          :class="{ cur: row.batch_id === activeBatchId }"
          @click="openHistory(row)"
        >
          <span class="h-id">{{ row.batch_id.slice(0, 8) }}</span>
          <span class="h-time">{{ row.created_at }}</span>
          <span class="h-prog">{{ Math.round(row.progress * 100) }}%</span>
          <span class="h-counts">
            {{ row.done }}/{{ row.total }}<em
              v-if="row.failed"
              class="h-fail"
            > 败{{ row.failed }}</em>
          </span>
          <span
            class="h-status"
            :class="row.status"
          >{{ row.status === "finished" ? "完成" : "进行中" }}</span>
        </button>
      </div>
    </div>
  </div>
  <ConfirmDialog
    :open="cancelConfirmOpen"
    title="取消批次确认"
    message="取消后该批次未处理的影像将停止处理，已完成结果保留；失败/取消的任务之后可重试。确定取消？"
    confirm-text="取消批次"
    danger
    @confirm="onCancelConfirmed"
    @cancel="cancelConfirmOpen = false"
  />
</template>

<style scoped>
.sec-label {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 14px;
  color: #22355c;
  margin-bottom: 12px;
}
.sec-state {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
}
.sec-state.run {
  background: rgba(47, 107, 255, 0.16);
  color: #2f6bff;
}
.sec-state.fin {
  background: rgba(42, 143, 74, 0.15);
  color: #1e7a3d;
}
.check {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: #44577a;
  margin: 10px 0 14px;
}
.hist {
  margin-top: 28px;
}
.hist-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.hist-row {
  display: flex;
  align-items: center;
  gap: 14px;
  width: 100%;
  text-align: left;
  padding: 8px 10px;
  border: 1px solid rgba(120, 140, 180, 0.25);
  border-radius: 8px;
  background: transparent;
  color: #22355c;
  font-size: 13px;
  cursor: pointer;
}
.hist-row:hover,
.hist-row.cur {
  border-color: #2f6bff;
  background: rgba(47, 107, 255, 0.06);
}
.h-id {
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  font-size: 12px;
  color: #2f6bff;
}
.h-time {
  color: #6a7b99;
}
.h-prog {
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}
.h-counts em {
  font-style: normal;
  color: #b03030;
}
.h-status {
  margin-left: auto;
  font-size: 12px;
}
.h-status.finished {
  color: #1e7a3d;
}
.h-status.running {
  color: #2f6bff;
}
.faint {
  color: #8a99b5;
}
</style>
