<script setup lang="ts">
/**
 * 批量处理视图：多底片/文件夹导入 → 异步队列 → 进度可视化 → 取消/重试 → 历史。
 * 数据全部来自真实后端 /batch 系列接口；进度经 2s 轮询实时更新。
 * 查重复核：提交命中重复（批内/与历史已检影像）时后端整批置 awaiting_review
 * 暂缓态，本页逐项确认「跳过/仍检测」后批次才继续执行。
 */
import { computed, onMounted, onUnmounted, ref } from "vue";
import { IMAGE_ACCEPT, IMAGE_EXTS as EXTS } from "../services/imageFormats";
import { toErrorMessage } from "../utils/errorMessage";
import { useViewerFilmsStore } from "../stores/viewerFilms";
import BatchProgress from "../components/BatchProgress.vue";
import ConfirmDialog from "../components/ConfirmDialog.vue";
import {
  cancelBatch,
  getBatchStatus,
  getReportDetections,
  listBatches,
  resolveBatchDuplicates,
  retryBatch,
  submitBatch,
} from "../services/api";
import type { BatchDuplicateItem, BatchStatusOut, BatchSummaryOut } from "../types/api";

const emit = defineEmits<{ archive: [] }>();

// 批量选片即时汇入「底片观察」（store 负责去重与 Blob 回收）
const viewerFilms = useViewerFilmsStore();

const MAX_PER_BATCH = 100;

type Phase = "upload" | "dedup" | "running" | "result";
const phase = ref<Phase>("upload");
const status = ref<BatchStatusOut | null>(null);
const files = ref<File[]>([]);
const activeBatchId = ref<string | null>(null);
const submitError = ref<string | null>(null);
const submitting = ref(false);
const history = ref<BatchSummaryOut[]>([]);

/* ── 查重复核（awaiting_review 阶段） ── */
const duplicates = ref<BatchDuplicateItem[]>([]);
const dupDecisions = ref<Record<string, "skip" | "keep">>({});
const resolving = ref(false);
const skipCount = computed(
  () => duplicates.value.filter((d) => (dupDecisions.value[d.task_id] ?? "skip") === "skip").length,
);
const keepCount = computed(() => duplicates.value.length - skipCount.value);

/** 进入复核阶段：决定缺省「跳过」（重复文件宁可不跑不重跑）。 */
function enterDedup(dups: BatchDuplicateItem[]): void {
  duplicates.value = dups;
  const dec: Record<string, "skip" | "keep"> = {};
  for (const d of dups) dec[d.task_id] = "skip";
  dupDecisions.value = dec;
  phase.value = "dedup";
}

function setDecision(taskId: string, action: "skip" | "keep"): void {
  dupDecisions.value = { ...dupDecisions.value, [taskId]: action };
}

function setAllDecisions(action: "skip" | "keep"): void {
  const dec: Record<string, "skip" | "keep"> = {};
  for (const d of duplicates.value) dec[d.task_id] = action;
  dupDecisions.value = dec;
}

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
    void enrichAnnotations(s);
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
    submitError.value = "所选文件/文件夹中未包含受支持的影像格式（DICOM .dcm / JPG / PNG / BMP / GIF / WebP / TIFF / HEIC 等）。";
    return;
  }
  if (accepted.length > MAX_PER_BATCH) {
    submitError.value = `单批上限 ${MAX_PER_BATCH} 幅，当前已选 ${accepted.length} 幅，请分批提交。`;
    return;
  }
  files.value = accepted;
  submitError.value = null;
  viewerFilms.add(accepted);
}

const fileSummary = () => {
  const n = files.value.length;
  if (n === 0) return "";
  const mb = files.value.reduce((s, f) => s + f.size, 0) / 1024 / 1024;
  return `${n} 幅 · ${mb.toFixed(1)} MB`;
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
    submitError.value = "请先导入底片文件或文件夹。";
    return;
  }
  if (!baseMetalThicknessMm.value.trim()) {
    submitError.value = "母材公称厚度 T 为必填项（分级评定依据）。";
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
    if (out.status === "awaiting_review") {
      // 查重命中：整批暂缓，先交人工逐项复核
      status.value = null;
      enterDedup(out.duplicates ?? []);
      void refreshHistory();
    } else {
      phase.value = "running";
      startPolling(out.batch_id);
    }
  } catch (e) {
    submitError.value = toErrorMessage(e);
  } finally {
    submitting.value = false;
  }
}

/** 人工复核确认：提交逐项决定，批次继续执行（全跳过则直接完成）。 */
async function onDedupConfirm(): Promise<void> {
  if (!activeBatchId.value || resolving.value) return;
  resolving.value = true;
  try {
    await resolveBatchDuplicates(
      activeBatchId.value,
      duplicates.value.map((d) => ({
        task_id: d.task_id,
        action: dupDecisions.value[d.task_id] ?? "skip",
      })),
    );
    duplicates.value = [];
    dupDecisions.value = {};
    status.value = null;
    phase.value = "running";
    startPolling(activeBatchId.value);
  } catch (e) {
    submitError.value = toErrorMessage(e);
  } finally {
    resolving.value = false;
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

/** 旧快照/缺 duplicates 字段时，从任务明细兜底还原重复清单。 */
function fallbackDups(s: BatchStatusOut): BatchDuplicateItem[] {
  return (s.tasks ?? [])
    .filter((t) => t.dup_kind)
    .map((t) => ({
      task_id: t.task_id,
      image_name: t.image_name,
      content_sha256: t.content_sha256 ?? null,
      kind: t.dup_kind ?? "batch",
      duplicate_of: t.dup_ref ?? null,
      history: null,
    }));
}

/* ── 批量结果 → 底片标注 ──
 * 批量上传的约定：有问题的底片（检出缺陷 > 0）在「底片观察」中叠加红色缺陷框，
 * 没有问题的底片不做任何标注。缺陷框来自真实检测记录（report/detections），
 * 按文件名匹配汇入查看工作区的同一张底片；判定与拉取每个报告只做一次。 */
const annotFetched = new Set<string>(); // 已判定/已拉取的 report_id（含"确认无缺陷"，防重复请求）

async function enrichAnnotations(s: BatchStatusOut): Promise<void> {
  for (const t of s.tasks ?? []) {
    if (t.status !== "done" || !t.report_id || annotFetched.has(t.report_id)) continue;
    annotFetched.add(t.report_id);
    // "有问题"判定：检出缺陷数 > 0。defect_count 缺失（旧批次快照）时退回
    // "有报告即拉取"，由返回的缺陷清单为空与否决定是否标注。
    if (t.defect_count != null && t.defect_count <= 0 && !t.need_review) continue;
    void fetchAnnotations(t.report_id, t.image_name);
  }
}

/* ── 批量结果 → 底片印字性质回填 ──
 * 任务完成后把印字识别快照（日期/编号、正/镜像）按文件名匹配进查看工作区，
 * 「底片观察」页即可查阅每张底片的这一性质；无印字底片同样回填（展示"无印字"）。 */
function enrichStamps(s: BatchStatusOut): void {
  for (const t of s.tasks ?? []) {
    if (t.status !== "done" || !t.stamp_status) continue;
    const film = viewerFilms.films.find((f) => f.name === t.image_name);
    if (!film || viewerFilms.stampOf(film.id)) continue; // 已移除或已回填
    viewerFilms.setStamp(film.id, {
      status: t.stamp_status,
      text: t.stamp_text ?? null,
      orientation: t.stamp_orientation ?? null,
      needReview: t.stamp_need_review === true,
    });
  }
}

async function fetchAnnotations(reportId: string, imageName: string): Promise<void> {
  try {
    const det = await getReportDetections(reportId);
    if (det.defects.length === 0) return; // 确认无缺陷：保持无标注
    const film = viewerFilms.films.find((f) => f.name === imageName);
    if (!film) return; // 底片已被移除/驱逐，或本机会话中无对应上传文件
    viewerFilms.setAnnotations(film.id, {
      imageW: det.image_w,
      imageH: det.image_h,
      reportId,
      boxes: det.defects.map((d) => ({
        id: d.id,
        classId: d.class_id,
        bbox: d.bbox,
        confidence: d.confidence,
        needReview: d.need_review,
      })),
    });
  } catch {
    /* 单张标注拉取失败不影响批次流程与轮询（影像仍可正常查看） */
  }
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
    void enrichAnnotations(s); // 逐任务完成后即时回填标注（有问题才标注）
    enrichStamps(s); // 逐任务完成后即时回填印字性质（正/镜像/无印字）
    if (s.status === "awaiting_review") {
      // 查重暂缓批（历史入口进入/轮询途中发现）：停止轮询，转人工复核
      stopPolling();
      enterDedup(s.duplicates ?? fallbackDups(s));
      return;
    }
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
  if (phase.value === "dedup") {
    // 暂缓批取消后立即终态（无 worker 收尾），拉一次状态展示结果
    duplicates.value = [];
    void fetchStatusOnce(activeBatchId.value);
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
  duplicates.value = [];
  dupDecisions.value = {};
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
      data-t="批量评定"
    >
      批量评定
    </h1>
    <div class="lede">
      多幅底片/整卷文件夹批量评定，支持异步队列调度、实时进度监视与历史批次管理
    </div>

    <!-- 阶段1：选择文件与参数 -->
    <div v-if="phase === 'upload'">
      <div class="guide">
        <div class="g">
          <div class="n">
            一 · 底片导入
          </div>
          <div class="t">
            支持多选文件或整卷文件夹导入（DICOM 及常见图像格式），单批上限 {{ MAX_PER_BATCH }} 幅。
          </div>
        </div>
        <div class="g">
          <div class="n">
            二 · 公共工艺参数
          </div>
          <div class="t">
            母材公称厚度 T 为必填项，统一应用于批内全部底片；像质不合格底片默认强制出片并标记「待人工复核」。
          </div>
        </div>
        <div class="g">
          <div class="n">
            三 · 异步批量执行
          </div>
          <div class="t">
            提交后即返回批次编号，多推理进程并行调度，界面实时刷新进度。
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
            文件夹导入将递归收集子目录中的影像；非影像文件自动跳过。
          </div>
          <div
            class="drop"
            @click="openFilePicker"
          >
            <div class="big">
              点击选择底片文件
            </div>
            <div class="hint">
              支持 Ctrl/Shift 多选；影像全程本机处理，不经外部网络传输
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
            或导入整个文件夹…
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
            <label for="spacing">空间像素标定（mm/px）</label>
            <input
              id="spacing"
              v-model="pixelSpacingMm"
            >
            <div class="why">
              默认 0.1000 mm/px；用于将像素尺寸换算为缺陷实际当量。
            </div>
          </div>
          <div class="field">
            <label for="thick">母材公称厚度 T（mm）<span class="req">*</span></label>
            <input
              id="thick"
              v-model="baseMetalThicknessMm"
              placeholder="如 20"
            >
            <div class="why">
              分级评定必备（NB/T 47013.2 依 T 划定评定区与各级限值），统一应用于批内全部底片。
            </div>
          </div>
          <div class="field">
            <label for="wp">工件编号（选填）</label>
            <input
              id="wp"
              v-model="workpieceNo"
              placeholder="如 WP-7781"
            >
          </div>
          <div class="field">
            <label for="wn">焊缝编号（选填）</label>
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
            强制出片（像质不合格底片标记「待人工复核」并继续处理，不阻断整批流程）
          </label>
          <button
            class="btn"
            type="button"
            :disabled="files.length === 0 || submitting"
            @click="onSubmit"
          >
            {{ submitting ? "提交中…" : "提交批量评定 →" }}
          </button>
        </div>
      </div>
    </div>

    <!-- 阶段1.5：查重复核（awaiting_review：有重复，先人工确认再执行） -->
    <div v-else-if="phase === 'dedup'">
      <div class="sec-label">
        批次 {{ activeBatchId ? activeBatchId.slice(0, 8) : "" }}
        <span class="sec-state hold">待重复性核查</span>
      </div>
      <div class="dedup-box">
        <div class="dd-head">
          检出 <b>{{ duplicates.length }}</b> 幅重复底片（与批内其他影像或历史已检影像内容指纹完全一致）。
          请逐项确认处置方式；非重复影像不受影响，确认后立即开始评定。
        </div>
        <div class="dd-list">
          <div
            v-for="d in duplicates"
            :key="d.task_id"
            class="dd-row"
          >
            <div class="dd-info">
              <div class="dd-name">
                {{ d.image_name }}
              </div>
              <div class="dd-meta">
                <span
                  class="dd-kind"
                  :class="d.kind"
                >{{ d.kind === "history" ? "与历史已检影像重复" : "批内重复" }}</span>
                <template v-if="d.kind === 'batch'">
                  与本批「{{ d.duplicate_of }}」内容一致
                </template>
                <template v-else-if="d.history">
                  首检 {{ d.history.image_id.slice(0, 8) }}
                  <template v-if="d.history.created_at"> · {{ d.history.created_at }}</template>
                  <template v-if="d.history.joint_level"> · 级别 {{ d.history.joint_level }}</template>
                </template>
              </div>
            </div>
            <div class="dd-actions">
              <button
                type="button"
                class="dd-btn"
                :class="{ on: (dupDecisions[d.task_id] ?? 'skip') === 'skip' }"
                @click="setDecision(d.task_id, 'skip')"
              >
                跳过（推荐）
              </button>
              <button
                type="button"
                class="dd-btn"
                :class="{ on: (dupDecisions[d.task_id] ?? 'skip') === 'keep' }"
                @click="setDecision(d.task_id, 'keep')"
              >
                强制评定
              </button>
            </div>
          </div>
        </div>
        <div class="dd-foot">
          <button
            type="button"
            class="btn ghost"
            @click="setAllDecisions('skip')"
          >
            全部跳过
          </button>
          <button
            type="button"
            class="btn ghost"
            @click="setAllDecisions('keep')"
          >
            全部强制评定
          </button>
          <span class="dd-hint">跳过的底片不重复评定、不出具报告；强制评定的底片正常评片并归档。</span>
          <button
            type="button"
            class="btn"
            :disabled="resolving"
            style="margin-left: auto"
            @click="onDedupConfirm"
          >
            {{ resolving ? "提交中…" : `确认并继续（跳过 ${skipCount} · 强制评定 ${keepCount}）→` }}
          </button>
          <button
            type="button"
            class="btn ghost dd-danger"
            @click="onCancel"
          >
            取消整批
          </button>
        </div>
        <div
          v-if="submitError"
          class="err show"
        >
          ⚠ {{ submitError }}
        </div>
      </div>
    </div>

    <!-- 阶段2/3：进度与结果 -->
    <div v-else>
      <div
        v-if="backendDown"
        class="err show"
      >
        ⚠ 推理服务无响应，已暂停进度轮询。<button
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
        >已完结</span>
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
          新建批次 →
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
            > 失败{{ row.failed }}</em>
          </span>
          <span
            class="h-status"
            :class="row.status"
          >{{
            row.status === "finished"
              ? "已完成"
              : row.status === "awaiting_review"
                ? "待重复性核查"
                : "进行中"
          }}</span>
        </button>
      </div>
    </div>
  </div>
  <ConfirmDialog
    :open="cancelConfirmOpen"
    title="取消批次确认"
    message="取消后，该批次尚未处理的底片将停止评定，已完成的结果予以保留；失败/已取消的任务后续可重试。确认取消？"
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
.sec-state.hold {
  background: rgba(214, 134, 21, 0.16);
  color: #b06a10;
}
/* ── 查重复核 ── */
.dedup-box {
  border: 1px solid rgba(214, 134, 21, 0.45);
  border-radius: 10px;
  background: rgba(214, 134, 21, 0.05);
  padding: 14px;
}
.dd-head {
  font-size: 13px;
  color: #44577a;
  margin-bottom: 12px;
}
.dd-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.dd-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 12px;
  border: 1px solid rgba(120, 140, 180, 0.25);
  border-radius: 8px;
  background: #fff;
}
.dd-name {
  font-size: 13px;
  font-weight: 700;
  color: #22355c;
  word-break: break-all;
}
.dd-meta {
  margin-top: 4px;
  font-size: 12px;
  color: #6a7b99;
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.dd-kind {
  font-size: 11px;
  padding: 1px 8px;
  border-radius: 10px;
  white-space: nowrap;
}
.dd-kind.history {
  background: rgba(176, 48, 48, 0.12);
  color: #b03030;
}
.dd-kind.batch {
  background: rgba(214, 134, 21, 0.16);
  color: #b06a10;
}
.dd-actions {
  display: flex;
  gap: 6px;
  flex-shrink: 0;
}
.dd-btn {
  border: 1px solid rgba(120, 140, 180, 0.35);
  border-radius: 8px;
  background: transparent;
  color: #6a7b99;
  font-size: 12px;
  padding: 5px 12px;
  cursor: pointer;
}
.dd-btn.on {
  border-color: #2f6bff;
  background: rgba(47, 107, 255, 0.1);
  color: #2f6bff;
  font-weight: 700;
}
.dd-foot {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 14px;
  flex-wrap: wrap;
}
.dd-hint {
  font-size: 12px;
  color: #8a99b5;
}
.dd-danger {
  color: #b03030;
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
.h-status.awaiting_review {
  color: #b06a10;
}
.faint {
  color: #8a99b5;
}
</style>
