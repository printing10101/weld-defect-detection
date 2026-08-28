<script setup lang="ts">
/**
 * 报告解读视图（设计稿：结论先行 + 影像对比 + 操作建议）。
 * 数据诚实性：全部内容来自 ReportOut 真实字段 + 用户上传文件真实 objectURL：
 * - 级别/待复核/可评片/缺陷数：ReportOut；
 * - 送检原图：用户上传文件的真实 objectURL（无标注）；
 * - 标注对比图：后端仅通过 PDF 提供（report 端点无独立标注图 URL），
 *   故如实提供 PDF 下载入口，不伪造标注图；
 * - 判定依据/黑度等：ReportOut 不包含，前端不构造展示。
 */
import { computed, ref } from "vue";
import ResultBanner from "./ResultBanner.vue";
import ReviewPanel from "./ReviewPanel.vue";
import DispositionPanel from "./DispositionPanel.vue";
import { activeExport, getReportDetections, verifyReport } from "../services/api";
import type {
  ActiveExportOut,
  ReportDetectionsOut,
  ReportOut,
  VerifyOut,
} from "../types/api";

/** 缺陷类别中文标签（镜像 backend/domain/dto.py DefectClass 0..5）。 */
const DEFECT_LABELS = ["气孔", "夹渣", "未焊透", "未熔合", "裂纹", "咬边"] as const;

interface ExportRow {
  id: string;
  class_id: number;
  bbox: [number, number, number, number];
  confidence: number;
  uncertainty: number;
  reviewed: boolean;
  need_review: boolean;
  checked: boolean;
  /** -1 = 不改判（沿用原类别）；否则为人工改判后的类别 id */
  override: number;
}

const props = defineProps<{
  result: ReportOut;
  sourceUrl: string | null;
  fileName: string | null;
}>();

const emit = defineEmits<{ archive: []; reset: [] }>();

function openPdf(): void {
  window.open(props.result.pdf_url, "_blank", "noopener");
}

/* ── 主动学习闭环回流：取明细 → 人工复核/改判 → 回流训练池 ── */
const exportOpen = ref(false);
const loadingDets = ref(false);
const detsErr = ref<string | null>(null);
const dets = ref<ReportDetectionsOut | null>(null);
const rows = ref<ExportRow[]>([]);
const exporting = ref(false);
const exportResult = ref<ActiveExportOut | null>(null);
const exportErr = ref<string | null>(null);

const selectedCount = computed(() => rows.value.filter((r) => r.checked).length);
const canExport = computed(() => selectedCount.value > 0 && !exporting.value);

async function openExport(): Promise<void> {
  exportOpen.value = true;
  loadingDets.value = true;
  detsErr.value = null;
  exportResult.value = null;
  exportErr.value = null;
  try {
    const d = await getReportDetections(props.result.report_id);
    dets.value = d;
    rows.value = d.defects.map((x) => ({
      id: x.id,
      class_id: x.class_id,
      bbox: x.bbox,
      confidence: x.confidence,
      uncertainty: x.uncertainty,
      reviewed: x.reviewed,
      need_review: x.need_review,
      checked: true,
      override: -1,
    }));
  } catch (e) {
    detsErr.value = e instanceof Error ? e.message : String(e);
  } finally {
    loadingDets.value = false;
  }
}

async function confirmExport(): Promise<void> {
  if (!dets.value || !canExport.value) return;
  exporting.value = true;
  exportErr.value = null;
  const sel = rows.value.filter((r) => r.checked);
  try {
    exportResult.value = await activeExport({
      image_stem: dets.value.image_stem,
      image_w: dets.value.image_w,
      image_h: dets.value.image_h,
      defects: sel.map((r) => ({
        id: r.id,
        class_id: r.override >= 0 ? r.override : r.class_id,
        bbox: r.bbox,
        confidence: r.confidence,
        uncertainty: r.uncertainty,
      })),
      class_overrides: {},
    });
  } catch (e) {
    exportErr.value = e instanceof Error ? e.message : String(e);
  } finally {
    exporting.value = false;
  }
}

function closeExport(): void {
  exportOpen.value = false;
}

/* ── 报告数字签名校验：比对内容指纹与签发记录，防篡改 ── */
const verifying = ref(false);
const verifyResult = ref<VerifyOut | null>(null);
const verifyError = ref<string | null>(null);

const verifyStateClass = computed(() => {
  if (!verifyResult.value) return "";
  if (verifyResult.value.valid === true) return "ok";
  if (verifyResult.value.valid === false) return "bad";
  return "na";
});

async function onVerify(): Promise<void> {
  verifying.value = true;
  verifyError.value = null;
  try {
    verifyResult.value = await verifyReport(props.result.report_id);
  } catch (e) {
    verifyError.value = e instanceof Error ? e.message : String(e);
  } finally {
    verifying.value = false;
  }
}
</script>

<template>
  <div>
    <h1
      class="title-zine"
      data-t="评片报告"
    >
      评片报告
    </h1>
    <div class="lede">
      报告编号 {{ result.report_id }} · 影像编号 {{ result.image_id }}
    </div>

    <ResultBanner :result="result" />
    <DispositionPanel :result="result" />

    <div class="section-h">
      <span class="no">影像</span>送检原始影像（来自你上传的文件）
    </div>
    <div class="compare">
      <div class="plate">
        <img
          v-if="sourceUrl"
          :src="sourceUrl"
          :alt="fileName ?? '上传影像'"
        >
        <span
          v-else
          class="ph"
        >影像不可用</span>
        <div class="cap">
          {{ fileName ?? "上传文件" }} · 未标注
        </div>
      </div>
      <div class="plate">
        <a
          :href="result.pdf_url"
          target="_blank"
          rel="noopener"
          style="display: block; text-decoration: none; color: inherit"
        >
          <span
            class="ph"
            style="display: grid; place-items: center; min-height: 160px; color: var(--accent); font-size: 11px"
          >
            检测标注影像见 PDF 报告 →
          </span>
          <div class="cap">打开 PDF/A 报告（含缺陷标注与明细）</div>
        </a>
      </div>
    </div>

    <div class="section-h">
      <span class="no">结果</span>本次检测结果
    </div>
    <div class="kv">
      <div class="k">
        报告编号
      </div><div class="v">
        {{ result.report_id }}
      </div>
      <div class="k">
        影像编号
      </div><div class="v">
        {{ result.image_id }}
      </div>
      <div class="k">
        缺陷数量
      </div><div class="v">
        {{ result.defect_count }} 处
      </div>
      <div class="k">
        可评片性
      </div><div class="v">
        {{ result.evaluable ? "可评片" : "不可评片（影像质量不达标）" }}
      </div>
      <div class="k">
        综合级别
      </div><div class="v">
        {{ result.joint_level ?? "（待复核/未输出）" }}
      </div>
    </div>

    <div
      v-if="result.need_review"
      class="section-h"
    >
      <span class="no">复核</span>人工复核（M7 闭环）
    </div>
    <ReviewPanel
      v-if="result.need_review"
      :image-id="result.image_id"
    />

    <div class="section-h">
      <span class="no">建议</span>接下来可以做什么
    </div>
    <div class="acts">
      <button
        class="act"
        type="button"
        @click="openPdf"
      >
        <div class="a">
          导出 PDF/A
        </div>
        <div class="d">
          下载归档合规报告（含缺陷标注、判定依据与明细）
        </div>
      </button>
      <button
        class="act"
        type="button"
        @click="emit('archive')"
      >
        <div class="a">
          查看档案
        </div>
        <div class="d">
          在档案检索中查看该影像的归档记录与统计
        </div>
      </button>
      <button
        class="act"
        type="button"
        @click="emit('reset')"
      >
        <div class="a">
          重新检测
        </div>
        <div class="d">
          返回上传步骤，提交下一份底片
        </div>
      </button>
      <button
        class="act"
        type="button"
        :disabled="verifying"
        @click="onVerify"
      >
        <div class="a">
          {{ verifying ? "校验中…" : "验证数字签名" }}
        </div>
        <div class="d">
          比对报告内容指纹与签发记录，防篡改（§7.2）
        </div>
      </button>
      <button
        class="act"
        type="button"
        :disabled="result.defect_count === 0"
        @click="openExport"
      >
        <div class="a">
          回流训练池
        </div>
        <div class="d">
          人工复核确认缺陷 → 标注回流主动学习训练池（§5.5 持续学习闭环）
        </div>
      </button>
    </div>

    <div
      v-if="exportOpen"
      class="modal-mask"
      @click.self="closeExport"
    >
      <div class="modal">
        <h3 class="m-title">
          回流训练池 · 人工复核确认
        </h3>
        <p
          v-if="loadingDets"
          class="hint"
        >
          加载缺陷明细…
        </p>
        <p
          v-else-if="detsErr"
          class="err show"
        >
          ⚠ 加载失败：{{ detsErr }}
        </p>
        <template v-else-if="dets">
          <p class="stat">
            影像 <b>{{ dets.image_stem }}</b> · {{ dets.image_w }}×{{ dets.image_h }}px ·
            共 {{ dets.defects.length }} 处缺陷。勾选需回流的样本，必要时改判类别后确认。
          </p>
          <table class="exp">
            <thead>
              <tr>
                <th>回流</th>
                <th>原类别</th>
                <th>人工改判</th>
                <th>置信度</th>
                <th>不确定性</th>
                <th>状态</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="r in rows"
                :key="r.id"
              >
                <td>
                  <input
                    v-model="r.checked"
                    type="checkbox"
                  >
                </td>
                <td>{{ DEFECT_LABELS[r.class_id] }} <span class="cid">({{ r.class_id }})</span></td>
                <td>
                  <select
                    v-model.number="r.override"
                    class="ov"
                  >
                    <option :value="-1">
                      不改判（{{ DEFECT_LABELS[r.class_id] }}）
                    </option>
                    <option
                      v-for="(lbl, i) in DEFECT_LABELS"
                      :key="i"
                      :value="i"
                    >
                      {{ lbl }}
                    </option>
                  </select>
                </td>
                <td>{{ (r.confidence * 100).toFixed(0) }}%</td>
                <td>
                  <span :class="r.uncertainty >= 0.5 ? 'warn' : ''">{{ (r.uncertainty * 100).toFixed(0) }}%</span>
                </td>
                <td>
                  <span
                    v-if="r.need_review"
                    class="need"
                  >待复核</span>
                  <span
                    v-else-if="r.reviewed"
                    class="ok"
                  >已复核</span>
                  <span v-else>—</span>
                </td>
              </tr>
            </tbody>
          </table>
          <div class="m-actions">
            <button
              class="btn primary"
              type="button"
              :disabled="!canExport"
              @click="confirmExport"
            >
              {{ exporting ? "回流中…" : `确认回流（${selectedCount}）` }}
            </button>
            <button
              class="btn ghost"
              type="button"
              @click="closeExport"
            >
              关闭
            </button>
          </div>
          <div
            v-if="exportResult"
            class="ok-msg"
          >
            ✓ 已回流训练池，当前共 <b>{{ exportResult.total_in_pool }}</b> 个样本（数据版本指纹
            <code>{{ exportResult.fingerprint ?? "—" }}</code>）
          </div>
          <div
            v-if="exportErr"
            class="err show"
          >
            ⚠ 回流失败：{{ exportErr }}
          </div>
        </template>
      </div>
    </div>

    <div
      v-if="verifyResult"
      class="sig-state"
      :class="verifyStateClass"
    >
      <template v-if="verifyResult.valid === true">
        ✓ 签名有效（签发者：{{ verifyResult.signer ?? "—" }}）
      </template>
      <template v-else-if="verifyResult.valid === false">
        ⚠ 签名无效或被篡改（{{ verifyResult.reason ?? "内容指纹不匹配" }}）
      </template>
      <template v-else>
        — 该报告未签署数字签名（{{ verifyResult.reason ?? "无签发记录" }}）
      </template>
    </div>
    <div
      v-if="verifyError"
      class="err show"
    >
      ⚠ 校验失败：{{ verifyError }}
    </div>

    <div class="sig">
      签字：____________　免责：本报告由系统自动生成，仅供质量追溯参考；需复核或不可评片时，结论不作为正式评片依据。
    </div>
  </div>
</template>

<style scoped>
.sig-state {
  margin-top: 12px;
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 13px;
}
.sig-state.ok {
  background: rgba(42, 143, 74, 0.12);
  color: #1e7a3d;
}
.sig-state.bad {
  background: rgba(176, 48, 48, 0.12);
  color: #b03030;
}
.sig-state.na {
  background: rgba(120, 140, 180, 0.12);
  color: #44577a;
}

/* ── 主动学习回流弹窗 ── */
.modal-mask {
  position: fixed;
  inset: 0;
  background: rgba(8, 12, 22, 0.66);
  display: grid;
  place-items: center;
  z-index: 50;
  padding: 24px;
}
.modal {
  width: min(720px, 100%);
  max-height: 86vh;
  overflow: auto;
  background: #161b2c;
  border: 1px solid rgba(140, 160, 200, 0.22);
  border-radius: 12px;
  padding: 20px 22px;
  box-shadow: 0 18px 48px rgba(0, 0, 0, 0.45);
}
.m-title {
  font-size: 16px;
  margin: 0 0 10px;
  letter-spacing: 0.04em;
}
.exp {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  margin: 8px 0 14px;
}
.exp th,
.exp td {
  border-bottom: 1px solid rgba(140, 160, 200, 0.16);
  padding: 8px 10px;
  text-align: left;
}
.exp th {
  color: #9fb0d0;
  font-weight: 600;
}
.cid {
  opacity: 0.55;
  font-size: 11px;
}
.ov {
  padding: 4px 6px;
  border: 1px solid rgba(140, 160, 200, 0.3);
  border-radius: 6px;
  background: #0f1424;
  color: #dde6f5;
}
.warn {
  color: #e0a13c;
  font-weight: 600;
}
.ok {
  color: #1e9e57;
}
.need {
  color: #d06b3a;
}
.m-actions {
  display: flex;
  gap: 10px;
  margin-top: 6px;
}
.btn {
  padding: 8px 14px;
  border-radius: 8px;
  border: 1px solid rgba(140, 160, 200, 0.3);
  background: transparent;
  color: #dde6f5;
  cursor: pointer;
  font-size: 13px;
}
.btn.primary {
  background: #2a6df0;
  border-color: #2a6df0;
  color: #fff;
}
.btn.primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.btn.ghost:hover {
  background: rgba(140, 160, 200, 0.12);
}
.ok-msg {
  margin-top: 12px;
  padding: 8px 12px;
  border-radius: 8px;
  background: rgba(42, 143, 74, 0.12);
  color: #1e7a3d;
  font-size: 13px;
}
</style>
