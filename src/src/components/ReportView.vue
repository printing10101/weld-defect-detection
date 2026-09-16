<script setup lang="ts">
/**
 * 报告解读视图（设计稿：结论先行 + 样张首页预览 + 影像对比 + 操作建议）。
 * 数据诚实性：全部内容来自 ReportOut 真实字段 + 用户上传文件真实 objectURL：
 * - 级别/待复核/可评片/缺陷数：ReportOut；
 * - 首页预览：ReportOut.workpiece_no/weld_no/report_meta 等真实回显，
 *   与打印 PDF 同一数据源；留空栏 = 报告留空（供手工补填），不伪造；
 * - 送检原图：用户上传文件的真实 objectURL（无标注）；
 * - 标注对比图：后端仅通过 PDF 提供（report 端点无独立标注图 URL），
 *   故如实提供 PDF 下载入口，不伪造标注图。
 */
import { computed, ref } from "vue";
import { toErrorMessage } from "../utils/errorMessage";
import { activeExport, getReportDetections, getReportNarrative, verifyReport } from "../services/api";
import { useControlledPdf } from "../composables/useControlledPdf";
import ResultBanner from "./ResultBanner.vue";
import ReportNotices from "./ReportNotices.vue";
import ReviewPanel from "./ReviewPanel.vue";
import DispositionPanel from "./DispositionPanel.vue";
import PdfGateModal from "./PdfGateModal.vue";
import { DEFECT_CLASS_LABELS } from "../types/api";
import type {
  ActiveExportOut,
  ReportDetectionsOut,
  ReportMeta,
  ReportNarrativeOut,
  ReportOut,
  VerifyOut,
} from "../types/api";

/** 缺陷类别中文标签（统一常量，types/api.ts 为唯一事实源）。 */
const DEFECT_LABELS = DEFECT_CLASS_LABELS;

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

/* ── C-14 受控导出：逻辑下沉到 useControlledPdf（与批量结果行共用，
 *  含审批状态自动轮询）；本页只保留下载文件名的业务语义。 ── */
const pdfCtrl = useControlledPdf();

/* ── 本地大模型评片结论（按需生成，不落库）─────────────────────────
 * 后端状态语义：ok=有正文；disabled/unavailable/failed=未生成且带可读 reason。
 * 失败**不弹错误框**——大模型不可用不是评片错误，界面如实说明即可；
 * 传输层失败（后端不可达/超时）才是异常，单独记录后如实展示。 */
const narrative = ref<ReportNarrativeOut | null>(null);
const narrativeLoading = ref(false);
const narrativeError = ref<string | null>(null);

async function loadNarrative(): Promise<void> {
  if (narrativeLoading.value) return;
  narrativeLoading.value = true;
  narrativeError.value = null;
  try {
    narrative.value = await getReportNarrative(props.result.report_id);
  } catch (e) {
    narrativeError.value = toErrorMessage(e);
  } finally {
    narrativeLoading.value = false;
  }
}

/** 下载文件名：工件编号_报告编号，便于操作员归档识别（此前是裸 report_id）。 */
function exportPdf(): void {
  const r = props.result;
  const name = r.workpiece_no ? `${r.workpiece_no}_${r.report_id}` : r.report_id;
  void pdfCtrl.openPdf(r.report_id, name);
}

/* ── 报告首页样张预览：与打印 PDF 汇总表同源同款 ── */
const RT_ROMAN: Record<string, string> = { I: "Ⅰ", II: "Ⅱ", III: "Ⅲ", IV: "Ⅳ" };

function m(key: string): string {
  return (props.result.report_meta as ReportMeta | undefined)?.[key] ?? "";
}

function roman(level: string | null | undefined): string {
  const lv = (level ?? "").trim().toUpperCase();
  return RT_ROMAN[lv] ?? lv;
}

function gradeRank(level: string | null | undefined): number | null {
  const lv = (level ?? "").trim().toUpperCase();
  const rank: Record<string, number> = { I: 1, II: 2, III: 3, IV: 4, "Ⅰ": 1, "Ⅱ": 2, "Ⅲ": 3, "Ⅳ": 4 };
  return rank[lv] ?? null;
}

/** 最终评定结果统计：单张底片评为哪级记 1 张（无级别以 / 占位，与 PDF 同逻辑） */
const gradeRow = computed(() => {
  const lv = (props.result.joint_level ?? "").trim().toUpperCase();
  return {
    i: lv === "I" ? "1" : "/",
    ii: lv === "II" ? "1" : "/",
    iii: lv === "III" ? "1" : "/",
    iv: lv === "IV" ? "1" : "/",
    total: lv ? "1" : "/",
  };
});

/** 结论第 1 条（与 PDF _conclusion_flowables 同逻辑，屏上预览保持一致） */
const conclusionLines = computed<string[]>(() => {
  const r = props.result;
  if (r.photo_mode && r.joint_level) {
    // 翻拍降级：级别确实算出来了，但黑度/IQI 未经验证——结论必须自带
    // 翻拍语境的强提示，不能复用"合格/不合格"话术，也不能说成"不可评片"。
    return [
      `1、本片为翻拍影像（绝对黑度不可测，质量门禁按翻拍策略降级），不构成正式评定依据；` +
        `AI 预筛级别为${roman(r.joint_level)}级，仅供参考，须由持证人工复核确认后方可作为评定结论。`,
    ];
  }
  if (r.grade_preliminary && r.joint_level) {
    // 预筛级别：底片质量未达标，级别不具合规效力——结论必须自带强提示，
    // 不能复用"合格/不合格"话术，也不能说成"不可评片"（级别确实算出来了）。
    return [
      `1、本片影像质量校验未通过（IQI/黑度不达标），不构成正式评定依据；` +
        `AI 预筛级别为${roman(r.joint_level)}级，仅供参考，不得作为验收结论，` +
        "须由持证人依标准原文重新评定。",
    ];
  }
  if (!r.evaluable) {
    return ["1、影像质量校验未通过（IQI/黑度不达标），本片不可评片，需人工复核处理。"];
  }
  const std = r.standard_ref || "验收标准";
  if (r.joint_level) {
    const grade = gradeRank(r.joint_level);
    const accept = gradeRank(m("accept_level"));
    if (grade !== null && accept !== null) {
      const ok = grade <= accept;
      return [
        `1、本工件（${r.workpiece_no || "—"}）焊缝质量经检测，依据${std}评为${roman(r.joint_level)}级，` +
          `${ok ? "满足" : "未满足"}验收要求（合格级别${roman(m("accept_level"))}级），结果${ok ? "合格" : "不合格"}。`,
      ];
    }
    const verdict = grade !== null && grade <= 2 ? "合格" : "不合格";
    return [
      `1、本工件（${r.workpiece_no || "—"}）焊缝质量经检测，依据${std} 评为${roman(r.joint_level)}级，结果${verdict}。`,
    ];
  }
  return ["1、本片暂无法自动评级（置信度不足或未标定），需人工评定。"];
});

/** 评片日期（报告即时生成，取当日；打印件以 PDF 内日期为准） */
const todayDot = computed(() => {
  const d = new Date();
  return `${d.getFullYear()}.${d.getMonth() + 1}.${d.getDate()}`;
});

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
    detsErr.value = toErrorMessage(e);
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
    exportErr.value = toErrorMessage(e);
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
    verifyError.value = toErrorMessage(e);
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
      射线检测报告
    </h1>
    <div class="lede">
      报告编号 {{ result.report_id }} · 影像编号 {{ result.image_id }}<template v-if="result.standard_ref">
        · {{ result.standard_ref }}
      </template>
    </div>

    <ResultBanner :result="result" />
    <ReportNotices :result="result" />
    <DispositionPanel :result="result" />

    <div class="section-h">
      <span class="no">报告</span>报告首页预览（与打印版式一致，空白栏以手工补填为准）
    </div>
    <div class="rt-wrap">
      <div class="rt-no">
        NO:{{ result.report_id }}
      </div>
      <table class="rt">
        <tbody>
          <tr>
            <td
              class="lbl"
              colspan="3"
            >
              委托单位
            </td>
            <td colspan="7">
              {{ m("client_unit") }}
            </td>
            <td
              class="lbl"
              colspan="7"
            >
              工程类别/检测时机
            </td>
            <td colspan="4">
              {{ m("project_category") }}
            </td>
          </tr>
          <tr>
            <td
              class="lbl"
              colspan="3"
            >
              工程名称
            </td>
            <td colspan="7">
              {{ m("project_name") }}
            </td>
            <td
              class="lbl"
              colspan="7"
            >
              检测地址
            </td>
            <td colspan="4">
              {{ m("test_address") }}
            </td>
          </tr>
          <tr>
            <td
              class="sec"
              rowspan="3"
            >
              工件概况
            </td>
            <td
              class="lbl"
              colspan="3"
            >
              工件名称
            </td>
            <td colspan="3">
              {{ result.workpiece_no ?? "" }}
            </td>
            <td
              class="lbl"
              colspan="4"
            >
              材　　质
            </td>
            <td colspan="5">
              {{ m("material") }}
            </td>
            <td
              class="lbl"
              colspan="3"
            >
              坡口形式
            </td>
            <td colspan="2">
              {{ m("groove_type") }}
            </td>
          </tr>
          <tr>
            <td
              class="lbl"
              colspan="3"
            >
              工件编号
            </td>
            <td colspan="3">
              {{ m("part_no") }}
            </td>
            <td
              class="lbl"
              colspan="4"
            >
              规　　格
            </td>
            <td colspan="5">
              —
            </td>
            <td
              class="lbl"
              colspan="3"
            >
              表面状况
            </td>
            <td colspan="2">
              {{ m("surface_status") }}
            </td>
          </tr>
          <tr>
            <td
              class="lbl"
              colspan="3"
            >
              检测部位
            </td>
            <td colspan="3">
              焊接接头
            </td>
            <td
              class="lbl"
              colspan="4"
            >
              焊接方式
            </td>
            <td colspan="5">
              {{ m("weld_process") }}
            </td>
            <td
              class="lbl"
              colspan="3"
            >
              热处理状态
            </td>
            <td colspan="2">
              {{ m("heat_treatment") }}
            </td>
          </tr>
          <tr>
            <td
              class="sec"
              rowspan="3"
            >
              技术要求
            </td>
            <td
              class="lbl"
              colspan="3"
            >
              验收标准
            </td>
            <td colspan="3">
              {{ result.standard_ref ?? "" }}
            </td>
            <td
              class="lbl"
              colspan="4"
            >
              检测标准
            </td>
            <td colspan="5">
              {{ result.standard_ref ?? "" }}
            </td>
            <td
              class="lbl"
              colspan="3"
            >
              检测比例
            </td>
            <td colspan="2">
              100%
            </td>
          </tr>
          <tr>
            <td
              class="lbl"
              colspan="3"
            >
              检测技术等级
            </td>
            <td colspan="3">
              {{ m("tech_level") }}
            </td>
            <td
              class="lbl"
              colspan="4"
            >
              合格级别
            </td>
            <td colspan="5">
              {{ m("accept_level") }}
            </td>
            <td
              class="lbl"
              colspan="3"
            >
              原始记录编号
            </td>
            <td colspan="2">
              {{ m("record_no") }}
            </td>
          </tr>
          <tr>
            <td
              class="lbl"
              colspan="3"
            >
              黑度范围
            </td>
            <td colspan="3">
              —
            </td>
            <td
              class="lbl"
              colspan="4"
            >
              应识别丝号
            </td>
            <td colspan="5">
              —
            </td>
            <td
              class="lbl"
              colspan="3"
            >
              散射线控制
            </td>
            <td colspan="2">
              {{ m("scatter_control") }}
            </td>
          </tr>
          <tr>
            <td
              class="sec"
              rowspan="7"
            >
              检测器材及工艺参数
            </td>
            <td
              class="lbl"
              colspan="3"
            >
              源种类
            </td>
            <td colspan="3">
              {{ m("source_kind") }}
            </td>
            <td
              class="lbl"
              colspan="4"
            >
              设备型号/编号
            </td>
            <td colspan="5">
              {{ m("device_no") }}
            </td>
            <td
              class="lbl"
              colspan="3"
            >
              焦点尺寸
            </td>
            <td colspan="2">
              {{ m("focus_size") }}
            </td>
          </tr>
          <tr>
            <td
              class="lbl"
              colspan="3"
            >
              胶片型号
            </td>
            <td colspan="3">
              {{ m("film_model") }}
            </td>
            <td
              class="lbl"
              colspan="4"
            >
              胶片规格
            </td>
            <td colspan="5">
              {{ m("film_size") }}
            </td>
            <td
              class="lbl"
              colspan="3"
            >
              胶片分类等级
            </td>
            <td colspan="2">
              {{ m("film_class") }}
            </td>
          </tr>
          <tr>
            <td
              class="lbl"
              colspan="3"
            >
              增感方式
            </td>
            <td colspan="3">
              {{ m("screen_way") }}
            </td>
            <td
              class="lbl"
              colspan="4"
            >
              像质计型号
            </td>
            <td colspan="5">
              —
            </td>
            <td
              class="lbl"
              colspan="3"
            >
              像质计摆放
            </td>
            <td colspan="2">
              {{ m("iqi_position") }}
            </td>
          </tr>
          <tr>
            <td
              class="lbl"
              colspan="3"
            >
              前屏/后屏
            </td>
            <td colspan="3">
              {{ m("screens") }}
            </td>
            <td
              class="lbl"
              colspan="4"
            >
              透照方式
            </td>
            <td colspan="5">
              {{ m("technique") }}
            </td>
            <td
              class="lbl"
              colspan="3"
            >
              透照厚度
            </td>
            <td colspan="2">
              —
            </td>
          </tr>
          <tr>
            <td
              class="lbl"
              colspan="3"
            >
              F（焦距）
            </td>
            <td colspan="3">
              {{ m("focus_distance") }}
            </td>
            <td
              class="lbl"
              colspan="4"
            >
              f（源至工件）
            </td>
            <td colspan="5">
              {{ m("source_distance") }}
            </td>
            <td
              class="lbl"
              colspan="3"
            >
              冲洗条件
            </td>
            <td colspan="2">
              {{ m("develop_method") }}
            </td>
          </tr>
          <tr>
            <td
              class="lbl"
              colspan="3"
            >
              b（工件至胶片）
            </td>
            <td colspan="3">
              {{ m("film_distance") }}
            </td>
            <td
              class="lbl"
              colspan="4"
            >
              管电压
            </td>
            <td colspan="5">
              {{ m("tube_voltage") }}
            </td>
            <td
              class="lbl"
              colspan="3"
            >
              显影液配方
            </td>
            <td colspan="2">
              {{ m("developer") }}
            </td>
          </tr>
          <tr>
            <td
              class="lbl"
              colspan="3"
            >
              管电流
            </td>
            <td colspan="3">
              {{ m("tube_current") }}
            </td>
            <td
              class="lbl"
              colspan="4"
            >
              曝光时间
            </td>
            <td colspan="5">
              {{ m("exposure_time") }}
            </td>
            <td
              class="lbl"
              colspan="3"
            >
              洗片温度
            </td>
            <td colspan="2">
              {{ m("develop_temp") }}
            </td>
          </tr>
          <tr>
            <td
              class="sec"
              rowspan="3"
            >
              检测情况
            </td>
            <td
              class="lbl"
              colspan="3"
            >
              焊缝总数
            </td>
            <td colspan="3">
              1道
            </td>
            <td
              class="sec2"
              rowspan="3"
            >
              最终评定结果
            </td>
            <td
              class="lbl"
              colspan="3"
              rowspan="2"
            >
              Ｉ级（张）
            </td>
            <td
              class="lbl"
              colspan="2"
              rowspan="2"
            >
              Ⅱ级（张）
            </td>
            <td
              class="lbl"
              rowspan="2"
            >
              Ⅲ级（张）
            </td>
            <td
              class="lbl"
              colspan="4"
              rowspan="2"
            >
              Ⅳ级（张）
            </td>
            <td
              class="lbl"
              rowspan="2"
            >
              总计（张）
            </td>
            <td
              class="lbl"
              rowspan="2"
            >
              返修数量（张）
            </td>
            <td
              class="lbl"
              rowspan="2"
            >
              最高返修次数（次）
            </td>
          </tr>
          <tr>
            <td
              class="lbl"
              colspan="3"
            >
              检测数量
            </td>
            <td colspan="3">
              1道
            </td>
          </tr>
          <tr>
            <td
              class="lbl"
              colspan="3"
            >
              检测比例
            </td>
            <td colspan="3">
              100%
            </td>
            <td colspan="3">
              {{ gradeRow.i }}
            </td>
            <td colspan="2">
              {{ gradeRow.ii }}
            </td>
            <td>
              {{ gradeRow.iii }}
            </td>
            <td colspan="4">
              {{ gradeRow.iv }}
            </td>
            <td>
              {{ gradeRow.total }}
            </td>
            <td>
              /
            </td>
            <td>
              /
            </td>
          </tr>
          <tr>
            <td
              class="concl"
              colspan="21"
            >
              <div class="c-head">
                检测结论及说明：
              </div>
              <div
                v-for="line in conclusionLines"
                :key="line"
                class="c-line"
              >
                {{ line }}
              </div>
              <div class="c-line">
                2、检测位置，返修部位，底片评定情况见射线检测位置示意图和底片评定表。
              </div>
              <div class="c-line">
                3、缺陷代号
              </div>
              <div class="c-line">
                A裂纹、B未焊透、C未熔合、D圆形缺陷（气孔、夹渣、夹钨、夹铜等）、E条形缺陷、F内凹、G咬边
              </div>
              <div class="c-line">
                4、本报告为AI辅助评定，级别须经责任工程师复核签核后方可采信。
              </div>
            </td>
          </tr>
          <tr>
            <td
              class="lbl"
              colspan="2"
            >
              检　测
            </td>
            <td colspan="3">
              {{ result.signer ?? "" }}
            </td>
            <td class="lbl">
              资　格
            </td>
            <td colspan="3">
              —
            </td>
            <td
              class="lbl"
              colspan="3"
            >
              日　期
            </td>
            <td colspan="3">
              {{ todayDot }}
            </td>
            <td
              class="seal"
              colspan="6"
              rowspan="2"
            >
              <div class="seal-t">
                检测单位检测专用章
              </div>
              <div class="seal-d">
                日期：{{ todayDot }}
              </div>
            </td>
          </tr>
          <tr>
            <td
              class="lbl"
              colspan="2"
            >
              审　核
            </td>
            <td colspan="3" />
            <td class="lbl">
              资　格
            </td>
            <td colspan="3">
              —
            </td>
            <td
              class="lbl"
              colspan="3"
            >
              日　期
            </td>
            <td colspan="3">
              {{ todayDot }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="section-h">
      <span class="no">影像</span>送检原始底片（用户导入原图，未经标注）
    </div>
    <div class="compare">
      <div class="plate">
        <img
          v-if="sourceUrl"
          :src="sourceUrl"
          :alt="fileName ?? '导入影像'"
        >
        <span
          v-else
          class="ph"
        >影像不可用</span>
        <div class="cap">
          {{ fileName ?? "导入文件" }} · 未标注
        </div>
      </div>
      <div class="plate">
        <button
          type="button"
          class="plate-btn"
          :disabled="pdfCtrl.gateBusy.value"
          style="display: block; width: 100%; border: 0; background: transparent; padding: 0; cursor: pointer; color: inherit; font-family: inherit"
          title="打开 PDF/A 归档报告（经受控导出通道）"
          @click="exportPdf"
        >
          <span
            class="ph"
            style="display: grid; place-items: center; min-height: 160px; color: var(--accent); font-size: 11px"
          >
            缺陷标注影像详见 PDF 报告 →
          </span>
          <div class="cap">
            打开 PDF/A 归档报告（含缺陷标注与检出明细）
          </div>
        </button>
      </div>
    </div>

    <div class="section-h">
      <span class="no">结果</span>本次评定结果
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
        缺陷检出数
      </div><div class="v">
        {{ result.defect_count }} 处
      </div>
      <div class="k">
        评片质量判定
      </div><div class="v">
        {{ result.photo_mode
          ? "翻拍影像（质量门禁降级，须人工复核）"
          : result.evaluable
            ? "底片质量合格，可评定"
            : "不可评片（底片质量不满足评定要求）" }}
      </div>
      <div class="k">
        质量级别（综合评定）
      </div><div class="v">
        {{ result.joint_level ?? "（待复核/未输出）" }}
        <span
          v-if="result.grade_preliminary"
          class="prelim-tag"
          title="底片质量未达标，该级别仅由算法预筛得出，不构成验收依据"
        >AI 预筛·非正式级别</span>
      </div>
    </div>

    <div class="section-h">
      <span class="no">AI</span>评片结论（本地大模型辅助撰述）
    </div>
    <div class="narr">
      <div
        v-if="narrativeLoading"
        class="narr-state"
      >
        本地大模型生成中…
      </div>
      <template v-else-if="narrative">
        <p
          v-if="narrative.status === 'ok'"
          class="narr-text"
        >
          {{ narrative.text }}
        </p>
        <p
          v-else
          class="narr-state"
        >
          AI 结论不可用：{{ narrative.reason }}
        </p>
        <div
          v-if="narrative.status === 'ok'"
          class="narr-meta"
        >
          模型 {{ narrative.model || "—" }} · 生成耗时 {{ (narrative.elapsed_ms / 1000).toFixed(1) }}s
        </div>
        <p class="narr-disc">
          {{ narrative.disclaimer }}
        </p>
      </template>
      <template v-else>
        <p
          v-if="narrativeError"
          class="narr-state"
        >
          生成失败：{{ narrativeError }}
        </p>
        <button
          class="act narr-btn"
          type="button"
          @click="loadNarrative"
        >
          <div class="a">
            {{ narrativeError ? "重试生成评片结论" : "生成评片结论" }}
          </div>
          <div class="d">
            由本机部署的本地大模型把检测结果与门禁结论整理成可读结论（不参与评级）
          </div>
        </button>
      </template>
    </div>

    <div
      v-if="result.need_review"
      class="section-h"
    >
      <span class="no">复核</span>人工复核（初评 / 复评 / 仲裁）
    </div>
    <ReviewPanel
      v-if="result.need_review"
      :image-id="result.image_id"
      :report-id="result.report_id"
    />

    <div class="section-h">
      <span class="no">处置</span>后续操作
    </div>
    <div class="acts">
      <button
        class="act"
        type="button"
        @click="exportPdf"
      >
        <div class="a">
          导出 PDF/A 报告
        </div>
        <div class="d">
          下载符合归档要求的检测报告（含缺陷标注、判定依据与检出明细）
        </div>
      </button>
      <button
        class="act"
        type="button"
        @click="emit('archive')"
      >
        <div class="a">
          查阅检测档案
        </div>
        <div class="d">
          在检测档案中查看该影像的归档记录与统计信息
        </div>
      </button>
      <button
        class="act"
        type="button"
        @click="emit('reset')"
      >
        <div class="a">
          新建评定任务
        </div>
        <div class="d">
          返回底片导入步骤，提交下一份待检底片
        </div>
      </button>
      <button
        class="act"
        type="button"
        :disabled="verifying"
        @click="onVerify"
      >
        <div class="a">
          {{ verifying ? "校验中…" : "校验报告数字签名" }}
        </div>
        <div class="d">
          比对报告内容指纹与签发记录，验证报告完整性与防篡改性（§7.2）
        </div>
      </button>
      <button
        class="act"
        type="button"
        :disabled="result.defect_count === 0"
        :title="result.defect_count === 0 ? '本片未检出缺陷，没有可回流的样本' : undefined"
        @click="openExport"
      >
        <div class="a">
          样本回流（主动学习）
        </div>
        <div class="d">
          {{
            result.defect_count === 0
              ? "本片未检出缺陷，无可回流样本（该按钮不可用）"
              : "经人工复核确认的缺陷标注回流至主动学习训练样本库（§5.5 持续学习闭环）"
          }}
        </div>
      </button>
    </div>

    <div
      v-if="exportOpen"
      class="modal-mask"
      @click.self="closeExport"
    >
      <div
        class="modal"
        role="dialog"
        aria-modal="true"
        aria-label="主动学习样本回流"
      >
        <h3 class="m-title">
          主动学习样本回流 · 人工确认
        </h3>
        <p
          v-if="loadingDets"
          class="hint"
        >
          正在载入缺陷检出明细…
        </p>
        <p
          v-else-if="detsErr"
          class="err show"
        >
          ⚠ 载入失败：{{ detsErr }}
        </p>
        <template v-else-if="dets">
          <p class="stat">
            影像 <b>{{ dets.image_stem }}</b> · {{ dets.image_w }}×{{ dets.image_h }}px ·
            共检出 {{ dets.defects.length }} 处缺陷。请勾选需回流的样本，必要时对类别进行人工改判后确认。
          </p>
          <table class="exp">
            <thead>
              <tr>
                <th>回流</th>
                <th>检出类别</th>
                <th>人工改判</th>
                <th>置信度</th>
                <th>不确定度</th>
                <th>复核状态</th>
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
                      维持原判（{{ DEFECT_LABELS[r.class_id] }}）
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
              {{ exporting ? "回流中…" : `确认回流（${selectedCount} 项）` }}
            </button>
            <button
              class="btn ghost"
              type="button"
              @click="closeExport"
            >
              关闭
            </button>
          </div>
          <p class="threshold-legend">
            判读口径：不确定度 ≥ <b>50%</b> 的检出（橙色）表示模型对该判定把握较低，
            建议优先人工复核后再回流；置信度为该缺陷框的检出把握，仅作参考，不构成级别结论。
          </p>
          <div
            v-if="exportResult"
            class="ok-msg"
          >
            ✓ 已回流至主动学习训练样本库，当前共 <b>{{ exportResult.total_in_pool }}</b> 个样本（数据集版本指纹
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
        ✓ 数字签名校验通过（签发者：{{ verifyResult.signer ?? "—" }}）
      </template>
      <template v-else-if="verifyResult.valid === false">
        ⚠ 数字签名无效或报告内容已被篡改（{{ verifyResult.reason ?? "内容指纹不匹配" }}）
      </template>
      <template v-else>
        — 本报告未附数字签名（{{ verifyResult.reason ?? "无签发记录" }}）
      </template>
    </div>
    <div
      v-if="verifyError"
      class="err show"
    >
      ⚠ 签名校验请求失败：{{ verifyError }}
    </div>

    <div class="sig">
      评片人签署：____________　声明：本报告由系统自动生成，仅供质量追溯参考；状态为「待人工复核」或「不可评片」时，不作为正式评片结论依据。
    </div>

    <!-- C-14 受控导出面板：申请 → 保密员审批（自动轮询）→ 领一次性令牌 → 下载 -->
    <PdfGateModal :ctrl="pdfCtrl" />
  </div>
</template>

<style scoped>
/* ── 预筛级别标识：必须显眼——级别不具合规效力，不得被误读为正式结论 ── */
.prelim-tag {
  display: inline-block;
  margin-left: 8px;
  padding: 1px 8px;
  border-radius: 999px;
  font-size: 11px;
  background: rgba(154, 103, 0, 0.12);
  color: var(--amber);
  border: 1px solid rgba(154, 103, 0, 0.42);
  white-space: nowrap;
}

/* ── 本地大模型评片结论（辅助撰述，不参与评级） ── */
.narr {
  margin-top: 10px;
  padding: 12px 14px;
  background: var(--panel-2);
  border: 1px solid var(--line-soft);
}
.narr-text {
  margin: 0 0 10px;
  font-size: 12px;
  line-height: 1.85;
  white-space: pre-wrap;
  color: var(--ink);
}
.narr-state {
  margin: 0 0 10px;
  font-size: 12px;
  line-height: 1.7;
  color: var(--ink-soft);
}
.narr-meta {
  margin-bottom: 8px;
  font-size: 11px;
  color: var(--ink-faint);
}
.narr-disc {
  margin: 0;
  padding-top: 8px;
  border-top: 1px dashed var(--line);
  font-size: 11px;
  line-height: 1.65;
  color: var(--ink-faint);
}
.narr-btn {
  width: 100%;
}

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

/* ── 主动学习回流弹窗（受控导出面板复用同一 mask/modal 样式） ── */
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
.threshold-legend {
  margin: 10px 0 0;
  font-size: 12px;
  line-height: 1.7;
  color: #9fb0d0;
}
</style>
