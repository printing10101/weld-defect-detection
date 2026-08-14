<script setup lang="ts">
/**
 * 报告解读视图（DESIGN.md：结论先行 + 影像对比 + 操作建议）。
 * 数据诚实性：全部内容来自 ReportOut 真实字段 + 用户上传文件真实 objectURL：
 * - 级别/待复核/可评片/缺陷数：ReportOut；
 * - 送检原图：用户上传文件的真实 objectURL（无标注）；
 * - 标注对比图：后端仅通过 PDF 提供（report 端点无独立标注图 URL），
 *   故如实提供 PDF 下载入口，不伪造标注图；
 * - 判定依据/黑度等：ReportOut 不包含，前端不构造展示。
 */
import ResultBanner from "./ResultBanner.vue";
import ReviewPanel from "./ReviewPanel.vue";
import type { ReportOut } from "../types/api";

const props = defineProps<{
  result: ReportOut;
  sourceUrl: string | null;
  fileName: string | null;
}>();

const emit = defineEmits<{ archive: []; reset: [] }>();

function openPdf(): void {
  window.open(props.result.pdf_url, "_blank", "noopener");
}
</script>

<template>
  <div>
    <h1 class="title-zine" data-t="评片报告">评片报告</h1>
    <div class="lede">
      REPORT {{ result.report_id }} · IMAGE {{ result.image_id }} · 数据来自真实检测流水线
    </div>

    <ResultBanner :result="result" />

    <div class="section-h"><span class="no">影像</span>送检原始影像（来自你上传的文件）</div>
    <div class="compare">
      <div class="plate">
        <img v-if="sourceUrl" :src="sourceUrl" :alt="fileName ?? '上传影像'" />
        <span v-else class="ph">影像不可用</span>
        <div class="cap">{{ fileName ?? "上传文件" }} · 未标注</div>
      </div>
      <div class="plate">
        <a
          :href="result.pdf_url"
          target="_blank"
          rel="noopener"
          style="display: block; text-decoration: none; color: inherit"
        >
          <span class="ph" style="display: grid; place-items: center; min-height: 160px; color: var(--accent); font-size: 11px">
            检测标注影像见 PDF 报告 →
          </span>
          <div class="cap">打开 PDF/A 报告（含缺陷标注与明细）</div>
        </a>
      </div>
    </div>

    <div class="section-h"><span class="no">结果</span>本次检测结果</div>
    <div class="kv">
      <div class="k">报告编号</div><div class="v">{{ result.report_id }}</div>
      <div class="k">影像编号</div><div class="v">{{ result.image_id }}</div>
      <div class="k">缺陷数量</div><div class="v">{{ result.defect_count }} 处</div>
      <div class="k">可评片性</div><div class="v">{{ result.evaluable ? "可评片" : "不可评片（影像质量不达标）" }}</div>
      <div class="k">综合级别</div><div class="v">{{ result.joint_level ?? "（待复核/未输出）" }}</div>
    </div>

    <div v-if="result.need_review" class="section-h"><span class="no">复核</span>人工复核（M7 闭环）</div>
    <ReviewPanel v-if="result.need_review" :image-id="result.image_id" />

    <div class="section-h"><span class="no">建议</span>接下来可以做什么</div>
    <div class="acts">
      <button class="act" type="button" @click="openPdf">
        <div class="a">导出 PDF/A</div>
        <div class="d">下载归档合规报告（含缺陷标注、判定依据与明细）</div>
      </button>
      <button class="act" type="button" @click="emit('archive')">
        <div class="a">查看档案</div>
        <div class="d">在档案检索中查看该影像的归档记录与统计</div>
      </button>
      <button class="act" type="button" @click="emit('reset')">
        <div class="a">重新检测</div>
        <div class="d">返回上传步骤，提交下一份底片</div>
      </button>
    </div>

    <div class="sig">签字：____________　免责：本报告由系统自动生成，仅供质量追溯参考；需复核或不可评片时，结论不作为正式评片依据。</div>
  </div>
</template>
