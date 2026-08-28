<script setup lang="ts">
/**
 * 系统评价工作区（DB50/T 1807-2025）：人员资质管理 + 附录A 记录表装配与 PDF 下载。
 * 指标来自 CLI（python -m backend.evaluation.run_std_eval）产出的
 * data/eval/std_eval.json；本页负责资质录入、记录表生成与结果展示。
 */
import { onMounted, ref } from "vue";
import {
  createStdRecord,
  getStdPersonnel,
  putStdPersonnel,
  stdRecordPdfUrl,
} from "../services/api";
import type { StdPersonnel, StdRecordOut } from "../types/api";

const qualified = ref<boolean | null>(null);
const issues = ref<string[]>([]);
const evaluators = ref<StdPersonnel[]>([]);
const labelers = ref<StdPersonnel[]>([]);
const busy = ref(false);
const msg = ref<string | null>(null);
const err = ref<string | null>(null);

const recordName = ref("std_record");
const record = ref<StdRecordOut | null>(null);

const systemName = ref("承压设备射线检测缺陷自动识别系统");
const systemVersion = ref("");
const developer = ref("");
const filmKind = ref("RT");
const exposureLayout = ref("");
const weldForm = ref<"single" | "double">("single");
const weldMethod = ref<"manual" | "auto">("manual");

// 新增人员表单
const pName = ref("");
const pCert = ref("");
const pRole = ref<"evaluator" | "labeler">("labeler");
const pCertNo = ref("");
const pValid = ref("");

async function reload(): Promise<void> {
  try {
    const out = await getStdPersonnel();
    qualified.value = out.qualified;
    issues.value = out.issues;
    evaluators.value = out.evaluators;
    labelers.value = out.labelers;
  } catch (e) {
    err.value = e instanceof Error ? e.message : String(e);
  }
}

onMounted(reload);

function collectPeople(): StdPersonnel[] {
  return [...evaluators.value, ...labelers.value].map((p) => ({
    name: p.name,
    cert_type: p.cert_type,
    role: p.role,
    cert_no: p.cert_no ?? "",
    valid_until: p.valid_until ?? "",
  }));
}

function addPerson(): void {
  err.value = null;
  if (!pName.value.trim() || !pCert.value.trim()) {
    err.value = "姓名与持证类型必填（如 张三 / RT(D)-II）。";
    return;
  }
  const person: StdPersonnel = {
    name: pName.value.trim(),
    cert_type: pCert.value.trim(),
    role: pRole.value,
    cert_no: pCertNo.value.trim(),
    valid_until: pValid.value.trim(),
  };
  if (pRole.value === "evaluator") evaluators.value = [...evaluators.value, person];
  else labelers.value = [...labelers.value, person];
  pName.value = "";
  pCert.value = "";
  pCertNo.value = "";
  pValid.value = "";
}

function removePerson(role: "evaluator" | "labeler", idx: number): void {
  if (role === "evaluator") evaluators.value = evaluators.value.filter((_, i) => i !== idx);
  else labelers.value = labelers.value.filter((_, i) => i !== idx);
}

async function savePersonnel(): Promise<void> {
  busy.value = true;
  err.value = null;
  msg.value = null;
  try {
    const out = await putStdPersonnel(collectPeople());
    qualified.value = out.qualified;
    issues.value = out.issues;
    msg.value = out.qualified ? "资质校验通过。" : "已保存，但存在资质问题（见下方列表）。";
  } catch (e) {
    err.value = e instanceof Error ? e.message : String(e);
  } finally {
    busy.value = false;
  }
}

async function buildRecord(): Promise<void> {
  busy.value = true;
  err.value = null;
  msg.value = null;
  try {
    record.value = await createStdRecord({
      system_name: systemName.value.trim() || "承压设备射线检测缺陷自动识别系统",
      system_version: systemVersion.value.trim(),
      developer: developer.value.trim(),
      film_kind: filmKind.value,
      exposure_layout: exposureLayout.value.trim(),
      weld_form: weldForm.value,
      weld_method: weldMethod.value,
      record_name: recordName.value.trim() || "std_record",
    });
    msg.value = record.value.grading.official
      ? "记录表已生成（正式分级结论）。"
      : "记录表已生成（资质或 FRR 未满足，仅参考值）。";
  } catch (e) {
    err.value = e instanceof Error ? e.message : String(e);
  } finally {
    busy.value = false;
  }
}

const pct = (v: number): string => `${(v * 100).toFixed(2)}%`;
</script>

<template>
  <div>
    <h1 class="title-zine">系统评价（DB50/T 1807-2025）</h1>
    <div class="lede">
      先用命令行产出指标：python -m backend.evaluation.run_std_eval --img-dir … --label-dir … --model …；
      本页录入资质并生成附录A 记录表。
    </div>

    <div class="section-h">
      <span class="no">1</span>评价/标注人员资质
    </div>
    <div class="panel">
      <div class="row">
        <div class="field">
          <label for="spn">姓名</label>
          <input
            id="spn"
            v-model="pName"
            placeholder="张三"
          >
        </div>
        <div class="field">
          <label for="spc">持证类型</label>
          <input
            id="spc"
            v-model="pCert"
            placeholder="RT(D)-II / RT-Ⅱ"
          >
        </div>
        <div class="field">
          <label for="spr">岗位</label>
          <select
            id="spr"
            v-model="pRole"
          >
            <option value="evaluator">评价人员</option>
            <option value="labeler">标注人员</option>
          </select>
        </div>
        <div class="field">
          <label for="spv">有效期（可选）</label>
          <input
            id="spv"
            v-model="pValid"
            placeholder="2099-12-31"
          >
        </div>
        <button
          class="btn"
          type="button"
          @click="addPerson"
        >
          添加
        </button>
      </div>
      <ul
        v-if="evaluators.length || labelers.length"
        class="plist"
      >
        <li
          v-for="(p, i) in evaluators"
          :key="`e${i}`"
        >
          评价人员：{{ p.name }}（{{ p.cert_type }}{{ p.valid_until ? "，至 " + p.valid_until : "" }}）
          <a
            href="#"
            @click.prevent="removePerson('evaluator', i)"
          >移除</a>
        </li>
        <li
          v-for="(p, i) in labelers"
          :key="`l${i}`"
        >
          标注人员：{{ p.name }}（{{ p.cert_type }}）
          <a
            href="#"
            @click.prevent="removePerson('labeler', i)"
          >移除</a>
        </li>
      </ul>
      <div
        v-if="qualified === false"
        class="err show"
      >
        资质问题：{{ issues.join("；") }}
      </div>
      <button
        class="btn"
        type="button"
        :disabled="busy"
        @click="savePersonnel"
      >
        保存资质
      </button>
      <span
        v-if="qualified === true"
        class="ok show"
      >校验通过</span>
    </div>

    <div class="section-h">
      <span class="no">2</span>记录表信息
    </div>
    <div class="panel">
      <div class="row">
        <div class="field grow">
          <label for="sen">系统名称</label>
          <input
            id="sen"
            v-model="systemName"
          >
        </div>
        <div class="field">
          <label for="sev">系统版本</label>
          <input
            id="sev"
            v-model="systemVersion"
            placeholder="1.0.0"
          >
        </div>
        <div class="field">
          <label for="sed">开发单位</label>
          <input
            id="sed"
            v-model="developer"
          >
        </div>
      </div>
      <div class="row">
        <div class="field">
          <label for="sek">检测方法</label>
          <select
            id="sek"
            v-model="filmKind"
          >
            <option>RT</option>
            <option>DR</option>
            <option>CR</option>
          </select>
        </div>
        <div class="field">
          <label for="sef">焊缝形式</label>
          <select
            id="sef"
            v-model="weldForm"
          >
            <option value="single">单面焊</option>
            <option value="double">双面焊</option>
          </select>
        </div>
        <div class="field">
          <label for="sem">焊接方法</label>
          <select
            id="sem"
            v-model="weldMethod"
          >
            <option value="manual">手工焊</option>
            <option value="auto">自动焊</option>
          </select>
        </div>
        <div class="field">
          <label for="ser">记录名</label>
          <input
            id="ser"
            v-model="recordName"
          >
        </div>
      </div>
      <button
        class="btn"
        type="button"
        :disabled="busy"
        @click="buildRecord"
      >
        生成记录表
      </button>
      <a
        v-if="record"
        class="btn"
        :href="stdRecordPdfUrl(recordName.trim() || 'std_record')"
        target="_blank"
      >
        下载 PDF
      </a>
    </div>

    <template v-if="record">
      <div class="section-h">
        <span class="no">3</span>评价结果
      </div>
      <div class="panel">
        <table class="dtable">
          <tbody>
            <tr>
              <th>KDR（重点关注）</th>
              <td>{{ pct(record.metrics.kdr) }}（严格口径 {{ pct(record.metrics.kdr_strict) }}）</td>
              <th>WDR（综合）</th>
              <td>{{ pct(record.metrics.wdr) }}（{{ pct(record.metrics.wdr_strict) }}）</td>
            </tr>
            <tr>
              <th>TDR（正检率）</th>
              <td>{{ pct(record.metrics.tdr) }}（{{ pct(record.metrics.tdr_strict) }}）</td>
              <th>FRR（底片误报率）</th>
              <td>{{ pct(record.metrics.frr) }}（{{ pct(record.metrics.frr_strict) }}）</td>
            </tr>
            <tr>
              <th>系统分级</th>
              <td colspan="3">
                {{ record.grading.level ?? "未定级" }}（标准口径 {{ record.grading.level_standard ?? "未定级" }} /
                严格口径 {{ record.grading.level_strict ?? "未定级" }}）
                <span v-if="!record.grading.official">（参考值）</span>
              </td>
            </tr>
            <tr>
              <th>风险分析</th>
              <td colspan="3">
                漏检 {{ record.risks.miss }}；误检 {{ record.risks.false_detect }}；误报
                {{ record.risks.false_report }}
              </td>
            </tr>
            <tr>
              <th>逐类指标</th>
              <td colspan="3">
                TDRn：{{ record.metrics.tdr_row }}<br>
                FDRn：{{ record.metrics.fdr_row }}<br>
                MDRn：{{ record.metrics.mdr_row }}<br>
                FRRn：{{ record.metrics.frr_row }}
              </td>
            </tr>
          </tbody>
        </table>
        <div
          v-if="record.grading.note"
          class="err show"
        >
          {{ record.grading.note }}
        </div>
      </div>
    </template>

    <div
      v-if="msg"
      class="ok show"
    >
      {{ msg }}
    </div>
    <div
      v-if="err"
      class="err show"
    >
      {{ err }}
    </div>
  </div>
</template>

<style scoped>
.panel {
  border: 1px solid #e2e2e2;
  border-radius: 6px;
  padding: 12px 14px;
  margin: 8px 0 18px;
}
.row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: flex-end;
}
.field {
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-size: 12px;
  color: #555;
}
.field input,
.field select {
  font-size: 13px;
  padding: 4px 6px;
}
.grow {
  flex: 1;
  min-width: 160px;
}
.plist {
  font-size: 12px;
  color: #444;
  margin: 8px 0;
  padding-left: 18px;
}
.plist a {
  color: #2c5aa0;
  margin-left: 6px;
}
.ok.show {
  color: #2c7a3d;
  font-size: 12px;
  margin-left: 10px;
}
.err.show {
  color: #b03030;
  font-size: 12px;
  margin: 8px 0;
}
.dtable {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
.dtable th,
.dtable td {
  border: 1px solid #d8d8d8;
  padding: 4px 8px;
  text-align: left;
  vertical-align: top;
}
.dtable th {
  background: #f7f7f7;
  width: 12em;
}
</style>
