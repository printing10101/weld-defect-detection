<script setup lang="ts">
/**
 * 系统评价工作区（DB50/T 1807-2025）：人员资质管理 + 附录A 记录表装配与 PDF 下载。
 * 指标来自 CLI（python -m backend.evaluation.run_std_eval）产出的
 * data/eval/std_eval.json；本页负责资质录入、记录表生成与结果展示。
 * 另含评价历史档案与等级曲线（E-15：GET /std-eval/history，版本-指标随时间）。
 */
import { computed, onMounted, ref } from "vue";
import { toErrorMessage } from "../utils/errorMessage";
import {
  createStdRecord,
  getStdEvalHistory,
  getStdPersonnel,
  putStdPersonnel,
  stdRecordPdfUrl,
} from "../services/api";
import type { StdEvalHistoryItem, StdPersonnel, StdRecordOut } from "../types/api";

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
    err.value = toErrorMessage(e);
  }
}

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
    err.value = toErrorMessage(e);
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
    err.value = toErrorMessage(e);
  } finally {
    busy.value = false;
  }
}

const pct = (v: number): string => `${(v * 100).toFixed(2)}%`;

// ---------------------------------------------------------------------------
// 评价历史与等级曲线（E-15）：GET /std-eval/history（降序）→ 曲线按时间升序。
// 无依赖 SVG 折线：TDR/WDR/FRR 三条百分比线 + 系统分级（L1-L4）标注点。
// ---------------------------------------------------------------------------

const history = ref<StdEvalHistoryItem[]>([]);
const historyErr = ref<string | null>(null);

async function loadHistory(): Promise<void> {
  try {
    const out = await getStdEvalHistory();
    history.value = [...out.items].reverse();
  } catch (e) {
    // 历史档案缺失/为空不阻断本页主功能，仅在曲线区提示
    historyErr.value = toErrorMessage(e);
  }
}
onMounted(() => {
  reload();
  loadHistory();
});

/** 曲线点：携带在历史时间线中的原始下标（x 轴统一按全量历史等距展开）。 */
const curvePoints = computed<{ item: StdEvalHistoryItem; idx: number }[]>(() =>
  history.value
    .map((item, idx) => ({ item, idx }))
    .filter(({ item }) => item.tdr != null || item.wdr != null || item.frr != null),
);

const CHART_W = 720;
const CHART_H = 240;
const PAD_L = 44;
const PAD_R = 16;
const PAD_T = 28;
const PAD_B = 46;

function xAt(idx: number, n: number): number {
  if (n <= 1) return PAD_L;
  return PAD_L + (idx * (CHART_W - PAD_L - PAD_R)) / (n - 1);
}

function yAt(v: number): number {
  return PAD_T + (1 - v) * (CHART_H - PAD_T - PAD_B);
}

/** 单指标折线 path（跳过 null 点：分段连续，避免断点被连线到 0）。 */
function seriesPath(key: "tdr" | "wdr" | "frr"): string {
  const n = history.value.length;
  if (n === 0) return "";
  let d = "";
  let pen = false;
  curvePoints.value.forEach(({ item, idx }) => {
    const v = item[key];
    if (v == null) {
      pen = false;
      return;
    }
    d += `${pen ? "L" : "M"}${xAt(idx, n).toFixed(1)},${yAt(v).toFixed(1)} `;
    pen = true;
  });
  return d.trim();
}

const tdrPath = computed(() => seriesPath("tdr"));
const wdrPath = computed(() => seriesPath("wdr"));
const frrPath = computed(() => seriesPath("frr"));

/** 网格线：0/25/50/75/100%。 */
const gridLines = computed(() =>
  [0, 0.25, 0.5, 0.75, 1].map((v) => ({ v, y: yAt(v) })),
);

/** X 轴刻度标签（时间，自动抽稀避免重叠）。 */
const xLabels = computed(() => {
  const n = history.value.length;
  const step = Math.max(1, Math.ceil(n / 8));
  return history.value
    .map((p, i) => ({ i, text: (p.evaluated_at ?? "").replace("T", " ").slice(0, 16) }))
    .filter((l) => l.i % step === 0 || l.i === n - 1);
});

/** 分级标注点（L1-L4）：有 level 的记录标在图顶。 */
const levelMarks = computed(() => {
  const n = history.value.length;
  return history.value
    .map((h, i) => ({ i, level: h.level ?? "" }))
    .filter((m) => m.level)
    .map((m) => ({ ...m, x: xAt(m.i, n) }));
});
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

    <div class="section-h">
      <span class="no">4</span>评价历史与等级曲线（E-15）
    </div>
    <div class="panel">
      <div
        v-if="historyErr"
        class="err show"
      >
        历史档案读取失败：{{ historyErr }}
      </div>
      <div
        v-else-if="history.length === 0"
        class="hint"
      >
        暂无评价历史：先用 CLI 产出指标或生成记录表，历史会自动聚合到这里。
      </div>
      <template v-else>
        <svg
          class="curve"
          :viewBox="`0 0 ${CHART_W} ${CHART_H}`"
          role="img"
          aria-label="评价等级曲线"
        >
          <!-- 网格与 Y 轴刻度 -->
          <g
            v-for="g in gridLines"
            :key="`g${g.v}`"
          >
            <line
              :x1="PAD_L"
              :y1="g.y"
              :x2="CHART_W - PAD_R"
              :y2="g.y"
              class="grid"
            />
            <text
              :x="PAD_L - 6"
              :y="g.y + 4"
              class="axis"
              text-anchor="end"
            >{{ Math.round(g.v * 100) }}%</text>
          </g>
          <!-- 指标折线 -->
          <path
            v-if="tdrPath"
            :d="tdrPath"
            class="line line-tdr"
          />
          <path
            v-if="wdrPath"
            :d="wdrPath"
            class="line line-wdr"
          />
          <path
            v-if="frrPath"
            :d="frrPath"
            class="line line-frr"
          />
          <!-- 系统分级标注（L1-L4） -->
          <g
            v-for="m in levelMarks"
            :key="`lv${m.i}`"
          >
            <text
              :x="m.x"
              :y="PAD_T - 8"
              class="lvl"
              text-anchor="middle"
            >{{ m.level }}</text>
          </g>
          <!-- X 轴刻度 -->
          <text
            v-for="l in xLabels"
            :key="`x${l.i}`"
            :x="xAt(l.i, history.length)"
            :y="CHART_H - 8"
            class="axis"
            text-anchor="middle"
          >{{ l.text }}</text>
        </svg>
        <div class="legend">
          <span class="key key-tdr">TDR</span>
          <span class="key key-wdr">WDR</span>
          <span class="key key-frr">FRR</span>
          <span class="key key-lvl">L1-L4=系统分级</span>
          <span class="key">共 {{ history.length }} 条评价记录</span>
        </div>
      </template>
    </div>

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
.curve {
  width: 100%;
  height: auto;
  display: block;
}
.curve .grid {
  stroke: #e4e4e4;
  stroke-width: 1;
}
.curve .axis {
  font-size: 10px;
  fill: #888;
}
.curve .line {
  fill: none;
  stroke-width: 2;
}
.line-tdr {
  stroke: #2c7a3d;
}
.line-wdr {
  stroke: #2c5aa0;
}
.line-frr {
  stroke: #b03030;
}
.curve .lvl {
  font-size: 11px;
  font-weight: 600;
  fill: #7a5a00;
}
.legend {
  font-size: 12px;
  color: #555;
  margin-top: 6px;
}
.legend .key {
  margin-right: 14px;
}
.legend .key::before {
  content: "—";
  margin-right: 4px;
  font-weight: 700;
}
.key-tdr::before {
  color: #2c7a3d;
}
.key-wdr::before {
  color: #2c5aa0;
}
.key-frr::before {
  color: #b03030;
}
.key-lvl::before {
  content: "▲";
  color: #7a5a00;
}
.hint {
  font-size: 12px;
  color: #777;
}
</style>
