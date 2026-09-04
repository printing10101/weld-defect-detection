<script setup lang="ts">
/**
 * 人工复核面板：级别复核（POST /api/v1/review）+ 缺陷增删改
 * （POST/PATCH/DELETE，DB50/T 1807 §6.1.4），变更后后端自动重评级。
 */
import { ref } from "vue";
import { toErrorMessage } from "../utils/errorMessage";
import ConfirmDialog from "./ConfirmDialog.vue";
import {
  addReviewDefect,
  deleteReviewDefect,
  editReviewDefect,
  getReportDetections,
  submitReview,
} from "../services/api";
import { DEFECT_CLASS_LABELS } from "../types/api";
import type { ReviewOut } from "../types/api";

const props = defineProps<{ imageId: string; reportId?: string }>();

const DEFECT_CLASSES = DEFECT_CLASS_LABELS.map((name, id) => ({ id, name }));

interface DefectRow {
  id: string;
  class_id: number;
  bbox_px: number[];
  source: string | null;
}

const reviewer = ref("");
const role = ref<"initial" | "secondary" | "arbitrator">("initial");
const overallLevel = ref("");
const note = ref("");
const submitting = ref(false);
const outcome = ref<ReviewOut | null>(null);
const error = ref<string | null>(null);

// ---- 缺陷增删改 ----
const defectOpen = ref(false);
const defectRows = ref<DefectRow[]>([]);
const defectBusy = ref(false);
const defectMsg = ref<string | null>(null);
const defectErr = ref<string | null>(null);
const reason = ref("");
const newClass = ref(0);
const newBox = ref("");

async function toggleDefects(): Promise<void> {
  defectOpen.value = !defectOpen.value;
  if (defectOpen.value && props.reportId) await reloadDefects();
}

async function reloadDefects(): Promise<void> {
  if (!props.reportId) return;
  defectErr.value = null;
  try {
    const d = await getReportDetections(props.reportId);
    defectRows.value = d.defects.map((x) => ({
      id: String(x.id),
      class_id: Number(x.class_id),
      bbox_px: (x.bbox as number[]) ?? [],
      source: (x.source as string | null) ?? null,
    }));
  } catch (e) {
    defectErr.value = toErrorMessage(e);
  }
}

function guardReason(): string | null {
  if (!reason.value.trim()) {
    defectErr.value = "请先填写变更理由（审计必填）。";
    return null;
  }
  return reason.value.trim();
}

async function onAddDefect(): Promise<void> {
  const r = guardReason();
  if (r === null) return;
  const parts = newBox.value.split(/[,,\s]+/).filter(Boolean).map(Number);
  if (parts.length !== 4 || parts.some((v) => !Number.isFinite(v) || v < 0)) {
    defectErr.value = "框坐标格式：x,y,w,h（非负数字）。";
    return;
  }
  defectBusy.value = true;
  defectErr.value = null;
  try {
    const out = await addReviewDefect(props.imageId, {
      class_id: newClass.value,
      bbox_px: parts,
      reason: r,
    });
    defectMsg.value = `已添加，综合级别 ${out.joint_level ?? "需人工"}（缺陷 ${out.defect_count}）`;
    await reloadDefects();
  } catch (e) {
    defectErr.value = toErrorMessage(e);
  } finally {
    defectBusy.value = false;
  }
}

async function onEditClass(row: DefectRow, classId: number): Promise<void> {
  const r = guardReason();
  if (r === null) return;
  defectBusy.value = true;
  defectErr.value = null;
  try {
    const out = await editReviewDefect(row.id, { class_id: classId, reason: r });
    defectMsg.value = `已修改类型，综合级别 ${out.joint_level ?? "需人工"}`;
    await reloadDefects();
  } catch (e) {
    defectErr.value = toErrorMessage(e);
  } finally {
    defectBusy.value = false;
  }
}

// 删除须二次确认（用户差错防御）：先选中目标 → 确认框 → 实际执行
const deleteTarget = ref<DefectRow | null>(null);

function onDeleteDefect(row: DefectRow): void {
  deleteTarget.value = row;
}

async function onDeleteDefectConfirmed(): Promise<void> {
  const row = deleteTarget.value;
  deleteTarget.value = null;
  if (!row) return;
  const r = guardReason();
  if (r === null) return;
  defectBusy.value = true;
  defectErr.value = null;
  try {
    const out = await deleteReviewDefect(row.id, r);
    defectMsg.value = `已删除，综合级别 ${out.joint_level ?? "需人工"}（缺陷 ${out.defect_count}）`;
    await reloadDefects();
  } catch (e) {
    defectErr.value = toErrorMessage(e);
  } finally {
    defectBusy.value = false;
  }
}

async function onSubmit(): Promise<void> {
  error.value = null;
  outcome.value = null;
  if (!reviewer.value.trim()) {
    error.value = "请填写评片员姓名/工号。";
    return;
  }
  submitting.value = true;
  try {
    outcome.value = await submitReview({
      image_id: props.imageId,
      reviewer: reviewer.value.trim(),
      role: role.value,
      overall_level: overallLevel.value || null,
      note: note.value.trim() || null,
    });
  } catch (e) {
    error.value = toErrorMessage(e);
  } finally {
    submitting.value = false;
  }
}
</script>

<template>
  <div
    class="track"
    style="margin-top: 12px"
  >
    <div class="stage-row">
      <span class="idx">·</span>
      <span
        class="nm"
        style="color: var(--amber)"
      >人工复核（初评/复评/仲裁）</span>
      <span
        class="st"
        style="color: var(--amber)"
      >need_review=true</span>
    </div>
    <div style="padding: 14px 16px">
      <div class="row">
        <div
          class="field grow"
          style="margin-top: 0"
        >
          <label for="rv">评片员（姓名/工号）<span class="req">*</span></label>
          <input
            id="rv"
            v-model="reviewer"
            placeholder="如 张三 / ZS-001"
          >
        </div>
        <div
          class="field grow"
          style="margin-top: 0"
        >
          <label for="rr">角色</label>
          <select
            id="rr"
            v-model="role"
          >
            <option value="initial">
              初评 initial
            </option>
            <option value="secondary">
              复评 secondary
            </option>
            <option value="arbitrator">
              仲裁 arbitrator
            </option>
          </select>
        </div>
        <div
          class="field grow"
          style="margin-top: 0"
        >
          <label for="rl">复核综合级别（可选）</label>
          <select
            id="rl"
            v-model="overallLevel"
          >
            <option value="">
              不指定
            </option>
            <option value="I">
              I 级
            </option>
            <option value="II">
              II 级
            </option>
            <option value="III">
              III 级
            </option>
            <option value="IV">
              IV 级
            </option>
          </select>
        </div>
      </div>
      <div class="field">
        <label for="rn">备注（可选）</label>
        <input
          id="rn"
          v-model="note"
          style="max-width: 100%"
        >
      </div>
      <button
        class="btn"
        type="button"
        :disabled="submitting"
        @click="onSubmit"
      >
        {{ submitting ? "提交中…" : "提交复核" }}
      </button>
      <div class="section-h">
        缺陷管理（增删改后自动重评级）
      </div>
      <button
        class="btn"
        type="button"
        @click="toggleDefects"
      >
        {{ defectOpen ? "收起" : "展开" }}缺陷管理
      </button>
      <div
        v-if="defectOpen"
        style="margin-top: 10px"
      >
        <div class="field">
          <label for="dfr">变更理由（审计必填）<span class="req">*</span></label>
          <input
            id="dfr"
            v-model="reason"
            placeholder="如：复核确认为裂纹 / 补录漏检气孔 / 确认为伪影像"
          >
        </div>
        <table
          v-if="defectRows.length"
          class="dtable"
        >
          <thead>
            <tr><th>类型</th><th>框 [x,y,w,h]</th><th>来源</th><th>操作</th></tr>
          </thead>
          <tbody>
            <tr
              v-for="row in defectRows"
              :key="row.id"
            >
              <td>
                <select
                  :value="row.class_id"
                  :disabled="defectBusy"
                  @change="onEditClass(row, Number(($event.target as HTMLSelectElement).value))"
                >
                  <option
                    v-for="c in DEFECT_CLASSES"
                    :key="c.id"
                    :value="c.id"
                  >
                    {{ c.name }}
                  </option>
                </select>
              </td>
              <td>{{ row.bbox_px.map((v: number) => Math.round(v)).join(", ") }}</td>
              <td>{{ row.source === "manual" ? "人工" : "检测" }}</td>
              <td>
                <button
                  class="btn danger"
                  type="button"
                  :disabled="defectBusy"
                  @click="onDeleteDefect(row)"
                >
                  删除
                </button>
              </td>
            </tr>
          </tbody>
        </table>
        <div
          v-else
          class="lede"
        >
          该影像暂无缺陷记录。
        </div>
        <div class="row">
          <div class="field">
            <label for="dfc">新增缺陷类型</label>
            <select
              id="dfc"
              v-model.number="newClass"
            >
              <option
                v-for="c in DEFECT_CLASSES"
                :key="c.id"
                :value="c.id"
              >
                {{ c.name }}
              </option>
            </select>
          </div>
          <div class="field grow">
            <label for="dfb">框坐标 x,y,w,h（像素）</label>
            <input
              id="dfb"
              v-model="newBox"
              placeholder="如 120,30,20,20"
            >
          </div>
          <button
            class="btn"
            type="button"
            :disabled="defectBusy"
            @click="onAddDefect"
          >
            添加缺陷
          </button>
        </div>
        <div
          v-if="defectMsg"
          class="ok show"
        >
          {{ defectMsg }}
        </div>
        <div
          v-if="defectErr"
          class="err show"
        >
          {{ defectErr }}
        </div>
      </div>
      <div
        v-if="error"
        class="err show"
        style="margin-top: 10px"
      >
        ⚠ {{ error }}
      </div>

      <div
        v-if="outcome"
        style="margin-top: 16px"
        class="kv"
      >
        <div class="k">
          一致性 consensus
        </div>
        <div class="v">
          {{ outcome.consensus ? "达成共识" : "未达成" }}
        </div>
        <div class="k">
          κ 系数
        </div>
        <div class="v">
          {{ outcome.kappa.toFixed(3) }}
        </div>
        <div class="k">
          阶段
        </div>
        <div class="v">
          {{ outcome.stage }}
        </div>
        <div class="k">
          最终级别
        </div>
        <div class="v">
          {{ outcome.joint_level ?? "—" }}
        </div>
        <div class="k">
          需仲裁
        </div>
        <div class="v">
          {{ outcome.needs_arbitration ? "是" : "否" }}
        </div>
        <div class="k">
          复核次数
        </div>
        <div class="v">
          {{ outcome.review_count }}
        </div>
      </div>
    </div>
  </div>
  <ConfirmDialog
    :open="deleteTarget !== null"
    title="删除缺陷确认"
    message="将删除该缺陷记录并触发重新评级、重出报告；操作入审计链且不可撤销。确定删除？"
    confirm-text="删除"
    danger
    @confirm="onDeleteDefectConfirmed"
    @cancel="deleteTarget = null"
  />
</template>

<style scoped>
.dtable {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  margin: 8px 0;
}
.dtable th,
.dtable td {
  border: 1px solid #d8d8d8;
  padding: 4px 6px;
  text-align: left;
}
.dtable select {
  font-size: 12px;
}
.btn.danger {
  color: #b03030;
}
.ok.show {
  color: #2c7a3d;
  font-size: 12px;
  margin-top: 6px;
}
</style>
