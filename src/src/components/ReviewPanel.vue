<script setup lang="ts">
/**
 * 人工复核面板（真实 POST /api/v1/review）。
 * 提交后展示后端真实返回：consensus / kappa / stage / joint_level / needs_arbitration。
 */
import { ref } from "vue";
import { submitReview } from "../services/api";
import type { ReviewOut } from "../types/api";

const props = defineProps<{ imageId: string }>();

const reviewer = ref("");
const role = ref<"initial" | "secondary" | "arbitrator">("initial");
const overallLevel = ref("");
const note = ref("");
const submitting = ref(false);
const outcome = ref<ReviewOut | null>(null);
const error = ref<string | null>(null);

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
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    submitting.value = false;
  }
}
</script>

<template>
  <div class="track" style="margin-top: 12px">
    <div class="stage-row">
      <span class="idx">·</span>
      <span class="nm" style="color: var(--amber)">人工复核（初评/复评/仲裁）</span>
      <span class="st" style="color: var(--amber)">need_review=true</span>
    </div>
    <div style="padding: 14px 16px">
      <div class="row">
        <div class="field grow" style="margin-top: 0">
          <label for="rv">评片员（姓名/工号）<span class="req">*</span></label>
          <input id="rv" v-model="reviewer" placeholder="如 张三 / ZS-001" />
        </div>
        <div class="field grow" style="margin-top: 0">
          <label for="rr">角色</label>
          <select id="rr" v-model="role">
            <option value="initial">初评 initial</option>
            <option value="secondary">复评 secondary</option>
            <option value="arbitrator">仲裁 arbitrator</option>
          </select>
        </div>
        <div class="field grow" style="margin-top: 0">
          <label for="rl">复核综合级别（可选）</label>
          <select id="rl" v-model="overallLevel">
            <option value="">不指定</option>
            <option value="I">I 级</option>
            <option value="II">II 级</option>
            <option value="III">III 级</option>
            <option value="IV">IV 级</option>
          </select>
        </div>
      </div>
      <div class="field">
        <label for="rn">备注（可选）</label>
        <input id="rn" v-model="note" style="max-width: 100%" />
      </div>
      <button class="btn" type="button" :disabled="submitting" @click="onSubmit">
        {{ submitting ? "提交中…" : "提交复核" }}
      </button>
      <div v-if="error" class="err show" style="margin-top: 10px">⚠ {{ error }}</div>

      <div v-if="outcome" style="margin-top: 16px" class="kv">
        <div class="k">一致性 consensus</div>
        <div class="v">{{ outcome.consensus ? "达成共识" : "未达成" }}</div>
        <div class="k">κ 系数</div>
        <div class="v">{{ outcome.kappa.toFixed(3) }}</div>
        <div class="k">阶段</div>
        <div class="v">{{ outcome.stage }}</div>
        <div class="k">最终级别</div>
        <div class="v">{{ outcome.joint_level ?? "—" }}</div>
        <div class="k">需仲裁</div>
        <div class="v">{{ outcome.needs_arbitration ? "是" : "否" }}</div>
        <div class="k">复核次数</div>
        <div class="v">{{ outcome.review_count }}</div>
      </div>
    </div>
  </div>
</template>
