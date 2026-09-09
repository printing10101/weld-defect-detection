<script setup lang="ts">
/** C-10 密级名称映射（与后端 classification_label 对齐）。 */
const SECRET_LEVEL_NAMES: Record<number, string> = { 0: "非密", 1: "内部", 2: "秘密", 3: "机密" };
/**
 * 档案检索视图（真实 GET /api/v1/records）。
 * 列表与统计全部来自后端查询；空态/错误态基于真实响应。
 * 附加：主动学习训练池状态（GET /api/v1/active/pool）。
 */
import { onMounted, ref } from "vue";
import { toErrorMessage } from "../utils/errorMessage";
import { stampBadge } from "../utils/filmStamp";
import { activePool, listRecords } from "../services/api";
import type { ActivePoolOut, RecordsResponse } from "../types/api";

const loading = ref(true);
const err = ref<string | null>(null);
const resp = ref<RecordsResponse | null>(null);
const level = ref("");
const workpiece = ref("");
const page = ref(1);
const pageSize = ref(50);
const total = ref(0);

// 请求序号守卫：切回 active / 翻页 / 过滤可能并发多个 load，
// 旧请求晚到会覆盖新响应。仅接受最新一次请求的返回。
let loadReqId = 0;

// 主动学习训练池
const poolLoading = ref(true);
const pool = ref<ActivePoolOut | null>(null);

async function loadPool(): Promise<void> {
  poolLoading.value = true;
  try {
    pool.value = await activePool();
  } catch {
    pool.value = null; // 训练池不可用不影响档案主流程
  } finally {
    poolLoading.value = false;
  }
}

async function load(): Promise<void> {
  const myId = ++loadReqId;
  loading.value = true;
  err.value = null;
  try {
    const r = await listRecords({
      level: level.value || undefined,
      workpiece: workpiece.value || undefined,
      page: page.value,
      size: pageSize.value,
    });
    if (myId !== loadReqId) return; // 旧请求晚到，丢弃，避免覆盖最新响应
    resp.value = r;
    total.value = r.total;
  } catch (e) {
    if (myId !== loadReqId) return; // 同上，非最新请求的错误不覆盖状态
    err.value = toErrorMessage(e);
  } finally {
    if (myId === loadReqId) loading.value = false;
  }
}

onMounted(() => {
  void load();
  void loadPool();
});

function onFilter(): void {
  page.value = 1; // 重新过滤回到首页
  void load();
}

function gotoPage(p: number): void {
  const last = Math.max(1, Math.ceil(total.value / pageSize.value));
  if (p < 1 || p > last) return;
  page.value = p;
  void load();
}

const levelOptions = ["", "I", "II", "III", "IV"] as const;
</script>

<template>
  <div>
    <h1
      class="title-zine"
      data-t="检测档案"
    >
      检测档案
    </h1>
    <div class="lede">
      检索本地归档的检测评定记录（数据来自本地检测档案数据库）
    </div>

    <!-- 主动学习训练样本库（M7） -->
    <section class="pool">
      <h2 class="pool-title">
        主动学习训练样本库
      </h2>
      <p
        v-if="poolLoading"
        class="hint"
      >
        加载中…
      </p>
      <p
        v-else-if="!pool"
        class="hint"
      >
        训练样本库暂不可用（后端未装配或目录不存在）。
      </p>
      <template v-else>
        <div class="pool-meta">
          <span>样本总数：<b>{{ pool.sample_count }}</b></span>
          <span>数据集版本指纹：<code class="fp">{{ pool.fingerprint ?? "—" }}</code></span>
          <span v-if="pool.exported_at">最近导出：{{ pool.exported_at }}</span>
        </div>
        <p class="stat">
          经人工复核确认的缺陷样本将回流至训练样本库（§5.5 持续学习闭环），标注文件：
          {{ pool.files.length ? pool.files.join(" · ") : "暂无，完成一次样本导出后生成" }}
        </p>
      </template>
    </section>

    <div class="search">
      <input
        v-model="workpiece"
        placeholder="按工件编号检索…"
        @keyup.enter="onFilter"
      >
      <select
        v-model="level"
        @change="onFilter"
      >
        <option
          v-for="l in levelOptions"
          :key="l"
          :value="l"
        >
          {{ l ? `${l} 级` : "全部级别" }}
        </option>
      </select>
      <button
        class="btn ghost"
        type="button"
        style="margin-top: 0"
        @click="onFilter"
      >
        检索
      </button>
    </div>

    <p
      v-if="loading"
      class="hint"
    >
      加载中…
    </p>
    <p
      v-else-if="err"
      class="err show"
    >
      ⚠ 加载失败：{{ err }}　<button
        class="btn link"
        type="button"
        @click="onFilter"
      >
        重试
      </button>
    </p>
    <p
      v-else-if="!resp || resp.items.length === 0"
      class="empty"
    >
      暂无归档记录 —— 完成检测评定后，结果将自动归档至此。
    </p>
    <template v-else>
      <table>
        <thead>
          <tr>
            <th>影像编号</th>
            <th>工件编号</th>
            <th>底片印字</th>
            <th>密级</th>
            <th>评定级别</th>
            <th>可评片</th>
            <th>待复核</th>
            <th>归档时间</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="item in resp.items"
            :key="item.image_id"
          >
            <td>{{ item.image_id }}</td>
            <td>{{ item.workpiece_no ?? "—" }}</td>
            <td>
              <!-- 底片印字性质（扫描日期/编号，正/镜像）；无识别数据的旧记录显示 — -->
              <span
                v-if="stampBadge(item)"
                class="stamp"
                :class="`stamp-${stampBadge(item)!.cls}`"
                :title="stampBadge(item)!.title"
              >{{ stampBadge(item)!.label }}</span>
              <span v-else>—</span>
            </td>
            <td>
              <!-- C-10 密级标识：0=非密 1=内部 2=秘密 3=机密（records.items 携带） -->
              <span
                class="lv"
                :class="{ secret: (item.secret_level ?? 0) >= 2 }"
              >{{ SECRET_LEVEL_NAMES[item.secret_level ?? 0] ?? "非密" }}</span>
            </td>
            <td>
              <span
                v-if="item.joint_level"
                class="lv"
              >{{ item.joint_level }}</span><span
                v-else
                class="need"
              >—</span>
            </td>
            <td>{{ item.evaluable ? "是" : "否" }}</td>
            <td>
              <span
                v-if="item.need_review"
                class="need"
              >待复核</span><span v-else>—</span>
            </td>
            <td>{{ item.created_at ?? "—" }}</td>
          </tr>
        </tbody>
      </table>
      <p class="stat">
        共 <b>{{ total }}</b> 条记录 · 级别分布：{{ Object.entries(resp.stats.by_level).map(([k, v]) => `${k}:${v}`).join(" ") || "—" }}
      </p>
      <div
        v-if="total > pageSize"
        class="pager"
      >
        <button
          class="btn ghost"
          type="button"
          :disabled="page <= 1"
          @click="gotoPage(page - 1)"
        >
          ← 上一页
        </button>
        <span class="pg">第 {{ page }} / {{ Math.max(1, Math.ceil(total / pageSize)) }} 页</span>
        <button
          class="btn ghost"
          type="button"
          :disabled="page >= Math.ceil(total / pageSize)"
          @click="gotoPage(page + 1)"
        >
          下一页 →
        </button>
        <select
          v-model.number="pageSize"
          class="ps"
          @change="onFilter"
        >
          <option :value="50">
            50/页
          </option>
          <option :value="100">
            100/页
          </option>
        </select>
      </div>
    </template>
  </div>
</template>

<style scoped>
/* C-10：秘密/机密级红色高亮 */
.lv.secret {
  color: #b3261e;
  font-weight: 700;
}
/* 底片印字徽标（正向/镜像=绿；缺印字待复核=红；缺印字未追责=灰） */
.stamp {
  display: inline-block;
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  white-space: nowrap;
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
.pool {
  border: 1px dashed rgba(140, 140, 140, 0.5);
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 18px;
}
.pool-title {
  font-size: 15px;
  margin: 0 0 8px;
  letter-spacing: 0.05em;
}
.pool-meta {
  display: flex;
  gap: 18px;
  flex-wrap: wrap;
  font-size: 13px;
  margin-bottom: 6px;
}
.fp {
  font-size: 12px;
  opacity: 0.75;
}
.pager {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 12px;
  font-size: 13px;
  color: #44577a;
}
.pager .pg {
  font-variant-numeric: tabular-nums;
}
.pager .ps {
  margin-left: auto;
  padding: 4px 6px;
  border: 1px solid rgba(120, 140, 180, 0.3);
  border-radius: 6px;
  background: transparent;
  color: #22355c;
}
</style>
