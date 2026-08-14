<script setup lang="ts">
/**
 * 档案检索视图（真实 GET /api/v1/records）。
 * 列表与统计全部来自后端查询；空态/错误态基于真实响应。
 * 附加：主动学习训练池状态（M7，GET /api/v1/active/pool）。
 */
import { onMounted, ref, watch } from "vue";
import { activePool, listRecords } from "../services/api";
import type { ActivePoolOut, RecordsResponse } from "../types/api";

const props = defineProps<{ active: boolean }>();

const loading = ref(true);
const err = ref<string | null>(null);
const resp = ref<RecordsResponse | null>(null);
const level = ref("");
const workpiece = ref("");

// 主动学习训练池（M7）
const poolLoading = ref(true);
const pool = ref<ActivePoolOut | null>(null);

async function loadPool(): Promise<void> {
  poolLoading.value = true;
  try {
    pool.value = await activePool();
  } catch (e) {
    pool.value = null; // 训练池不可用不影响档案主流程
  } finally {
    poolLoading.value = false;
  }
}

async function load(): Promise<void> {
  loading.value = true;
  err.value = null;
  try {
    resp.value = await listRecords({
      level: level.value || undefined,
      workpiece: workpiece.value || undefined,
      page: 1,
      size: 50,
    });
  } catch (e) {
    err.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

onMounted(() => {
  void load();
  void loadPool();
});

// 视图被切换到档案时重新拉取，确保展示最新真实归档
watch(
  () => props.active,
  (v) => {
    if (v) {
      void load();
      void loadPool();
    }
  },
);

function onFilter(): void {
  void load();
}

const levelOptions = ["", "I", "II", "III", "IV"] as const;
</script>

<template>
  <div>
    <h1 class="title-zine" data-t="档案检索">档案检索</h1>
    <div class="lede">RECORDS · 处理结果自动归档，可随时回来查看（真实后端数据）</div>

    <!-- 主动学习训练池（M7） -->
    <section class="pool">
      <h2 class="pool-title">主动学习训练池</h2>
      <p v-if="poolLoading" class="hint">加载中…</p>
      <p v-else-if="!pool" class="hint">训练池暂不可用（后端未装配或目录不存在）。</p>
      <template v-else>
        <div class="pool-meta">
          <span>样本数：<b>{{ pool.sample_count }}</b></span>
          <span>数据版本：<code class="fp">{{ pool.fingerprint ?? "—" }}</code></span>
          <span v-if="pool.exported_at">最近导出：{{ pool.exported_at }}</span>
        </div>
        <p class="stat">
          人工复核确认的缺陷会回流此训练池（§5.5 闭环），标注文件：
          {{ pool.files.length ? pool.files.join(" · ") : "暂无，完成一次主动导出后出现" }}
        </p>
      </template>
    </section>

    <div class="search">
      <input v-model="workpiece" placeholder="检索 工件号…" @keyup.enter="onFilter" />
      <select v-model="level" @change="onFilter">
        <option v-for="l in levelOptions" :key="l" :value="l">{{ l ? `${l} 级` : "全部级别" }}</option>
      </select>
      <button class="btn ghost" type="button" style="margin-top: 0" @click="onFilter">查询</button>
    </div>

    <p v-if="loading" class="hint">加载中…</p>
    <p v-else-if="err" class="err show">⚠ 加载失败：{{ err }}　<button class="btn link" type="button" @click="onFilter">重试</button></p>
    <p v-else-if="!resp || resp.items.length === 0" class="empty">暂无记录 —— 完成一次检测后，结果会自动归档到这里。</p>
    <template v-else>
      <table>
        <thead>
          <tr>
            <th>影像编号</th>
            <th>工件号</th>
            <th>级别</th>
            <th>可评片</th>
            <th>待复核</th>
            <th>生成时间</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="item in resp.items" :key="item.image_id">
            <td>{{ item.image_id }}</td>
            <td>{{ item.workpiece_no ?? "—" }}</td>
            <td><span v-if="item.joint_level" class="lv">{{ item.joint_level }}</span><span v-else class="need">—</span></td>
            <td>{{ item.evaluable ? "是" : "否" }}</td>
            <td><span v-if="item.need_review" class="need">待复核</span><span v-else>—</span></td>
            <td>{{ item.created_at ?? "—" }}</td>
          </tr>
        </tbody>
      </table>
      <p class="stat">
        共 <b>{{ resp.total }}</b> 条 · 级别分布：{{ Object.entries(resp.stats.by_level).map(([k, v]) => `${k}:${v}`).join(" ") || "—" }}
      </p>
    </template>
  </div>
</template>

<style scoped>
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
</style>
