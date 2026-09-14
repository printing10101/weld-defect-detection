<script setup lang="ts">
/**
 * 本地大模型（发现 + 选择 + 连接）。
 *
 * 解决的问题：安装包**不随包分发**模型运行时（权重 2.4GB 起、llama.cpp CUDA
 * 运行时 1.3GB，均超 NSIS 2GB 打包上限），所以模型必须来自「本机已有的」——
 * 本面板就是那条路径的入口：自动发现本机 GGUF、探测已在跑的服务端点、选中后
 * 热应用到引擎。
 *
 * 几个刻意的显示口径（都是踩过的坑，不要"优化"掉）：
 * - **不折叠"跑不动"的模型**：全部列出并标注显存可行性。筛选会让用户以为
 *   模型不存在，而实际上"跑不动"只是当前显存下的运行时判断，不是资产缺失。
 * - **"模型上限 ctx"与"服务实际 -c"分开显示**：前者是权重文件声明的能力上限
 *   （如 262144），后者是 llama-server 启动参数（如 8192）。混在一起会让人
 *   以为选了 256K 上下文的模型就能跑 256K，实际显存瞬间就爆。
 * - **`ready` 不等于"能对话"**：引擎状态由后端按 /health 与 /v1/models 双判据
 *   给出；`auth_required` 表示进程活着但要求 API Key，`unavailable` 表示端点
 *   没在跑（是预期状态，不是故障）。
 */
import { computed, onMounted, onUnmounted, ref } from "vue";
import {
  addLlmDir,
  cancelLlmScan,
  clearLlmSelection,
  getLlmStatus,
  listLlmModels,
  removeLlmDir,
  selectLlmModel,
  startLlmScan,
} from "../services/api";
import { useAuthStore } from "../stores/auth";
import { toErrorMessage } from "../utils/errorMessage";
import type { LlmModelOut, LlmScanOut, LlmServiceOut, LlmStatusOut } from "../types/api";

const auth = useAuthStore();
/** 读取对所有登录角色开放；写操作（选中/改目录/扫描）由后端限 sysadmin。 */
const canWrite = computed(() => auth.role === "sysadmin");

const status = ref<LlmStatusOut | null>(null);
const models = ref<LlmModelOut[]>([]);
const services = ref<LlmServiceOut[]>([]);
const counts = ref<{ total: number; local: number; service: number; infeasible: number }>({
  total: 0,
  local: 0,
  service: 0,
  infeasible: 0,
});
const loading = ref(false);
const busy = ref(false);
const err = ref<string | null>(null);
const notice = ref<string | null>(null);
const newDir = ref("");
const onlyAvailable = ref(false);

let poll: ReturnType<typeof setInterval> | null = null;

const ENGINE_LABEL: Record<string, string> = {
  starting: "启动中",
  ready: "就绪",
  error: "故障",
  unavailable: "端点未运行",
  auth_required: "需要 API Key",
  disabled: "已关闭",
  stopped: "已停止",
};
const ENGINE_CLASS: Record<string, string> = {
  ready: "ok",
  starting: "warn",
  unavailable: "warn",
  auth_required: "warn",
  error: "bad",
  disabled: "muted",
  stopped: "muted",
};
const VERDICT_LABEL: Record<string, string> = {
  full_gpu: "可全量上卡",
  tight: "余量紧张",
  partial_offload: "需部分 offload",
  infeasible: "跑不动（走 CPU）",
  unknown: "未知",
};
const VERDICT_CLASS: Record<string, string> = {
  full_gpu: "ok",
  tight: "warn",
  partial_offload: "warn",
  infeasible: "bad",
  unknown: "muted",
};

const scan = computed<LlmScanOut | null>(() => status.value?.scan ?? null);
const scanning = computed(() => scan.value?.state === "running");
const engine = computed(() => status.value?.engine ?? null);
const selection = computed(() => status.value?.selection ?? null);

/** 显存可行性优先；"跑不动"不代表不可用（服务条目不占本机显存）。 */
const visible = computed(() =>
  onlyAvailable.value
    ? models.value.filter((m) => m.available && m.vram?.verdict !== "infeasible")
    : models.value,
);

function gib(n: number | null | undefined): string {
  if (!n || n <= 0) return "—";
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

function ctxOf(m: LlmModelOut): string {
  if (!m.max_ctx) return "—";
  return `${(m.max_ctx / 1024).toFixed(0)}K`;
}

async function load(): Promise<void> {
  loading.value = true;
  err.value = null;
  try {
    const [st, mo] = await Promise.all([getLlmStatus(), listLlmModels(true)]);
    status.value = st;
    models.value = mo.models;
    services.value = mo.services;
    counts.value = {
      total: mo.counts.total,
      local: mo.counts.local,
      service: mo.counts.service,
      infeasible: mo.counts.infeasible,
    };
  } catch (e) {
    err.value = toErrorMessage(e);
  } finally {
    loading.value = false;
  }
}

async function guard(fn: () => Promise<unknown>, okMsg: string): Promise<void> {
  busy.value = true;
  err.value = null;
  notice.value = null;
  try {
    await fn();
    notice.value = okMsg;
    await load();
  } catch (e) {
    err.value = toErrorMessage(e);
  } finally {
    busy.value = false;
  }
}

function onSelect(m: LlmModelOut): void {
  void guard(
    () => selectLlmModel(m.id),
    `已选中「${m.display_name}」（引擎热应用，加载需数秒到数十秒）`,
  );
}

function onClear(): void {
  void guard(clearLlmSelection, "已回落到配置文件默认");
}

function onScan(): void {
  void guard(startLlmScan, "已启动模型扫描（后台进行，可继续操作）");
}

function onCancelScan(): void {
  void guard(cancelLlmScan, "已请求取消扫描（已扫到的结果保留）");
}

function onAddDir(): void {
  const p = newDir.value.trim();
  if (!p) return;
  void guard(async () => {
    await addLlmDir(p);
    newDir.value = "";
  }, `已添加模型目录：${p}`);
}

function onRemoveDir(p: string): void {
  void guard(() => removeLlmDir(p), `已移除目录登记（磁盘文件未动）：${p}`);
}

onMounted(() => {
  void load();
  // 扫描/引擎加载都是异步的：轮询状态而不是让用户手动刷新。
  poll = setInterval(async () => {
    if (busy.value) return;
    try {
      status.value = await getLlmStatus();
      if (scanning.value || engine.value?.state === "starting") return;
      if (scan.value?.state === "done") void load();
    } catch {
      /* 轮询失败静默：主请求路径的错误已单独呈现 */
    }
  }, 3000);
});

onUnmounted(() => {
  if (poll !== null) clearInterval(poll);
});
</script>

<template>
  <div>
    <h1
      class="title-zine"
      data-t="本地大模型"
    >
      本地大模型
    </h1>
    <div class="lede">
      发现本机已有的 GGUF
      模型与在跑的推理服务，选中后热应用为本次会话的本地模型（全程只读扫描，不复制、不上传）
    </div>

    <p
      v-if="err"
      class="err show"
    >
      ⚠ {{ err }}
    </p>
    <p
      v-else-if="notice"
      class="ok-msg"
    >
      ✓ {{ notice }}
    </p>

    <!-- 引擎状态 -->
    <div class="section-h">
      <span class="no">1</span>引擎状态
    </div>
    <div
      v-if="engine"
      class="cards"
    >
      <div class="card">
        <div class="card-t">
          运行状态
        </div>
        <div class="rowline">
          <span
            class="chip"
            :class="ENGINE_CLASS[engine.state] ?? 'muted'"
          >{{
            ENGINE_LABEL[engine.state] ?? engine.state
          }}</span>
          <span class="kv">方式：{{ engine.mode === "managed" ? "本进程拉起" : "连接已有服务" }}</span>
        </div>
        <div class="kv">
          端点：{{ engine.endpoint || "—" }}
          <span v-if="engine.adopted">（复用外部实例，退出时不回收）</span>
        </div>
        <p
          v-if="engine.error"
          class="hint errline"
        >
          {{ engine.error }}
        </p>
        <p
          v-if="engine.advice"
          class="hint"
        >
          {{ engine.advice }}
        </p>
      </div>

      <div class="card">
        <div class="card-t">
          本机显存
        </div>
        <template v-if="status?.gpu">
          <div class="kv">
            {{ status.gpu.name }}
          </div>
          <div class="kv">
            空闲 {{ gib(status.gpu.free_bytes) }} / 共 {{ gib(status.gpu.total_bytes) }}
          </div>
        </template>
        <div
          v-else
          class="kv"
        >
          未检测到 GPU 信息
        </div>
      </div>

      <div class="card">
        <div class="card-t">
          已选中
        </div>
        <template v-if="selection?.active_id">
          <div class="kv">
            {{ selection.model_path || selection.endpoint || selection.active_id }}
          </div>
          <button
            class="btn"
            type="button"
            :disabled="busy || !canWrite"
            @click="onClear"
          >
            回落到配置默认
          </button>
        </template>
        <div
          v-else
          class="kv"
        >
          未显式选中（按配置文件默认运行）
        </div>
      </div>
    </div>

    <!-- 已有服务 -->
    <div class="section-h">
      <span class="no">2</span>本机推理服务
    </div>
    <div class="svc">
      <div
        v-for="s in services"
        :key="s.base_url"
        class="svc-row"
      >
        <span
          class="chip"
          :class="s.reachable && s.health_ok ? 'ok' : 'muted'"
        >{{
          s.reachable && s.health_ok ? "在线" : "未运行"
        }}</span>
        <code>{{ s.base_url }}</code>
        <span class="kv">{{ s.models.length }} 个模型</span>
        <span
          v-if="s.needs_auth"
          class="chip warn"
        >需要 Key</span>
        <span
          v-if="s.error && !s.models.length"
          class="kv muted"
        >{{ s.error }}</span>
      </div>
      <div
        v-if="!services.length"
        class="empty"
      >
        未探测到可用的本机推理服务。
      </div>
    </div>

    <!-- 模型清单 -->
    <div class="section-h">
      <span class="no">3</span>可选模型（{{ counts.total }} 个：本地 {{ counts.local }} / 服务
      {{ counts.service }}）
    </div>
    <div class="actions">
      <button
        class="btn"
        type="button"
        :disabled="busy || !canWrite || scanning"
        @click="onScan"
      >
        {{ scanning ? "扫描中…" : "开始扫描本机模型" }}
      </button>
      <button
        v-if="scanning"
        class="btn"
        type="button"
        :disabled="busy"
        @click="onCancelScan"
      >
        取消扫描
      </button>
      <label class="kv inline">
        <input
          v-model="onlyAvailable"
          type="checkbox"
        >
        只显示当前跑得动的
      </label>
      <span
        v-if="scanning && scan"
        class="kv"
      >{{ scan.progress.current || "遍历中…" }}（已扫 {{ scan.progress.dirs }} 个目录 / 命中
        {{ scan.found }}）</span>
    </div>
    <p
      v-if="counts.infeasible > 0"
      class="hint"
    >
      其中 {{ counts.infeasible }} 个在所有本机显存条件下都无法全量上卡（会退回 CPU
      推理）——仍会列出，不做隐藏。
    </p>

    <div
      v-if="loading && !models.length"
      class="empty"
    >
      正在读取模型清单…
    </div>
    <div
      v-else-if="!visible.length"
      class="empty"
    >
      未发现模型。可在下方添加模型目录，或先启动本机的推理服务。
    </div>

    <table
      v-else
      class="mlist"
    >
      <thead>
        <tr>
          <th>模型</th>
          <th>来源</th>
          <th>量化</th>
          <th title="权重文件声明的能力上限，不等于服务实际启动的 -c">
            上限 ctx
          </th>
          <th title="按当前空闲显存估算的运行占用">
            显存
          </th>
          <th>操作</th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="m in visible"
          :key="m.id"
          :class="{ active: m.active }"
        >
          <td>
            <div class="mname">
              {{ m.display_name }}
              <span
                v-if="m.is_embedding"
                class="chip muted"
              >向量</span>
              <span
                v-if="m.duplicate_count"
                class="chip muted"
              >另有 {{ m.duplicate_count }} 份副本</span>
              <span
                v-if="!m.available"
                class="chip bad"
              >不可用</span>
            </div>
            <div class="mpath">
              {{ m.source === "local" ? m.path : m.endpoint }}
            </div>
            <div
              v-if="m.error"
              class="hint errline"
            >
              {{ m.error }}
            </div>
          </td>
          <td>{{ m.source === "local" ? "本地文件" : "已有服务" }}</td>
          <td>{{ m.quant || "—" }}</td>
          <td>{{ ctxOf(m) }}</td>
          <td>
            <template v-if="m.vram">
              <span
                class="chip"
                :class="VERDICT_CLASS[m.vram.verdict] ?? 'muted'"
              >{{
                VERDICT_LABEL[m.vram.verdict] ?? m.vram.verdict
              }}</span>
              <div class="hint">
                {{ gib(m.vram.needed_bytes) }} / 可用 {{ gib(m.vram.free_vram_bytes) }}
              </div>
            </template>
            <span
              v-else
              class="kv"
            >不占本机显存</span>
          </td>
          <td>
            <button
              class="btn"
              type="button"
              :disabled="busy || !canWrite || !m.available || m.active"
              @click="onSelect(m)"
            >
              {{ m.active ? "使用中" : "选中" }}
            </button>
          </td>
        </tr>
      </tbody>
    </table>

    <!-- 模型目录 -->
    <div class="section-h">
      <span class="no">4</span>模型目录
    </div>
    <div class="actions">
      <input
        v-model="newDir"
        placeholder="添加目录，如 E:\llama-cpp\models"
        @keyup.enter="onAddDir"
      >
      <button
        class="btn"
        type="button"
        :disabled="busy || !canWrite || !newDir.trim()"
        @click="onAddDir"
      >
        添加目录
      </button>
    </div>
    <div class="dirs">
      <div
        v-for="d in status?.model_dirs ?? []"
        :key="d"
        class="dir-row"
      >
        <code>{{ d }}</code>
        <button
          class="btn"
          type="button"
          :disabled="busy || !canWrite"
          @click="onRemoveDir(d)"
        >
          移除登记
        </button>
      </div>
      <div
        v-if="!status?.model_dirs?.length"
        class="kv"
      >
        未添加额外目录（扫描范围默认覆盖本机全部固定盘）。
      </div>
    </div>
    <p
      v-if="!canWrite"
      class="hint"
    >
      当前身份只读：选中模型、增删目录与扫描需系统管理员权限。
    </p>
  </div>
</template>

<style scoped>
.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
  margin-bottom: 8px;
}
.card {
  border: 1px solid var(--line, #2a2f3a);
  border-radius: 8px;
  padding: 10px 12px;
}
.card-t {
  font-weight: 600;
  margin-bottom: 6px;
}
.rowline {
  display: flex;
  align-items: center;
  gap: 8px;
}
.kv {
  opacity: 0.85;
  font-size: 0.9em;
  margin-top: 4px;
}
.inline {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.svc,
.dirs {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 8px;
}
.svc-row,
.dir-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 10px;
}
.mlist {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 12px;
}
.mlist th,
.mlist td {
  text-align: left;
  padding: 8px 10px;
  border-bottom: 1px solid var(--line, #2a2f3a);
  vertical-align: top;
}
.mlist tr.active {
  background: rgba(64, 158, 255, 0.08);
}
.mname {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}
.mpath {
  opacity: 0.6;
  font-size: 0.82em;
  word-break: break-all;
  max-width: 420px;
}
.hint {
  opacity: 0.75;
  font-size: 0.85em;
  margin: 4px 0 0;
}
.errline {
  color: #e07a7a;
}
.ok-msg {
  color: #5cc98a;
}
</style>
