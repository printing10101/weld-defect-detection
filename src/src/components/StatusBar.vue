<script setup lang="ts">
/** 状态栏（AutoCAD 范式）：底部常驻状态格。
 *  就绪 | 后端连接状态 | 模型状态 | 记录总数 | 操作员 | 系统时间。
 *  后端状态从 backend store 的响应式状态派生——UP/DOWN 是一次性窗口事件，
 *  本组件常在登录页之后才挂载，读状态才不会永远停在「正在连接」；时间每秒刷新。 */
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { apiHostLabel } from "../services/api";
import { useBackendStore } from "../stores/backend";

const store = useBackendStore();
const backend = computed<"connecting" | "up" | "down">(() =>
  store.backendUp ? "up" : store.backendDown ? "down" : "connecting",
);
const modelStatus = computed(() =>
  backend.value === "up" ? "就绪" : backend.value === "down" ? "不可用" : "加载中",
);
const now = ref(new Date());
/** 服务地址随配置解析（此前硬编码 127.0.0.1:18773，改 VITE_API_BASE 即误导）。 */
const host = apiHostLabel();
let timer: number | undefined;

onMounted(() => {
  timer = window.setInterval(() => (now.value = new Date()), 1000);
});
onBeforeUnmount(() => {
  if (timer !== undefined) window.clearInterval(timer);
});

function fmtTime(d: Date): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}
</script>

<template>
  <footer class="statusbar">
    <div class="cell">
      <span
        class="lamp"
        :class="backend"
      />
      {{ backend === "up" ? "系统就绪" : backend === "down" ? "推理服务未连接" : "正在连接推理服务" }}
    </div>
    <div class="cell sep">
      服务端 {{ host }}
    </div>
    <div class="cell sep">
      评定模型：<span
        :class="modelStatus === '就绪' ? 'ok' : 'warn'"
      >{{ modelStatus }}</span>
    </div>
    <div class="cell sep">
      本地化部署 · 数据不出机
    </div>
    <div class="spacer" />
    <div class="cell">
      {{ fmtTime(now) }}
    </div>
  </footer>
</template>

<style scoped>
.statusbar {
  display: flex;
  align-items: center;
  height: 24px;
  background: linear-gradient(180deg, #f0f0f2, #e8e8ea);
  border-top: 1px solid var(--line);
  font-size: 11px;
  color: var(--ink-soft);
  user-select: none;
  flex: none;
}
.cell {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0 10px;
  height: 100%;
}
.cell.sep {
  border-left: 1px solid var(--line-soft);
}
.spacer {
  flex: 1;
}
.lamp {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #c8c8cc;
}
.lamp.up {
  background: var(--ok);
}
.lamp.down {
  background: var(--signal);
}
.lamp.connecting {
  background: var(--amber);
}
.ok {
  color: var(--ok);
}
.warn {
  color: var(--amber);
}
</style>
