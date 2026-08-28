<script setup lang="ts">
/** 状态栏（AutoCAD 范式）：底部常驻状态格。
 *  就绪 | 后端连接状态 | 模型状态 | 记录总数 | 操作员 | 系统时间。
 *  后端状态复用 App.vue 的 BACKEND_UP/DOWN 窗口事件；时间每秒刷新。 */
import { onBeforeUnmount, onMounted, ref } from "vue";
import { BACKEND_DOWN_EVENT, BACKEND_UP_EVENT } from "../services/api";

const backend = ref<"connecting" | "up" | "down">("connecting");
const modelStatus = ref("加载中");
const now = ref(new Date());
let timer: number | undefined;

function onUp(): void {
  backend.value = "up";
  modelStatus.value = "就绪";
}
function onDown(): void {
  backend.value = "down";
  modelStatus.value = "不可用";
}

onMounted(() => {
  window.addEventListener(BACKEND_UP_EVENT, onUp);
  window.addEventListener(BACKEND_DOWN_EVENT, onDown);
  timer = window.setInterval(() => (now.value = new Date()), 1000);
});
onBeforeUnmount(() => {
  window.removeEventListener(BACKEND_UP_EVENT, onUp);
  window.removeEventListener(BACKEND_DOWN_EVENT, onDown);
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
      {{ backend === "up" ? "就绪" : backend === "down" ? "后端未连接" : "正在连接后端" }}
    </div>
    <div class="cell sep">
      后端 127.0.0.1:18773
    </div>
    <div class="cell sep">
      模型：<span
        :class="modelStatus === '就绪' ? 'ok' : 'warn'"
      >{{ modelStatus }}</span>
    </div>
    <div class="cell sep">
      本地优先 · 数据不出机
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
