<script setup lang="ts">
/** 工具栏（WPS 功能区范式）：大图标分组命令 + 紧凑小按钮混合。
 *  组1「检测」：打开影像 / 批量导入（主命令）
 *  组2「工作区」：四个视图切换（等价于标签页，鼠标可达性优先）
 *  动作上抛 AppShell 分发；视图切换走 view 事件。 */
import type { ViewId } from "../types/api";

defineProps<{ activeView: ViewId; operator: string }>();
const emit = defineEmits<{
  action: [id: string];
  view: [id: ViewId];
}>();

const VIEWS: { id: ViewId; label: string; icon: string }[] = [
  { id: "journey", label: "单张检测", icon: "M4 3h16v14H4z M4 17l5-5 4 4 3-3 4 4" },
  { id: "batch", label: "批量检测", icon: "M3 4h8v8H3z M13 4h8v8h-8z M3 14h8v8H3z M13 14h8v8h-8z" },
  { id: "archive", label: "档案检索", icon: "M4 4h16v4H4z M4 10h16v10H4z M8 7h.01" },
  { id: "device", label: "设备标定", icon: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z M12 2v3 M12 19v3 M2 12h3 M19 12h3" },
];
</script>

<template>
  <div class="ribbon">
    <!-- 组1：检测主命令 -->
    <div class="group">
      <button
        type="button"
        class="big"
        title="选择单张影像提交检测 (Ctrl+O)"
        @click="emit('action', 'open-image')"
      >
        <svg
          viewBox="0 0 24 24"
          class="ic"
        ><path d="M12 3 5 9v12h14V9z M9 21v-7h6v7" /></svg>
        <span>打开影像</span>
      </button>
      <button
        type="button"
        class="big"
        title="多底片/文件夹批量检测 (Ctrl+Shift+O)"
        @click="emit('action', 'open-batch')"
      >
        <svg
          viewBox="0 0 24 24"
          class="ic"
        ><path d="M3 5h7l2 2h9v12H3z M3 5v14" /></svg>
        <span>批量导入</span>
      </button>
    </div>

    <div class="divider" />

    <!-- 组2：工作区切换 -->
    <div class="group views">
      <button
        v-for="v in VIEWS"
        :key="v.id"
        type="button"
        :class="{ on: activeView === v.id }"
        :title="`${v.label} (Ctrl+${VIEWS.indexOf(v) + 1})`"
        @click="emit('view', v.id)"
      >
        <svg
          viewBox="0 0 24 24"
          class="ic s"
        ><path :d="v.icon" /></svg>
        <span>{{ v.label }}</span>
      </button>
    </div>

    <div class="spacer" />

    <!-- 右侧：操作员（点击修改，审计留痕入口） -->
    <button
      type="button"
      class="operator"
      title="点击修改操作员姓名（用于报告签名与审计留痕）"
      @click="emit('action', 'operator')"
    >
      操作员：{{ operator }}
    </button>
  </div>
</template>

<style scoped>
.ribbon {
  display: flex;
  align-items: stretch;
  gap: 4px;
  height: 56px;
  padding: 4px 8px;
  background: var(--panel-2);
  border-bottom: 1px solid var(--line);
  flex: none;
}
.group {
  display: flex;
  gap: 4px;
  align-items: center;
}
.divider {
  width: 1px;
  background: var(--line);
  margin: 6px 6px;
}
.big {
  appearance: none;
  border: 1px solid transparent;
  background: transparent;
  font-family: var(--font);
  font-size: 12px;
  color: var(--ink);
  width: 62px;
  height: 46px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 3px;
  cursor: pointer;
  border-radius: 2px;
}
.big:hover {
  border-color: var(--line);
  background: #fff;
}
.big .ic {
  width: 20px;
  height: 20px;
  fill: none;
  stroke: var(--accent);
  stroke-width: 1.6;
}
.views button {
  appearance: none;
  border: 1px solid transparent;
  background: transparent;
  font-family: var(--font);
  font-size: 12px;
  color: var(--ink-soft);
  padding: 0 10px;
  height: 46px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 3px;
  cursor: pointer;
  border-radius: 2px;
}
.views button:hover {
  background: #fff;
  border-color: var(--line);
}
.views button.on {
  background: var(--accent-soft);
  border-color: var(--accent);
  color: var(--accent);
}
.views .ic.s {
  width: 14px;
  height: 14px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.6;
}
.spacer {
  flex: 1;
}
.operator {
  appearance: none;
  border: 1px solid var(--line);
  background: #fff;
  font-family: var(--font);
  font-size: 12px;
  color: var(--ink);
  padding: 0 10px;
  align-self: center;
  height: 26px;
  cursor: pointer;
  border-radius: 2px;
}
.operator:hover {
  border-color: var(--accent);
  color: var(--accent);
}
</style>
