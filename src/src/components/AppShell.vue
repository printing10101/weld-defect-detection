<script setup lang="ts">
/** 应用壳（AutoCAD/WPS 桌面软件范式）：
 *  MenuBar（菜单栏）→ RibbonBar（工具栏）→ 内联文档标签区（doctabs）→ main（RouterView 工作区）→ StatusBar（状态栏）。
 *  当前工作区由 Vue Router 驱动，操作员状态来自 Pinia workspace store。
 *  快捷键：Ctrl+1..6 切换工作区，Ctrl+O/Ctrl+Shift+O 打开影像/批量导入。
 *  帮助菜单提供「快捷键」「关于」模态（桌面软件标准 About 对话框）。 */
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { RouterView, useRoute, useRouter } from "vue-router";
import type { ViewId } from "../types/api";
import MenuBar from "./MenuBar.vue";
import RibbonBar from "./RibbonBar.vue";
import StatusBar from "./StatusBar.vue";
import { routeNameToViewId } from "../router";
import { useWorkspaceStore } from "../stores/workspace";

const route = useRoute();
const router = useRouter();
const workspace = useWorkspaceStore();

/** 当前工作区（= 路由名，与 ViewId 一一对应）。 */
const view = computed<ViewId>(() => routeNameToViewId(route.name));
const operator = computed(() => workspace.operator);
const helpModal = ref<"none" | "shortcuts" | "about">("none");

const TABS: { id: ViewId; label: string }[] = [
  { id: "journey", label: "单张检测" },
  { id: "batch", label: "批量检测" },
  { id: "archive", label: "档案检索" },
  { id: "device", label: "设备标定" },
  { id: "viewer", label: "底片查看" },
  { id: "std-eval", label: "系统评价" },
];
function goto(v: ViewId): void {
  router.push({ name: v });
}

function editOperator(): void {
  const name = window.prompt("操作员姓名（用于报告签名与审计留痕）", operator.value);
  if (name === null) return; // 取消
  workspace.setOperator(name);
}

function onAction(id: string): void {
  switch (id) {
    case "open-image":
      goto("journey");
      break;
    case "open-batch":
      goto("batch");
      break;
    case "view-journey":
      goto("journey");
      break;
    case "view-batch":
      goto("batch");
      break;
    case "view-archive":
      goto("archive");
      break;
    case "view-device":
      goto("device");
      break;
    case "view-viewer":
      goto("viewer");
      break;
    case "view-std-eval":
      goto("std-eval");
      break;
    case "operator":
      editOperator();
      break;
    case "shortcuts":
      helpModal.value = "shortcuts";
      break;
    case "about":
      helpModal.value = "about";
      break;
    case "exit":
      window.close();
      break;
  }
}

function onKeydown(e: KeyboardEvent): void {
  if (!(e.ctrlKey || e.metaKey)) return;
  const k = e.key.toLowerCase();
  if (k === "1") goto("journey");
  else if (k === "2") goto("batch");
  else if (k === "3") goto("archive");
  else if (k === "4") goto("device");
  else if (k === "5") goto("viewer");
  else if (k === "6") goto("std-eval");
  else if (k === "o" && e.shiftKey) goto("batch");
  else if (k === "o") goto("journey");
  else return;
  e.preventDefault();
}

onMounted(() => window.addEventListener("keydown", onKeydown));
onBeforeUnmount(() => window.removeEventListener("keydown", onKeydown));
</script>

<template>
  <div class="app-shell">
    <MenuBar
      :active-view="view"
      @action="onAction"
    />
    <RibbonBar
      :active-view="view"
      :operator="operator"
      @action="onAction"
      @view="goto"
    />

    <!-- 文档标签页（WPS 多文档范式）：四个工作区标签 → 真实路由导航（T4-3） -->
    <div class="doctabs">
      <router-link
        v-for="t in TABS"
        :key="t.id"
        :to="{ name: t.id }"
        class="doctab"
        :class="{ on: view === t.id }"
      >
        {{ t.label }}
      </router-link>
    </div>

    <main class="main">
      <RouterView v-slot="{ Component }">
        <component
          :is="Component"
          :active="true"
          @archive="goto('archive')"
        />
      </RouterView>
    </main>

    <StatusBar />

    <!-- 帮助模态（桌面软件标准 About/快捷键对话框） -->
    <div
      v-if="helpModal !== 'none'"
      class="overlay"
      @click.self="helpModal = 'none'"
    >
      <div class="dialog">
        <div class="d-head">
          {{ helpModal === "shortcuts" ? "键盘快捷键" : "关于" }}
          <button
            type="button"
            class="x"
            @click="helpModal = 'none'"
          >
            ×
          </button>
        </div>
        <div
          v-if="helpModal === 'shortcuts'"
          class="d-body"
        >
          <table class="keys">
            <tbody>
              <tr><td>打开影像</td><td>Ctrl+O</td></tr>
              <tr><td>批量导入</td><td>Ctrl+Shift+O</td></tr>
              <tr><td>单张检测</td><td>Ctrl+1</td></tr>
              <tr><td>批量检测</td><td>Ctrl+2</td></tr>
              <tr><td>档案检索</td><td>Ctrl+3</td></tr>
              <tr><td>设备标定</td><td>Ctrl+4</td></tr>
              <tr><td>底片查看</td><td>Ctrl+5</td></tr>
              <tr><td>系统评价</td><td>Ctrl+6</td></tr>
              <tr><td>退出</td><td>Alt+F4</td></tr>
            </tbody>
          </table>
        </div>
        <div
          v-else
          class="d-body about"
        >
          <div class="logo" />
          <p class="app-name">
            射线焊缝缺陷智能检测系统
          </p>
          <p class="ver">
            版本 0.1.0 · 本地优先
          </p>
          <p class="ver">
            依据 NB/T47013.2-2015 进行焊缝缺陷智能评定<br>
            检测数据全程本地处理，不联网上传
          </p>
        </div>
        <div class="d-foot">
          <button
            type="button"
            class="ok"
            @click="helpModal = 'none'"
          >
            确定
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.doctabs {
  display: flex;
  align-items: flex-end;
  background: var(--panel-3);
  border-bottom: 1px solid var(--line);
  padding: 4px 6px 0;
  gap: 2px;
  flex: none;
}
.doctabs .doctab {
  appearance: none;
  text-decoration: none;
  border: 1px solid transparent;
  border-bottom: 0;
  background: transparent;
  font-family: var(--font);
  font-size: 12px;
  color: var(--ink-soft);
  padding: 5px 16px;
  cursor: pointer;
  border-radius: 3px 3px 0 0;
}
.doctabs .doctab:hover {
  color: var(--ink);
}
.doctabs .doctab.on {
  background: var(--panel);
  border-color: var(--line);
  color: var(--ink);
  position: relative;
  top: 1px;
}
.overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.25);
  display: grid;
  place-items: center;
  z-index: 500;
}
.dialog {
  min-width: 340px;
  max-width: 420px;
  background: var(--panel);
  border: 1px solid var(--line-strong);
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.2);
}
.d-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  font-size: 12px;
  font-weight: 600;
  background: linear-gradient(180deg, #fafafa, #f0f0f2);
  border-bottom: 1px solid var(--line);
}
.d-head .x {
  appearance: none;
  border: 0;
  background: transparent;
  font-size: 14px;
  color: var(--ink-soft);
  cursor: pointer;
  padding: 0 2px;
}
.d-head .x:hover {
  color: var(--signal);
}
.d-body {
  padding: 14px;
  font-size: 12px;
}
.keys {
  width: 100%;
}
.keys td {
  padding: 5px 6px;
}
.keys td:last-child {
  font-family: var(--mono);
  color: var(--accent);
  text-align: right;
}
.about {
  text-align: center;
}
.logo {
  width: 36px;
  height: 36px;
  background: var(--accent);
  margin: 4px auto 10px;
}
.app-name {
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 4px;
}
.ver {
  color: var(--ink-faint);
  line-height: 1.7;
  font-size: 12px;
}
.d-foot {
  padding: 8px 12px;
  border-top: 1px solid var(--line-soft);
  text-align: right;
}
.d-foot .ok {
  appearance: none;
  border: 1px solid var(--line-strong);
  background: linear-gradient(180deg, #fdfdfd, #f0f0f2);
  font-family: var(--font);
  font-size: 12px;
  padding: 5px 18px;
  cursor: pointer;
  border-radius: 2px;
}
.d-foot .ok:hover {
  border-color: var(--accent);
  color: var(--accent);
}
</style>
