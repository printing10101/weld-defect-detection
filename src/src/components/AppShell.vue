<script setup lang="ts">
/** 应用壳（AutoCAD/WPS 桌面软件范式）：
 *  MenuBar（菜单栏）→ RibbonBar（工具栏）→ 内联文档标签区（doctabs）→ main（RouterView 工作区）→ StatusBar（状态栏）。
 *  当前工作区由 Vue Router 驱动，操作员状态来自 Pinia workspace store。
 *  快捷键：Ctrl+1..6 切换工作区，Ctrl+O/Ctrl+Shift+O 打开影像/批量导入。
 *  帮助菜单提供「快捷键」「关于」模态（桌面软件标准 About 对话框）。 */
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { RouterView, useRoute, useRouter } from "vue-router";
import pkg from "../../package.json";
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
const appVersion = pkg.version;

const TABS: { id: ViewId; label: string }[] = [
  { id: "journey", label: "单幅评定" },
  { id: "batch", label: "批量评定" },
  { id: "archive", label: "检测档案" },
  { id: "device", label: "设备标定" },
  { id: "viewer", label: "底片观察" },
  { id: "std-eval", label: "系统评价" },
  { id: "llm", label: "本地大模型" },
];
function goto(v: ViewId): void {
  router.push({ name: v });
}

/** 菜单/快捷键「打开影像/批量导入」：先下发意图再跳转，
 *  目标视图（含已挂载场景）经 workspace 消费意图并弹出文件选择器。 */
function openImage(): void {
  workspace.requestFileOpen("image");
  goto("journey");
}
function openBatch(): void {
  workspace.requestFileOpen("batch");
  goto("batch");
}

/* ── 检测人员信息：自建输入对话框。
 *  Tauri WebView 不实现 window.prompt（恒返回 null，此前该菜单点了永远没反应），
 *  项目内已有同款结论（ConfirmDialog 注释），交互一律走自建对话框。 ── */
const operatorDialogOpen = ref(false);
const operatorDraft = ref("");
const operatorInput = ref<HTMLInputElement | null>(null);

function editOperator(): void {
  operatorDraft.value = operator.value;
  operatorDialogOpen.value = true;
  void nextTick(() => operatorInput.value?.focus());
}

function confirmOperator(): void {
  workspace.setOperator(operatorDraft.value.trim());
  operatorDialogOpen.value = false;
}

/* ── 退出：尽力 window.close()；Tauri 主窗口通常不允许脚本关闭，
 *  300ms 后仍在运行则如实告知操作员正确的退出方式（此前点了毫无反应）。 ── */
const exitHintOpen = ref(false);

function exitApp(): void {
  window.close();
  window.setTimeout(() => {
    exitHintOpen.value = true;
  }, 300);
}

function onAction(id: string): void {
  switch (id) {
    case "open-image":
      openImage();
      break;
    case "open-batch":
      openBatch();
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
    case "view-llm":
      goto("llm");
      break;
    case "view-admin":
      goto("admin");
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
      exitApp();
      break;
  }
}

/* ── 帮助模态键盘支持：ESC 关闭 + 打开时焦点落「确定」（此前焦点留在触发按钮，
 *  Enter 会再次打开模态；也没有任何键盘关闭途径） ── */
const helpOkBtn = ref<HTMLButtonElement | null>(null);

function onModalKeydown(e: KeyboardEvent): void {
  if (e.key === "Escape" && helpModal.value !== "none") {
    helpModal.value = "none";
  }
}

watch(helpModal, (v) => {
  if (v !== "none") void nextTick(() => helpOkBtn.value?.focus());
});

function onKeydown(e: KeyboardEvent): void {
  if (!(e.ctrlKey || e.metaKey)) return;
  const k = e.key.toLowerCase();
  if (k === "1") goto("journey");
  else if (k === "2") goto("batch");
  else if (k === "3") goto("archive");
  else if (k === "4") goto("device");
  else if (k === "5") goto("viewer");
  else if (k === "6") goto("std-eval");
  else if (k === "7") goto("llm");
  else if (k === "o" && e.shiftKey) openBatch();
  else if (k === "o") openImage();
  else return;
  e.preventDefault();
}

onMounted(() => {
  window.addEventListener("keydown", onKeydown);
  window.addEventListener("keydown", onModalKeydown);
});
onBeforeUnmount(() => {
  window.removeEventListener("keydown", onKeydown);
  window.removeEventListener("keydown", onModalKeydown);
});
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
      <div
        class="dialog"
        role="dialog"
        aria-modal="true"
        :aria-label="helpModal === 'shortcuts' ? '键盘快捷键' : '关于'"
      >
        <div class="d-head">
          {{ helpModal === "shortcuts" ? "键盘快捷键" : "关于" }}
          <button
            type="button"
            class="x"
            aria-label="关闭"
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
              <tr><td>单幅评定</td><td>Ctrl+1</td></tr>
              <tr><td>批量评定</td><td>Ctrl+2</td></tr>
              <tr><td>检测档案</td><td>Ctrl+3</td></tr>
              <tr><td>设备标定</td><td>Ctrl+4</td></tr>
              <tr><td>底片观察</td><td>Ctrl+5</td></tr>
              <tr><td>系统评价</td><td>Ctrl+6</td></tr>
              <tr><td>本地大模型</td><td>Ctrl+7</td></tr>
              <tr><td>退出</td><td>Alt+F4 / 窗口关闭按钮</td></tr>
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
            版本 {{ appVersion }} · 本地化部署
          </p>
          <p class="ver">
            依据 NB/T 47013.2-2015 执行焊缝缺陷智能评定<br>
            检测数据全程本地化处理，不经外部网络传输
          </p>
        </div>
        <div class="d-foot">
          <button
            ref="helpOkBtn"
            type="button"
            class="ok"
            @click="helpModal = 'none'"
          >
            确定
          </button>
        </div>
      </div>
    </div>

    <!-- 检测人员信息对话框（替代 Tauri 下失效的 window.prompt） -->
    <div
      v-if="operatorDialogOpen"
      class="overlay"
      @click.self="operatorDialogOpen = false"
      @keydown.escape="operatorDialogOpen = false"
    >
      <div
        class="dialog"
        role="dialog"
        aria-modal="true"
        aria-label="检测人员信息"
      >
        <div class="d-head">
          检测人员信息
          <button
            type="button"
            class="x"
            aria-label="关闭"
            @click="operatorDialogOpen = false"
          >
            ×
          </button>
        </div>
        <div class="d-body">
          <label
            class="op-field"
            for="operator-name"
          >检测人员姓名（用于报告签署与审计追溯）</label>
          <input
            id="operator-name"
            ref="operatorInput"
            v-model="operatorDraft"
            type="text"
            @keyup.enter="confirmOperator"
          >
        </div>
        <div class="d-foot">
          <button
            type="button"
            class="ok"
            @click="operatorDialogOpen = false"
          >
            取消
          </button>
          <button
            type="button"
            class="ok primary"
            @click="confirmOperator"
          >
            保存
          </button>
        </div>
      </div>
    </div>

    <!-- 退出提示：脚本关闭被运行环境拦截时的如实告知 -->
    <div
      v-if="exitHintOpen"
      class="overlay"
      @click.self="exitHintOpen = false"
    >
      <div
        class="dialog"
        role="alertdialog"
        aria-modal="true"
        aria-label="退出应用"
      >
        <div class="d-head">
          退出应用
        </div>
        <div class="d-body">
          当前运行环境不允许本页自动关闭窗口。请点击窗口右上角的关闭按钮 ×，或按
          Alt+F4 退出应用。
        </div>
        <div class="d-foot">
          <button
            type="button"
            class="ok"
            @click="exitHintOpen = false"
          >
            知道了
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
.d-foot .ok.primary {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
.op-field {
  display: block;
  font-size: 12px;
  color: var(--ink-soft);
  margin-bottom: 6px;
}
.d-body input[type="text"] {
  width: 100%;
  box-sizing: border-box;
  padding: 7px 9px;
  font-size: 13px;
  font-family: var(--font);
  border: 1px solid var(--line-strong);
  border-radius: 3px;
}
</style>
