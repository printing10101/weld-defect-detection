<script setup lang="ts">
/** 经典菜单栏（WPS/AutoCAD 范式）：文件/视图/工具/帮助 四组下拉菜单。
 *  交互约定：单击菜单名展开，展开后悬停切换，单击外部或选中项后收起；
 *  所有动作以 action 事件上抛，由 AppShell 统一分发。 */
import { onBeforeUnmount, onMounted, ref } from "vue";
import { useRouter } from "vue-router";
import { useAuthStore } from "../stores/auth";

interface MenuItem {
  id: string;
  label: string;
  shortcut?: string;
  separator?: boolean;
}
interface Menu {
  id: string;
  label: string;
  items: MenuItem[];
}

defineProps<{ activeView: string }>();
const emit = defineEmits<{ action: [id: string] }>();

// 工作区名映射（表驱动：三元链漏分支时 viewer/std-eval 等页会错显示成"设备标定"）。
// 键与 ViewId（types/api.ts）一致。
const WORKSPACE_NAMES: Record<string, string> = {
  journey: "单张检测",
  batch: "批量检测",
  archive: "档案检索",
  viewer: "底片查看",
  "std-eval": "系统评价",
  device: "设备标定",
};

// 三员认证（C-06）：顶栏展示当前登录身份，支持手动登出
const auth = useAuthStore();
const router = useRouter();
const ROLE_NAMES: Record<string, string> = {
  sysadmin: "系统管理员",
  secadmin: "安全保密管理员",
  auditor: "安全审计员",
};
function logout(): void {
  void auth.logout().then(() => router.push("/login"));
}

const MENUS: Menu[] = [
  {
    id: "file",
    label: "文件",
    items: [
      { id: "open-image", label: "打开影像…", shortcut: "Ctrl+O" },
      { id: "open-batch", label: "批量导入…", shortcut: "Ctrl+Shift+O" },
      { id: "sep1", label: "", separator: true },
      { id: "exit", label: "退出", shortcut: "Alt+F4" },
    ],
  },
  {
    id: "view",
    label: "视图",
    items: [
      { id: "view-journey", label: "单张检测", shortcut: "Ctrl+1" },
      { id: "view-batch", label: "批量检测", shortcut: "Ctrl+2" },
      { id: "view-archive", label: "档案检索", shortcut: "Ctrl+3" },
      { id: "view-device", label: "设备标定", shortcut: "Ctrl+4" },
      { id: "view-viewer", label: "底片查看", shortcut: "Ctrl+5" },
      { id: "view-std-eval", label: "系统评价", shortcut: "Ctrl+6" },
    ],
  },
  {
    id: "tools",
    label: "工具",
    items: [{ id: "operator", label: "操作员设置…" }],
  },
  {
    id: "help",
    label: "帮助",
    items: [
      { id: "shortcuts", label: "快捷键…" },
      { id: "about", label: "关于…" },
    ],
  },
];

const openMenu = ref<string | null>(null);

function toggle(menuId: string): void {
  openMenu.value = openMenu.value === menuId ? null : menuId;
}
function enter(menuId: string): void {
  // 已有菜单展开时，悬停切换到其它菜单（桌面软件标准行为）
  if (openMenu.value !== null) openMenu.value = menuId;
}
function pick(id: string): void {
  openMenu.value = null;
  emit("action", id);
}
function onDocClick(): void {
  openMenu.value = null;
}
onMounted(() => document.addEventListener("click", onDocClick));
onBeforeUnmount(() => document.removeEventListener("click", onDocClick));
</script>

<template>
  <div
    class="menubar"
    @click.stop
  >
    <!-- 品牌区（AutoCAD 左上角应用标识位） -->
    <div class="brand">
      <span
        class="mark"
        aria-hidden="true"
      />
      <span class="name">射线焊缝缺陷智能检测系统</span>
    </div>

    <div
      v-for="menu in MENUS"
      :key="menu.id"
      class="menu"
      @mouseenter="enter(menu.id)"
    >
      <button
        type="button"
        :class="{ open: openMenu === menu.id }"
        @click="toggle(menu.id)"
      >
        {{ menu.label }}
      </button>
      <div
        v-if="openMenu === menu.id"
        class="dropdown"
      >
        <template
          v-for="item in menu.items"
          :key="item.id"
        >
          <div
            v-if="item.separator"
            class="sep"
          />
          <button
            v-else
            type="button"
            class="item"
            @click="pick(item.id)"
          >
            <span class="lab">{{ item.label }}</span>
            <span
              v-if="item.shortcut"
              class="key"
            >{{ item.shortcut }}</span>
          </button>
        </template>
      </div>
    </div>

    <div class="spacer" />
    <!-- 右侧：当前工作区指示（AutoCAD 顶栏上下文信息） -->
    <div class="ctx">
      工作区：{{
        WORKSPACE_NAMES[activeView] ?? "设备标定"
      }}
    </div>
    <!-- 右侧：当前登录身份（三员之一）+ 登出 -->
    <div
      v-if="auth.isLoggedIn"
      class="ctx user"
    >
      {{ auth.username }}（{{ ROLE_NAMES[auth.role] ?? auth.role }}）
      <button
        type="button"
        class="logout"
        @click="logout"
      >登出</button>
    </div>
  </div>
</template>

<style scoped>
.logout {
  margin-left: 8px;
  padding: 1px 8px;
  font-size: 11px;
  cursor: pointer;
  border: 1px solid var(--line, #ccc);
  border-radius: 2px;
  background: #fff;
}
.menubar {
  display: flex;
  align-items: stretch;
  height: 30px;
  background: linear-gradient(180deg, #fafafa, #f0f0f2);
  border-bottom: 1px solid var(--line);
  user-select: none;
  flex: none;
}
.brand {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 0 14px 0 10px;
  font-size: 12px;
  font-weight: 600;
  color: var(--ink);
  border-right: 1px solid var(--line-soft);
}
.brand .mark {
  width: 12px;
  height: 12px;
  background: var(--accent);
}
.menu {
  position: relative;
  display: flex;
}
.menu > button {
  appearance: none;
  border: 0;
  background: transparent;
  font-family: var(--font);
  font-size: 12px;
  color: var(--ink-soft);
  padding: 0 12px;
  cursor: pointer;
}
.menu > button:hover,
.menu > button.open {
  background: var(--accent-soft);
  color: var(--ink);
}
.dropdown {
  position: absolute;
  top: 100%;
  left: 0;
  min-width: 180px;
  background: #fff;
  border: 1px solid var(--line);
  box-shadow: 0 3px 10px rgba(0, 0, 0, 0.14);
  padding: 3px;
  z-index: 200;
}
.dropdown .sep {
  height: 1px;
  background: var(--line-soft);
  margin: 3px 6px;
}
.dropdown .item {
  appearance: none;
  width: 100%;
  border: 0;
  background: transparent;
  font-family: var(--font);
  font-size: 12px;
  color: var(--ink);
  padding: 5px 10px;
  display: flex;
  justify-content: space-between;
  gap: 24px;
  cursor: pointer;
  text-align: left;
}
.dropdown .item:hover {
  background: var(--accent);
  color: #fff;
}
.dropdown .item .key {
  color: var(--ink-faint);
  font-size: 11px;
}
.dropdown .item:hover .key {
  color: rgba(255, 255, 255, 0.8);
}
.spacer {
  flex: 1;
}
.ctx {
  display: flex;
  align-items: center;
  padding: 0 12px;
  font-size: 11px;
  color: var(--ink-faint);
  border-left: 1px solid var(--line-soft);
}
</style>
