<script setup lang="ts">
/** 全局离线/模型加载提示的宿主；状态与轮询逻辑已下沉到 Pinia backend store。 */
import { onMounted, onUnmounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import AppShell from "./components/AppShell.vue";
import ConfirmDialog from "./components/ConfirmDialog.vue";
import LoginView from "./views/LoginView.vue";
import { useAuthStore } from "./stores/auth";
import { useBackendStore } from "./stores/backend";
import { busyTask } from "./stores/busy";

const backend = useBackendStore();
const auth = useAuthStore();
const route = useRoute();
const router = useRouter();

onMounted(() => {
  // 后端不可达/超时 → 全局离线横幅 + 自动轮询恢复；任意成功响应 → 清除
  backend.bind();
  backend.start();
  // 三员认证（C-06/C-07）：空闲超时登出 + 401 会话失效监听
  auth.bindIdleWatch();
  // 刷新后恢复会话身份（token 失效时由 401 事件统一清除并跳登录页）
  void auth.restore();
  bindDesktopCloseGuard();
});

/** preload 注入的桌面桥（Electron 环境）；web 环境为 undefined。 */
const desktopBridge = (
  window as unknown as {
    desktop?: {
      setBusy: (busy: boolean) => void;
      onCloseRequested: (cb: () => void) => () => void;
      confirmQuit: () => void;
    };
  }
).desktop;

// Electron 桌面环境的窗口关闭拦截（web 环境无 window.desktop，静默跳过）。
// 点 × 的关闭请求由主进程拦下（close 事件 preventDefault），经 desktop 桥
// 通知这里弹自建确认框；确认后经 confirmQuit() 真正关闭。
function bindDesktopCloseGuard(): void {
  if (!desktopBridge) return;
  desktopBridge.onCloseRequested(() => {
    closeGuardOpen.value = true; // 弹自建确认框；确认后经 confirmQuit() 真正关闭
  });
}

/** 关闭确认对话框状态：主进程拦截到长任务时的点 × 关闭后置位。 */
const closeGuardOpen = ref(false);

function confirmClose(): void {
  closeGuardOpen.value = false;
  busyTask.value = null; // 用户已确认放弃任务
  desktopBridge?.confirmQuit();
}

// 长任务（批量/单幅评定）进行中的关闭/刷新保护：
//   Electron 桌面态 → 主进程 close 拦截（beforeunload 在 Electron 会静默取消
//   close、弹不出确认框，故改走 desktop 桥，且不设 beforeunload 以免抢先
//   取消 close 让确认框永远弹不出来）；
//   web 态（vite dev 浏览器调试）→ beforeunload 兜底。
watch(busyTask, (busy) => {
  if (desktopBridge) {
    desktopBridge.setBusy(Boolean(busy));
    window.onbeforeunload = null;
    return;
  }
  window.onbeforeunload = busy
    ? (e) => {
        e.preventDefault();
        e.returnValue = "评定任务正在进行中，关闭将中断任务。";
        return e.returnValue;
      }
    : null;
});

// 登出（手动/空闲超时/401 失效）→ 跳转登录页
watch(
  () => auth.isLoggedIn,
  (authed) => {
    if (!authed && route.name !== "login") void router.push("/login");
  },
);

onUnmounted(() => {
  backend.unbind();
  auth.unbindIdleWatch();
});
</script>

<template>
  <!-- 登录页独立成页（不套工作台外壳）；登录后进入标准工作台 -->
  <LoginView v-if="route.name === 'login'" />
  <AppShell v-else />

  <!-- 全局状态横幅：离线（红）优先于模型加载中（琥珀）；恢复后自动隐藏（§优化 F18）。
       登录页同样显示——冷启动导入阶段（/health 不可达）用户停在登录页，
       没有横幅会误以为"后端没启动"。 -->
  <transition name="fade">
    <div
      v-if="backend.backendDown"
      class="offline-banner"
      role="alert"
    >
      <span class="dot" />
      推理服务正在启动或未连接，系统自动重试中…（冷启动需导入推理依赖，可能需要
      1~3 分钟，请保持窗口开启）
    </div>
    <div
      v-else-if="backend.modelLoading"
      class="offline-banner loading"
      role="status"
    >
      <span class="dot" />
      评定模型加载中，评定功能稍后可用（档案查阅不受影响）…
    </div>
  </transition>

  <!-- 空闲登出预警（琥珀色，活动即消失）：给操作员"动一下鼠标"的机会，
       避免填到一半的报告补充信息随自动登出静默丢失 -->
  <transition name="fade">
    <div
      v-if="auth.isLoggedIn && auth.idleWarning"
      class="offline-banner idle-warn"
      role="alert"
    >
      <span class="dot" />
      已空闲较久，即将自动退出登录（约 1 分钟内）；移动鼠标或按任意键可继续保持登录
    </div>
  </transition>

  <!-- Electron 点 × 关闭窗口时的任务保护确认（主进程拦截后经 desktop 桥弹此处） -->
  <ConfirmDialog
    :open="closeGuardOpen"
    title="退出确认"
    :message="
      busyTask === 'batch'
        ? '批量评定正在进行中，关闭窗口将中断本机评定任务（已完成的结果已保存，未完成的可重新提交）。确定要关闭吗？'
        : '单幅评定正在进行中，关闭窗口将放弃本次评定。确定要关闭吗？'
    "
    confirm-text="关闭窗口"
    danger
    @confirm="confirmClose"
    @cancel="closeGuardOpen = false"
  />
</template>

<style scoped>
.offline-banner {
  position: fixed;
  /* 贴在状态栏上方（底部），不遮挡菜单栏/工具栏操作（桌面软件通知惯例） */
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 1000;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border-radius: 2px;
  background: #b3261e;
  color: #fff;
  font-size: 12px;
  font-weight: 600;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
/* 模型加载中：信息性提示（琥珀色），区别于错误（红色） */
.offline-banner.loading {
  background: #7a5900;
}
/* 空闲登出预警：中性琥珀，与模型加载同色系但独立状态 */
.offline-banner.idle-warn {
  background: #7a5900;
}
.offline-banner .dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #fff;
  animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
  0%,
  100% {
    opacity: 0.4;
  }
  50% {
    opacity: 1;
  }
}
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.25s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
