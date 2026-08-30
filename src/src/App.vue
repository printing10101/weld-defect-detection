<script setup lang="ts">
/** 全局离线/模型加载提示的宿主；状态与轮询逻辑已下沉到 Pinia backend store。 */
import { onMounted, onUnmounted, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import AppShell from "./components/AppShell.vue";
import LoginView from "./views/LoginView.vue";
import { useAuthStore } from "./stores/auth";
import { useBackendStore } from "./stores/backend";

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

  <!-- 全局状态横幅：离线（红）优先于模型加载中（琥珀）；恢复后自动隐藏（§优化 F18） -->
  <transition name="fade">
    <div
      v-if="backend.backendDown && route.name !== 'login'"
      class="offline-banner"
      role="alert"
    >
      <span class="dot" />
      后端正在启动或未连接，正在自动重试…（首次启动加载模型可能需要 1~2 分钟）
    </div>
    <div
      v-else-if="backend.modelLoading && route.name !== 'login'"
      class="offline-banner loading"
      role="status"
    >
      <span class="dot" />
      模型加载中，检测功能稍后可用（浏览档案不受影响）…
    </div>
  </transition>
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
