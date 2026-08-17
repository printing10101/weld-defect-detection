<script setup lang="ts">
import { onMounted, onUnmounted, ref } from "vue";
import AppShell from "./components/AppShell.vue";
import LoginView from "./views/LoginView.vue";
import { useAuth } from "./composables/useAuth";
import { AUTH_UNAUTHORIZED_EVENT, BACKEND_DOWN_EVENT, BACKEND_UP_EVENT } from "./services/api";

const auth = useAuth();
const backendDown = ref(false);

function onBackendDown(): void {
  backendDown.value = true;
}
function onBackendUp(): void {
  backendDown.value = false;
}

onMounted(() => {
  void auth.bootstrap();
  // 令牌在服务端失效（过期/被禁用）时，任何请求遇 401 都会广播该事件 → 立即返回登录态
  window.addEventListener(AUTH_UNAUTHORIZED_EVENT, () => auth.logout());
  // 后端不可达/超时 → 全局离线横幅；任意成功响应 → 清除（§优化 F18）
  window.addEventListener(BACKEND_DOWN_EVENT, onBackendDown);
  window.addEventListener(BACKEND_UP_EVENT, onBackendUp);
});

onUnmounted(() => {
  window.removeEventListener(AUTH_UNAUTHORIZED_EVENT, () => auth.logout());
  window.removeEventListener(BACKEND_DOWN_EVENT, onBackendDown);
  window.removeEventListener(BACKEND_UP_EVENT, onBackendUp);
});
</script>

<template>
  <div v-if="!auth.isAuthenticated.value">
    <LoginView />
  </div>
  <AppShell v-else />

  <!-- 全局离线横幅：任意请求后端不可达/超时时显示，恢复后自动隐藏（§优化 F18） -->
  <transition name="fade">
    <div v-if="backendDown" class="offline-banner" role="alert">
      <span class="dot" />
      后端未响应：请确认本地服务已启动（默认 127.0.0.1:18773）
    </div>
  </transition>
</template>

<style scoped>
.offline-banner {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1000;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  background: #b3261e;
  color: #fff;
  font-size: 13px;
  font-weight: 600;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
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
