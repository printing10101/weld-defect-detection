<script setup lang="ts">
import { onMounted } from "vue";
import AppShell from "./components/AppShell.vue";
import LoginView from "./views/LoginView.vue";
import { useAuth } from "./composables/useAuth";
import { AUTH_UNAUTHORIZED_EVENT } from "./services/api";

const auth = useAuth();

onMounted(() => {
  void auth.bootstrap();
  // 令牌在服务端失效（过期/被禁用）时，任何请求遇 401 都会广播该事件 → 立即返回登录态
  window.addEventListener(AUTH_UNAUTHORIZED_EVENT, () => auth.logout());
});
</script>

<template>
  <LoginView v-if="!auth.isAuthenticated.value" />
  <AppShell v-else />
</template>
