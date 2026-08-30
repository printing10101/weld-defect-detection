/**
 * 三员身份认证（Pinia，C-06/C-07）——登录态、会话令牌与空闲超时的唯一状态源。
 *
 * 职责：
 * - 登录：SM2 挑战-响应（私钥文件内容交本机后端代签，前端不碰密码学——
 *   单机本地软件可接受的简化，见 services/api.login 注释）；
 * - 令牌持久化（localStorage），实际请求头注入统一在 services/api.ts；
 * - 空闲超时（默认 15min，以后端返回 idle_timeout_min 为准）：监听用户活动，
 *   超时同步登出（后端会话同样已滑动过期，双保险）；
 * - 401 会话失效（后端过期/注销）：清除登录态，由路由守卫跳转登录页。
 */
import { defineStore } from "pinia";
import { computed, ref } from "vue";

import { AUTH_UNAUTHORIZED_EVENT, getChallenge, getMe, login as apiLogin, logout as apiLogout } from "../services/api";
import { clearToken, getToken, setToken } from "../services/authToken";

const IDLE_CHECK_MS = 30_000; // 空闲检查周期

export const useAuthStore = defineStore("auth", () => {
  const token = ref<string>(getToken());
  const username = ref<string>("");
  const role = ref<string>("");
  const accountId = ref<string>("");

  const isLoggedIn = computed(() => token.value !== "");

  let idleTimer: number | undefined;
  let lastActivity = Date.now();
  let idleTimeoutMs = 15 * 60_000; // 默认 15min，登录成功后以后端配置为准

  /** 登录：用户名 + SM2 私钥文件内容（后端代签）。 */
  async function login(name: string, privateKey: string): Promise<void> {
    const challenge = await getChallenge();
    const out = await apiLogin(name, challenge.challenge_id, privateKey);
    token.value = out.token;
    username.value = out.username;
    role.value = out.role;
    accountId.value = out.account_id;
    setToken(out.token);
    idleTimeoutMs = Math.max(1, out.idle_timeout_min) * 60_000;
    lastActivity = Date.now();
  }

  /** 恢复会话：页面刷新后用本地 token 校验身份（失败由 401 事件统一登出）。 */
  async function restore(): Promise<boolean> {
    if (!token.value) return false;
    try {
      const me = await getMe();
      username.value = me.username;
      role.value = me.role;
      accountId.value = me.account_id;
      return true;
    } catch {
      return false;
    }
  }

  /** 登出（尽力通知后端吊销会话；本地状态一律清除）。 */
  async function logout(): Promise<void> {
    try {
      if (token.value) await apiLogout();
    } catch {
      /* 后端不可达也照常本地登出 */
    }
    _clear();
  }

  function _clear(): void {
    token.value = "";
    username.value = "";
    role.value = "";
    accountId.value = "";
    clearToken();
    stopIdleWatch();
  }

  // ---- 空闲超时（前端同步登出，与后端滑动过期双保险） ----

  function markActivity(): void {
    lastActivity = Date.now();
  }

  function onIdleCheck(): void {
    if (token.value && Date.now() - lastActivity > idleTimeoutMs) {
      void logout();
    }
  }

  function onUnauthorized(): void {
    // 后端 401（过期/注销）：仅清本地态，不回呼后端
    _clear();
  }

  /** 绑定全局活动监听 + 空闲检查（App 挂载时调用一次）。 */
  function bindIdleWatch(): void {
    window.addEventListener("mousemove", markActivity, { passive: true });
    window.addEventListener("keydown", markActivity);
    window.addEventListener(AUTH_UNAUTHORIZED_EVENT, onUnauthorized);
    idleTimer = window.setInterval(onIdleCheck, IDLE_CHECK_MS);
  }

  function stopIdleWatch(): void {
    if (idleTimer !== undefined) {
      window.clearInterval(idleTimer);
      idleTimer = undefined;
    }
  }

  function unbindIdleWatch(): void {
    stopIdleWatch();
    window.removeEventListener("mousemove", markActivity);
    window.removeEventListener("keydown", markActivity);
    window.removeEventListener(AUTH_UNAUTHORIZED_EVENT, onUnauthorized);
  }

  return {
    token,
    username,
    role,
    accountId,
    isLoggedIn,
    login,
    restore,
    logout,
    bindIdleWatch,
    unbindIdleWatch,
  };
});
