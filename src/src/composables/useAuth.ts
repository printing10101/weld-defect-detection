/**
 * 鉴权会话（§T3，前端）。
 * 模块级响应式单例：登录态在 Journey/Archive 视图间共享；不引入 Pinia/vue-router，
 * 与现有单页架构一致。令牌持久化见 services/auth.ts，请求头注入见 services/api.ts。
 */
import { computed, reactive } from "vue";
import { getMe, login as apiLogin, logout as apiLogout } from "../services/api";
import { getToken } from "../services/auth";
import type { UserOut } from "../types/api";

interface AuthState {
  user: UserOut | null;
  /** bootstrap 是否完成（已用本地令牌校验过有效性）。 */
  ready: boolean;
}

const state = reactive<AuthState>({ user: null, ready: false });

export function useAuth() {
  const isAuthenticated = computed(() => state.user !== null);

  /** 启动时若本地有令牌，拉取当前用户校验有效性；失败则清除（令牌过期/服务端重置）。 */
  async function bootstrap(): Promise<void> {
    if (!getToken()) {
      state.ready = true;
      return;
    }
    try {
      state.user = await getMe();
    } catch {
      state.user = null;
    } finally {
      state.ready = true;
    }
  }

  async function login(username: string, password: string): Promise<void> {
    const out = await apiLogin({ username, password });
    state.user = out.user;
  }

  function logout(): void {
    apiLogout();
    state.user = null;
  }

  return { state, isAuthenticated, bootstrap, login, logout };
}
