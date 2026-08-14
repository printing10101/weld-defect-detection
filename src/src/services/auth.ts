/**
 * 鉴权令牌本地持久化（§T3）。
 * 仅存无状态访问令牌（HMAC 签名、到期失效），绝不持久化密码；令牌缺失/不可用时降级为内存态。
 */

const TOKEN_KEY = "scan_auth_token";

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string): void {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* 隐私模式/存储不可用时：静默降级，令牌仅留当前会话内存 */
  }
}

export function clearToken(): void {
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* ignore */
  }
}

/** 请求头：已登录则附带 X-Scan-Token（后端 get_current_user 读取该头）。 */
export function authHeaders(): Record<string, string> {
  const t = getToken();
  return t ? { "X-Scan-Token": t } : {};
}
