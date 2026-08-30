/**
 * 会话令牌存取（localStorage）——供 services/api.ts 与 stores/auth.ts 共用。
 * C-06/C-07：登录态下后端以账号为准（X-Operator-Name 头不再构成身份）。
 */

const TOKEN_KEY = "scan_auth_token";

export function getToken(): string {
  return (localStorage.getItem(TOKEN_KEY) ?? "").trim();
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}
