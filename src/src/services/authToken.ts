/**
 * 会话令牌存取（sessionStorage）——供 services/api.ts 与 stores/auth.ts 共用。
 * C-06/C-07：登录态下后端以账号为准（X-Operator-Name 头不再构成身份）。
 *
 * 安全口径（GB/T 25000.51 信息安全性）：令牌存 sessionStorage 而非
 * localStorage——应用窗口关闭即随会话清除，不持久驻留 WebView 存储区；
 * 每次启动应用须重新登录（对涉密单机属预期管控）。
 */

const TOKEN_KEY = "scan_auth_token";
const GUEST_KEY = "scan_guest_mode";

export function getToken(): string {
  return (sessionStorage.getItem(TOKEN_KEY) ?? "").trim();
}

export function setToken(token: string): void {
  sessionStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  sessionStorage.removeItem(TOKEN_KEY);
}

/** 访客模式（后端 guest_mode 开启时免登录浏览全部功能）：随会话清除。 */
export function isGuestMode(): boolean {
  return sessionStorage.getItem(GUEST_KEY) === "1";
}

export function setGuestMode(): void {
  sessionStorage.setItem(GUEST_KEY, "1");
}

export function clearGuestMode(): void {
  sessionStorage.removeItem(GUEST_KEY);
}
