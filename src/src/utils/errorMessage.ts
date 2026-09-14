/** 错误 → 用户可读文案（全仓统一出口，替代散落的 instanceof 三元判断）。
 *  ApiRequestError 的 message 已是纯中文（技术 code 走独立字段，见 services/api.ts）；
 *  非 Error 对象不再直接 String()（会显示 [object Object]），给出通用兜底。 */
export function toErrorMessage(e: unknown): string {
  if (e instanceof Error && e.message) return e.message;
  if (typeof e === "string" && e.trim()) return e;
  return "操作失败，请重试；若持续出现请联系系统管理员查看服务日志。";
}
