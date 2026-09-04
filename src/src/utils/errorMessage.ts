/** 错误 → 用户可读文案（全仓统一出口，替代散落的 instanceof 三元判断）。 */
export function toErrorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
