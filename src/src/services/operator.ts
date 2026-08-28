/** 操作员姓名（单机科研自用，无用户系统）：存 localStorage，空时回退 "local"。 */

const KEY = "scan_operator_name";

export function getOperatorName(): string {
  return (localStorage.getItem(KEY) ?? "").trim() || "local";
}

export function setOperatorName(name: string): void {
  const trimmed = name.trim();
  if (trimmed) {
    localStorage.setItem(KEY, trimmed);
  } else {
    localStorage.removeItem(KEY);
  }
}
