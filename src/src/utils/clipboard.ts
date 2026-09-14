/** 剪贴板复制（带降级）：WebView 剪贴板权限缺失/非安全上下文时退回
 *  execCommand；调用方据返回值给操作员「已复制/请手动复制」反馈，
 *  不允许静默失败（私钥这类一次性内容复制失败代价极高）。 */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* 权限拒绝/上下文不允许 → 走降级 */
  }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  } catch {
    return false;
  }
}
