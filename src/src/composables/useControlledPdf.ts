/**
 * 受控导出（C-14）闭环：申请 → （保密员审批，自动轮询结论）→ 领一次性令牌 → 下载。
 * 从 ReportView 内联逻辑抽取为可复用组合式函数：报告页与批量结果行共用同一套
 * 「导出 PDF」交互；此前批量任务与直链入口绕过受控通道，需审批模式下会看到裸 401。
 */
import { onScopeDispose, ref } from "vue";

import {
  ApiRequestError,
  createExportRequest,
  downloadExportBlob,
  getExportRequest,
  issueExportToken,
  reportPdfUrl,
} from "../services/api";
import { useAuthStore } from "../stores/auth";
import { toErrorMessage } from "../utils/errorMessage";

/** 下载文件名安全化：去掉路径分隔与 Windows 非法字符。 */
function safeName(name: string): string {
  return name.replace(/[\\/:*?"<>|]/g, "_").slice(0, 120);
}

export function useControlledPdf() {
  const auth = useAuthStore();

  const gateOpen = ref(false);
  const gateBusy = ref(false);
  const gateMsg = ref<string | null>(null);
  const gateErr = ref<string | null>(null);
  const gateReason = ref("");
  /** 当前导出目标报告；null = 面板关闭。 */
  const reportId = ref<string | null>(null);
  const requestId = ref<string | null>(null);
  /** 审批状态（轮询 /export/requests/{id}；approved/rejected/pending）。 */
  const approval = ref<"pending" | "approved" | "rejected" | null>(null);

  let pollTimer: number | undefined;

  function requestKey(rid: string): string {
    return `export_req_${rid}`;
  }

  function savePdfBlob(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    // revoke 延迟到下一个宏任务之后：点击后立刻回收在部分内核下会截断下载
    window.setTimeout(() => URL.revokeObjectURL(url), 4000);
  }

  function stopPolling(): void {
    if (pollTimer !== undefined) {
      window.clearInterval(pollTimer);
      pollTimer = undefined;
    }
  }

  async function pollApproval(): Promise<void> {
    if (!requestId.value || !gateOpen.value) return;
    try {
      const out = await getExportRequest(requestId.value);
      if (out.status !== approval.value) {
        approval.value = out.status === "approved" ? "approved" : out.status === "rejected" ? "rejected" : "pending";
        if (approval.value === "approved") gateMsg.value = "申请已批准，点击「领取令牌并下载」完成导出。";
        else if (approval.value === "rejected") gateErr.value = "导出申请被安全保密管理员驳回，如需导出请重新申请并补充理由。";
      }
    } catch {
      /* 单次轮询失败不打断：申请人仍可手动点「领取令牌并下载」盲试 */
    }
  }

  function startPolling(): void {
    stopPolling();
    pollTimer = window.setInterval(() => void pollApproval(), 5000);
  }

  /** 打开报告 PDF：保密员直接预览；免审批/已持凭据直接下载；
   *  需审批则弹出受控导出面板（恢复本报告未决的申请）。 */
  async function openPdf(targetReportId: string, suggestName?: string): Promise<void> {
    reportId.value = targetReportId;
    const filename = `${safeName(suggestName ?? targetReportId)}.pdf`;
    if (auth.role === "secadmin") {
      // 安全保密管理员是审批人：端点内预授权直接放行，保留直链打开（新标签页预览）
      window.open(reportPdfUrl(targetReportId), "_blank", "noopener");
      reportId.value = null;
      return;
    }
    gateBusy.value = true;
    gateErr.value = null;
    gateMsg.value = null;
    try {
      savePdfBlob(
        await downloadExportBlob(`/report/${encodeURIComponent(targetReportId)}/pdf`),
        filename,
      );
      reportId.value = null;
      return;
    } catch (e) {
      if (e instanceof ApiRequestError && e.code === "EXPORT_TOKEN_REQUIRED") {
        requestId.value = sessionStorage.getItem(requestKey(targetReportId));
        approval.value = null;
        gateOpen.value = true;
        startPolling();
      } else {
        gateOpen.value = true;
        gateErr.value = toErrorMessage(e);
      }
    } finally {
      gateBusy.value = false;
    }
  }

  async function applyExportRequest(): Promise<void> {
    if (!reportId.value) return;
    gateBusy.value = true;
    gateErr.value = null;
    gateMsg.value = null;
    try {
      const out = await createExportRequest(
        `report:${reportId.value}`,
        gateReason.value.trim() || undefined,
      );
      requestId.value = out.request_id;
      sessionStorage.setItem(requestKey(reportId.value), out.request_id);
      approval.value = "pending";
      gateMsg.value = "申请已提交，等待安全保密管理员审批；批准后本面板会提示，届时点击「领取令牌并下载」。";
      startPolling();
    } catch (e) {
      gateErr.value = toErrorMessage(e);
    } finally {
      gateBusy.value = false;
    }
  }

  async function fetchTokenAndDownload(): Promise<void> {
    if (!requestId.value || !reportId.value) return;
    gateBusy.value = true;
    gateErr.value = null;
    gateMsg.value = null;
    try {
      const t = await issueExportToken(requestId.value);
      savePdfBlob(
        await downloadExportBlob(`/report/${encodeURIComponent(reportId.value)}/pdf`, t.token),
        `${safeName(reportId.value)}.pdf`,
      );
      sessionStorage.removeItem(requestKey(reportId.value));
      requestId.value = null;
      gateOpen.value = false;
      stopPolling();
      reportId.value = null;
    } catch (e) {
      gateErr.value = toErrorMessage(e);
    } finally {
      gateBusy.value = false;
    }
  }

  /** 理由输入回写（子组件经此更新，避免模板直改 prop）。 */
  function setReason(v: string): void {
    gateReason.value = v;
  }

  function closeGate(): void {
    gateOpen.value = false;
    stopPolling();
    gateErr.value = null;
    gateMsg.value = null;
    gateReason.value = "";
    reportId.value = null;
  }

  onScopeDispose(stopPolling);

  return {
    gateOpen,
    gateBusy,
    gateMsg,
    gateErr,
    gateReason,
    reportId,
    requestId,
    approval,
    openPdf,
    applyExportRequest,
    fetchTokenAndDownload,
    setReason,
    closeGate,
  };
}

export type ControlledPdf = ReturnType<typeof useControlledPdf>;
