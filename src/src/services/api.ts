/**
 * 唯一 API 客户端。
 * 所有请求必须经此文件；字段与后端 openapi.json 对齐。
 * 响应数据一律来自真实后端，前端不做任何构造/模拟。
 */
import type {
  ActiveExportIn,
  AccountOut,
  BootstrapOut,
  ChallengeOut,
  LoginOut,
  MeOut,
  ActiveExportOut,
  ActivePoolOut,
  ExportRequestOut,
  ExportTokenOut,
  BatchRetryOut,
  BatchStatusOut,
  BatchSubmitOut,
  BatchSummaryOut,
  BatchDedupDecision,
  CalibrationIn,
  CalibrationOut,
  DeviceDetailOut,
  DeviceIn,
  DeviceOut,
  HealthResponse,
  LlmDirsOut,
  LlmModelsOut,
  LlmScanResponseOut,
  LlmSelectOut,
  LlmServicesOut,
  LlmStatusOut,
  RecordsResponse,
  ReportOut,
  ReportDetectionsOut,
  ReportNarrativeOut,
  ReviewDefectMutateOut,
  ReviewIn,
  ReviewOut,
  StdPersonnel,
  StdPersonnelOut,
  StdEvalHistoryOut,
  StdRecordIn,
  StdRecordOut,
  VerifyOut,
} from "../types/api";
import { getOperatorName } from "./operator";
import { clearToken, getToken } from "./authToken";

const BASE = import.meta.env.VITE_API_BASE ?? "/api/v1";

/** 服务地址（状态栏展示用）：把配置的 BASE 解析成可读的 host:port。 */
export function apiHostLabel(): string {
  if (/^https?:\/\//.test(BASE)) {
    try {
      return new URL(BASE).host;
    } catch {
      /* fallthrough */
    }
  }
  return `${window.location.host || "127.0.0.1"}（同源代理）`;
}

/** 后端统一错误包：{error:{code,message,detail}} 或 HTTPException 的 {detail:{code,message}}。
 *  message 面向操作员直接展示（纯中文文案），机器可读的 code 走独立字段，
 *  不再拼进 message（此前界面会出现「HTTP_ERROR: HTTP 500」这类技术前缀）。 */
export class ApiRequestError extends Error {
  readonly status: number;
  readonly code: string;
  readonly detail: unknown;

  constructor(status: number, code: string, message: string, detail: unknown) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

/** 后端离线信号：request 遇连接不可达/超时派发，供 App 显示全局离线横幅。 */
export const BACKEND_DOWN_EVENT = "backend:down";
/** 后端恢复信号：任意成功响应派发，供 App 清除离线横幅。 */
export const BACKEND_UP_EVENT = "backend:up";
/** 会话失效信号（C-06/C-07）：任意 401 派发，App 清除登录态并跳转登录页。 */
export const AUTH_UNAUTHORIZED_EVENT = "auth:unauthorized";

/** 单次请求超时（ms）：本地推理通常数秒，批量/复杂报告留 30s 余量。 */
const REQUEST_TIMEOUT_MS = 30_000;
/** 上传/批量/报告生成：大底片或百张批量易超 30s，单独放宽超时。 */
const UPLOAD_TIMEOUT_MS = 120_000;
/** 本地大模型评片结论：本地 4B 模型纯 CPU 推理可能数十秒（GPU 通常 <10s），再放宽一档。 */
const NARRATIVE_TIMEOUT_MS = 180_000;
/** 仅对「后端不可达（连接被拒）」做指数退避重试；超时与 HTTP 错误不重试（避免重复提交）。 */
const MAX_NETWORK_RETRIES = 2;
const RETRY_BASE_MS = 400;

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

/** IPC 一次性令牌（C-17）：Tauri 外壳在后端就绪后注入 window.__IPC_TOKEN__，
 *  本机后端要求业务请求统一携带 X-IPC-Token（防其他本机进程误调/网页 CSRF
 *  式调用）；浏览器开发环境无此值，仅调试时由后端关闭 ipc.enforce。 */
function getIpcToken(): string | null {
  return (window as unknown as { __IPC_TOKEN__?: string }).__IPC_TOKEN__ ?? null;
}

async function rawRequest<T>(path: string, init: RequestInit, timeoutMs = REQUEST_TIMEOUT_MS): Promise<T> {
  const ctrl = new AbortController();
  const timer = window.setTimeout(() => ctrl.abort(), timeoutMs);
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { ...init, signal: ctrl.signal });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      // 超时 ≠ 后端离线（长任务处理中同样会超时），不派发全局 DOWN 事件，
      // 避免慢请求把全局横幅误打成「后端未连接」。
      throw new ApiRequestError(
        0,
        "TIMEOUT",
        `请求超时（>${timeoutMs / 1000}s），后端可能未响应或正在处理`,
        null,
      );
    }
    // TypeError（连接被拒等网络层错误）：给出可操作的「后端未启动」提示
    window.dispatchEvent(new CustomEvent(BACKEND_DOWN_EVENT));
    throw new ApiRequestError(
      0,
      "BACKEND_UNREACHABLE",
      "无法连接本地推理服务，请确认服务已启动（默认地址 127.0.0.1:18773）",
      null,
    );
  } finally {
    window.clearTimeout(timer);
  }
  // 任意成功响应（含 4xx/5xx 已被上层转换为错误前）都说明后端在线，清除离线态
  window.dispatchEvent(new CustomEvent(BACKEND_UP_EVENT));

  if (!res.ok) await throwForResponse(res);
  return (await res.json()) as T;
}

/** 统一错误包解析（rawRequest / downloadExportBlob 共用）：非 2xx 抛 ApiRequestError。 */
async function throwForResponse(res: Response): Promise<never> {
  const statusTextZh: Record<number, string> = {
    400: "请求参数有误",
    401: "登录状态已失效，请重新登录",
    403: "当前账号无权限执行此操作",
    404: "请求的资源不存在",
    409: "与当前状态冲突，请刷新后重试",
    413: "文件超过大小上限",
    422: "提交的内容未通过校验",
    429: "请求过于频繁，请稍后再试",
    500: "服务内部错误",
    502: "推理服务不可用",
    503: "服务暂不可用，正在处理中",
  };
  let code = "HTTP_ERROR";
  let message = statusTextZh[res.status] ?? `请求失败（HTTP ${res.status}）`;
  let detail: unknown = null;
  try {
    const body = (await res.json()) as {
      error?: { code?: string; message?: string; detail?: unknown };
      detail?: { code?: string; message?: string };
    };
    if (body?.error) {
      code = body.error.code ?? code;
      message = body.error.message ?? message;
      detail = body.error.detail ?? null;
    } else if (body?.detail) {
      // HTTPException 默认包体为 {detail:{code,message}}（401/403 等）
      code = body.detail.code ?? code;
      message = body.detail.message ?? message;
    }
  } catch {
    /* 非 JSON 响应：保留上方中文兜底文案 */
  }
  if (res.status === 401) {
    // 会话无效/过期（C-07 空闲超时由后端判定）：清除本地登录态并通知 App 跳转
    clearToken();
    window.dispatchEvent(new CustomEvent(AUTH_UNAUTHORIZED_EVENT));
  }
  throw new ApiRequestError(res.status, code, message, detail);
}

/** JSON POST 样板统一出口（headers/body 构造收敛一处）。 */
function postJson<T>(path: string, body: unknown, timeoutMs?: number): Promise<T> {
  return request<T>(
    path,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
    timeoutMs,
  );
}

async function request<T>(path: string, init?: RequestInit, timeoutMs = REQUEST_TIMEOUT_MS): Promise<T> {
  // 会话令牌（C-06）：登录后统一携带 Authorization；调用方显式传入的头优先
  const token = getToken();
  // 操作员姓名（X-Operator-Name）：仅登录前场景作审计 actor 记录
  const headers: Record<string, string> = { "X-Operator-Name": getOperatorName() };
  if (token) headers.Authorization = `Bearer ${token}`;
  // IPC 一次性令牌（C-17）：每次请求实时读取（Tauri 注入时机晚于前端启动）
  const ipcToken = getIpcToken();
  if (ipcToken) headers["X-IPC-Token"] = ipcToken;
  Object.assign(headers, init?.headers ?? {});
  const merged: RequestInit = { ...init, headers };
  let lastErr: unknown;
  // 仅后端不可达时重试（连接刚启动时短暂抖动）；超时/HTTP 错误直接抛，不重试。
  for (let attempt = 0; attempt <= MAX_NETWORK_RETRIES; attempt++) {
    try {
      return await rawRequest<T>(path, merged, timeoutMs);
    } catch (e) {
      lastErr = e;
      const retryable = e instanceof ApiRequestError && e.code === "BACKEND_UNREACHABLE";
      if (!retryable || attempt === MAX_NETWORK_RETRIES) break;
      await delay(RETRY_BASE_MS * 2 ** attempt);
    }
  }
  throw lastErr;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

/** 新评片全链路：上传影像 + 表单参数 → 真实报告结果（同步流水线，等待期间为处理中）。
 *  大底片处理可能超过 30s，使用上传专用超时。 */
export function createReport(form: FormData): Promise<ReportOut> {
  return request<ReportOut>("/report", { method: "POST", body: form }, UPLOAD_TIMEOUT_MS);
}

/** 档案检索：多条件过滤 + 分页 + 统计，全部来自后端 records 查询。 */
export function listRecords(params?: {
  level?: string;
  workpiece?: string;
  classId?: number;
  dateFrom?: string;
  dateTo?: string;
  needReview?: boolean;
  page?: number;
  size?: number;
}): Promise<RecordsResponse> {
  const q = new URLSearchParams();
  if (params?.level) q.set("level", params.level);
  if (params?.workpiece) q.set("workpiece", params.workpiece);
  if (params?.classId !== undefined) q.set("class", String(params.classId));
  if (params?.dateFrom) q.set("from", params.dateFrom);
  if (params?.dateTo) q.set("to", params.dateTo);
  if (params?.needReview !== undefined) q.set("need_review", params.needReview ? "true" : "false");
  q.set("page", String(params?.page ?? 1));
  q.set("size", String(params?.size ?? 50));
  const qs = q.toString();
  return request<RecordsResponse>(`/records?${qs}`);
}

/** 提交一次人工复核（初评/复评/仲裁），结果由后端计算并返回。 */
export function submitReview(body: ReviewIn): Promise<ReviewOut> {
  return postJson<ReviewOut>("/review", body);
}

/** 主动学习：人工确认缺陷回流训练池（YOLO 标注 + 版本指纹，）。 */
export function activeExport(body: ActiveExportIn): Promise<ActiveExportOut> {
  return postJson<ActiveExportOut>("/active/export", body);
}

/** 主动学习：训练池状态（样本数 / 数据版本指纹 / 最近导出）。 */
export function activePool(): Promise<ActivePoolOut> {
  return request<ActivePoolOut>("/active/pool");
}

/* ── 批量处理：多图/文件夹导入 → 异步队列 → 进度 → 取消/重试 ── */

/** 提交批量评片：FormData 含 images[]（多文件）+ 公共参数 → batch_id（异步执行）。
 *  百张大底片批量易超 30s，使用上传专用超时。 */
export function submitBatch(form: FormData): Promise<BatchSubmitOut> {
  return request<BatchSubmitOut>("/batch", { method: "POST", body: form }, UPLOAD_TIMEOUT_MS);
}

/** 提交批量评片（带上传进度回调）：XMLHttpRequest 才能拿到上传字节数，
 *  百张大底片上传期间给操作员百分比反馈；timeout 10 分钟（上传完即返回）。 */
export function submitBatchWithProgress(
  form: FormData,
  onProgress: (pct: number) => void,
  signal?: AbortSignal,
): Promise<BatchSubmitOut> {
  return new Promise<BatchSubmitOut>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}/batch`);
    xhr.timeout = 600_000;
    const token = getToken();
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    const op = getOperatorName();
    if (op) xhr.setRequestHeader("X-Operator-Name", op);
    const ipcToken = getIpcToken();
    if (ipcToken) xhr.setRequestHeader("X-IPC-Token", ipcToken);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as BatchSubmitOut);
        } catch {
          reject(new ApiRequestError(0, "BAD_RESPONSE", "服务响应格式异常", null));
        }
        return;
      }
      let code = "HTTP_ERROR";
      let message = `提交失败（HTTP ${xhr.status}）`;
      try {
        const body = JSON.parse(xhr.responseText) as {
          error?: { code?: string; message?: string };
          detail?: { code?: string; message?: string };
        };
        code = body?.error?.code ?? body?.detail?.code ?? code;
        message = body?.error?.message ?? body?.detail?.message ?? message;
      } catch {
        /* 非 JSON 响应：保留兜底文案 */
      }
      if (xhr.status === 401) {
        clearToken();
        window.dispatchEvent(new CustomEvent(AUTH_UNAUTHORIZED_EVENT));
      }
      reject(new ApiRequestError(xhr.status, code, message, null));
    };
    xhr.onerror = () => {
      window.dispatchEvent(new CustomEvent(BACKEND_DOWN_EVENT));
      reject(
        new ApiRequestError(0, "BACKEND_UNREACHABLE", "无法连接本地推理服务，请确认服务已启动", null),
      );
    };
    xhr.ontimeout = () => {
      reject(new ApiRequestError(0, "TIMEOUT", "上传超时（>10 分钟），请减少单批数量后重试", null));
    };
    if (signal) {
      signal.addEventListener("abort", () => {
        xhr.abort();
        reject(new ApiRequestError(0, "ABORTED", "上传已取消", null));
      });
    }
    xhr.send(form);
  });
}

/** 批次进度与逐任务结果。 */
export function getBatchStatus(batchId: string): Promise<BatchStatusOut> {
  return request<BatchStatusOut>(`/batch/${batchId}`);
}

/** 历史批次摘要列表（最近在前），断点续跑入口。 */
export function listBatches(): Promise<BatchSummaryOut[]> {
  return request<BatchSummaryOut[]>("/batches");
}

/** 取消批次：未启动任务不再执行。 */
export function cancelBatch(batchId: string): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>(`/batch/${batchId}/cancel`, { method: "POST" });
}

/** 暂停批次：未启动任务停止派发（running 任务自然结束），可恢复。 */
export function pauseBatch(batchId: string): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>(`/batch/${batchId}/pause`, { method: "POST" });
}

/** 恢复暂停的批次：被退回的待派发任务重新入队。 */
export function resumeBatch(batchId: string): Promise<{ ok: boolean; resumed: number }> {
  return request<{ ok: boolean; resumed: number }>(`/batch/${batchId}/resume`, { method: "POST" });
}

/** 断点续跑：重跑本批 failed/cancelled 任务。 */
export function retryBatch(batchId: string): Promise<BatchRetryOut> {
  return request<BatchRetryOut>(`/batch/${batchId}/retry`, { method: "POST" });
}

/** 人工查重复核：逐项决定重复文件跳过/仍检测，确认后批次继续执行。 */
export function resolveBatchDuplicates(
  batchId: string,
  decisions: BatchDedupDecision[],
): Promise<{ ok: boolean; skipped: number; kept: number }> {
  return request<{ ok: boolean; skipped: number; kept: number }>(
    `/batch/${batchId}/dedup/resolve`,
    { method: "POST", body: JSON.stringify({ decisions }) },
  );
}

/* ── 设备标定与报告数字签名校验 ── */

/** 设备列表（含最近标定摘要与一致性状态）。 */
export function listDevices(): Promise<DeviceOut[]> {
  return request<DeviceOut[]>("/devices");
}

/** 注册检测设备。 */
export function registerDevice(body: DeviceIn): Promise<DeviceOut> {
  return postJson<DeviceOut>("/devices", body);
}

/** 设备详情：档案 + 完整标定档案。 */
export function getDevice(deviceId: string): Promise<DeviceDetailOut> {
  return request<DeviceDetailOut>(`/devices/${deviceId}`);
}

/** 记录一次标定（跨设备一致率 ≤5% 判定）。 */
export function addCalibration(deviceId: string, body: CalibrationIn): Promise<CalibrationOut> {
  return request<CalibrationOut>(`/devices/${deviceId}/calibrations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** 报告数字签名校验：重算内容指纹与签发时比对（防篡改）。 */
export function verifyReport(reportId: string): Promise<VerifyOut> {
  return request<VerifyOut>(`/report/${reportId}/verify`, { method: "POST" });
}

/** 主动学习：取报告对应影像的缺陷明细（像素 bbox + 置信度/不确定性），供人工复核后回流训练池。 */
export function getReportDetections(reportId: string): Promise<ReportDetectionsOut> {
  return request<ReportDetectionsOut>(`/report/${reportId}/detections`);
}

/**
 * 本地大模型评片结论（按需生成，不落库）。
 *
 * 生成失败不抛错：后端以 status（ok/disabled/unavailable/failed）+ reason
 * 如实返回，调用方据此展示原因而非报错——大模型不可用不应影响评片主链路。
 * 超时给足：本地 4B 模型单轮结论在 CPU 上可能数十秒。
 */
export function getReportNarrative(reportId: string): Promise<ReportNarrativeOut> {
  return request<ReportNarrativeOut>(
    `/report/${reportId}/narrative`,
    {},
    NARRATIVE_TIMEOUT_MS,
  );
}

/** 复核添加缺陷框（operator 取请求头操作员，reason 审计必填）。 */
export function addReviewDefect(
  imageId: string,
  body: { class_id: number; bbox_px: number[]; reason: string },
): Promise<ReviewDefectMutateOut> {
  return request<ReviewDefectMutateOut>(`/review/${imageId}/defects`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }, UPLOAD_TIMEOUT_MS);
}

/** 复核修改缺陷类型/位置（至少一项）。 */
export function editReviewDefect(
  defectId: string,
  body: { class_id?: number; bbox_px?: number[]; reason: string },
): Promise<ReviewDefectMutateOut> {
  return request<ReviewDefectMutateOut>(`/review/defects/${defectId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }, UPLOAD_TIMEOUT_MS);
}

/** 复核删除缺陷（软删除，后端重评级）。 */
export function deleteReviewDefect(defectId: string, reason: string): Promise<ReviewDefectMutateOut> {
  const qs = new URLSearchParams({ reason });
  return request<ReviewDefectMutateOut>(`/review/defects/${defectId}?${qs}`, {
    method: "DELETE",
  }, UPLOAD_TIMEOUT_MS);
}

/** 标准评价：读取人员资质。 */
export function getStdPersonnel(): Promise<StdPersonnelOut> {
  return request<StdPersonnelOut>("/std-eval/personnel");
}

/** 标准评价：保存人员资质。 */
export function putStdPersonnel(people: StdPersonnel[]): Promise<StdPersonnelOut> {
  return request<StdPersonnelOut>("/std-eval/personnel", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ people }),
  });
}

/** 标准评价：装配附录A 记录表（JSON）。 */
export function createStdRecord(body: StdRecordIn): Promise<StdRecordOut> {
  return postJson<StdRecordOut>("/std-eval/record", body);
}

/** 标准评价：附录A 记录表 PDF 下载地址（直链经 access_token 鉴权）。 */
export function stdRecordPdfUrl(recordName: string): string {
  return withAccessToken(`${BASE}/std-eval/record/pdf?record_name=${encodeURIComponent(recordName)}`);
}

/** 标准评价：评价历史时间线（E-15 等级曲线数据源），按 evaluated_at 降序。 */
export function getStdEvalHistory(): Promise<StdEvalHistoryOut> {
  return request<StdEvalHistoryOut>("/std-eval/history");
}

/** 库内影像 PNG 预览地址（浏览器不解码 TIFF/DICOM，由后端统一转换；直链经 access_token 鉴权）。 */
export function imagePreviewUrl(imageId: string): string {
  return withAccessToken(`${BASE}/images/${encodeURIComponent(imageId)}/preview.png`);
}

/** 报告 PDF 下载地址（C-14：默认需导出审批，直链经 access_token 鉴权）。 */
export function reportPdfUrl(reportId: string): string {
  return withAccessToken(`${BASE}/report/${encodeURIComponent(reportId)}/pdf`);
}

/* ── C-14 受控导出：申请 → 保密员审批 → 领一次性令牌 → 携令牌下载 ── */

/** 申请导出（任意已登录角色）；subject 形如 report:<report_id>。 */
export function createExportRequest(subject: string, reason?: string): Promise<ExportRequestOut> {
  return postJson<ExportRequestOut>("/export/requests", { subject, reason: reason || null });
}

/** 领取一次性导出令牌（申请批准后；明文仅本次返回，后端只存 SM3 哈希）。 */
export function issueExportToken(requestId: string): Promise<ExportTokenOut> {
  return postJson<ExportTokenOut>(`/export/requests/${encodeURIComponent(requestId)}/token`, {});
}

/** 查询导出申请状态（受控导出面板轮询审批结论，申请人不必盲试领取）。 */
export function getExportRequest(requestId: string): Promise<ExportRequestOut> {
  return request<ExportRequestOut>(`/export/requests/${encodeURIComponent(requestId)}`);
}

/** 受控导出下载（Blob）：window.open 无法携带 X-Export-Token 自定义头，一次性
 *  令牌下载走 fetch→Blob→objectURL；登录会话/IPC 凭据与普通请求同源注入。 */
export async function downloadExportBlob(path: string, exportToken?: string): Promise<Blob> {
  const headers: Record<string, string> = { "X-Operator-Name": getOperatorName() };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  const ipcToken = getIpcToken();
  if (ipcToken) headers["X-IPC-Token"] = ipcToken;
  const et = exportToken?.trim();
  if (et) headers["X-Export-Token"] = et;
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { headers });
  } catch {
    window.dispatchEvent(new CustomEvent(BACKEND_DOWN_EVENT));
    throw new ApiRequestError(
      0,
      "BACKEND_UNREACHABLE",
      "无法连接本地推理服务，请确认服务已启动（默认地址 127.0.0.1:18773）",
      null,
    );
  }
  window.dispatchEvent(new CustomEvent(BACKEND_UP_EVENT));
  if (!res.ok) await throwForResponse(res);
  return await res.blob();
}

/** 为直链 URL 追加 access_token 查询参数（已登录时）；未登录原样返回。 */
function withAccessToken(url: string): string {
  const token = getToken();
  if (!token) return url;
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}access_token=${encodeURIComponent(token)}`;
}

/* ── 三员身份认证（C-06/C-07）────────────────────────────── */

/** 签发登录挑战（一次一用，60s 有效）。 */
export function getChallenge(): Promise<ChallengeOut> {
  return request<ChallengeOut>("/auth/challenge");
}

/**
 * SM2 挑战-响应登录。
 * 软件模式简化流程（诚实声明）：私钥文件内容提交给本机后端代签后验签——
 * 单机本地软件可接受；私钥仅在本机进程内存中出现，不落日志/审计。
 */
export function login(
  username: string,
  challengeId: string,
  privateKey: string,
): Promise<LoginOut> {
  return request<LoginOut>("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      username,
      challenge_id: challengeId,
      private_key: privateKey,
    }),
  });
}

/** 引导窗口：仅系统尚无账号时可用（创建后永久关闭）。 */
export function bootstrap(body: {
  username: string;
  role: string;
  public_key?: string;
}): Promise<BootstrapOut> {
  return postJson<BootstrapOut>("/auth/bootstrap", body);
}

/** 注销当前会话。 */
export function logout(): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>("/auth/logout", { method: "POST" });
}

/** 当前登录身份。 */
export function getMe(): Promise<MeOut> {
  return request<MeOut>("/auth/me");
}

/* ── 账号管理（sysadmin 专属；私钥丢失/人员变动在界面内自助处置） ── */

/** 列出全部三员账号（含锁定状态/失败计数）。 */
export function listAccounts(): Promise<AccountOut[]> {
  return request<AccountOut[]>("/auth/accounts");
}

/** 创建三员账号（返回账号档案；私钥随后用 issueKeypair 一次性签发展示）。 */
export function createAccount(body: {
  username: string;
  role: "sysadmin" | "secadmin" | "auditor";
}): Promise<AccountOut> {
  return postJson<AccountOut>("/auth/accounts", body);
}

/** 为账号重新签发 SM2 软证书（私钥丢失/疑似泄露后的补救；私钥一次性返回）。 */
export function issueKeypair(
  accountId: string,
): Promise<{ account_id: string; public_key: string; private_key: string }> {
  return request<{ account_id: string; public_key: string; private_key: string }>(
    `/auth/accounts/${encodeURIComponent(accountId)}/keypair`,
    { method: "POST" },
  );
}

/** 启用/停用账号（停用同时吊销其全部会话）。 */
export function setAccountStatus(
  accountId: string,
  status: "active" | "disabled",
): Promise<AccountOut> {
  return request<AccountOut>(`/auth/accounts/${encodeURIComponent(accountId)}/status`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
}

/** 引导状态查询（公开）：系统是否还没有任何账号（登录页据此自动展开引导）。 */
export function bootstrapStatus(): Promise<{ needs_bootstrap: boolean; guest_mode: boolean }> {
  return request<{ needs_bootstrap: boolean; guest_mode: boolean }>("/auth/bootstrap/status");
}

/** 库内影像 PNG 预览拉取（Blob）：走统一请求管道，404/401 等错误可直接
 *  归因（此前直链 <img> 的 onerror 把 404/401 都误报成「格式不支持」）。 */
export async function fetchImagePreviewBlob(imageId: string): Promise<Blob> {
  const res = await fetch(`${BASE}/images/${encodeURIComponent(imageId)}/preview.png`, {
    headers: await authedHeaders(),
  });
  if (!res.ok) await throwForResponse(res);
  return await res.blob();
}

/** 业务请求头（Authorization / X-IPC-Token / X-Operator-Name），供 fetch 直连场景复用。 */
async function authedHeaders(): Promise<Record<string, string>> {
  const headers: Record<string, string> = { "X-Operator-Name": getOperatorName() };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  const ipcToken = getIpcToken();
  if (ipcToken) headers["X-IPC-Token"] = ipcToken;
  return headers;
}

/* ── 本地大模型（backend/app/routers/llm.py）────────────────────────────
 * 超时口径：/llm/models 与 /llm/services 要串探测多个本机端点，
 * 单端点超时 3s、端点数为 4 → 给 60s 留足余量；/llm/status 不探测端点，30s 足够。
 */

/** 引擎 + 选中 + 扫描 + GPU 汇总（不探测端点，响应快）。 */
export function getLlmStatus(): Promise<LlmStatusOut> {
  return request<LlmStatusOut>("/llm/status", { method: "GET" }, 30_000);
}

/** 全部可选模型（本地 GGUF + 已有服务）。includeServices=false 时零网络调用。 */
export function listLlmModels(includeServices = true): Promise<LlmModelsOut> {
  return request<LlmModelsOut>(
    `/llm/models?include_services=${includeServices ? "true" : "false"}`,
    { method: "GET" },
    60_000,
  );
}

/** 探测本机已有的 llama 兼容端点（Ollama / LM Studio / 自建 llama.cpp）。 */
export function listLlmServices(): Promise<LlmServicesOut> {
  return request<LlmServicesOut>("/llm/services", { method: "GET" }, 60_000);
}

/** 启动模型发现扫描（后台任务，立即返回；进度看 /llm/status 的 scan 字段）。 */
export function startLlmScan(): Promise<LlmScanResponseOut> {
  return request<LlmScanResponseOut>("/llm/scan", { method: "POST" });
}

/** 取消进行中的扫描（已扫到的部分结果保留，取消不等于丢数据）。 */
export function cancelLlmScan(): Promise<LlmScanResponseOut> {
  return request<LlmScanResponseOut>("/llm/scan/cancel", { method: "POST" });
}

/** 选中模型并热应用到引擎（sysadmin）。不可用条目后端返回 409，不静默接受。 */
export function selectLlmModel(modelId: string): Promise<LlmSelectOut> {
  return request<LlmSelectOut>(
    `/llm/models/${encodeURIComponent(modelId)}/select`,
    { method: "POST" },
    60_000,
  );
}

/** 放弃显式选中，回落到配置文件默认（sysadmin）。 */
export function clearLlmSelection(): Promise<LlmSelectOut> {
  return request<LlmSelectOut>("/llm/selection", { method: "DELETE" }, 60_000);
}

/** 添加模型目录（须真实存在）；仅登记扫描范围，不复制任何文件。 */
export function addLlmDir(path: string): Promise<LlmDirsOut> {
  return request<LlmDirsOut>("/llm/dirs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
}

/** 移除模型目录记录（只移除登记，不删除磁盘文件）。 */
export function removeLlmDir(path: string): Promise<LlmDirsOut> {
  return request<LlmDirsOut>(`/llm/dirs?path=${encodeURIComponent(path)}`, { method: "DELETE" });
}
