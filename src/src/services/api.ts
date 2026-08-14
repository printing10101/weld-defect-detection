/**
 * 唯一 API 客户端（§T6）。
 * 所有请求必须经此文件；字段与后端 openapi.json 对齐（§14）。
 * 响应数据一律来自真实后端，前端不做任何构造/模拟。
 */
import type {
  ActiveExportIn,
  ActiveExportOut,
  ActivePoolOut,
  ActiveSampleIn,
  ActiveSampleOut,
  BatchRetryOut,
  BatchStatusOut,
  BatchSubmitOut,
  BatchSummaryOut,
  CalibrationIn,
  CalibrationOut,
  DeviceDetailOut,
  DeviceIn,
  DeviceOut,
  HealthResponse,
  LoginIn,
  LoginOut,
  RecordsResponse,
  ReportOut,
  ReviewIn,
  ReviewOut,
  UserOut,
  VerifyOut,
} from "../types/api";
import { authHeaders, clearToken, setToken } from "./auth";

const BASE = import.meta.env.VITE_API_BASE ?? "/api/v1";

/** 后端统一错误包（§13.4）：{error:{code,message,detail}} 或 HTTPException 的 {detail:{code,message}}。 */
export class ApiRequestError extends Error {
  readonly status: number;
  readonly code: string;
  readonly detail: unknown;

  constructor(status: number, code: string, message: string, detail: unknown) {
    super(`${code}: ${message}`);
    this.name = "ApiRequestError";
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

/** 令牌失效统一信号：request() 遇 401 派发，供 App 切回登录态。 */
export const AUTH_UNAUTHORIZED_EVENT = "auth:unauthorized";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = { ...authHeaders(), ...(init?.headers ?? {}) };
  const res = await fetch(`${BASE}${path}`, { ...init, headers });
  if (!res.ok) {
    let code = "HTTP_ERROR";
    let message = res.statusText || `HTTP ${res.status}`;
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
      /* 非 JSON 响应：保留 statusText */
    }
    if (res.status === 401) {
      // 令牌无效/过期：清除本地令牌并广播，驱动 UI 返回登录态（不静默保留失效会话）
      clearToken();
      window.dispatchEvent(new CustomEvent(AUTH_UNAUTHORIZED_EVENT));
    }
    throw new ApiRequestError(res.status, code, message, detail);
  }
  return (await res.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

/** 登录：用户名+密码 → 存储令牌并返回当前用户（§T3）。 */
export function login(body: LoginIn): Promise<LoginOut> {
  return request<LoginOut>("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((out) => {
    setToken(out.access_token);
    return out;
  });
}

/** 当前登录用户信息（§T3）。 */
export function getMe(): Promise<UserOut> {
  return request<UserOut>("/auth/me");
}

/** 退出登录：清除本地令牌（§T3）。 */
export function logout(): void {
  clearToken();
}

/** 新评片全链路：上传影像 + 表单参数 → 真实报告结果（同步流水线，等待期间为处理中）。 */
export function createReport(form: FormData): Promise<ReportOut> {
  return request<ReportOut>("/report", { method: "POST", body: form });
}

/** 档案检索：多条件过滤 + 分页 + 统计，全部来自后端 records 查询。 */
export function listRecords(params?: {
  level?: string;
  workpiece?: string;
  page?: number;
  size?: number;
}): Promise<RecordsResponse> {
  const q = new URLSearchParams();
  if (params?.level) q.set("level", params.level);
  if (params?.workpiece) q.set("workpiece", params.workpiece);
  q.set("page", String(params?.page ?? 1));
  q.set("size", String(params?.size ?? 50));
  const qs = q.toString();
  return request<RecordsResponse>(`/records?${qs}`);
}

/** 提交一次人工复核（初评/复评/仲裁），结果由后端计算并返回。 */
export function submitReview(body: ReviewIn): Promise<ReviewOut> {
  return request<ReviewOut>("/review", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** 主动学习：从一次评片检出中采样高价值样本（优先人工标注，§5.6）。 */
export function activeSample(body: ActiveSampleIn): Promise<ActiveSampleOut> {
  return request<ActiveSampleOut>("/active/sample", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** 主动学习：人工确认缺陷回流训练池（YOLO 标注 + 版本指纹，§5.5）。 */
export function activeExport(body: ActiveExportIn): Promise<ActiveExportOut> {
  return request<ActiveExportOut>("/active/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** 主动学习：训练池状态（样本数 / 数据版本指纹 / 最近导出）。 */
export function activePool(): Promise<ActivePoolOut> {
  return request<ActivePoolOut>("/active/pool");
}

/* ── 批量处理（§12.1）：多图/文件夹导入 → 异步队列 → 进度 → 取消/重试 ── */

/** 提交批量评片：FormData 含 images[]（多文件）+ 公共参数 → batch_id（异步执行）。 */
export function submitBatch(form: FormData): Promise<BatchSubmitOut> {
  return request<BatchSubmitOut>("/batch", { method: "POST", body: form });
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

/** 断点续跑：重跑本批 failed/cancelled 任务。 */
export function retryBatch(batchId: string): Promise<BatchRetryOut> {
  return request<BatchRetryOut>(`/batch/${batchId}/retry`, { method: "POST" });
}

/* ── 设备标定（§12.4）与报告数字签名校验（§7.2） ── */

/** 设备列表（含最近标定摘要与一致性状态）。 */
export function listDevices(): Promise<DeviceOut[]> {
  return request<DeviceOut[]>("/devices");
}

/** 注册检测设备。 */
export function registerDevice(body: DeviceIn): Promise<DeviceOut> {
  return request<DeviceOut>("/devices", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
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
