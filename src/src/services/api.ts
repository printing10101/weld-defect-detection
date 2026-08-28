/**
 * 唯一 API 客户端。
 * 所有请求必须经此文件；字段与后端 openapi.json 对齐。
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
  RecordsResponse,
  ReportOut,
  ReportDetectionsOut,
  ReviewIn,
  ReviewOut,
  VerifyOut,
} from "../types/api";
import { getOperatorName } from "./operator";

const BASE = import.meta.env.VITE_API_BASE ?? "/api/v1";

/** 后端统一错误包：{error:{code,message,detail}} 或 HTTPException 的 {detail:{code,message}}。 */
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

/** 后端离线信号：request 遇连接不可达/超时派发，供 App 显示全局离线横幅。 */
export const BACKEND_DOWN_EVENT = "backend:down";
/** 后端恢复信号：任意成功响应派发，供 App 清除离线横幅。 */
export const BACKEND_UP_EVENT = "backend:up";

/** 单次请求超时（ms）：本地推理通常数秒，批量/复杂报告留 30s 余量。 */
const REQUEST_TIMEOUT_MS = 30_000;
/** 上传/批量/报告生成：大底片或百张批量易超 30s，单独放宽超时。 */
const UPLOAD_TIMEOUT_MS = 120_000;
/** 仅对「后端不可达（连接被拒）」做指数退避重试；超时与 HTTP 错误不重试（避免重复提交）。 */
const MAX_NETWORK_RETRIES = 2;
const RETRY_BASE_MS = 400;

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function rawRequest<T>(path: string, init: RequestInit, timeoutMs = REQUEST_TIMEOUT_MS): Promise<T> {
  const ctrl = new AbortController();
  const timer = window.setTimeout(() => ctrl.abort(), timeoutMs);
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { ...init, signal: ctrl.signal });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") {
      window.dispatchEvent(new CustomEvent(BACKEND_DOWN_EVENT));
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
      "无法连接后端，请确认本地服务已启动（默认 127.0.0.1:18773）",
      null,
    );
  } finally {
    window.clearTimeout(timer);
  }
  // 任意成功响应（含 4xx/5xx 已被上层转换为错误前）都说明后端在线，清除离线态
  window.dispatchEvent(new CustomEvent(BACKEND_UP_EVENT));

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
    throw new ApiRequestError(res.status, code, message, detail);
  }
  return (await res.json()) as T;
}

async function request<T>(path: string, init?: RequestInit, timeoutMs = REQUEST_TIMEOUT_MS): Promise<T> {
  // 操作员姓名（X-Operator-Name）：单机无用户系统，由设置中的操作员姓名标识"谁在操作"
  const headers = { "X-Operator-Name": getOperatorName(), ...(init?.headers ?? {}) };
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

/** 主动学习：从一次评片检出中采样高价值样本（优先人工标注，）。 */
export function activeSample(body: ActiveSampleIn): Promise<ActiveSampleOut> {
  return request<ActiveSampleOut>("/active/sample", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** 主动学习：人工确认缺陷回流训练池（YOLO 标注 + 版本指纹，）。 */
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

/* ── 批量处理：多图/文件夹导入 → 异步队列 → 进度 → 取消/重试 ── */

/** 提交批量评片：FormData 含 images[]（多文件）+ 公共参数 → batch_id（异步执行）。
 *  百张大底片批量易超 30s，使用上传专用超时。 */
export function submitBatch(form: FormData): Promise<BatchSubmitOut> {
  return request<BatchSubmitOut>("/batch", { method: "POST", body: form }, UPLOAD_TIMEOUT_MS);
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

/* ── 设备标定与报告数字签名校验 ── */

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

/** 主动学习：取报告对应影像的缺陷明细（像素 bbox + 置信度/不确定性），供人工复核后回流训练池。 */
export function getReportDetections(reportId: string): Promise<ReportDetectionsOut> {
  return request<ReportDetectionsOut>(`/report/${reportId}/detections`);
}
