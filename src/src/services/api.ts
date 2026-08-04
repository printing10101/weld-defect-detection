/**
 * 唯一 API 客户端（§T6）。
 * 所有请求必须经此文件；字段与后端 openapi.json 对齐（§14）。
 * TODO(T6 收尾)：由 openapi 生成客户端，替换手写实现。
 */
import type { HealthResponse } from "../types/api";

const BASE = import.meta.env.VITE_API_BASE ?? "/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return (await res.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}
