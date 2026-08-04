/**
 * 前端类型（镜像 backend/domain/dto.py，§T6）。
 * TODO(T6 收尾)：由后端 openapi.json 经 openapi-typescript 生成，
 * 本文件随后替换为生成产物，禁止手工维护分叉。
 */
export const DefectClass = {
  POROSITY: 0,
  SLAG: 1,
  INCOMPLETE_PENETRATION: 2,
  LACK_OF_FUSION: 3,
  CRACK: 4,
} as const;
export type DefectClassId = (typeof DefectClass)[keyof typeof DefectClass];

export type DefectShape = "round" | "linear";
export type JointLevel = "I" | "II" | "III" | "IV";
export type Modality = "CR" | "DR" | "DICOM" | "GENERIC";

export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Detection {
  id: string;
  bbox: BBox;
  class_id: number;
  score: number;
  uncertainty: number;
  shape?: DefectShape | null;
  mask_ref?: string | null;
}

export interface HealthResponse {
  status: string;
  app_version: string;
  uri: string;
  backend: string;
  active_version: string | null;
}

export interface ApiError {
  error: { code: string; message: string; detail: unknown };
}
