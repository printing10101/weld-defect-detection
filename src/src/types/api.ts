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
  UNDERCUT: 5,
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

/* ── 真实后端契约（镜像 backend/app/routers/report.py · records.py · review.py）── */

/** 顶层视图（RailNav 导航目标） */
export type ViewId = "journey" | "archive";

/** POST /api/v1/report → ReportOut */
export interface ReportOut {
  report_id: string;
  image_id: string;
  joint_level: string | null;
  need_review: boolean;
  evaluable: boolean;
  defect_count: number;
  /** 标准来源免责声明（工业过渡路径）：authorized_copy=false 时为强声明 */
  disclaimer: string | null;
  pdf_url: string;
}

/** GET /api/v1/records → items[]（镜像 repository._image_to_dict） */
export interface RecordItem {
  image_id: string;
  path: string;
  source_type: string;
  modality: string;
  workpiece_no: string | null;
  weld_no: string | null;
  pixel_spacing_mm: number | null;
  base_metal_thickness_mm: number | null;
  iqi_pass: boolean | null;
  iqi_detail: Record<string, unknown> | null;
  density: number | null;
  density_ok: boolean | null;
  evaluable: boolean;
  joint_level: string | null;
  need_review: boolean;
  standard_id: string | null;
  standard_version: string | null;
  created_at: string | null;
}

/** GET /api/v1/records → stats（镜像 repository.stats） */
export interface RecordsStats {
  total: number;
  by_level: Record<string, number>;
  by_class: Record<string, number>;
}

export interface RecordsResponse {
  items: RecordItem[];
  total: number;
  stats: RecordsStats;
}

/** POST /api/v1/review → ReviewOut */
export interface ReviewIn {
  image_id: string;
  reviewer: string;
  role: "initial" | "secondary" | "arbitrator";
  defect_grades?: { defect_id: string; joint_level: string }[];
  overall_level?: string | null;
  note?: string | null;
}

export interface ReviewOut {
  image_id: string;
  reviewer: string;
  role: string;
  consensus: boolean;
  kappa: number;
  needs_arbitration: boolean;
  joint_level: string | null;
  reviewed_by: string | null;
  stage: string;
  need_review: boolean;
  review_count: number;
}

/**
 * 真实流水线阶段（镜像 backend/app/pipelines.py 的执行顺序，非模拟数据；
 * 仅作处理中视图的流程说明，进度/状态一律来自真实请求）。
 */
export const PIPELINE_STAGES: readonly string[] = [
  "影像加载",
  "影像质量校验（黑度 + IQI）",
  "缺陷检测 + 当量量化",
  "标准判定（NB/T47013.2）",
  "落库归档",
  "生成报告（PDF/A）",
] as const;


/* ── 主动学习（M7 · POST /api/v1/active/…）── */

/** 高价值样本候选（主动学习采样结果） */
export interface ActiveCandidate {
  detection_id: string;
  class_id: number;
  score: number;
  uncertainty: number;
  value_score: number;
  reasons: string[];
}

export interface ActiveSampleOut {
  candidates: ActiveCandidate[];
  total: number;
}

export interface ActiveSampleIn {
  image_id?: string | null;
  defects: {
    id: string;
    class_id: number;
    bbox: [number, number, number, number];
    confidence: number;
    uncertainty: number;
  }[];
}

export interface ActiveExportIn {
  image_stem: string;
  image_w: number;
  image_h: number;
  defects: ActiveSampleIn["defects"];
  class_overrides?: Record<string, number>;
}

export interface ActiveExportOut {
  label_file: string;
  sample_count: number;
  fingerprint: string | null;
  total_in_pool: number;
}

export interface ActivePoolOut {
  sample_count: number;
  fingerprint: string | null;
  files: string[];
  exported_at: string | null;
}

/* ── 鉴权（§T3 · POST /api/v1/auth/…）── */

export interface LoginIn {
  username: string;
  password: string;
}

export interface UserOut {
  id: string;
  username: string;
  display_name: string | null;
  role: string;
  disabled: boolean;
  created_at: string | null;
  created_by: string | null;
  last_login_at: string | null;
}

export interface LoginOut {
  access_token: string;
  token_type: string;
  user: UserOut;
}
