/**
 * 前端类型：镜像后端 Pydantic 响应模型（backend/app/routers/*.py）
 * 与领域枚举/结构（backend/domain/dto.py 的 BBox、Detection、DefectClass 等）。
 * 本文件是仓库唯一的前端契约真相；后端同名 response_model 新增/改名时，
 * 须同步本文件，由 backend/tests/test_frontend_contract.py 做字段对账防漂移。
 * 不依赖 openapi 自动生成产物。
 */
export const DefectClass = {
  POROSITY: 0,
  SLAG: 1,
  INCOMPLETE_PENETRATION: 2,
  LACK_OF_FUSION: 3,
  CRACK: 4,
  UNDERCUT: 5,
  CONCAVITY: 6,
} as const;

/** 缺陷类别中文标签（镜像 backend/domain/dto.py DefectClass 0..6，全前端唯一事实源）。 */
export const DEFECT_CLASS_LABELS: readonly string[] = [
  "气孔",
  "夹渣",
  "未焊透",
  "未熔合",
  "裂纹",
  "咬边",
  "内凹",
];
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

/** 顶层视图（菜单栏/工具栏/标签页导航目标） */
export type ViewId = "journey" | "archive" | "batch" | "device" | "viewer" | "std-eval";

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
  /** 合规处置建议：accept | conditional | rework | recheck */
  disposition: string | null;
  disposition_label: string | null;
  /** readonly：useJourney 的 readonly 深度只读化后保持可赋值 */
  disposition_actions: readonly string[];
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
  /** C-10 密级：0=非密 1=内部 2=秘密 3=机密 */
  secret_level: number;
  classification_basis: string | null;
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


/* ── 主动学习（ · POST /api/v1/active/…）── */

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

/* ── 批量处理── */

export type BatchTaskStatus = "pending" | "running" | "done" | "failed" | "cancelled";

/** POST /api/v1/batch → BatchSubmitOut */
export interface BatchSubmitOut {
  batch_id: string;
  total: number;
  estimated_sec: number;
}

/** GET /api/v1/batch/{id} → tasks[] 项 */
export interface BatchTaskOut {
  task_id: string;
  image_name: string;
  status: BatchTaskStatus;
  error: string | null;
  image_id: string | null;
  report_id: string | null;
  joint_level: string | null;
  need_review: boolean | null;
}

/** GET /api/v1/batch/{id} → BatchStatusOut */
export interface BatchStatusOut {
  batch_id: string;
  status: string;
  total: number;
  done: number;
  failed: number;
  cancelled: number;
  estimated_sec: number;
  progress: number;
  tasks: BatchTaskOut[];
}

/** GET /api/v1/batches → 列表项（历史/断点续跑入口） */
export interface BatchSummaryOut {
  batch_id: string;
  status: string;
  total: number;
  done: number;
  failed: number;
  cancelled: number;
  progress: number;
  estimated_sec: number;
  created_at: string | null;
  finished_at: string | null;
}

/** POST /api/v1/batch/{id}/retry → BatchRetryOut */
export interface BatchRetryOut {
  ok: boolean;
  retried: number;
}


/* ── 设备标定与报告数字签名校验── */

export type CalibrationStatus = "ok" | "over";

export interface CalibrationOut {
  calibration_id: string;
  device_id: string;
  calibrator: string;
  pixel_spacing_mm: number;
  ref_pixel_spacing_mm: number | null;
  deviation_pct: number | null;
  status: CalibrationStatus;
  density_ref: number | null;
  notes: string | null;
  calibrated_at: string | null;
}

export interface DeviceOut {
  device_id: string;
  name: string;
  model: string | null;
  serial_no: string | null;
  notes: string | null;
  created_by: string | null;
  created_at: string | null;
  calibration_count: number;
  last_calibration: CalibrationOut | null;
}

export interface DeviceDetailOut extends DeviceOut {
  calibrations: CalibrationOut[];
}

export interface DeviceIn {
  name: string;
  model?: string | null;
  serial_no?: string | null;
  notes?: string | null;
}

export interface CalibrationIn {
  calibrator: string;
  pixel_spacing_mm: number;
  ref_pixel_spacing_mm?: number | null;
  density_ref?: number | null;
  notes?: string | null;
}

/** SM2 验签结果（VerifyOut.signature；独立于指纹比对的双结果之一） */
export interface SignatureCheckOut {
  valid: boolean | null; // true=验签通过；false=不通过；null=无签名（legacy 旧报告）
  algo: string | null; // 签名算法（SM2）
  public_key: string | null; // 签名方 SM2 公钥（128 hex）
  reason: string | null; // missing | invalid_sidecar | fingerprint_mismatch | mismatch
}

/** POST /api/v1/report/{id}/verify → VerifyOut */
export interface VerifyOut {
  report_id: string;
  valid: boolean | null;
  hash: string | null;
  signer: string | null;
  generated_at: string | null;
  reason: string | null;
  signature: SignatureCheckOut | null; // SM2 验签结果（null=sidecar 不可用）
}

/** 主动学习回流用：单条缺陷明细（镜像后端 GET /report/{id}/detections） */
export interface ReportDetection {
  id: string;
  class_id: number;
  bbox: [number, number, number, number];
  confidence: number;
  uncertainty: number;
  reviewed: boolean;
  need_review: boolean;
  source?: string | null;
}

/** GET /api/v1/report/{id}/detections → ReportDetectionsOut */
export interface ReportDetectionsOut {
  report_id: string;
  image_id: string;
  image_stem: string;
  image_w: number;
  image_h: number;
  defects: ReportDetection[];
}

/** 底片查看器变换状态（FilmViewer 双片对比同步用，DB50/T 1807 ） */
export interface Transform {
  scale: number;
  tx: number;
  ty: number;
  rotation: number;
  flipH: boolean;
  flipV: boolean;
}

/** POST /api/v1/review/{image_id}/defects、PATCH/DELETE /api/v1/review/defects/{id} → 缺陷行 */
export interface ReviewDefectOut {
  id: string;
  image_id: string;
  class_id: number;
  bbox_px: number[];
  confidence: number;
  uncertainty: number;
  joint_level: string | null;
  need_review: boolean;
  reviewed_by: string | null;
  source: string | null;
  deleted_at: string | null;
}

/** 缺陷增删改的响应（含重评级结果） */
export interface ReviewDefectMutateOut {
  defect: ReviewDefectOut;
  image_id: string;
  joint_level: string | null;
  need_review: boolean;
  defect_count: number;
}

/** 人员资质（GET/PUT /api/v1/std-eval/personnel） */
export interface StdPersonnel {
  name: string;
  cert_type: string;
  role: "evaluator" | "labeler";
  cert_no?: string;
  valid_until?: string;
  level?: number | null;
}
export interface StdPersonnelOut {
  qualified: boolean;
  issues: string[];
  evaluators: StdPersonnel[];
  labelers: StdPersonnel[];
}

/** POST /api/v1/std-eval/record 入参 */
export interface StdRecordIn {
  eval_result_path?: string;
  system_name: string;
  system_version: string;
  developer: string;
  contact?: string;
  address?: string;
  film_kind?: string;
  exposure_layout?: string;
  weld_form: "single" | "double";
  weld_method: "manual" | "auto";
  n_defect_images?: number;
  n_no_defect_images?: number;
  record_name?: string;
}

/** 附录A 记录表（POST /api/v1/std-eval/record 响应，字段较宽，展示按需取用） */
export interface StdRecordOut {
  meta: {
    system_name: string;
    system_version: string;
    developer: string;
    eval_date: string;
    operator: string;
  };
  film: {
    weld_form: string;
    weld_method: string;
    n_defect_images: number;
    n_defects: number;
    class_distribution: string;
    n_no_defect_images: number;
  };
  personnel: { qualified: boolean; issues: string[] };
  metrics: {
    tdr_row: string;
    fdr_row: string;
    mdr_row: string;
    frr_row: string;
    kdr: number;
    wdr: number;
    tdr: number;
    frr: number;
    iou_standard: number;
    iou_strict: number;
    kdr_strict: number;
    wdr_strict: number;
    tdr_strict: number;
    frr_strict: number;
  };
  grading: {
    level: string | null;
    level_standard: string | null;
    level_strict: string | null;
    official: boolean;
    note?: string;
  };
  risks: { miss: string; false_detect: string; false_report: string };
}

/** GET /std-eval/history 条目（E-15）：历次评价时间线，缺省字段为 null。 */
export interface StdEvalHistoryItem {
  evaluated_at: string | null;
  model_version: string | null;
  level: string | null;
  tdr: number | null;
  wdr: number | null;
  frr: number | null;
  map50: number | null;
  recall: number | null;
  source: string | null;
}

/** GET /std-eval/history 响应（E-15）：按 evaluated_at 降序的时间线。 */
export interface StdEvalHistoryOut {
  total: number;
  items: StdEvalHistoryItem[];
}

/* ── 三员身份认证（C-06/C-07/C-09）与合规治理（C-10~C-14）新增契约 ── */

/** GET /auth/challenge 响应：nonce 需以账号 SM2 私钥签名（或上传私钥由后端代签） */
export interface ChallengeOut {
  challenge_id: string;
  nonce: string;
}

/** POST /auth/login 响应（token 明文仅此一次返回） */
export interface LoginOut {
  token: string;
  account_id: string;
  username: string;
  role: "sysadmin" | "secadmin" | "auditor";
  idle_timeout_min: number;
}

/** GET /auth/me 响应 */
export interface MeOut {
  account_id: string;
  username: string;
  role: "sysadmin" | "secadmin" | "auditor";
}

/** 三员账号（GET/POST /auth/accounts） */
export interface AccountOut {
  account_id: string;
  username: string;
  role: "sysadmin" | "secadmin" | "auditor";
  sm2_public_key: string | null;
  auth_mode: string;
  status: string;
  failed_attempts: number;
  locked_until: string | null;
  created_by: string | null;
  created_at: string | null;
}

/** POST /auth/bootstrap 响应（引导窗口，私钥一次性下发） */
export interface BootstrapOut extends AccountOut {
  private_key: string | null;
}

/** POST /auth/accounts/{id}/keypair 响应（私钥一次性下发） */
export interface KeyPairOut {
  account_id: string;
  public_key: string;
  private_key: string;
}

/** 安全告警（GET /auth/alerts，C-19） */
export interface AlertOut {
  alert_id: string;
  kind: string;
  level: string;
  message: string;
  detail: unknown;
  status: string;
  resolved_by: string | null;
  resolved_at: string | null;
  note: string | null;
  created_at: string | null;
}

/** 密级（C-10）：0=非密 1=内部 2=秘密 3=机密 */
export interface SecretLevelOut {
  image_id: string;
  secret_level: 0 | 1 | 2 | 3;
  secret_level_name: string;
  classification_basis: string | null;
}

/** 涉密载体（C-12） */
export interface CarrierOut {
  carrier_id: string;
  kind: "film" | "report" | "backup";
  object_id: string | null;
  secret_level: number;
  owner: string | null;
  status: "in_stock" | "borrowed" | "returned" | "pending_destroy" | "destroyed";
  borrow_history: Array<{ action: string; operator: string | null; at: string | null; note: string | null }>;
  destroy_method: string | null;
  destroy_note: string | null;
  destroy_requested_by: string | null;
  destroy_confirmed_by: string | null;
  destroyed_at: string | null;
  created_at: string | null;
}

/** 导出审批（C-14） */
export interface ExportRequestOut {
  request_id: string;
  subject: string;
  reason: string | null;
  requested_by: string;
  status: "pending" | "approved" | "rejected" | "consumed";
  decided_by: string | null;
  decided_at: string | null;
  token_expires_at: string | null;
  used_at: string | null;
  created_at: string | null;
}

/** POST /export/requests/{id}/token 响应（明文令牌仅此一次返回） */
export interface ExportTokenOut {
  token: string;
  expires_in_sec: number;
}

/** 脱敏残留审计（C-13，POST /privacy/audit 响应） */
export interface PrivacyAuditOut {
  generated_at: string;
  directory: string;
  scanned: number;
  n_findings: number;
  clean: boolean;
  findings: Array<{ file: string; kind: string; residues: string[] }>;
  errors: Array<{ file: string; error: string }>;
  report_files: { json: string; pdf: string };
}
