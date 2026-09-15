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
export type DefectShape = "round" | "linear";
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
  /** 访客模式开关（登录页据此显示/隐藏「访客入口」按钮） */
  guest_mode?: boolean;
}

/* ── 真实后端契约（镜像 backend/app/routers/report.py · records.py · review.py）── */

/** 顶层视图（菜单栏/工具栏/标签页导航目标） */
export type ViewId =
  | "journey"
  | "archive"
  | "batch"
  | "device"
  | "viewer"
  | "std-eval"
  | "admin"
  | "llm";

/* ── 本地大模型（镜像 backend/app/routers/llm.py · infra/llm_registry.py）── */

/** 模型来源：本机 GGUF 文件 / 已有 llama 兼容服务。 */
export type LlmSource = "local" | "service";

/** 引擎加载方式：managed=本进程拉起 llama-server；external=只连已有服务。 */
export type LlmMode = "managed" | "external";

/**
 * 引擎状态。注意 `ready` 的判据是 `/health` 与 `/v1/models` **双双**可用，
 * 不是只看前者——llama-server 的 /health 免鉴权，单看它会把「要密钥、
 * 客户端根本用不了」的实例报成就绪（假绿）。
 */
export type LlmEngineState =
  | "starting"
  | "ready"
  | "error"
  | "unavailable"
  | "auth_required"
  | "disabled"
  | "stopped";

/** 显存可行性判定（backend/infra/gguf_meta.estimate_vram 的结论）。 */
export interface LlmVramOut {
  weights_bytes: number;
  kv_cache_bytes: number | null;
  overhead_bytes: number;
  needed_bytes: number | null;
  free_vram_bytes: number | null;
  /** full_gpu / tight / partial_offload / infeasible / unknown */
  verdict: string;
  advice: string;
  notes: string[];
}

/** 一条可选模型（本地 GGUF 或服务端已加载模型）。 */
export interface LlmModelOut {
  id: string;
  name: string;
  display_name: string;
  source: LlmSource;
  /** 本地模型的文件路径；服务条目为空串。 */
  path: string;
  /** 服务条目的端点；本地条目为空串。 */
  endpoint: string;
  size_bytes: number;
  architecture: string;
  quant: string;
  /** **模型支持上限**，不是运行时 `-c` 实际值，两者不可混为一谈。 */
  max_ctx: number | null;
  is_embedding: boolean;
  available: boolean;
  error: string | null;
  active: boolean;
  /** 同一模型多副本时，`primary` 为择一保留的主条目。 */
  primary: boolean;
  duplicate_of: string | null;
  duplicate_count: number;
  /** 服务条目的显存判定为 null（不占本机额外显存）。 */
  vram: LlmVramOut | null;
  notes: string[];
}

/** 引擎（llama-server 进程 / 外部端点）运行状态。 */
export interface LlmEngineOut {
  enabled: boolean;
  mode: LlmMode | string;
  state: LlmEngineState | string;
  endpoint: string;
  /** true = 复用外部已在跑的实例，退出时不回收。 */
  adopted: boolean;
  error: string | null;
  /** 面向用户的可执行指引（如「先启动本机 llama 服务」）。 */
  advice: string | null;
  exit_code?: number;
}

export interface LlmSelectionOut {
  active_id: string | null;
  mode: string | null;
  endpoint: string;
  model_path: string;
}

export interface LlmScanProgressOut {
  dirs: number;
  found: number;
  current: string;
}

export interface LlmScanOut {
  /** idle / running / done / cancelled / error */
  state: string;
  roots: string[];
  declared_dirs: string[];
  progress: LlmScanProgressOut;
  found: number;
  elapsed_sec: number | null;
  error: string | null;
}

export interface LlmGpuOut {
  name: string;
  total_bytes: number;
  free_bytes: number;
}

/** 已有 llama 兼容端点探测结果。 */
export interface LlmServiceOut {
  host: string;
  port: number;
  base_url: string;
  openai_base_url: string;
  reachable: boolean;
  needs_auth: boolean;
  health_ok: boolean;
  models: string[];
  error: string | null;
}

export interface LlmStatusOut {
  engine: LlmEngineOut;
  selection: LlmSelectionOut;
  scan: LlmScanOut;
  gpu: LlmGpuOut | null;
  model_dirs: string[];
  counts: { local: number; service: number; available: number; total: number };
}

export interface LlmModelsOut {
  active_id: string | null;
  selection: LlmSelectionOut;
  gpu: LlmGpuOut | null;
  scan: LlmScanOut;
  /** 是否含"跑不动"的条目（前端不折叠、只标注）。 */
  counts: {
    local: number;
    service: number;
    available: number;
    total: number;
    infeasible: number;
  };
  services: LlmServiceOut[];
  models: LlmModelOut[];
}

export interface LlmServicesOut {
  services: LlmServiceOut[];
}

export interface LlmScanResponseOut {
  ok: boolean;
  started: boolean;
  status: LlmScanOut;
}

export interface LlmSelectOut {
  ok: boolean;
  active: string | null;
  mode: string;
  endpoint: string;
  model_path: string;
  reloaded: boolean;
  engine: LlmEngineOut;
  warning: string | null;
}

export interface LlmDirsOut {
  ok: boolean;
  model_dirs: string[];
  config_model_dirs: string[];
}

/** 底片印字（扫描日期/编号）识别结论快照（镜像 ReportOut.stamp / run_inspection 结果） */
export interface FilmStampOut {
  status: "present" | "missing" | "unavailable" | "off";
  text: string | null;
  orientation: "normal" | "mirrored" | null;
  confidence: number | null;
  need_review: boolean;
}

/**
 * 《射线检测报告》汇总表补充信息字段（镜像 backend/domain/report/meta_fields.py，
 * 键为前后端+PDF 填充三方约定；缺省/留空栏在报告中留空供手工补填）。
 */
export interface ReportMetaField {
  key: string;
  label: string;
  /** 输入示例/占位提示 */
  ph?: string;
}

export const REPORT_META_GROUPS: readonly { title: string; fields: readonly ReportMetaField[] }[] = [
  {
    title: "工程信息",
    fields: [
      { key: "client_unit", label: "委托单位" },
      { key: "project_name", label: "工程名称" },
      { key: "project_category", label: "工程类别/检测时机", ph: "如 锅炉安装/焊后" },
      { key: "test_address", label: "检测地址", ph: "如 施工现场" },
    ],
  },
  {
    title: "工件概况",
    fields: [
      { key: "material", label: "材质", ph: "如 20G" },
      { key: "part_no", label: "工件编号" },
      { key: "groove_type", label: "坡口形式", ph: "如 V" },
      { key: "surface_status", label: "表面状况", ph: "如 符合要求" },
      { key: "weld_process", label: "焊接方式", ph: "如 GTAW" },
      { key: "heat_treatment", label: "热处理状态" },
    ],
  },
  {
    title: "技术要求",
    fields: [
      { key: "tech_level", label: "检测技术等级", ph: "A / AB / B" },
      { key: "accept_level", label: "合格级别（验收要求）", ph: "Ⅰ / Ⅱ / Ⅲ / Ⅳ" },
      { key: "record_no", label: "原始记录编号" },
      { key: "scatter_control", label: "散射线控制" },
    ],
  },
  {
    title: "检测器材及工艺参数",
    fields: [
      { key: "source_kind", label: "源种类", ph: "X射线 / γ源" },
      { key: "device_no", label: "设备型号/编号" },
      { key: "focus_size", label: "焦点尺寸", ph: "如 2.0×2.0mm" },
      { key: "film_model", label: "胶片型号" },
      { key: "film_size", label: "胶片规格", ph: "如 180×80mm" },
      { key: "film_class", label: "胶片分类等级", ph: "如 C5" },
      { key: "screen_way", label: "增感方式", ph: "如 Pb" },
      { key: "iqi_position", label: "像质计摆放", ph: "源侧 / 胶片侧" },
      { key: "screens", label: "前屏/后屏", ph: "如 0.03/0.03mm" },
      { key: "technique", label: "透照方式", ph: "如 双壁双影" },
      { key: "focus_distance", label: "F（焦距）" },
      { key: "source_distance", label: "f（源至工件）" },
      { key: "film_distance", label: "b（工件至胶片）" },
      { key: "tube_voltage", label: "管电压", ph: "如 190kV" },
      { key: "tube_current", label: "管电流", ph: "如 5mA" },
      { key: "exposure_time", label: "曝光时间", ph: "如 1.5min" },
      { key: "develop_method", label: "冲洗条件", ph: "手工 / 自动" },
      { key: "developer", label: "显影液配方" },
      { key: "develop_temp", label: "洗片温度", ph: "如 22℃" },
    ],
  },
];

/** 报告补充信息（键 → 用户填写值；进入 POST /report 的 report_meta 表单字段） */
export type ReportMeta = Record<string, string>;

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
  /** 门禁降级/屏蔽告警（黑度越界/翻拍降级/印字区屏蔽等"为什么转人工"） */
  warnings?: readonly string[];
  /** 判定依据/熔断原因快照（与报告 PDF"判定依据"章节同源） */
  basis?: readonly string[];
  /** 底片黑度与门禁结论 */
  density?: number | null;
  density_ok?: boolean | null;
  iqi_pass?: boolean | null;
  /** IQI 验证明细 {type, achieved, required, grade} */
  iqi_detail?: Record<string, unknown> | null;
  /** 翻拍影像降级模式（绝对黑度不可测） */
  photo_mode?: boolean;
  /** 检测工作模式：balanced | recall_first | precision_first */
  detect_mode?: string | null;
  /** 印字区误检屏蔽数量（detect.mask_stamp_zone） */
  stamp_zone_masked?: number;
  /** 单张评片查重：与历史影像内容完全相同的记录摘要 */
  duplicates?: readonly Record<string, unknown>[];
  /** 报告补充信息回显（清洗后快照，报告页渲染样张式首页预览用） */
  report_meta?: ReportMeta;
  /** 首页预览所需表单回显（工件名称/焊缝编号/签字人/标准引用） */
  workpiece_no?: string | null;
  weld_no?: string | null;
  signer?: string | null;
  standard_ref?: string | null;
  pdf_url: string;
  /** 底片印字性质快照（重新生成模式无 fresh 识别结果时为 null） */
  stamp?: FilmStampOut | null;
  /**
   * AI 预筛级别标记：true 表示 joint_level 来自"底片质量未达标但用户显式请求
   * 预筛"的通道，级别不具合规效力（basis 首条给出降级原因，界面须显著标识）。
   */
  grade_preliminary?: boolean;
}

/** GET /api/v1/report/{report_id}/narrative → ReportNarrativeOut */
export interface ReportNarrativeOut {
  report_id: string;
  /** ok=已生成；disabled/unavailable/failed/empty=未生成（见 reason） */
  status: "ok" | "disabled" | "unavailable" | "failed" | "empty";
  /** 评片结论正文（status=ok 时有效） */
  text: string;
  /** 实际使用的本地模型标识（llama-server 为权重文件名） */
  model: string;
  /** 未生成原因（面向评片员的可读说明，非堆栈） */
  reason: string;
  elapsed_ms: number;
  /** 强制免责声明：AI 撰述不构成等级/合格判定 */
  disclaimer: string;
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
  /** 底片印字（扫描日期/编号）性质快照（历史数据/未启用时为 null） */
  stamp_status?: string | null;
  stamp_text?: string | null;
  stamp_orientation?: "normal" | "mirrored" | null;
  stamp_confidence?: number | null;
  stamp_need_review?: boolean;
  /** 最新报告编号（档案行「查看报告」入口；旧版本后端无此字段） */
  report_id?: string | null;
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
  /** 人工确认缺陷是否成功自动回流训练池（false 时复核面板展示告警） */
  training_pool_synced: boolean;
}

/**
 * 真实流水线阶段（镜像 backend/app/pipelines.py 的执行顺序，非模拟数据；
 * 仅作处理中视图的流程说明，进度/状态一律来自真实请求）。
 */
export const PIPELINE_STAGES: readonly string[] = [
  "底片影像加载",
  "影像质量校验（黑度 D · 像质计 IQI）",
  "缺陷检出与当量测定",
  "标准符合性判定（NB/T 47013.2）",
  "检测数据归档",
  "评片报告签发（PDF/A）",
] as const;


/* ── 主动学习（ · POST /api/v1/active/…）── */

/** 高价值样本候选（主动学习采样结果，POST /active/sample 响应） */
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

/** 单个重复文件的查重复核信息（submit 响应与批次 status 共用） */
export interface BatchDuplicateItem {
  task_id: string;
  image_name: string;
  content_sha256: string | null;
  kind: "history" | "batch"; // history=与历史已检影像重复 | batch=批内重复
  duplicate_of: string | null; // 批内原文件名或历史 image_id
  history: {
    image_id: string;
    workpiece_no: string | null;
    weld_no: string | null;
    joint_level: string | null;
    created_at: string | null;
  } | null;
}

/** POST /api/v1/batch → BatchSubmitOut */
export interface BatchSubmitOut {
  batch_id: string;
  total: number;
  estimated_sec: number;
  status: "running" | "awaiting_review";
  duplicates: BatchDuplicateItem[];
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
  /** 检出缺陷数（null=未执行或旧批次快照无此字段） */
  defect_count?: number | null;
  /** 底片印字（扫描日期/编号）识别结论（null=未执行或旧批次快照） */
  stamp_status?: string | null;
  stamp_text?: string | null;
  stamp_orientation?: "normal" | "mirrored" | null;
  stamp_need_review?: boolean | null;
  content_sha256?: string | null;
  dup_kind?: "history" | "batch" | null;
  dup_ref?: string | null;
}

/** 批次收尾的印字占比裁决摘要（缺印字复核的批量豁免结论） */
export interface BatchStampSummary {
  evaluated: number;
  present: number;
  ratio: number;
  suppressed: boolean;
  flagged: number;
}

/** 批次级状态机（backend/app/batch_queue.py）：awaiting_review=查重命中待人工复核 */
export type BatchStatus = "awaiting_review" | "running" | "paused" | "finished";

/** GET /api/v1/batch/{id} → BatchStatusOut */
export interface BatchStatusOut {
  batch_id: string;
  status: BatchStatus;
  total: number;
  done: number;
  failed: number;
  cancelled: number;
  estimated_sec: number;
  progress: number;
  tasks: BatchTaskOut[];
  duplicates?: BatchDuplicateItem[];
  stamp_summary?: BatchStampSummary | null;
}

/** 人工查重复核的单项决定：skip=跳过不检测 | keep=人工确认后仍检测 */
export interface BatchDedupDecision {
  task_id: string;
  action: "skip" | "keep";
}

/** GET /api/v1/batches → 列表项（历史/断点续跑入口） */
export interface BatchSummaryOut {
  batch_id: string;
  status: BatchStatus;
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

/** 底片查看器的单条缺陷标注框（批量检测完成后回填，bbox 为原图像素坐标） */
export interface FilmAnnotationBox {
  id: string;
  classId: number;
  bbox: [number, number, number, number];
  confidence: number;
  /** 该缺陷本身被标记为需人工复核 */
  needReview: boolean;
}

/** 一幅底片的标注集合：imageW/H 为检测时后端记录的原图尺寸（坐标换算基准） */
export interface FilmAnnotations {
  imageW: number;
  imageH: number;
  reportId: string;
  boxes: FilmAnnotationBox[];
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

/* ── C-14 受控导出（POST /api/v1/export/…）── */

/** 导出申请状态机：pending（待审批）→ approved/rejected（保密员决策）→ consumed（令牌已使用） */
export type ExportRequestStatus = "pending" | "approved" | "rejected" | "consumed";

/** 导出申请（POST /export/requests 响应 / GET /export/requests/{id}） */
export interface ExportRequestOut {
  request_id: string;
  subject: string;
  reason: string | null;
  requested_by: string;
  status: ExportRequestStatus;
  decided_by: string | null;
  decided_at: string | null;
  token_expires_at: string | null;
  used_at: string | null;
  created_at: string | null;
}

/** POST /export/requests/{id}/token 响应：一次性令牌（明文仅本次返回） */
export interface ExportTokenOut {
  token: string;
  expires_in_sec: number;
}

