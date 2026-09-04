"""配置加载。

- 所有可调参数入 configs/*.yaml，环境变量以 SCAN_ 前缀覆盖；
- 禁止硬编码端口/路径/密钥；
- 新增配置键必须同时更新 schema.yaml 与本模块字段。
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings

_BASE = Path(__file__).resolve().parents[1] / "configs"
_LOG = logging.getLogger("scandetection.config")


class ServerCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 18773
    # CORS 允许源：Tauri webview + 本地开发源。
    # 部署新增前端源（如公司内网门户）改配置即可，不改代码；禁 "*"，
    # 否则任意外部网站均可跨源读取本机 API（含审计链 / 报告）。
    cors_origins: list[str] = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "tauri://localhost",
        "https://tauri.localhost",
        "http://tauri.localhost",
    ]


class ModelCfg(BaseModel):
    default_uri: str = "models/weights/best.onnx"
    backend: str = "onnx"  # onnx | torch | tensorrt
    weights_dir: str = "models/weights"  # 模型注册表扫描目录（§7.4，M4）
    registry_state_file: str = "data/model_registry.json"  # 活跃模型指针持久化
    # ONNX Runtime 执行提供者（S-04 推理后端可插拔）。默认 CPU 不变；
    # 换 onnxruntime-gpu 后可配 ["CUDAExecutionProvider", "CPUExecutionProvider"]。
    # 昇腾 CANN / 寒武纪 / DCU 等国产后端为预留写法，未真机验证（见 default.yaml 注释）。
    providers: list[str] = ["CPUExecutionProvider"]
    # S-17 运行期模型回退：连续推理异常达到阈值 → 自动回退上一稳定版本权重
    # （fail-safe 语义），并落告警+审计+degraded 标记。
    infer_failure_threshold: int = 3
    auto_rollback: bool = True


class SecurityCfg(BaseModel):
    encrypt: bool = True


class AuthCfg(BaseModel):
    """三员身份认证配置（C-06/C-07/C-09）。

    challenge_ttl_sec      : 登录挑战有效期（一次一用，防重放）；
    idle_timeout_min       : 会话空闲超时（分钟，滑动过期）；
    session_ttl_min        : 会话绝对有效期（分钟，从签发起算上限）；
    max_sessions           : 单账号并发会话上限（超限吊销最旧会话，单点登录语义）；
    max_failed_attempts    : 连续挑战失败锁定阈值；
    lockout_min            : 触发锁定后的锁定时长（分钟），并落安全告警。
    """

    challenge_ttl_sec: int = 60
    idle_timeout_min: int = 15
    session_ttl_min: int = 720
    max_sessions: int = 1
    max_failed_attempts: int = 5
    lockout_min: int = 30


class BatchExportAlertCfg(BaseModel):
    """批量导出告警规则（C-22）。

    enabled     : 是否启用"窗口期内批量导出"异常行为告警；
    window_min  : 计数窗口（分钟）；
    threshold   : 窗口期内同一操作者的导出下载次数达到该值即触发一次 high 告警
                  （仅在跨越阈值的那一刻告警一次，防告警刷屏）。
    """

    enabled: bool = True
    window_min: int = 10
    threshold: int = 5


class SimpleAlertCfg(BaseModel):
    """布尔开关型告警规则（C-22）：每次事件触发即落一条告警。"""

    enabled: bool = True


class AlertsCfg(BaseModel):
    """安全告警规则配置（C-22 异常行为告警完善）。

    各告警 kind 的开关/阈值集中于此，替代散落硬编码：
    - batch_export        : 批量导出（窗口期内导出≥threshold 次 → high 告警）；
    - unauthorized_access : 越权访问（require_role 403 → warn 告警）；
    - account_lockout     : 登录失败锁定（account_locked → critical 告警）。
    """

    batch_export: BatchExportAlertCfg = BatchExportAlertCfg()
    unauthorized_access: SimpleAlertCfg = SimpleAlertCfg()
    account_lockout: SimpleAlertCfg = SimpleAlertCfg()


class ExportCfg(BaseModel):
    """导出管控配置（C-14）。

    require_approval : true=报告 PDF/清单导出需保密员预授权（申请→批准→
                       一次性令牌→凭令下载）或导出审批令牌；false=仅登录即可导出
                       （单机调试用，生产保持 true）。
    token_ttl_sec    : 一次性导出令牌有效期（秒）。
    """

    require_approval: bool = True
    token_ttl_sec: int = 600


class EvalCfg(BaseModel):
    """评估 / 漂移 / 实验追踪配置。

    golden_dir        : 固定、版本化评估集（禁止用于训练）；缺失则 evaluate 端点返回 409。
    drift_baseline_path: 漂移监控参考基线（尺寸/置信度/类别分布），首跑自动建立。
    experiments_dir  : 实验追踪 JSONL 落盘目录。
    auto_on_activate : 模型激活后自动跑 Golden Set 评估（使 metric_map 有值）。
    """

    golden_dir: str = "data/eval/golden"
    drift_baseline_path: str = "data/eval/drift_baseline.json"
    experiments_dir: str = "data/experiments"
    auto_on_activate: bool = True


class SyncCfg(BaseModel):
    """端边云同步适配器选择。

    kind=local：LocalAdapter（数据不出本机，默认）；
    kind=http：HttpSyncAdapter（推送到可配置端点，需自行保证传输加密）。
    """

    kind: str = "local"  # local | http | cloud（v3 联邦占位，未实现，勿生产启用）
    http_endpoint: str | None = None  # kind=http 时的推送 URL
    http_token: str | None = None  # 可选 Bearer Token（建议走环境变量注入，勿入版本库）
    http_timeout: float = 10.0  # HTTP 同步推送超时（秒，§13.6 配置中心化）


class EgressCfg(BaseModel):
    """进程级外联防护（C-16）。

    enabled     : true=启动时装配外联拦截（socket/urllib 层，进程级）；
    allow_cidrs : 允许外连的目的网段（CIDR）。本机回环 127.0.0.0/8 与 ::1/128
                  在代码中恒放行（本机前后端/标注器通信必需，不可配置关闭），
                  此处只需登记额外放行的内网网段；默认空 = 除回环外全拦截
                  （纯离线部署的从严默认）。配置 sync.kind=http 推送端点时，
                  须将其网段显式加入白名单。
    """

    enabled: bool = True
    allow_cidrs: list[str] = []


class IpcCfg(BaseModel):
    """IPC 一次性启动令牌（C-17）。

    enforce: true=除存活/指标/认证与静态资源外，所有请求须携带 X-IPC-Token
             头（或已带 Bearer 会话）——防其他本机进程误调/网页 CSRF 式调用。
             后端启动时生成一次性令牌写入 data/ipc_token（进程生命周期有效），
             Tauri 外壳读取后注入 WebView。单机调试/测试可置 false（诚实声明：
             本机回环为明文 HTTP，令牌不解决传输加密；需 TLS 时挂本机证书，
             不在本次范围）。
    """

    enforce: bool = True


class AnnotatorCfg(BaseModel):
    """人工标注服务是否随主应用同进程启动。

    enabled=True 时，lifespan 内以守护线程拉起标注器（默认 8899），
    使标注→训练池回流在单一进程内闭环，无需手动另开终端。
    """

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8899


class PathsCfg(BaseModel):
    data_dir: str = "data"
    tmp_dir: str = "data/tmp"
    db_path: str = "data/scan.db"  # SQLite 数据库文件（§7.1）
    # S-03 数据库方言：非空时作为完整 SQLAlchemy URL 优先于 db_path 使用
    # （如达梦/人大金仓，示例见 infra/db.py docstring；未真机验证）。
    # 默认空 = 保持 sqlite:///<db_path> 语义不变。
    db_url: str = ""
    images_dir: str = "data/images"  # 原图副本目录（报告缺陷图谱数据源）
    reports_dir: str = "data/reports"  # PDF 报告输出目录（§7.2）


class DensityCfg(BaseModel):
    low: float = 2.0  # AB 级黑度下限
    high: float = 4.5  # AB 级黑度上限
    # 翻拍影像（相机拍灯箱，8bit 且黑度物理上限 2.41、绝对黑度不可测）门禁策略：
    # warn=黑度/IQI/质量门禁降级为告警+强制人工复核（不阻断出片）；
    # block=与扫描件同等严格（不通过即阻断评片）。
    photo_policy: str = "warn"


class FilmRegionCfg(BaseModel):
    """底片区域分割配置（backend/domain/film_region.py 的配置镜像）。

    enabled=True 时评片前先分割胶片有效区：黑度按掩膜计算、IQI/伪缺陷/
    质量门禁在胶片区上评估、检测时屏蔽胶片区外背景（坐标系不变）。
    """

    enabled: bool = True
    min_area_frac: float = 0.08  # 胶片区最小占画面比例（低于视为分割失败）
    max_photo_area_frac: float = 0.88  # 胶片占比低于此值才可能判翻拍（满幅=扫描件）
    surround_bright_gray: float = 200.0  # 环绕背景"亮"的灰度下限（灯箱过曝特征）
    surround_min_frac: float = 0.05  # 亮背景占**整幅**最小占比（低阈偏安全：误判翻拍仅多人工复核）


class IqiCfg(BaseModel):
    # 线型像质计丝号 1..N 直径(mm)，递增（公开参考，待官方复核）
    type: str = "wire"  # wire | hole（线型/孔型像质计，§4.2）
    wire_diameters_mm: tuple[float, ...] = (
        3.2,
        2.5,
        2.0,
        1.6,
        1.25,
        1.0,
        0.8,
        0.63,
        0.5,
        0.4,
        0.32,
        0.25,
        0.2,
        0.16,
        0.125,
        0.1,
        0.08,
        0.063,
        0.05,
    )
    required_wire_no: int = 10
    # 孔型像质计孔径(mm)，递增（公开参考，待官方复核）
    hole_diameters_mm: tuple[float, ...] = (
        1.0,
        0.8,
        0.63,
        0.5,
        0.4,
        0.32,
        0.25,
        0.2,
        0.16,
        0.125,
    )
    required_hole_no: int = 6
    min_contrast_ratio: float = 3.0
    auto_locate: bool = True  # 自动定位像质计（模板匹配），关闭则须前端/人工给 ROI
    locate_threshold: float = 0.3  # 像质计带垂直周期性强度下限（低于视为未找到）
    # A/AB/B 影像质量等级 → 线型像质计要求丝号（公开参考，待官方复核）：
    # [透照厚度上限 mm, A级要求丝号, AB级要求丝号, B级要求丝号]，等级越严要求越细。
    sensitivity: tuple[tuple[float, int, int, int], ...] = (
        (2.0, 14, 13, 12),
        (4.0, 13, 12, 11),
        (8.0, 12, 11, 10),
        (12.0, 11, 10, 9),
        (18.0, 10, 9, 8),
        (30.0, 9, 8, 7),
        (50.0, 8, 7, 6),
        (80.0, 7, 6, 5),
        (120.0, 6, 5, 4),
        (200.0, 5, 4, 3),
        (350.0, 4, 3, 2),
        (9999.0, 3, 2, 1),
    )


class PseudoDefectCfg(BaseModel):
    """伪缺陷筛查阈值。

        字段集须与 backend/domain/pseudo_defect.py 的 PseudoDefectCfg 对齐
    。
    """

    hough_threshold: int = 60
    scratch_min_ratio: float = 0.5
    scratch_grating_min_lines: int = 5  # 长直线段数 ≥ 此值视为周期性光栅（像质计），非孤立划痕
    canny_lo: int = 40
    canny_hi: int = 120
    uniformity_low_freq: float = 0.012
    uniformity_max_ratio: float = 6.0
    dust_tophat_k: int = 15
    dust_min_area: int = 10  # 尘点最小面积（px，过滤本底噪声小斑）
    dust_max_count: int = 400  # 尘点连通域上限（超则判"尘点密集/污渍"）
    block_on_scratch: bool = True
    block_on_uniformity: bool = False
    block_on_dust: bool = False


class PreprocessCfg(BaseModel):
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    median_k: int = 3
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    canny_kernel: int = 5
    morph_k_open: int = 3
    morph_k_close: int = 3
    gamma: float = 1.0  # 默认伽马（请求未显式传参时使用；1.0 = 不做伽马校正）
    enabled: bool = True  # 是否在 run_inspection 中应用降噪+增强（关闭则对原始 gray 推理）


class QualityCfg(BaseModel):
    """射线底片质量门禁配置。

    门禁用可解释 RQI 复合分（0–100，越高越好）= Σ wᵢ·sᵢ，sᵢ∈[0,1]。
    min_score 以下判不合格；block_on_quality=True 时不合格阻断评片（默认 True，
    反证测试确认真实好底片 0 误杀；不合格即阻断评片并提示重拍/换片）。
    """

    w_noise: float = 0.25
    w_sharp: float = 0.20
    w_contrast: float = 0.20
    w_dynamic: float = 0.15
    w_uniform: float = 0.10
    w_artifact: float = 0.10
    min_score: float = 70.0  # RQI 门限（0–100），已按真实底片标定
    block_on_quality: bool = True  # True=不合格阻断评片；False=仅告警+need_review
    noise_good: float = 4.0  # 噪声σ ≤ 此值满分
    noise_bad: float = 14.0  # 噪声σ ≥ 此值 0 分
    sharp_good: float = 1.5  # 平均梯度幅值满分阈值（真实底片实测 ~0.7–8.7，原 18 严重偏高）
    contrast_good: float = 25.0  # 信号对比度（中值滤波后 std）满分阈值
    dr_good: float = 0.6  # 动态范围利用率（(p99−p1)/255）满分阈值
    uniformity_low_freq: float = 0.012  # 低频核占比（相对短边）
    uniformity_max_ratio: float = 6.0  # 低频漂移 σ / 局部 σ 上限
    dust_tophat_k: int = 15  # 尘点 top-hat 核（奇数）
    dust_min_area: int = 10  # 尘点最小面积（px）
    dust_max_count: int = 400  # 尘点连通域上限
    # —— 三类硬门禁（反证测试后补强；任一触发即判不合格，不扰动已标定 RQI）——
    blur_lap_bad: float = 30.0  # Laplacian 方差低于此值判失焦/模糊（真实底片实测 ≥45）
    exposure_entropy_bad: float = 0.62  # 直方图熵(归一化)低于此值判过/欠曝（真实底片实测 ≥0.73）
    stain_smooth_bad: float = 0.15  # 平滑异常斑块占比高于此值判污渍（真实底片实测 ≤0.14）


class MaskRefineCfg(BaseModel):
    """掩膜精修量化配置（， 四地同步）。

        字段集须与 backend/domain/quantify.py 的 MaskRefineCfg dataclass 对齐
    。
    """

    enabled: bool = True
    blur_k: int = 5  # ROI 高斯平滑核（奇数）
    adaptive_block: int = 31  # 自适应阈值窗口（奇数，≤ROI 短边）
    adaptive_c: float = 8.0  # 自适应阈值常数 C（0–255 量纲）
    min_mask_abs_area_px: int = 4  # 轮廓面积小于此值丢弃
    min_mask_rel_area: float = 0.25  # 掩膜面积须 ≥ 此比例×框面积，否则回退包围盒
    close_k: int = 5  # 形态学闭运算核
    round_aspect_max: float = 3.0  # 圆形/条形分界（L/W<=3 为圆形）


class DetectCfg(BaseModel):
    # 安全默认 = 训练模型路径（缺失权重时按 allow_baseline_fallback 策略显式回退并记日志）。
    # 原默认 True：一旦 default.yaml 缺键，会静默落 blob 基线而无人告警。
    kind: str = "trained_yolo"  # 检测器种类（注册表键）：trained_yolo=YOLO 训练模型，baseline_blob=连通域基线
    quantifier_kind: str = (
        "mask"  # 量化器种类（注册表键，§T8）：mask=掩膜精修(M4b)，bbox=包围盒近似(M4a)
    )
    baseline_enabled: bool = (
        False  # M4a 基线检测器开关（kind 的兼容别名；显式 kind 优先）；训练模型就绪后保持 false
    )
    allow_baseline_fallback: bool = True  # 训练模型加载失败时是否回退基线（False=启动即失败）
    infer_conf: float = 0.3  # 推理置信度阈值（§T8：禁硬编码，统一入口）
    infer_iou: float = 0.5  # NMS IoU 阈值
    # 逐类置信度阈值（ 扩展）：稀有且安全关键缺陷设更低阈值优先召回，
    # 气孔设更高阈值抑制海量误检。未在此映射中的类回落 infer_conf。
    # 依据：稀有平衡实验（runs/yolo11n_real_rare）显示稀有类信号弱、需低阈值放行，
    # 而气孔占训练样本 94.6%，高阈值可压低误检。裂纹/未熔合属 NB/T47013 重大缺陷，
    # 漏检代价远高于误检，故取最低阈值；低置信候选经 review_conf 兜底转人工复核。
    class_conf: dict[int, float] = {
        0: 0.30,  # POROSITY 气孔：高阈值抑制海量误检
        1: 0.12,  # SLAG 夹渣
        2: 0.12,  # INCOMPLETE_PENETRATION 未焊透
        3: 0.08,  # LACK_OF_FUSION 未熔合（重大缺陷，低阈值优先召回）
        4: 0.05,  # CRACK 裂纹（最危险缺陷，最低阈值）
        5: 0.18,  # UNDERCUT 咬边
        6: 0.10,  # CONCAVITY 内凹（DB50/T 1807 重点关注缺陷，低阈值优先召回）
    }
    round_aspect_max: float = 3.0  # 圆形/条形长宽比分界（NB/T47013：L/W<=3 为圆形）
    min_area_px: int = 30
    max_area_px: int = 200_000
    min_size_px: int = 3
    noise_sigma_ratio: float = 2.5
    abs_threshold: float = 8.0
    dark_only: bool = False
    review_conf: float = 0.5  # 检测不确定性阈值，超过则判定 need_review（M4b 人工兜底）


class UploadCfg(BaseModel):
    """上传限额：无上限的 multipart 读取会被单请求打爆内存/磁盘。"""

    max_bytes: int = 200 * 1024 * 1024  # 单文件上限 200 MiB（大幅面 DR 底片留余量）
    allowed_suffixes: tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".dcm",
        ".dicom",
        ".ima",
    )


class StandardCfg(BaseModel):
    default_id: str = "NB/T47013.2-2015"
    tables_filename: str = "nb47013.yaml"


class ReviewCfg(BaseModel):
    kappa_threshold: float = 0.8  # Cohen's κ 高度一致阈值（§15.3）；低于则升级仲裁


class GateCfg(BaseModel):
    """底片合格性门禁补充配置（DB50/T 1807-2025 §5 扫描参数，评片硬前置）。

    dpi      : 扫描分辨率下限。DICOM 按 PixelSpacing 推算；通用图像文件
               元数据加载器未解析、无法确定时由 require_dpi 决定处置
               （true=从严拦截，false=放行并告警留档）。
    bit_depth: 位深硬门禁。8bit 底片灰度精度不足默认拦截，allow_8bit=true
               可降级放行（告警留档，仍强制人工复核语义由上层决定）。
    """

    min_dpi: int = 600
    require_dpi: bool = False  # dpi 无法确定时：true=拦截（从严），false=放行+告警
    min_bit_depth: int = 16
    allow_8bit: bool = False
    rejects_dir: str = "data/rejects"  # 不合格底片留档目录（密文归档）


class StdEvalCfg(BaseModel):
    """DB50/T 1807-2025 标准评价配置（backend/evaluation/std501807.py）。

    iou_standard/strict : 标准口径 0.1 与严格口径（双阈值并行评估，
                          记录分级取两者较差）。
    weld_form/method    : single=单面焊（重点关注含内凹/咬边）；
                          manual/auto 决定 FRR 分级线。
    strict_frr          : true=FRR 分级线默认取收紧值（自动 3%/手工 4%，
                          即标准 L2 线），严于标准 L1 线。
    personnel_path      : 评价/标注人员资质记录（TSG Z8001 证书，附录A 用）。
    """

    iou_standard: float = 0.1
    iou_strict: float = 0.3
    weld_form: str = "single"
    weld_method: str = "manual"
    strict_frr: bool = True
    frr_l1_auto: float = 0.08
    frr_l1_manual: float = 0.10
    frr_strict_auto: float = 0.03
    frr_strict_manual: float = 0.04
    aspect_round_max: float = 3.0
    eval_dir: str = "data/eval"
    personnel_path: str = "data/eval/std_personnel.json"


class ObservabilityCfg(BaseModel):
    """可观测性配置。

    log_format     : text=人类可读（本地开发默认）；json=单行结构化日志，
                     便于接入日志采集（十年数据积累的可观测基础）。
    enable_metrics : 是否启用进程内指标采集（/api/v1/metrics 导出）。
                     true 时每请求计数/计时（内存开销极小、恒定）。
    """

    log_format: str = "text"  # text | json
    enable_metrics: bool = True


class BatchCfg(BaseModel):
    """批量任务队列配置。

    桌面单机场景：线程池多 worker 并行推理（run_inspection 全链路逐图跑），
    单图失败隔离不拖垮整批。断点续跑 = 状态快照持久化到 data/batch/，
    重启后可查询历史与未完成标记（自动重跑未完成项为增强项，v1 不实现）。
    """

    workers: int = 2  # 并行 worker 数（IO/推理并行度；过大会加重 CPU/内存）
    max_per_batch: int = 100  # 单批最大图数（防一次性打爆资源）
    per_image_estimate_sec: float = 8.0  # 单图预估耗时（进度条/预计时间展示用）
    max_retained_batches: int = 50  # 内存保留终结批次上限（S-*：防 100h 长跑内存随批次无界增长；
    # 超过后只驱逐"已完成且无失败可重试任务"的最旧批次出内存）
    max_retained_snapshot_files: int = 200  # 磁盘快照保留上限（S-21 防磁盘随批次无界增长）：
    # data/batch/*.json 超过该值后，只剪枝"已完成且无失败可重试任务"的最旧快照文件；
    # 保留有 retry 价值（failed/cancelled 可断点续跑）的批次，running/新建批次恒保留。


class BackupCfg(BaseModel):
    """备份策略配置（S-12 备份增强）。

    interval_hours : 自动备份间隔（小时）；0=关闭定期调度（默认，不影响测试）。
                     >0 时应用启动后由后台线程按间隔 create_backup 并记审计。
    include_images : 备份是否纳入影像目录（paths.images_dir）。默认 false
                     （影像体积大，归档策略 v1 不并入）；true 时逐文件 SM3 校验。
    """

    interval_hours: float = 0.0
    include_images: bool = False


class WatchdogCfg(BaseModel):
    """内存看门狗（S-09）：后台线程周期采样 RSS，超阈值告警/可选标记重启。

    enabled          : 默认 false（不影响既有测试/部署）。
    interval_sec     : 采样周期（秒）。
    rss_warn_mb      : RSS 告警阈值（MB），超限落 security alert + 审计。
    rss_restart_mb   : RSS 重启标记阈值（MB）；仅 graceful_restart=true 时写
                       data/restart_required 标记文件（由 Tauri 壳检测重启，
                       当前壳侧集成待做——诚实边界，仅告警+审计兜底）。
    graceful_restart : 是否允许写重启标记文件。
    """

    enabled: bool = False
    interval_sec: float = 30.0
    rss_warn_mb: float = 2048.0
    rss_restart_mb: float = 4096.0
    graceful_restart: bool = False


class DiskSpaceCfg(BaseModel):
    """磁盘水位看门狗（S-20）：后台线程周期统计 data 分区剩余空间，低水位告警。

    enabled        : 默认 true（生产开启磁盘层兜底，防磁盘写满导致 DB/WAL/报告故障）。
    interval_sec   : 采样周期（秒），低频（默认 300s）避免无谓 I/O。
    warn_ratio_pct : 剩余空间比例阈值（%），低于即告警。
    warn_min_bytes : 剩余空间绝对阈值（字节），低于即告警（默认 1 GiB）。
                     任一触发即告警；同一次连续低水位只告警一次。
    """

    enabled: bool = True
    interval_sec: float = 300.0
    warn_ratio_pct: float = 10.0
    warn_min_bytes: int = 1073741824  # 1 GiB


class ModelGateCfg(BaseModel):
    """模型投产门禁状态机（E-14：更新即重评投产门禁）。

    enabled    : 门禁开关。默认 false = 保持旧行为（activate 即切换，评估仅
                 激活后补跑）；true = activate 先跑 Golden 评估，达标进入
                 candidate 状态，需 sysadmin 经 POST /models/{id}/approve
                 审批后才真正切换（审批留审计）。
    min_map    : Golden 评估 mAP@0.5 下限（默认 0 = 宽松，仅要求评估可完成）。
    min_recall : Golden 评估召回下限（默认 0 = 宽松）。
    """

    enabled: bool = False
    min_map: float = 0.0
    min_recall: float = 0.0


class AppConfig(BaseSettings):
    server: ServerCfg = ServerCfg()
    model: ModelCfg = ModelCfg()
    security: SecurityCfg = SecurityCfg()
    auth: AuthCfg = AuthCfg()
    alerts: AlertsCfg = AlertsCfg()
    export: ExportCfg = ExportCfg()
    eval: EvalCfg = EvalCfg()
    sync: SyncCfg = SyncCfg()
    egress: EgressCfg = EgressCfg()
    ipc: IpcCfg = IpcCfg()
    annotator: AnnotatorCfg = AnnotatorCfg()
    paths: PathsCfg = PathsCfg()
    density: DensityCfg = DensityCfg()
    film_region: FilmRegionCfg = FilmRegionCfg()
    iqi: IqiCfg = IqiCfg()
    pseudo_defect: PseudoDefectCfg = PseudoDefectCfg()
    preprocess: PreprocessCfg = PreprocessCfg()
    quality: QualityCfg = QualityCfg()
    detect: DetectCfg = DetectCfg()
    mask_refine: MaskRefineCfg = MaskRefineCfg()
    upload: UploadCfg = UploadCfg()
    standard: StandardCfg = StandardCfg()
    review: ReviewCfg = ReviewCfg()
    gate: GateCfg = GateCfg()
    observability: ObservabilityCfg = ObservabilityCfg()
    batch: BatchCfg = BatchCfg()
    backup: BackupCfg = BackupCfg()
    watchdog: WatchdogCfg = WatchdogCfg()
    disk_space: DiskSpaceCfg = DiskSpaceCfg()
    # 注意字段名 modelgate（非 model_gate）：E-14 专项测试/部署用环境变量
    # SCAN_MODELGATE__ENABLED=true 开启完整门禁链（env 解析按小写段匹配）。
    modelgate: ModelGateCfg = ModelGateCfg()
    std_eval: StdEvalCfg = StdEvalCfg()

    model_config = {"env_prefix": "SCAN_"}


def load_config() -> AppConfig:
    """加载配置：configs/default.yaml 为基础，SCAN_* 环境变量优先覆盖。

    注意：pydantic-settings 的 init 参数优先于环境变量，因此不能直接
    `AppConfig(**raw)`（env 会被 yaml 值吞掉）——这里手动把 SCAN_* 合并进
    raw 再构造，保证"环境变量以 SCAN_ 前缀覆盖"承诺真正生效
    （测试隔离、部署注入均依赖此语义）。
    """
    raw: dict = {}
    cfg_file = _BASE / "default.yaml"
    if cfg_file.exists():
        raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    known = _known_paths(AppConfig)
    for key, value in os.environ.items():
        if key.startswith("SCAN_"):
            # pydantic-settings 环境变量大小写不敏感 → 转小写匹配字段路径
            path = tuple(k.lower() for k in key[5:].split("__"))
            # 仅合并叶子字段（排除 'paths' 这类段级路径，避免把整段覆盖成字符串导致启动崩溃）
            if len(path) >= 2 and path in known:
                _apply_env(raw, list(path), value)
    # 启动期捕获配置漂移（schema 与 default.yaml 不一致 → 静默落默认的高危场景）
    drift = validate_config_against_schema(raw)
    if drift:
        _LOG.error("配置漂移检测（%d 项）:\n  - %s", len(drift), "\n  - ".join(drift))
    return AppConfig(**raw)


def _known_paths(model: type[BaseModel], prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    """收集配置模型的合法嵌套字段路径（如 ('paths','db_path')）。"""
    paths: set[tuple[str, ...]] = set()
    for name, field in model.model_fields.items():
        path = prefix + (name,)
        paths.add(path)
        if isinstance(field.default, BaseModel):  # 嵌套配置段（均有默认实例）
            paths |= _known_paths(type(field.default), path)
    return paths


def _apply_env(target: dict, keys: list[str], value: str) -> None:
    """按嵌套键写入 target（如 SCAN_PATHS__DB_PATH → paths.db_path）。

    容错：遇到非 dict 中间节点（如 yaml 中该段为 null）时重建为 dict，
    避免 `None.setdefault` 抛 AttributeError 导致服务启动即崩溃。
    """
    cur: dict = target
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value


def _leaf_paths(node: Any, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    """递归收集配置树的叶子字段路径（如 ('detect','baseline_enabled')）。

    分支判定：dict 且**所有键均为合法标识符**（即"配置段"，键为字段名）→ 继续下钻；
    否则（标量 / list / 非标识符键的映射如 class_conf 的 {0:0.30,...}）→ 视为单个叶子。

    这避免了 schema.yaml（用 `bool`/`str` 等类型注解作叶值）与 default.yaml
    （用真实值，class_conf 是 int 键映射）在形状上的结构性误报：两侧都按
    "标识符键=配置段、其余=单叶子" 归一化，路径精确对齐。
    """
    if isinstance(node, dict) and all(isinstance(k, str) and _IDENT_RE.match(k) for k in node):
        out: set[tuple[str, ...]] = set()
        for k, v in node.items():
            out |= _leaf_paths(v, prefix + (str(k),))
        return out
    # 叶子：标量 / list / 非标识符键映射（dict-of-scalars）一并视作单个配置键
    return {prefix} if prefix else set()


def validate_config_against_schema(raw: dict) -> list[str]:
    """schema.yaml 是权威契约；与合并后的 raw 比对，返回漂移问题列表。

    捕获两类漂移：
    - ① schema 要求但 default.yaml 缺失 → 将静默落 pydantic 默认（如 baseline_enabled
      缺键会无声发 blob 检测器），属高危；
    - ② default.yaml 有但 schema 未登记 → 可能是新增键漏补 schema，仅告警。

    非致命：仅返回问题列表由调用方记录（启动期 ERROR 可见，不阻断启动）。
    新增键未补 schema 属渐进过程，不在此阻断；高危的①类缺失会被上层启动检查捕获。
    """
    schema_file = _BASE / "schema.yaml"
    if not schema_file.exists():
        return []
    schema = yaml.safe_load(schema_file.read_text(encoding="utf-8")) or {}
    expected = _leaf_paths(schema)
    actual = _leaf_paths(raw)
    issues: list[str] = []
    for p in sorted(expected - actual):
        issues.append(f"schema 要求但 default.yaml 缺失: {'.'.join(p)} (将静默落 pydantic 默认)")
    for p in sorted(actual - expected):
        issues.append(f"default.yaml 存在但 schema.yaml 未登记: {'.'.join(p)} (新增键未补 schema?)")
    return issues


# 安装根目录锚点：backend/infra/config.py -> parents[2] = 安装根目录
# （与 dependencies._INSTALL_ROOT / model_store._INSTALL_ROOT 同源，用于相对路径配置落盘）
from backend.infra.paths import INSTALL_ROOT as _INSTALL_ROOT


def resolve_config_path(p: str) -> Path:
    """相对配置路径锚定安装根目录（与 dependencies._resolve_path 语义一致）。

    绝对路径原样返回；相对路径解析为 _INSTALL_ROOT / p，使产品以任意方式启动
    （安装程序.exe / Tauri 外壳 / 启动脚本，CWD 各异）都能落到正确的运行时目录。
    """
    if os.path.isabs(p):
        return Path(p)
    return _INSTALL_ROOT / p


def ensure_runtime_dirs(config: AppConfig) -> list[Path]:
    """部署硬化：启动时创建全部运行时目录，保证干净环境可立即启动。

    覆盖 paths（data/tmp/images/reports）、模型权重目录、批量队列与同步队列目录。
    评估 Golden Set 属固定资产，不在此创建（缺失由 /evaluate 返回 409）。
    返回实际创建的目录列表（绝对路径）。
    """
    p = config.paths
    spec: list[str] = [
        p.data_dir,
        p.tmp_dir,
        p.images_dir,
        p.reports_dir,
        config.model.weights_dir,
        str(Path(p.data_dir) / "batch"),
        str(Path(p.data_dir) / "sync"),
    ]
    created: list[Path] = []
    for raw in spec:
        d = resolve_config_path(raw)
        d.mkdir(parents=True, exist_ok=True)
        created.append(d)
    return created
