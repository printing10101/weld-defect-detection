"""配置加载（§13.6 / §T8）。

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
    # CORS 允许源（§13.6 配置中心化，P2）：Tauri webview + 本地开发源。
    # 部署新增前端源（如公司内网门户）改配置即可，不改代码；禁 "*"，
    # 否则任意外部网站均可跨源读取本机 API（含审计链 / 报告）。
    cors_origins: list[str] = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "tauri://localhost",
        "https://tauri.localhost",
    ]


class ModelCfg(BaseModel):
    default_uri: str = "models/weights/best.onnx"
    backend: str = "onnx"  # onnx | torch | tensorrt
    weights_dir: str = "models/weights"  # 模型注册表扫描目录（§7.4，M4）
    registry_state_file: str = "data/model_registry.json"  # 活跃模型指针持久化


class SecurityCfg(BaseModel):
    encrypt: bool = True


class EvalCfg(BaseModel):
    """评估 / 漂移 / 实验追踪配置（§7.4 / §15.6 MLOps 闭环）。

    golden_dir        : 固定、版本化评估集（禁止用于训练）；缺失则 evaluate 端点返回 409。
    drift_baseline_path: 漂移监控参考基线（尺寸/置信度/类别分布），首跑自动建立。
    experiments_dir  : 实验追踪 JSONL 落盘目录（§7.4，MLflow 可演进）。
    auto_on_activate : 模型激活后自动跑 Golden Set 评估（使 metric_map 有值）。
    """

    golden_dir: str = "data/eval/golden"
    drift_baseline_path: str = "data/eval/drift_baseline.json"
    experiments_dir: str = "data/experiments"
    auto_on_activate: bool = True


class SyncCfg(BaseModel):
    """端边云同步适配器选择（§7.6）。

    kind=local：LocalAdapter（数据不出本机，默认）；
    kind=http ：HttpSyncAdapter（推送到可配置端点，需自行保证传输加密）。
    """

    kind: str = "local"  # local | http | cloud（v3 联邦占位，未实现，勿生产启用）
    http_endpoint: str | None = None  # kind=http 时的推送 URL
    http_token: str | None = None  # 可选 Bearer Token（建议走环境变量注入，勿入版本库）
    http_timeout: float = 10.0  # HTTP 同步推送超时（秒，§13.6 配置中心化）


class AnnotatorCfg(BaseModel):
    """人工标注服务（§12.2 主动学习闭环）是否随主应用同进程启动。

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
    images_dir: str = "data/images"  # 原图副本目录（报告缺陷图谱数据源）
    reports_dir: str = "data/reports"  # PDF 报告输出目录（§7.2）


class DensityCfg(BaseModel):
    low: float = 2.0  # AB 级黑度下限
    high: float = 4.5  # AB 级黑度上限


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
    """伪缺陷筛查阈值（§4.2，默认仅长直划痕阻断评片）。

    字段集须与 backend/domain/pseudo_defect.py 的 PseudoDefectCfg 对齐
    （§T8 三处同步：config.py / default.yaml / schema.yaml / domain 默认）。
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
    """射线底片质量门禁配置（§4.4，§T8 三处同步）。

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
    """掩膜精修量化配置（M4b，§T8 四地同步）。

    字段集须与 backend/domain/quantify.py 的 MaskRefineCfg dataclass 对齐
    （§T8：domain 默认 + 本模块 pydantic + default.yaml + schema.yaml）。
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
    # 原默认 True：一旦 default.yaml 缺键，会静默落 blob 基线而无人告警（§部署硬化 配置漂移）。
    kind: str = "trained_yolo"  # 检测器种类（注册表键）：trained_yolo=YOLO 训练模型，baseline_blob=连通域基线
    quantifier_kind: str = "mask"  # 量化器种类（注册表键，§T8）：mask=掩膜精修(M4b)，bbox=包围盒近似(M4a)
    baseline_enabled: bool = False  # M4a 基线检测器开关（kind 的兼容别名；显式 kind 优先）；训练模型就绪后保持 false
    allow_baseline_fallback: bool = True  # 训练模型加载失败时是否回退基线（False=启动即失败）
    infer_conf: float = 0.3  # 推理置信度阈值（§T8：禁硬编码，统一入口）
    infer_iou: float = 0.5  # NMS IoU 阈值
    # 逐类置信度阈值（ADR-010 扩展）：稀有且安全关键缺陷设更低阈值优先召回，
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
    """上传限额（§13.9）：无上限的 multipart 读取会被单请求打爆内存/磁盘。"""

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


class BatchCfg(BaseModel):
    """批量任务队列配置（§12.1，§T8 三处同步）。

    桌面单机场景：线程池多 worker 并行推理（run_inspection 全链路逐图跑），
    单图失败隔离不拖垮整批。断点续跑 = 状态快照持久化到 data/batch/，
    重启后可查询历史与未完成标记（自动重跑未完成项为增强项，v1 不实现）。
    """

    workers: int = 2  # 并行 worker 数（IO/推理并行度；过大会加重 CPU/内存）
    max_per_batch: int = 100  # 单批最大图数（防一次性打爆资源）
    per_image_estimate_sec: float = 8.0  # 单图预估耗时（进度条/预计时间展示用）



class AppConfig(BaseSettings):
    server: ServerCfg = ServerCfg()
    model: ModelCfg = ModelCfg()
    security: SecurityCfg = SecurityCfg()
    eval: EvalCfg = EvalCfg()
    sync: SyncCfg = SyncCfg()
    annotator: AnnotatorCfg = AnnotatorCfg()
    paths: PathsCfg = PathsCfg()
    density: DensityCfg = DensityCfg()
    iqi: IqiCfg = IqiCfg()
    pseudo_defect: PseudoDefectCfg = PseudoDefectCfg()
    preprocess: PreprocessCfg = PreprocessCfg()
    quality: QualityCfg = QualityCfg()
    detect: DetectCfg = DetectCfg()
    mask_refine: MaskRefineCfg = MaskRefineCfg()
    upload: UploadCfg = UploadCfg()
    standard: StandardCfg = StandardCfg()
    review: ReviewCfg = ReviewCfg()
    batch: BatchCfg = BatchCfg()

    model_config = {"env_prefix": "SCAN_"}


def load_config() -> AppConfig:
    """加载配置：configs/default.yaml 为基础，SCAN_* 环境变量优先覆盖（§13.6）。

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
    # §部署硬化：启动期捕获配置漂移（schema 与 default.yaml 不一致 → 静默落默认的高危场景）
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
    if isinstance(node, dict) and all(
        isinstance(k, str) and _IDENT_RE.match(k) for k in node
    ):
        out: set[tuple[str, ...]] = set()
        for k, v in node.items():
            out |= _leaf_paths(v, prefix + (str(k),))
        return out
    # 叶子：标量 / list / 非标识符键映射（dict-of-scalars）一并视作单个配置键
    return {prefix} if prefix else set()


def validate_config_against_schema(raw: dict) -> list[str]:
    """schema.yaml 是权威契约（§T8 三处同步）；与合并后的 raw 比对，返回漂移问题列表。

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
        issues.append(
            f"schema 要求但 default.yaml 缺失: {'.'.join(p)} (将静默落 pydantic 默认)"
        )
    for p in sorted(actual - expected):
        issues.append(f"default.yaml 存在但 schema.yaml 未登记: {'.'.join(p)} (新增键未补 schema?)")
    return issues


# 安装根目录锚点：backend/infra/config.py -> parents[2] = 安装根目录
# （与 dependencies._INSTALL_ROOT / model_store._INSTALL_ROOT 同源，用于相对路径配置落盘）
_INSTALL_ROOT = Path(__file__).resolve().parents[2]


def resolve_config_path(p: str) -> Path:
    """相对配置路径锚定安装根目录（与 dependencies._resolve_path 语义一致）。

    绝对路径原样返回；相对路径解析为 _INSTALL_ROOT / p，使产品以任意方式启动
    （安装程序.exe / Tauri 外壳 / 启动脚本，CWD 各异）都能落到正确的运行时目录。
    """
    if os.path.isabs(p):
        return Path(p)
    return _INSTALL_ROOT / p


def ensure_runtime_dirs(config: AppConfig) -> list[Path]:
    """部署硬化（§部署硬化 #6）：启动时创建全部运行时目录，保证干净环境可立即启动。

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
