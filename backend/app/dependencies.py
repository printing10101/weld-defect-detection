"""DI / registry 单例（§T4）。

共享状态（模型、队列）唯一入口；线程安全。
禁止在 router 内直接 new 模型或绕过 registry（§19.3）。
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from fastapi import Header

from backend.app.batch_queue import BatchManager
from backend.app.plugins import bootstrap_plugins
from backend.domain.detect import BlobConfig, get_detector
from backend.domain.errors import ModelUnavailableError
from backend.domain.grade.registry import get_grader
from backend.domain.interfaces import DefectDetector, Reporter, StandardGrader
from backend.domain.preprocess.pipeline import OpencvPreprocessor
from backend.domain.standards.tables.loader import load_standard_tables, set_default_table_source
from backend.domain.sync import LocalAdapter
from backend.infra.config import AppConfig, ensure_runtime_dirs, load_config
from backend.infra.model_registry import ModelEntry, ModelRegistry
from backend.infra.model_store import LocalModelStore
from backend.infra.repository import InspectionRepository
from backend.infra.standards.tables_source import FileTableSource

_LOG = logging.getLogger("scandetection.dependencies")

# 安装根目录锚点：backend/app/dependencies.py -> parents[2] = 安装根目录
_INSTALL_ROOT = Path(__file__).resolve().parents[2]
# backend 包根目录锚点：parents[1] = backend/。
# Tauri 打包时模型随 ``backend`` 资源一同分发，落在 <安装目录>/backend/models/weights/，
# 而非安装根目录下的 models/weights；解析时回退到此处，避免找不到权重而静默降级。
_BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _resolve_model_uri(uri: str) -> str:
    """将配置中的相对模型路径解析为绝对路径。

    依次尝试以下锚点（任一存在即采用），保证 dev 与打包两种布局都能命中权重：
      1. 安装根目录（dev/repo 布局：``<root>/models/weights/...``）
      2. backend 包根目录（Tauri 打包布局：``<root>/backend/models/weights/...``，
         模型随 backend 资源分发，而非安装根目录下的 models/weights）
    否则回退相对当前工作目录（启动脚本以安装根目录为 CWD 的场景），
    最后保持原样以便报出原始路径错误。
    """
    if os.path.isabs(uri):
        return uri
    for anchor in (_INSTALL_ROOT, _BACKEND_ROOT):
        candidate = anchor / uri
        if candidate.exists():
            return str(candidate)
    if os.path.exists(uri):
        return uri
    return uri


def _resolve_path(p: str) -> str:
    """将配置中的相对路径解析为绝对路径（锚定安装根目录，避免受 CWD 影响）。"""
    if os.path.isabs(p):
        return p
    root_candidate = _INSTALL_ROOT / p
    if root_candidate.exists():
        return str(root_candidate)
    return p


class Registry:
    """应用共享状态容器（单例）。"""

    def __init__(self) -> None:
        # §19.4 插件发现（P2）：先于检测器/判定器装配，使插件注册的种类立即可用；
        # 幂等，未安装插件时静默无操作。
        bootstrap_plugins()
        self._lock = threading.Lock()
        self.config: AppConfig = load_config()
        # §T8 依赖倒置：将 infra.FileTableSource 注册为默认标准表数据源，
        # 使 domain 在运行期完全不接触文件系统（测试/独立场景回退域内置引导默认）。
        set_default_table_source(FileTableSource())
        # 部署硬化（#6）：启动即创建运行时目录，保证干净环境（首次安装/容器）可立即运行。
        ensure_runtime_dirs(self.config)
        self.detector_kind: str = "unknown"
        self.detector_degraded: bool = False
        model_uri = _resolve_model_uri(self.config.model.default_uri)
        self.model = LocalModelStore(model_uri, self.config.model.backend)
        self.model.load()
        self.model_registry = ModelRegistry(
            self.config.model.weights_dir, self.config.model.registry_state_file
        )
        self.detector: DefectDetector = self._build_detector()
        # 启动期同步：活跃指针对齐当前实际加载的权重 uri（§7.4，M4）。
        self.model_registry.mark_active_by_uri(_resolve_model_uri(self.config.model.default_uri))
        self.grader: StandardGrader = self._build_grader()
        self.preprocessor = self._build_preprocessor()
        self.repository = InspectionRepository(_resolve_path(self.config.paths.db_path))
        # reportlab 导入 ~0.4s，延迟到 Registry 装配时（Registry 本身在后台线程初始化），
        # 不占用进程导入→端口绑定的关键路径。
        from backend.infra.reporting.pdf_reporter import PdfReporter

        self.reporter: Reporter = PdfReporter(
            self.repository, _resolve_path(self.config.paths.reports_dir)
        )
        self.batch_manager = self._build_batch_manager()
        self.syncer = self._build_syncer()
        self.eval_dir = _resolve_path(str(Path(self.config.paths.data_dir) / "eval"))
        # §12.4 设备标定档案（跨设备一致性 ≤5%）
        from backend.infra.device_store import DeviceStore

        self.device_store = DeviceStore(_resolve_path(self.config.paths.db_path))

    def eval_report(self, model_id: str) -> dict | None:
        """读取某模型的评估报告（§7.4 模型卡 metric_map 数据源；无则 None）。"""
        from backend.evaluation.harness import load_eval_report

        return load_eval_report(model_id, self.eval_dir)

    def _build_batch_manager(self) -> BatchManager:
        """按配置装配批量任务队列（§12.1，M6）。

        pipeline_factory 返回新 InspectionPipeline（每个 worker 独立实例，
        避免跨线程共享 pipeline 内部状态）；状态快照落 data/batch/。
        """
        from backend.app.pipelines import InspectionPipeline

        bm = BatchManager(
            lambda: InspectionPipeline(self),
            workers=self.config.batch.workers,
            per_image_estimate_sec=self.config.batch.per_image_estimate_sec,
            batch_dir=_resolve_path(str(Path(self.config.paths.data_dir) / "batch")),
        )
        bm._load_existing()
        return bm

    def _build_syncer(self):
        """装配端边云同步适配器（§7.6）：按 SyncCfg.kind 选择 local / http。

        - local：数据不出本机，待同步队列落 data/sync/pending.jsonl（可观测）；
        - http ：本地留档 + POST 到 http_endpoint（尽力而为，失败仅告警）。

        默认 local（数据不出本机）；仅当显式配置 sync.kind=http 才发起网络调用。
        IO 依赖倒置（Task #9）：JSONL 落盘（JsonlQueue）与 HTTP 传输（UrllibJsonPoster）
        均由 infra 提供并注入，domain 适配器不触碰文件系统/网络。
        """
        from backend.domain.sync import CloudAdapter, HttpSyncAdapter
        from backend.infra.sync_io import JsonlQueue, UrllibJsonPoster

        queue_path = _resolve_path(str(Path(self.config.paths.data_dir) / "sync" / "pending.jsonl"))
        queue = JsonlQueue(queue_path)
        if self.config.sync.kind == "cloud":
            # v3 联邦占位（P3）：契约完整但未实现，push/pull/federate 显式
            # NotImplementedError（fail-loud，绝不静默假装已同步/已联邦）。
            try:
                return CloudAdapter(
                    endpoint=self.config.sync.http_endpoint or "",
                    token=self.config.sync.http_token,
                )
            except ValueError as exc:
                _LOG.error("CloudAdapter 配置无效，回退 local：%s", exc)
        if self.config.sync.kind == "http":
            try:
                return HttpSyncAdapter(
                    endpoint=self.config.sync.http_endpoint,
                    token=self.config.sync.http_token,
                    queue=queue,
                    transport=UrllibJsonPoster(timeout=self.config.sync.http_timeout),
                )
            except ValueError as exc:
                _LOG.error("HttpSyncAdapter 配置无效，回退 local：%s", exc)
        return LocalAdapter(queue)

    def _build_detector(self) -> DefectDetector:
        """按 config.detect.kind 经注册表装配检测器（模型无关，ADR-002）。

        决策（选哪种）仍由本方法负责，构造（如何建）收敛到 get_detector，
        兑现"换检测器不改主干"。训练模型加载失败时按 allow_baseline_fallback 回退基线。
        """
        dc = self.config.detect
        # baseline_enabled 作为 kind 的兼容别名：显式 kind 优先。
        if dc.kind == "baseline_blob" or dc.baseline_enabled:
            self.detector_kind = "baseline_blob"
            self.detector_degraded = False
            return get_detector("baseline_blob", blob_cfg=self._blob_cfg(dc))
        # 训练模型检测器（默认）。权重缺失/加载失败按策略处理。
        uri = _resolve_model_uri(self.config.model.default_uri)
        try:
            det = get_detector("trained_yolo", model_uri=uri, backend=self.config.model.backend)
            self.detector_kind = "trained_yolo"
            self.detector_degraded = False
            _LOG.info("detector loaded: trained_yolo (uri=%s)", uri)
            return det
        except Exception as exc:
            if not dc.allow_baseline_fallback:
                raise ModelUnavailableError(f"训练模型加载失败: {exc}") from exc
            _LOG.error("M4b 权重加载失败，已回退 M4a 基线（评级不可用于正式判定）：%s", exc)
            self.detector_kind = "baseline_blob"
            self.detector_degraded = True
            return get_detector("baseline_blob", blob_cfg=self._blob_cfg(dc))

    @staticmethod
    def _blob_cfg(dc) -> BlobConfig:
        return BlobConfig(
            min_area_px=dc.min_area_px,
            max_area_px=dc.max_area_px,
            min_size_px=dc.min_size_px,
            noise_sigma_ratio=dc.noise_sigma_ratio,
            abs_threshold=dc.abs_threshold,
            dark_only=dc.dark_only,
        )

    def _build_preprocessor(self) -> OpencvPreprocessor:
        """按配置装配预处理实例（§4.3，与 detector/grader 同模式经 Registry 单例）。"""
        pc = self.config.preprocess
        return OpencvPreprocessor(
            bilateral_d=pc.bilateral_d,
            bilateral_sigma_color=pc.bilateral_sigma_color,
            bilateral_sigma_space=pc.bilateral_sigma_space,
            median_k=pc.median_k,
            clahe_clip=pc.clahe_clip,
            clahe_grid=pc.clahe_grid,
            canny_kernel=pc.canny_kernel,
            morph_k_open=pc.morph_k_open,
            morph_k_close=pc.morph_k_close,
        )

    def _build_grader(self) -> StandardGrader:
        """按配置装配默认标准判定器（NB/T47013，多标准适配见 grader_for）。

        统一经 get_grader 装配，使 config.detect.review_conf 在所有路径生效
        （此前 _build_grader 与 registry.get_grader 的 Nb47013Grader 构造签名分叉）。
        """
        sc = self.config.standard
        tables = load_standard_tables(sc.default_id, filename=sc.tables_filename)
        return get_grader(sc.default_id, tables, review_uncertainty=self.config.detect.review_conf)

    def grader_for(self, standard_id: str) -> StandardGrader:
        """按 standard_id 路由判定器（§6.1 多标准适配）。

        默认标准返回已装配实例（含已授权数值表）；其余标准经 registry 装配
        （骨架适配器 grade() 熔断 422；未知标准抛 GradingAmbiguousError）。
        """
        if standard_id == self.config.standard.default_id:
            return self.grader
        return get_grader(standard_id)

    def activate_model(self, model_id: str, actor: str | None = None) -> ModelEntry:
        """运行时热切换检测器权重（§7.4，M4）。

        在 registry 锁内执行切换（串行化并发切换请求）；失败时抛出（调用方转 HTTP 错误），
        当前检测器保持不变（fail-safe）。成功持久化活跃指针。
        """
        with self._lock:
            entry = self.model_registry.activate(
                model_id,
                loader=lambda uri: self.detector.load(uri, self.config.model.backend),
            )
        # 不可变审计日志（§12.5）：模型热切换记入，工业合规追溯。
        # actor = 请求头操作员（X-Operator-Name）；缺省回退 "system"。
        self.repository.append_audit(
            actor=actor or "system",
            action="model_activate",
            object_type="model",
            object_id=entry.id,
            before=None,
            after={"model_id": model_id},
            note=f"activated {model_id}",
        )
        # 激活后自动跑 Golden Set 评估（§7.4 MLOps 闭环）：使 metric_map 有值，
        # 形成"激活→评估→模型卡"的可观测闭环。后台线程执行，不阻塞激活响应；
        # 失败（如 Golden Set 缺失）仅记录告警，不改变已成功的激活。
        if self.config.eval.auto_on_activate:
            self._auto_evaluate(entry.id)
        return entry

    def _auto_evaluate(self, model_id: str) -> None:
        """后台线程：对刚激活的模型跑 Golden Set 评估（非阻塞、fail-soft）。"""
        import threading

        def _job() -> None:
            try:
                from backend.evaluation.run_eval import run_golden_evaluation

                pp_fn = None
                if self.config.preprocess.enabled:
                    pp = self.preprocessor
                    gamma = self.config.preprocess.gamma

                    pp_fn = lambda gray: pp.enhance(pp.denoise(gray), gamma)

                run_golden_evaluation(
                    model_id,
                    self.detector,
                    golden_dir=_resolve_path(self.config.eval.golden_dir),
                    eval_dir=self.eval_dir,
                    experiments_dir=_resolve_path(self.config.eval.experiments_dir),
                    drift_baseline_path=_resolve_path(self.config.eval.drift_baseline_path),
                    conf=self.config.detect.infer_conf,
                    iou=self.config.detect.infer_iou,
                    class_conf=self.config.detect.class_conf,
                    preprocess_fn=pp_fn,
                )
            except FileNotFoundError as exc:
                _LOG.warning("自动评估跳过（Golden Set 缺失）: %s", exc)
            except Exception as exc:  # noqa: BLE001 - 评估失败不应影响已成功的激活
                _LOG.warning("自动评估失败（不影响激活）: %s", exc)

        threading.Thread(target=_job, name="golden-eval", daemon=True).start()

    @property
    def health(self) -> dict:
        """存活/版本/模型状态（§14）。

        status 表达"存活"（liveness），服务能应答即为 ok；模型降级另用
        degraded/detector_degraded 显式暴露，避免把可用服务误判为不可用，
        同时不再隐藏"训练模型没加载上、正在用基线"这一关键事实。
        """
        with self._lock:
            return {
                "status": "ok",
                "degraded": self.detector_degraded,
                "app_version": "0.1.0",
                "detector": self.detector_kind,
                "detector_degraded": self.detector_degraded,
                "sync": {
                    "adapter": self.syncer.name,
                    "pending": self.syncer.pending_count,
                },
                **self.model.status,
            }


def get_operator_name(
    x_operator_name: str | None = Header(default=None, alias="X-Operator-Name"),
) -> str:
    """从请求头解析操作员姓名（单机科研自用，无用户系统）；缺省返回 "local"。"""
    name = (x_operator_name or "").strip()
    return name or "local"


_registry: Registry | None = None
_registry_lock = threading.Lock()


def get_registry() -> Registry:
    """获取全局 registry（懒初始化单例）；已初始化后走无锁快路径。

    若后台初始化线程（main.lifespan 启动）正在进行，调用方在此阻塞直至就绪
    （与原先同步初始化语义一致，仅等待点提前到了首个需要 registry 的请求）。
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = Registry()
    return _registry


def try_get_registry() -> Registry | None:
    """非阻塞获取 registry：未就绪（初始化中/未开始）返回 None。

    供 /health 等存活探针使用——启动期即能应答（HTTP 200 + status=starting），
    无需等待模型加载完成；业务端点仍走 get_registry() 阻塞等待。
    """
    if _registry is not None:
        return _registry
    if _registry_lock.acquire(blocking=False):
        try:
            return _registry
        finally:
            _registry_lock.release()
    return None
