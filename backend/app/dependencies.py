"""DI / registry 单例（§T4）。

共享状态（模型、队列）唯一入口；线程安全。
禁止在 router 内直接 new 模型或绕过 registry（§19.3）。
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from backend.app.auth import ROLE_ADMIN, hash_password
from backend.app.batch_queue import BatchManager
from backend.domain.detect.blob_detector import BlobConfig, BlobDetector
from backend.domain.detect.yolo_detector import YoloDetector
from backend.domain.grade.nb47013 import Nb47013Grader
from backend.domain.grade.registry import get_grader
from backend.domain.interfaces import DefectDetector, Reporter, StandardGrader
from backend.domain.preprocess.pipeline import OpencvPreprocessor
from backend.domain.standards.tables.loader import load_standard_tables
from backend.domain.sync import LocalAdapter
from backend.infra.config import AppConfig, ensure_runtime_dirs, load_config
from backend.infra.model_registry import ModelEntry, ModelRegistry
from backend.infra.model_store import LocalModelStore
from backend.infra.reporting.pdf_reporter import PdfReporter
from backend.infra.repository import InspectionRepository

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
        self._lock = threading.Lock()
        self.config: AppConfig = load_config()
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
        # 首启动引导管理员（§T3，P0）：无任何用户时依环境变量创建初始 admin，
        # 使产品开箱可用；已存在用户则跳过（不覆盖）。闭合工业合规"谁在操作"的硬前置。
        self._seed_bootstrap_admin()
        self.reporter: Reporter = PdfReporter(
            self.repository, _resolve_path(self.config.paths.reports_dir)
        )
        self.batch_manager = self._build_batch_manager()
        self.syncer = self._build_syncer()
        self.eval_dir = _resolve_path(str(Path(self.config.paths.data_dir) / "eval"))
        # §12.4 设备标定档案（跨设备一致性 ≤5%）
        from backend.infra.device_store import DeviceStore

        self.device_store = DeviceStore(_resolve_path(self.config.paths.db_path))

    def _seed_bootstrap_admin(self) -> None:
        """首启动引导管理员（§T3，P0）。

        当系统无任何用户时，依 SCAN_ADMIN_USERNAME / SCAN_ADMIN_PASSWORD 创建初始
        admin（密码缺失则生成随机 16 位并打印至日志，仅首次）。已存在用户则跳过
        （不覆盖、不重复创建）。保证产品开箱即可登录，闭合"操作者身份"合规前置。
        """
        if self.repository.count_users() > 0:
            return
        import os
        import secrets as _secrets

        username = (
            os.environ.get(self.config.auth.bootstrap_username_env)
            or self.config.auth.bootstrap_default_username
        )
        pw = os.environ.get(self.config.auth.bootstrap_password_env)
        if not pw:
            pw = _secrets.token_urlsafe(16)
            _LOG.warning(
                "首启动已生成引导管理员 用户名=%s 密码=%s（请立即登录修改；"
                "生产建议用 %s 环境变量预设）",
                username,
                pw,
                self.config.auth.bootstrap_password_env,
            )
        self.repository.create_user(
            username=username,
            display_name="管理员",
            role=ROLE_ADMIN,
            password_hash=hash_password(pw),
            created_by="system",
        )
        _LOG.info("引导管理员已创建：%s", username)

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
        - http ：LocalAdapter 本地留档 + POST 到 http_endpoint（尽力而为，失败仅告警）。

        默认 local（数据不出本机）；仅当显式配置 sync.kind=http 才发起网络调用。
        """
        from backend.domain.sync import HttpSyncAdapter

        queue_path = _resolve_path(str(Path(self.config.paths.data_dir) / "sync" / "pending.jsonl"))
        if self.config.sync.kind == "http":
            try:
                return HttpSyncAdapter(
                    endpoint=self.config.sync.http_endpoint,
                    token=self.config.sync.http_token,
                    queue_path=queue_path,
                )
            except ValueError as exc:
                _LOG.error("HttpSyncAdapter 配置无效，回退 local：%s", exc)
        return LocalAdapter(queue_path)

    def _build_detector(self) -> DefectDetector:
        """按配置装配检测器：M4a 基线 / M4b 训练模型（模型无关接口，ADR-002）。"""
        dc = self.config.detect
        if dc.baseline_enabled:
            self.detector_kind = "baseline_blob"
            self.detector_degraded = False
            return self._build_blob(dc)
        # M4b：训练模型检测器。权重缺失/加载失败按策略处理。
        try:
            det = YoloDetector()
            det.load(_resolve_model_uri(self.config.model.default_uri), self.config.model.backend)
            self.detector_kind = "trained_yolo"
            self.detector_degraded = False
            _LOG.info("detector loaded: trained_yolo (uri=%s)", self.config.model.default_uri)
            return det
        except Exception as exc:
            if not dc.allow_baseline_fallback:
                from backend.domain.errors import ModelUnavailableError

                raise ModelUnavailableError(f"训练模型加载失败: {exc}") from exc
            _LOG.error("M4b 权重加载失败，已回退 M4a 基线（评级不可用于正式判定）：%s", exc)
            self.detector_kind = "baseline_blob"
            self.detector_degraded = True
            return self._build_blob(dc)

    @staticmethod
    def _build_blob(dc) -> BlobDetector:
        return BlobDetector(
            BlobConfig(
                min_area_px=dc.min_area_px,
                max_area_px=dc.max_area_px,
                min_size_px=dc.min_size_px,
                noise_sigma_ratio=dc.noise_sigma_ratio,
                abs_threshold=dc.abs_threshold,
                dark_only=dc.dark_only,
            )
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
        """按配置装配默认标准判定器（NB/T47013，多标准适配见 grader_for）。"""
        sc = self.config.standard
        tables = load_standard_tables(sc.default_id, filename=sc.tables_filename)
        return Nb47013Grader(tables, review_uncertainty=self.config.detect.review_conf)

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
        # actor = 登录操作员（T3 鉴权闭环）；缺省回退 "system"。
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

                    def pp_fn(gray):
                        return pp.enhance(pp.denoise(gray), gamma)

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


_registry: Registry | None = None
_registry_lock = threading.Lock()


def get_registry() -> Registry:
    """获取全局 registry（懒初始化单例）；已初始化后走无锁快路径。"""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = Registry()
    return _registry
