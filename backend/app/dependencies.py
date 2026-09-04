"""DI / registry 单例。

共享状态（模型、队列）唯一入口；线程安全。
禁止在 router 内直接 new 模型或绕过 registry。
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import Header, Request

from backend.app.batch_queue import BatchManager
from backend.app.plugins import bootstrap_plugins
from backend.domain.detect import BlobConfig, get_detector
from backend.domain.errors import ModelUnavailableError
from backend.domain.grade.registry import get_grader
from backend.domain.interfaces import DefectDetector, Reporter, StandardGrader
from backend.domain.preprocess.pipeline import OpencvPreprocessor
from backend.domain.standards.tables.loader import load_standard_tables, set_default_table_source
from backend.domain.sync import LocalAdapter
from backend.infra.config import AppConfig, ensure_runtime_dirs, load_config, resolve_config_path
from backend.infra.model_registry import ModelEntry, ModelRegistry
from backend.infra.model_store import LocalModelStore
from backend.infra.paths import resolve_model_uri as _resolve_model_uri
from backend.infra.repository import InspectionRepository
from backend.infra.standards.tables_source import FileTableSource

if TYPE_CHECKING:
    from backend.infra.backup import BackupScheduler
    from backend.infra.disk_space import DiskWatchdog
    from backend.infra.watchdog import MemoryWatchdog

_LOG = logging.getLogger("scandetection.dependencies")


def _resolve_path(p: str) -> str:
    """将配置中的相对路径解析为绝对路径（恒锚定安装根，不受 CWD 影响）。

    历史"存在才锚定"语义已废除：相对路径一律解析为 ``<安装根>/<p>``——
    旧语义在 CWD≠安装根（打包启动）时会把新建路径落到错误位置。
    """
    return str(resolve_config_path(p))


class ResilientDetector:
    """S-17 运行期模型回退：推理异常计数与自动回退上一稳定版本。

    包装真实检测器：``infer`` 抛异常时累计连续失败次数，达到
    ``threshold`` 且 auto_rollback 允许时回调 ``on_threshold``（由 Registry
    实装：回退上一稳定权重 + 告警 + 审计 + degraded 标记）。异常始终原样
    上抛——本类只负责"计数与触发回退"，不吞错、不伪造空结果。

    ``load`` 透传：成功加载（含热切换）即清零计数并回调 ``on_load_success``
    （Registry 据此更新"上一稳定版本"指针与 degraded 复位）。
    """

    def __init__(
        self,
        inner: DefectDetector,
        *,
        threshold: int,
        auto_rollback: bool,
        on_threshold,
        on_load_success,
    ) -> None:
        self._inner = inner
        self._threshold = max(1, int(threshold))
        self._auto_rollback = bool(auto_rollback)
        self._on_threshold = on_threshold
        self._on_load_success = on_load_success
        self.consecutive_failures = 0
        self.rollback_count = 0
        self.last_rollback_at: str | None = None

    def load(self, model_uri: str, backend: str = "onnx") -> None:
        """透传加载；成功即视为新稳定版本（清零计数、通知 Registry）。"""
        self._inner.load(model_uri, backend)
        self.consecutive_failures = 0
        self._on_load_success(model_uri)

    def infer(
        self,
        image,
        conf: float,
        iou: float,
        class_conf: dict[int, float] | None = None,
    ):
        try:
            dets = self._inner.infer(image, conf, iou, class_conf)
        except Exception:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self._threshold and self._auto_rollback:
                self.consecutive_failures = 0
                self.rollback_count += 1
                from datetime import UTC, datetime

                self.last_rollback_at = (
                    datetime.now(UTC).replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%S")
                )
                try:
                    self._on_threshold()
                except Exception as exc:  # noqa: BLE001 - 回退失败不掩盖原始推理异常
                    _LOG.error("自动回退执行失败: %s", exc)
            raise
        # 推理成功：连续失败计数清零（"连续"语义）。
        self.consecutive_failures = 0
        return dets

    def infer_tta(self, image, conf, iou, class_conf=None, scales=(0.8, 1.0, 1.25)):
        # infer_tta 是 YoloDetector 的增强能力、非 DefectDetector 协议契约
        return self._inner.infer_tta(image, conf, iou, class_conf, scales)  # type: ignore[attr-defined]


class Registry:
    """应用共享状态容器（单例）。"""

    def __init__(self) -> None:
        # 插件发现（P2）：先于检测器/判定器装配，使插件注册的种类立即可用；
        # 幂等，未安装插件时静默无操作。
        bootstrap_plugins()
        self._lock = threading.Lock()
        self.config: AppConfig = load_config()
        # 依赖倒置：将 infra.FileTableSource 注册为默认标准表数据源，
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
        # S-17：上一稳定版本权重指针（启动期=配置默认权重，成功热切换后更新）。
        self._last_stable_uri: str | None = model_uri
        self.detector: DefectDetector = self._build_detector()
        # 启动期同步：活跃指针对齐当前实际加载的权重 uri。
        self.model_registry.mark_active_by_uri(_resolve_model_uri(self.config.model.default_uri))
        # E-14 投产门禁：candidate 状态（评估结果）暂存，进程内存态
        # （重启后 candidate 失效，须重新评估——评估本身可重放，不丢一致性）。
        self.pending_activations: dict[str, dict] = {}
        self.grader: StandardGrader = self._build_grader()
        self.preprocessor = self._build_preprocessor()
        # S-03：paths.db_url 非空时优先作为完整 SQLAlchemy URL（达梦/金仓，未真机验证）；
        # 默认空 = 传 db_path，保持 sqlite:/// 语义不变。
        self.repository = InspectionRepository(
            self.config.paths.db_url or self.config.paths.db_path
        )
        # 安全治理存储（C-06/C-19）：三员账号/会话/告警/独立安全审计链
        from backend.infra.security_store import SecurityStore

        self.security_store = SecurityStore(_resolve_path(self.config.paths.db_path))
        # 合规存储（C-12/C-14）：涉密载体台账 + 导出审批
        from backend.infra.compliance_store import CarrierStore, ExportStore

        self.carrier_store = CarrierStore(_resolve_path(self.config.paths.db_path))
        self.export_store = ExportStore(_resolve_path(self.config.paths.db_path))
        # reportlab 导入 ~0.4s，延迟到 Registry 装配时（Registry 本身在后台线程初始化），
        # 不占用进程导入→端口绑定的关键路径。
        from backend.infra.reporting.pdf_reporter import PdfReporter

        self.reporter: Reporter = PdfReporter(
            self.repository, _resolve_path(self.config.paths.reports_dir)
        )
        self.batch_manager = self._build_batch_manager()
        self.syncer = self._build_syncer()
        self.eval_dir = _resolve_path(str(Path(self.config.paths.data_dir) / "eval"))
        # 设备标定档案（跨设备一致性 ≤5%）
        from backend.infra.device_store import DeviceStore

        self.device_store = DeviceStore(_resolve_path(self.config.paths.db_path))
        # S-09 内存看门狗：默认 None（lifespan 按 config.watchdog.enabled 装配启动）。
        self.watchdog: MemoryWatchdog | None = None
        # S-20 磁盘水位看门狗：默认 None（lifespan 按 config.disk_space.enabled 装配启动）。
        self.disk_watchdog: DiskWatchdog | None = None
        # S-12 定期备份调度器：默认 None（lifespan 按 config.backup.interval_hours 装配）。
        self.backup_scheduler: BackupScheduler | None = None
        # 登录服务实例（app.auth 在首次登录流程挂载，见 auth.get_auth_service）
        self._auth_service: object | None = None

    def eval_report(self, model_id: str) -> dict | None:
        """读取某模型的评估报告。"""
        from backend.evaluation.harness import load_eval_report

        return load_eval_report(model_id, self.eval_dir)

    def _build_batch_manager(self) -> BatchManager:
        """按配置装配批量任务队列。

        pipeline_factory 返回新 InspectionPipeline（每个 worker 独立实例，
        避免跨线程共享 pipeline 内部状态）；状态快照落 data/batch/。
        """
        from backend.app.pipelines import InspectionPipeline

        bm = BatchManager(
            lambda: InspectionPipeline(self),
            workers=self.config.batch.workers,
            per_image_estimate_sec=self.config.batch.per_image_estimate_sec,
            batch_dir=_resolve_path(str(Path(self.config.paths.data_dir) / "batch")),
            max_retained_batches=self.config.batch.max_retained_batches,
            max_retained_snapshot_files=self.config.batch.max_retained_snapshot_files,
        )
        bm._load_existing()
        return bm

    def _build_syncer(self):
        """装配端边云同步适配器：按 SyncCfg.kind 选择 local / http。

        - local：数据不出本机，待同步队列落 data/sync/pending.jsonl（可观测）；
        - http：本地留档 + POST 到 http_endpoint（尽力而为，失败仅告警）。

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
            # NotImplementedError（未实现即抛错）。
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
        """按 config.detect.kind 经注册表装配检测器（模型无关，）。

        决策（选哪种）仍由本方法负责，构造（如何建）收敛到 get_detector，
        兑现"换检测器不改主干"。训练模型加载失败时按 allow_baseline_fallback 回退基线。
        S-17：装配结果统一以 ResilientDetector 包装（推理异常计数→自动回退）。
        """
        dc = self.config.detect
        # baseline_enabled 作为 kind 的兼容别名：显式 kind 优先。
        if dc.kind == "baseline_blob" or dc.baseline_enabled:
            self.detector_kind = "baseline_blob"
            self.detector_degraded = False
            return self._wrap_resilient(get_detector("baseline_blob", blob_cfg=self._blob_cfg(dc)))
        # 训练模型检测器（默认）。权重缺失/加载失败按策略处理。
        uri = _resolve_model_uri(self.config.model.default_uri)
        try:
            det = get_detector(
                "trained_yolo",
                model_uri=uri,
                backend=self.config.model.backend,
                providers=self.config.model.providers,
            )
            self.detector_kind = "trained_yolo"
            self.detector_degraded = False
            _LOG.info("detector loaded: trained_yolo (uri=%s)", uri)
            return self._wrap_resilient(det)
        except Exception as exc:
            if not dc.allow_baseline_fallback:
                raise ModelUnavailableError(f"训练模型加载失败: {exc}") from exc
            _LOG.error("M4b 权重加载失败，已回退 M4a 基线（评级不可用于正式判定）：%s", exc)
            self.detector_kind = "baseline_blob"
            self.detector_degraded = True
            return self._wrap_resilient(get_detector("baseline_blob", blob_cfg=self._blob_cfg(dc)))

    def _wrap_resilient(self, det: DefectDetector) -> DefectDetector:
        """S-17：以 ResilientDetector 包装检测器，接通告警/审计/回退回调。"""
        return ResilientDetector(
            det,
            threshold=self.config.model.infer_failure_threshold,
            auto_rollback=self.config.model.auto_rollback,
            on_threshold=self._on_infer_failure_threshold,
            on_load_success=self._on_detector_load_success,
        )

    def _on_infer_failure_threshold(self) -> None:
        """S-17：连续推理异常达到阈值 → 回退上一稳定版本 + 告警 + 审计 + degraded。"""
        last = self._last_stable_uri
        _LOG.error("连续推理异常达到阈值，尝试回退上一稳定版本: %s", last)
        if last:
            # fail-safe：回退失败抛错由 ResilientDetector 记录，不吞掉原始异常
            self.detector.load(last, self.config.model.backend)
        self.detector_degraded = True
        try:
            self.security_store.raise_alert(
                kind="model_rollback",
                level="high",
                message=f"推理连续异常，已自动回退上一稳定版本: {Path(last).name if last else '（无记录）'}",
                detail={"last_stable_uri": last},
            )
        except Exception as exc:  # noqa: BLE001 - 告警失败不影响回退本身
            _LOG.warning("model_rollback 告警落库失败: %s", exc)
        try:
            self.repository.append_audit(
                actor="system",
                action="model_rollback",
                object_type="model",
                object_id=Path(last).name if last else "unknown",
                before=None,
                after={"last_stable_uri": last, "degraded": True},
                note="S-17 运行期模型自动回退（连续推理异常）",
            )
        except Exception as exc:  # noqa: BLE001 - 审计失败不影响回退本身
            _LOG.warning("model_rollback 审计落库失败: %s", exc)

    def _on_detector_load_success(self, uri: str) -> None:
        """S-17：检测器成功加载（启动/热切换/回退）→ 更新稳定版本指针。"""
        self._last_stable_uri = uri
        # 主动加载成功（热切换/启动回退）视为恢复稳定；回退路径随后会再置
        # degraded=True（_on_infer_failure_threshold 在 load 之后设置），不冲突。
        if self.detector_degraded:
            self.detector_degraded = False

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
        """按配置装配预处理实例。"""
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
        """按 standard_id 路由判定器。

        默认标准返回已装配实例（含已授权数值表）；其余标准经 registry 装配
        （骨架适配器 grade 熔断 422；未知标准抛 GradingAmbiguousError）。
        """
        if standard_id == self.config.standard.default_id:
            return self.grader
        return get_grader(standard_id)

    def activate_model(self, model_id: str, actor: str | None = None) -> ModelEntry:
        """运行时热切换检测器权重。

        在 registry 锁内执行切换（串行化并发切换请求）；失败时抛出（调用方转 HTTP 错误），
        当前检测器保持不变（fail-safe）。成功持久化活跃指针。
        """
        with self._lock:
            entry = self.model_registry.activate(
                model_id,
                loader=lambda uri: self.detector.load(uri, self.config.model.backend),
            )
        # 不可变审计日志：模型热切换记入，工业合规追溯。
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
        # 激活后自动跑 Golden Set 评估：使 metric_map 有值，
        # 形成"激活→评估→模型卡"的可观测闭环。后台线程执行，不阻塞激活响应；
        # 失败（如 Golden Set 缺失）仅记录告警，不改变已成功的激活。
        if self.config.eval.auto_on_activate:
            self._auto_evaluate(entry.id)
        return entry

    # ---- E-14 更新即重评投产门禁（状态机：candidate → approve → active）-------

    def run_candidate_evaluation(self, model_id: str) -> dict:
        """对候选模型跑 Golden 评估（同步；复用 /models/{id}/evaluate 逻辑）。

        使用独立检测器实例（不干扰当前活跃检测器）；Golden Set 缺失抛
        FileNotFoundError；其余评估失败原样上抛，由路由转 4xx/5xx。
        """
        entry = self.model_registry.get(model_id)
        if entry is None:
            raise KeyError(model_id)
        from backend.evaluation.run_eval import run_golden_evaluation

        det = get_detector(
            "trained_yolo",
            model_uri=entry.uri,
            backend=self.config.model.backend,
            providers=self.config.model.providers,
        )
        return run_golden_evaluation(
            model_id,
            det,
            golden_dir=_resolve_path(self.config.eval.golden_dir),
            eval_dir=self.eval_dir,
            experiments_dir=_resolve_path(self.config.eval.experiments_dir),
            drift_baseline_path=_resolve_path(self.config.eval.drift_baseline_path),
            conf=self.config.detect.infer_conf,
            iou=self.config.detect.infer_iou,
            class_conf=self.config.detect.class_conf,
            preprocess_fn=self._preprocess_fn(),
        )

    def _preprocess_fn(self):
        """生产增强链路（与 activate 后自动评估一致）；preprocess 关闭返回 None。"""
        if not self.config.preprocess.enabled:
            return None
        pp = self.preprocessor
        gamma = self.config.preprocess.gamma
        return lambda gray: pp.enhance(pp.denoise(gray), gamma)

    def evaluate_activation_gate(self, model_id: str) -> dict:
        """E-14 门禁判定：跑 Golden 评估并对照 modelgate 阈值。

        返回 {model_id, passed, metrics, reason}；结果写入 pending_activations
        （candidate 状态）。评估不可完成（Golden 缺失/加载失败）向上抛由路由转
        HTTP 错误——诚实门禁：评估不了就不允许投产，不留"默认放行"。
        """
        summary = self.run_candidate_evaluation(model_id)
        metrics = summary.get("metrics", {})
        gate = self.config.modelgate
        map50 = float(metrics.get("mAP50") or 0.0)
        recall = float(metrics.get("recall") or 0.0)
        reasons: list[str] = []
        if map50 < gate.min_map:
            reasons.append(f"mAP50={map50:.4f} < min_map={gate.min_map}")
        if recall < gate.min_recall:
            reasons.append(f"recall={recall:.4f} < min_recall={gate.min_recall}")
        passed = not reasons
        record = {
            "model_id": model_id,
            "passed": passed,
            "reason": "；".join(reasons) if reasons else "",
            "metrics": metrics,
            "golden_fingerprint": summary.get("golden_fingerprint"),
            "experiment_run_id": summary.get("experiment_run_id"),
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.pending_activations[model_id] = record
        return record

    def approve_model(self, model_id: str, actor: str | None = None) -> ModelEntry:
        """E-14 投产审批：sysadmin 确认 candidate 评估通过后执行真正热切换。

        - 门禁未启用 → ValueError（路由转 409，避免绕过状态机的"万能审批"）；
        - 无通过评估记录 / 评估未达标 → ValueError（路由转 409/422）。
        成功后审计 action=model_approve（actor=审批人）。
        """
        if not self.config.modelgate.enabled:
            raise ValueError("模型投产门禁未启用（modelgate.enabled=false），无需审批")
        record = self.pending_activations.get(model_id)
        if record is None:
            raise ValueError(f"模型 {model_id} 无 candidate 评估记录（先 POST /activate 触发评估）")
        if not record.get("passed"):
            raise ValueError(f"模型 {model_id} 门禁评估未通过: {record.get('reason')}")
        entry = self.activate_model(model_id, actor=actor)
        self.repository.append_audit(
            actor=actor or "system",
            action="model_approve",
            object_type="model",
            object_id=entry.id,
            before=None,
            after={
                "model_id": model_id,
                "metrics": record.get("metrics"),
                "evaluated_at": record.get("evaluated_at"),
            },
            note="E-14 投产审批：Golden 评估达标后由 sysadmin 批准投产",
        )
        # 投产完成后 candidate 记录使命完成，移除（防止重复审批产生重复切换）。
        self.pending_activations.pop(model_id, None)
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
        """存活/版本/模型状态。

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
                # S-17 运行期回退可观测：degraded 标记已含于上，附回退计数/时间。
                "resilience": {
                    "rollback_count": getattr(self.detector, "rollback_count", 0),
                    "last_rollback_at": getattr(self.detector, "last_rollback_at", None),
                    "consecutive_failures": getattr(self.detector, "consecutive_failures", 0),
                },
                # S-09 内存看门狗状态（未启用时 enabled=false 显式呈现）。
                "watchdog": (
                    self.watchdog.snapshot() if self.watchdog is not None else {"enabled": False}
                ),
                # S-20 磁盘水位看门狗状态（未启用时 enabled=false 显式呈现）。
                "disk_space": (
                    self.disk_watchdog.snapshot()
                    if self.disk_watchdog is not None
                    else {"enabled": False}
                ),
            }


def get_operator_name(
    request: Request,
    x_operator_name: str | None = Header(default=None, alias="X-Operator-Name"),
) -> str:
    """解析操作者身份（审计 actor）。

    C-06/C-19 兼容语义：登录态下以账号名为准——生产环境中 get_principal
    （路由级依赖，先于本依赖执行）会把 Principal 写入 request.state；
    未登录请求 X-Operator-Name 头仅作审计 actor 记录，不构成身份
    （缺省 "local"）。测试经 dependency_overrides 注入 principal 时同样
    会写入 request.state（见 tests/conftest.py）。
    """
    principal = getattr(request.state, "principal", None)
    if principal is not None and getattr(principal, "username", ""):
        return principal.username
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
    无需等待模型加载完成；业务端点仍走 get_registry 阻塞等待。
    """
    if _registry is not None:
        return _registry
    if _registry_lock.acquire(blocking=False):
        try:
            return _registry
        finally:
            _registry_lock.release()
    return None
