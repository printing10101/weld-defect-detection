"""模型列表与热切换（§7.4，M4）。

- ``GET /api/v1/models``：列出权重目录中所有可用模型（id/名称/版本指纹/大小/是否活跃）。
- ``POST /api/v1/models/{id}/activate``：运行时热切换检测器权重（重载 ONNX 会话），
  成功持久化活跃指针；失败（模型不存在/加载错误）返回 404/500 且**不改变**当前检测器。

热切换在 registry 锁内执行（仅串行化并发切换）；检测器推理与切换之间无全局读写锁，
属于已知限制（生产建议切换期暂停推理或请求排空），见 §7.4 注释。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from backend.app.auth import CurrentUser, get_current_user
from backend.app.dependencies import Registry, _resolve_path, get_registry
from backend.domain.detect.yolo_detector import YoloDetector
from backend.evaluation.run_eval import run_golden_evaluation

router = APIRouter(tags=["models"], dependencies=[Depends(get_current_user)])


class ModelInfo(BaseModel):
    id: str
    name: str
    version: str
    size_bytes: int
    active: bool
    uri: str
    metric_map: dict | None = None  # 评估指标 + Golden Set 指纹（§7.4 模型卡；无评估为 None）


class ModelsResponse(BaseModel):
    active_id: str | None
    models: list[ModelInfo]


class ActivateResponse(BaseModel):
    ok: bool
    active: str


class EvaluateResponse(BaseModel):
    model_id: str
    metrics: dict
    golden_fingerprint: str
    drift: dict
    experiment_run_id: str | None = None


def _preprocess_factory(reg: Registry):
    """复现生产增强链路（与 run_inspection 一致），使 Golden Set mAP 可信。

    preprocess 关闭时返回 None（在原始 gray 上推理）。
    """
    if reg.config.preprocess.enabled:
        pp = reg.preprocessor
        gamma = reg.config.preprocess.gamma

        def _fn(gray):
            return pp.enhance(pp.denoise(gray), gamma)

        return _fn
    return None


@router.get("/models", response_model=ModelsResponse)
def list_models(reg: Annotated[Registry, Depends(get_registry)]) -> ModelsResponse:
    entries = reg.model_registry.scan()
    return ModelsResponse(
        active_id=reg.model_registry.active_id,
        models=[
            ModelInfo(
                id=e.id,
                name=e.name,
                version=e.version,
                size_bytes=e.size_bytes,
                active=e.active,
                uri=e.uri,
                metric_map=reg.eval_report(e.id),
            )
            for e in entries
        ],
    )


@router.post("/models/{model_id}/activate", response_model=ActivateResponse)
async def activate_model(
    model_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
    current_user: Annotated[CurrentUser, Depends(get_current_user)],
) -> ActivateResponse:
    try:
        # registry 锁内重载检测器会话（DefectDetector.load 已预热校验）；失败抛出→保持原检测器。
        # actor 记录为登录操作员（T3 合规闭环）。
        entry = await run_in_threadpool(reg.activate_model, model_id, current_user.username)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={"code": "MODEL_NOT_FOUND", "message": f"未找到模型: {model_id}"},
        )
    except (RuntimeError, OSError) as exc:  # 加载失败：保持原检测器不变
        raise HTTPException(
            status_code=500,
            detail={"code": "MODEL_ACTIVATE_FAILED", "message": str(exc)},
        )
    return ActivateResponse(ok=True, active=entry.id)


@router.post("/models/{model_id}/evaluate", response_model=EvaluateResponse)
async def evaluate_model(
    model_id: str,
    reg: Annotated[Registry, Depends(get_registry)],
) -> EvaluateResponse:
    """在固定 Golden Set 上评估指定模型并闭环写报告/漂移/模型卡/实验。

    - 模型不存在 → 404；
    - Golden Set 未准备（data/eval/golden 缺失）→ 409，提示先建立评估集；
    - 评估在后台线程执行（CPU 密集），不阻塞请求。
    """
    entry = reg.model_registry.get(model_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "MODEL_NOT_FOUND", "message": f"未找到模型: {model_id}"},
        )
    # 评估该模型权重：加载独立检测器（不干扰当前活跃检测器）
    det = YoloDetector()
    try:
        det.load(entry.uri, reg.config.model.backend)
    except (RuntimeError, OSError) as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "MODEL_LOAD_FAILED", "message": str(exc)},
        ) from exc
    try:
        summary = await run_in_threadpool(
            run_golden_evaluation,
            model_id,
            det,
            golden_dir=_resolve_path(reg.config.eval.golden_dir),
            eval_dir=reg.eval_dir,
            experiments_dir=_resolve_path(reg.config.eval.experiments_dir),
            drift_baseline_path=_resolve_path(reg.config.eval.drift_baseline_path),
            conf=reg.config.detect.infer_conf,
            iou=reg.config.detect.infer_iou,
            class_conf=reg.config.detect.class_conf,
            preprocess_fn=_preprocess_factory(reg),
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "NO_GOLDEN_SET", "message": str(exc)},
        ) from exc
    return EvaluateResponse(
        model_id=summary["model_id"],
        metrics=summary["metrics"],
        golden_fingerprint=summary["golden_fingerprint"],
        drift=summary["drift"],
        experiment_run_id=summary.get("experiment_run_id"),
    )
