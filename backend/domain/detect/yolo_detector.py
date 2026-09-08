"""YoloDetector： 训练模型检测器（实现 DefectDetector，）。

- load(model_uri, backend):
    backend="onnx"  → ONNX Runtime 推理（部署默认，）
    backend="torch"/"yolo" → Ultralytics YOLOv8/v11 推理（训练后验证/开发）
- infer(image, conf, iou, class_conf=None) → list[Detection]
    image: 单通道灰度 (H,W)/(H,W,1) 或 3 通道 (H,W,3) uint8
    输出像素坐标 BBox(x, y, w, h, 左上角)，class_id=DefectClass(value)，
    score，uncertainty 由 estimate_uncertainty 综合（置信度余量+尺寸+类别安全关键度，
    见；非 MC Dropout，属可解释代理），shape 按长宽比（round/linear）。
    class_conf: 可选逐类置信度阈值 {class_id: threshold}，指定后该类用专属
        阈值过滤、未指定的类回落全局 conf。稀有且安全关键缺陷（裂纹/未熔合）
        设更低阈值以优先召回，气孔设更高阈值抑制海量误检（见 DetectCfg.class_conf）。

人工兜底：检测器只输出 Detection；need_review 由 StandardGrader
（Nb47013Grader）按 uncertainty 阈值（detect.review_conf）与零容忍规则综合判定
。torch/ultralytics 与 onnxruntime 均为延迟导入，未安装不阻断模块加载。
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from backend.domain.detect.calibration import (
    apply_class_temperature,
    temperature_transform,
)
from backend.domain.detect.uncertainty import (
    ensemble_uncertainty_stats,
    estimate_ensemble_uncertainty,
    estimate_uncertainty,
)
from backend.domain.dto import BBox, DefectClass, DefectShape, Detection

_LOG = logging.getLogger("scandetection.detector")


class YoloDetector:
    """模型无关检测器实现，可热替换主干而不动编排层。"""

    def __init__(self) -> None:
        self._backend: str = "onnx"
        self._onnx_session = None
        self._onnx_input: str | None = None
        self._onnx_shape = (640, 640)
        self._yolo_model = None
        # S-04 推理后端可插拔：ONNX Runtime 执行提供者清单（由 get_detector/
        # Registry 从 config.model.providers 注入）；None = 默认 CPU 不变。
        self.providers: list[str] | None = None
        # Tiling 分块推理（大底片小缺陷召回）：参数由 get_detector/Registry 从
        # config.detect.tile_* 注入（鸭子类型，不改 DefectDetector 契约）。
        # tile_size=0 关闭；trigger_side 控制仅在最长边超限的大图上启用，
        # max_count 限制单图瓦片数上限（超出时平滑放大瓦片，控耗时）。
        self.tile_size: int = 0
        self.tile_overlap: float = 0.2
        self.tile_trigger_side: int = 2400
        self.tile_max_count: int = 400
        # 跨瓦片合并 NMS 的 IoU：比推理 NMS 宽松（相邻瓦片对同一缺陷的回归框
        # 不完全重合，取 infer_iou 会漏合并成双检）；按类独立合并防跨类互吞。
        self.tile_merge_iou: float = 0.3
        # 逐类温度校准（置信度校准，§15.4 ECE 门禁）：{class_id: T}，由
        # get_detector/Registry 从 config.detect.calibration_file 注入（鸭子
        # 类型，不改 DefectDetector 契约）。None/空 = 不校准（原始分数）。
        # 校准在 sigmoid 分数上做 argmax/阈值比较**之前**，infer_conf/class_conf
        # 的语义因此统一落在校准尺度上。None/空 = 不校准。
        self.class_temperature: dict[int, float] | None = None

    # ---- 加载 ----------------------------------------------------------------
    @property
    def cam_model(self) -> object | None:
        """真 Grad-CAM 所需的 torch 模型（仅 torch 后端已加载时非 None）。

        ONNX Runtime 部署路径无梯度回传，返回 None——调用方（domain/explain）
        回退 Sobel 显著性近似。鸭子类型属性：BlobDetector 等其他检测器无此
        属性，调用方以 getattr(…, "cam_model", None) 探测。
        """
        if self._backend in ("torch", "yolo") and self._yolo_model is not None:
            return self._yolo_model
        return None

    def load(self, model_uri: str, backend: str = "onnx") -> None:
        self._backend = backend
        if backend in ("torch", "yolo"):
            self._load_torch(model_uri)
        else:
            self._load_onnx(model_uri)

    def _load_torch(self, model_uri: str) -> None:
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("未安装 ultralytics，无法加载 torch/yolo 权重") from e
        self._yolo_model = YOLO(model_uri)

    def _load_onnx(self, model_uri: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("未安装 onnxruntime，无法加载 onnx 权重") from e
        # 执行提供者由配置注入（S-04）：默认 CPU 不变；CUDA/昇腾 CANN/寒武纪/DCU
        # 等后端为预留写法，需对应 onnxruntime 分发版与运行时库（未真机验证）。
        providers = list(self.providers) if self.providers else ["CPUExecutionProvider"]
        sess = ort.InferenceSession(model_uri, providers=providers)
        self._onnx_session = sess
        inp = sess.get_inputs()[0]
        self._onnx_input = inp.name
        shape = inp.shape
        h = shape[2] if isinstance(shape[2], int) else 640
        w = shape[3] if isinstance(shape[3], int) else 640
        self._onnx_shape = (int(h), int(w))
        # 预热：用一张全零输入跑一次前向，触发执行提供方（CPUExecutionProvider）
        # 初始化并验证图可运行。避免首张真实底片在请求路径上承担初始化开销与
        # 潜在初始化失败，把"模型坏了"的问题提前到启动期暴露。
        dummy = np.zeros((1, 3, h, w), dtype=np.float32)
        sess.run(None, {self._onnx_input: dummy})
        _LOG.info(
            "ONNX detector loaded and warmed up: %s (input=%s, shape=%s)",
            model_uri,
            self._onnx_input,
            self._onnx_shape,
        )

    # ---- 推理 ----------------------------------------------------------------
    def infer(
        self,
        image: np.ndarray,
        conf: float,
        iou: float,
        class_conf: dict[int, float] | None = None,
    ) -> list[Detection]:
        # 空/退化输入（0 尺寸）直接返回空结果，避免 letterbox 缩放中的除零
        # （r = min(nw/w, nh/h)）与下游张量形状错误，把"坏图"变成一次安全空检出。
        if image is None or image.size == 0:
            return []
        # Tiling：配置开启且最长边超触发阈值时按瓦片推理（小图整图路径不受影响）。
        tile = self._effective_tile(image)
        if tile is not None:
            return self._infer_tiled(image, conf, iou, class_conf, tile)
        return self._infer_single(image, conf, iou, class_conf)

    def _infer_single(
        self,
        image: np.ndarray,
        conf: float,
        iou: float,
        class_conf: dict[int, float] | None = None,
    ) -> list[Detection]:
        """整图单次推理（letterbox 到模型输入尺寸）——infer 的不分块路径。"""
        if self._backend in ("torch", "yolo"):
            return self._infer_torch(image, conf, iou, class_conf)
        return self._infer_onnx(image, conf, iou, class_conf)

    # ---- Tiling 分块推理 -----------------------------------------------------
    def _effective_tile(self, image: np.ndarray) -> int | None:
        """返回本图应使用的瓦片边长；不分块返回 None。

        触发条件：tile_size>0 且最长边 > tile_trigger_side（焊缝长条底片仅按
        最短边判断会漏触发——300×8000 的长条整图 letterbox 后高度只剩 ~24px）。
        瓦片数超过 tile_max_count 时按 1.5× 递增瓦片边长直至预算内（平滑降级：
        超大底片自动降低放大倍率而不是把耗时拖到分钟级），并留告警日志。
        """
        tile = int(self.tile_size or 0)
        if tile <= 0:
            return None
        h, w = image.shape[:2]
        if max(h, w) <= self.tile_trigger_side:
            return None
        base = tile
        while self._tile_count(h, w, tile, self.tile_overlap) > max(1, int(self.tile_max_count)):
            tile = int(tile * 1.5)
        if tile != base:
            # 降级完成后告警一次（不在循环里逐档刷屏）。
            _LOG.warning(
                "tiling: 瓦片数超上限（max_count=%d），瓦片边长 %d→%d（召回率相应下降）",
                self.tile_max_count,
                base,
                tile,
            )
        return tile

    @staticmethod
    def _tile_count(h: int, w: int, tile: int, overlap: float = 0.2) -> int:
        """给定时边的瓦片网格数量（含重叠步进与贴边收尾）。"""
        ny = len(YoloDetector._grid_origins(h, tile, overlap))
        nx = len(YoloDetector._grid_origins(w, tile, overlap))
        return ny * nx

    @staticmethod
    def _grid_origins(span: int, tile: int, overlap: float) -> list[int]:
        """单维瓦片起点序列：重叠步进覆盖 [0, span)，末块贴边（不足一整块也覆盖到）。"""
        t = max(1, int(tile))
        # round 而非截断：1-0.9 的浮点漂移会把 step=100 算成 99，瓦片数虚增
        step = max(1, round(t * (1 - min(max(overlap, 0.0), 0.9))))
        if span <= t:
            return [0]
        origins = list(range(0, span - t + 1, step))
        if origins[-1] != span - t:
            origins.append(span - t)
        return origins

    def _infer_tiled(
        self,
        image: np.ndarray,
        conf: float,
        iou: float,
        class_conf: dict[int, float] | None,
        tile: int,
    ) -> list[Detection]:
        """分块推理：瓦片内单独 letterbox（小缺陷保留原生分辨率）→ 坐标平移回
        全图 → 跨瓦片 NMS 合并重叠区重复检出。

        上下文外扩：每块瓦片向四周多裁 overlap/2 的上下文（贴边收窄），跨瓦片
        边界的缺陷在相邻瓦片的扩展区内完整可见（硬切会把边界缺陷截成两半，
        各自都是不可检的局部特征）；检出仅保留中心落在核心区的部分——扩展区
        只提供"看得全"，归属仍按核心区划分，避免与邻瓦片重复计数。

        代价：推理次数 ≈ 瓦片数（tile_size=1280 的 8K 底片约 64 次）；收益：
        整图 letterbox 到 640 时 8K→640 的 ~12 倍缩放对小尺寸缺陷（细裂纹、
        小气孔）是分辨率天花板，分块后单瓦片内缩放仅 ~2 倍。
        """
        h, w = image.shape[:2]
        ov = min(max(self.tile_overlap, 0.0), 0.9)
        pad = max(0, int(tile * ov / 2))
        all_dets: list[Detection] = []
        for y0 in self._grid_origins(h, tile, ov):
            for x0 in self._grid_origins(w, tile, ov):
                th = min(tile, h - y0)
                tw = min(tile, w - x0)
                y0p, x0p = max(0, y0 - pad), max(0, x0 - pad)
                y1p, x1p = min(h, y0 + th + pad), min(w, x0 + tw + pad)
                tile_img = image[y0p:y1p, x0p:x1p]
                for d in self._infer_single(tile_img, conf, iou, class_conf):
                    bx = d.bbox.x + x0p  # 平移回全图坐标系
                    by = d.bbox.y + y0p
                    cx, cy = bx + d.bbox.w / 2, by + d.bbox.h / 2
                    # 中心不在核心区 → 该检出归属邻瓦片（扩展区检出不重复计）
                    if not (x0 <= cx < x0 + tw and y0 <= cy < y0 + th):
                        continue
                    all_dets.append(
                        Detection(
                            id=f"{d.class_id.name}-{len(all_dets)}",
                            bbox=BBox(x=bx, y=by, w=d.bbox.w, h=d.bbox.h),
                            class_id=d.class_id,
                            score=d.score,
                            uncertainty=d.uncertainty,
                            shape=d.shape,
                            deep_hole=d.deep_hole,
                        )
                    )
        if not all_dets:
            return []
        # 跨瓦片 NMS：重叠区同一缺陷被相邻瓦片各检一次 → 合并为一条。
        # 按类独立 + 独立的宽松合并阈值：相邻瓦片回归框不完全重合，取推理
        # NMS 的严格阈值会漏合并（双检推高缺陷计数）；类无关合并则可能把
        # 重叠区的气孔框与夹渣框互吞（类别以高分者为准）。
        raw = [
            (
                d.bbox.x,
                d.bbox.y,
                d.bbox.x + d.bbox.w,
                d.bbox.y + d.bbox.h,
                d.score,
                d.class_id.value,
            )
            for d in all_dets
        ]
        keep = self._nms(raw, self.tile_merge_iou, class_aware=True)
        # 重编全局唯一 id（各瓦片内局部序号在合并后会重复）并按置信度降序
        merged = [
            Detection(
                id=f"{all_dets[i].class_id.name}-{j}",
                bbox=all_dets[i].bbox,
                class_id=all_dets[i].class_id,
                score=all_dets[i].score,
                uncertainty=all_dets[i].uncertainty,
                shape=all_dets[i].shape,
                deep_hole=all_dets[i].deep_hole,
            )
            for j, i in enumerate(sorted(keep, key=lambda k: all_dets[k].score, reverse=True))
        ]
        return merged

    @staticmethod
    def _thr_for(cls_id: int, conf: float, class_conf: dict[int, float] | None) -> float:
        """返回某类的有效置信度阈值：class_conf 指定则优先，否则回落全局 conf。

        单测可独立验证，无需 ONNX/torch 运行时。
        """
        if class_conf:
            return class_conf.get(int(cls_id), conf)
        return conf

    def _eff_thr(self, cls_id: int, conf: float, class_conf: dict[int, float] | None) -> float:
        """温度校准后的有效阈值：与分数同一逐类单调变换。

        只校准分数不校准阈值会静默改变检测工作点（阈值两侧的框集合漂移）；
        分数与阈值同变换 → 保留的检出集合与未校准完全一致，仅置信度数值
        落在校准尺度上（review_conf/u_score 语义随之一致）。
        """
        thr = self._thr_for(cls_id, conf, class_conf)
        if self.class_temperature:
            t = self.class_temperature.get(int(cls_id), 1.0)
            thr = float(temperature_transform(np.array([thr]), t)[0])
        return thr

    # ---- 共用：后处理 --------------------------------------------------------
    # 约定：boxes 中每个元素为 (x, y, w, h, cls, score)
    # x,y = 左上角像素坐标（未 letterbox 还原后的原图坐标）
    # w,h = 框宽/高（像素）  cls = 类别索引  score = 置信度
    def _to_detections(
        self, boxes, conf: float, class_conf: dict[int, float] | None = None
    ) -> list[Detection]:
        dets: list[Detection] = []
        for x, y, w, h, cls, score in boxes:
            ci = int(cls)
            cid = DefectClass(ci) if 0 <= ci < len(DefectClass) else DefectClass.POROSITY
            aspect = max(w, h) / max(min(w, h), 1e-6)
            shape = DefectShape.ROUND if aspect <= 3.0 else DefectShape.LINEAR
            eff = self._eff_thr(ci, conf, class_conf)
            area = max(float(w) * float(h), 0.0)
            u = estimate_uncertainty(score, eff, ci, area)
            dets.append(
                Detection(
                    id=f"{cid.name}-{len(dets)}",
                    bbox=BBox(x=float(x), y=float(y), w=float(w), h=float(h)),
                    class_id=cid,
                    score=float(score),
                    uncertainty=u,
                    shape=shape,
                )
            )
        return dets

    def infer_tta(
        self,
        image: np.ndarray,
        conf: float,
        iou: float,
        class_conf: dict[int, float] | None = None,
        scales: tuple[float, ...] = (0.8, 1.0, 1.25),
    ) -> list[Detection]:
        """多尺度推理 TTA（，借鉴 LF-YOLO 多尺度策略）。

        在各尺度（0.8 / 1.0 / 1.25）下分别 letterbox 推理，坐标还原到原图后
        跨尺度 NMS 去重。小幅提升小目标（气孔）与细长缺陷（裂纹）召回；
        代价是推理耗时 ×len(scales)，默认关闭（调用方显式开启）。

        不确定性升级：合并时统计每条保留检出在**各视角**中的检出比例与得分
        标准差（``ensemble_uncertainty_stats``），与单视角启发式不确定性 max
        融合——多视角分歧是 Deep Ensemble 认知不确定性的测试时增广近似，
        "仅单个尺度冒出"的候选会自动获得更高的 ``uncertainty``，从而更早触发
        人工复核。

        约定：`infer` 内部 letterbox 到固定尺寸并还原坐标到**输入图**坐标系，
        故按均匀缩放 s 预缩放输入后，输出坐标除以 s 即回到原图坐标系。
        """
        if image is None or image.size == 0:
            return []
        if len(scales) <= 1:
            return self.infer(image, conf, iou, class_conf)
        h, w = image.shape[:2]
        views: list[list[tuple[BBox, int, float]]] = []
        results: list[Detection] = []
        for s in scales:
            resized = cv2.resize(image, (max(1, int(w * s)), max(1, int(h * s))))
            dets = self.infer(resized, conf, iou, class_conf)
            # views 必须与保留检出同处原图坐标系（还原坐标后再入列），否则跨视角
            # IoU 匹配错位、集成不确定性全部虚高。
            views.append(
                [
                    (
                        BBox(x=d.bbox.x / s, y=d.bbox.y / s, w=d.bbox.w / s, h=d.bbox.h / s),
                        d.class_id.value,
                        d.score,
                    )
                    for d in dets
                ]
            )
            for d in dets:
                results.append(
                    Detection(
                        id=f"{d.id}@s{s}",
                        bbox=BBox(
                            x=float(d.bbox.x) / s,
                            y=float(d.bbox.y) / s,
                            w=float(d.bbox.w) / s,
                            h=float(d.bbox.h) / s,
                        ),
                        class_id=d.class_id,
                        score=d.score,
                        uncertainty=d.uncertainty,
                        shape=d.shape,
                        deep_hole=d.deep_hole,
                    )
                )
        if not results:
            return []
        # 跨尺度 NMS：相同区域的多尺度检出合并为一条（保留最高置信候选）
        raw = [
            (
                d.bbox.x,
                d.bbox.y,
                d.bbox.x + d.bbox.w,
                d.bbox.y + d.bbox.h,
                d.score,
                d.class_id.value,
            )
            for d in results
        ]
        keep = self._nms(raw, iou)
        # 跨视角集成不确定性：检出比例/得分标准差与单视角信号 max 融合
        kept_tuples = [(results[i].bbox, results[i].class_id.value) for i in keep]
        stats = ensemble_uncertainty_stats(kept_tuples, views, match_iou=iou)
        merged: list[Detection] = []
        for i, (vote_frac, std) in zip(keep, stats, strict=True):
            d = results[i]
            u_aug = estimate_ensemble_uncertainty(vote_frac, std)
            merged.append(
                Detection(
                    id=d.id,
                    bbox=d.bbox,
                    class_id=d.class_id,
                    score=d.score,
                    uncertainty=max(d.uncertainty, u_aug),
                    shape=d.shape,
                    deep_hole=d.deep_hole,
                )
            )
        return sorted(merged, key=lambda d: d.score, reverse=True)

    @staticmethod
    def _to_rgb(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        # 3 通道：假设 BGR（cv2 读入）→ RGB
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ---- torch / ultralytics ------------------------------------------------
    def _infer_torch(self, image, conf, iou, class_conf=None) -> list[Detection]:
        rgb = self._to_rgb(image)
        # ultralytics 仅支持单一 conf；逐类阈值时先以"最低类别阈值"取全部候选，再逐类后过滤。
        pred_conf = min(min(class_conf.values()), conf) if class_conf else conf
        assert self._yolo_model is not None
        # res 显式 Any：ultralytics 的 YOLO.__call__ 返回 Results|Tensor 联合类型，
        # 其类型存根在本地 ml venv 会触发误报（CI 无 ml 依赖不解析）；鸭子类型访问。
        pred_iter: Any = self._yolo_model(rgb, conf=pred_conf, iou=iou, verbose=False)
        res: Any = pred_iter[0]
        # ultralytics 已做 NMS；直接转 (x, y, w, h, cls, score)
        raw: list[tuple[float, float, float, float, int, float]] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy()
            for (x1, y1, x2, y2), s, c in zip(xyxy, scores, clss):
                sc = float(s)
                if self.class_temperature:
                    # 分数与阈值同一逐类温度变换（保持检出集合不变）
                    sc = float(
                        temperature_transform(
                            np.array([sc]), self.class_temperature.get(int(c), 1.0)
                        )[0]
                    )
                raw.append((float(x1), float(y1), float(x2 - x1), float(y2 - y1), int(c), sc))
        # 逐类后过滤：class_conf 指定阈值的类按其专属阈值，未指定回落全局 conf。
        # 阈值与分数同一逐类温度变换（与 ONNX 路径同序同语义）。
        if class_conf:
            raw = [b for b in raw if b[5] >= self._eff_thr(b[4], conf, class_conf)]
        return self._to_detections(raw, conf, class_conf)

    # ---- onnx ----------------------------------------------------------------
    def _infer_onnx(self, image, conf, iou, class_conf=None) -> list[Detection]:
        sess = self._onnx_session
        if sess is None:
            raise RuntimeError("ONNX 模型未加载（先调用 load）")
        nh, nw = self._onnx_shape
        rgb = self._to_rgb(image)
        h, w = rgb.shape[:2]
        r = min(nw / w, nh / h)
        new_w, new_h = int(w * r), int(h * r)
        resized = cv2.resize(rgb, (new_w, new_h))
        top = (nh - new_h) // 2
        bottom = nh - new_h - top
        left = (nw - new_w) // 2
        right = nw - new_w - left
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=114
        )
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        out = np.asarray(
            sess.run(None, {self._onnx_input: blob})[0]
        )  # [1, 4+nc, 8400] 或 [1, 8400, 4+nc]
        # 归一为 (anchors, 4+nc)：通道维远小于锚框维，据此判断是否转置。
        # 不按当前类别数推断——旧模型输出通道数与枚举类数可以不一致。
        if out.ndim == 3 and out.shape[1] < out.shape[2]:
            # 通道优先 (batch, 4+nc, anchors) → 转置为 (batch, anchors, 4+nc)
            out = out.transpose(0, 2, 1)
        preds = out[0]  # [anchors, 4+nc]
        boxes_xywh = preds[:, :4]
        scores_all = preds[:, 4:]
        # 分类通道语义自适应：
        # - 含负值 → 原始 logits，需 sigmoid 还原概率；
        # - 全部 ∈ [0,1] → 已是概率，二次 sigmoid 会把背景锚框抬到 ~0.5，
        #   造成全图误检。
        # 用 min<0 判定：背景锚框的 logit 必为负，概率恒 ≥0。
        if scores_all.min() < -1e-3:
            scores_all = 1.0 / (1.0 + np.exp(-np.clip(scores_all, -50.0, 50.0)))
        # 温度校准只改变**输出置信度**，不参与类别指派/阈值筛选/NMS 排序：
        # 这些决策全部沿用原始分数——逐类不同温度会改变跨类相对排序（类无关
        # NMS 里锐化类压掉软化类），静默改变检出集合（实测 mAP -7.8 点、
        # 召回 -6.7 点，对漏检敏感的系统不可接受）。分数与阈值同变换保证
        # 逐类筛选结果与未校准一致，NMS 用原始分保证跨类竞争与基线一致。
        scores_cal = (
            apply_class_temperature(scores_all, self.class_temperature, class_axis=-1)
            if self.class_temperature
            else scores_all
        )
        cls = scores_all.argmax(1)
        score = scores_all.max(1)
        score_cal = scores_cal[np.arange(len(cls)), cls]
        # 逐类置信度阈值（与分数同变换 → 筛选结果与未校准一致）。
        thr = np.array([self._eff_thr(c, conf, class_conf) for c in cls], dtype=np.float32)
        mask = score_cal >= thr
        if not np.any(mask):
            return []
        bx = boxes_xywh[mask]
        sc = score[mask]  # NMS 排序用原始分（跨类竞争与基线一致）
        sc_cal = score_cal[mask]  # 输出置信度用校准分
        cl = cls[mask]
        x1 = bx[:, 0] - bx[:, 2] / 2
        y1 = bx[:, 1] - bx[:, 3] / 2
        x2 = bx[:, 0] + bx[:, 2] / 2
        y2 = bx[:, 1] + bx[:, 3] / 2
        x1 = (x1 - left) / r
        y1 = (y1 - top) / r
        x2 = (x2 - left) / r
        y2 = (y2 - top) / r
        # NMS 需要 (x1,y1,x2,y2,score,cls) 格式（score=原始分，保持基线排序）
        raw = list(
            zip(x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist(), sc.tolist(), cl.tolist())
        )
        keep = self._nms(raw, iou)
        # 转成 (x,y,w,h,cls,score) 交给 _to_detections（score=校准分，仅输出语义）
        converted = []
        for i in keep:
            x1i, y1i, x2i, y2i, _sci, cli = raw[i]
            converted.append((x1i, y1i, x2i - x1i, y2i - y1i, cli, float(sc_cal[i])))
        return self._to_detections(converted, conf, class_conf)

    @staticmethod
    def _nms(boxes, iou_thr, class_aware: bool = False) -> list[int]:
        """NMS：boxes 元素 (x1, y1, x2, y2, score, cls)，返回保留的下标。

        class_aware=True 时按类别独立做 NMS（不同类的框互不抑制）——供跨瓦片
        合并使用；cv2.dnn.NMSBoxes 无类别参数，按类分组后各跑一遍。
        """
        if not boxes:
            return []
        if class_aware:
            keep: list[int] = []
            for c in sorted({int(b[5]) for b in boxes}):
                idxs_c = [i for i, b in enumerate(boxes) if int(b[5]) == c]
                sub = [boxes[i] for i in idxs_c]
                keep.extend(idxs_c[k] for k in YoloDetector._nms(sub, iou_thr))
            return keep
        try:
            # cv2.dnn.NMSBoxes 要求 (x, y, w, h) 格式，需将 (x1,y1,x2,y2) 转为宽高
            xywh = np.array(
                [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes], dtype=np.float32
            )
            scores = np.array([b[4] for b in boxes], dtype=np.float32)
            idxs = cv2.dnn.NMSBoxes(xywh, scores, 0.0, iou_thr)  # type: ignore[arg-type]
            if isinstance(idxs, tuple):
                idxs = idxs[0] if idxs else []
            if len(idxs) == 0:  # type: ignore[reportArgumentType]
                # conf=0 合法（/detect 显式允许），全零分候选经 NMS 可能返回空。
                return []
            return [int(i) for i in idxs]  # type: ignore[reportGeneralTypeIssues]
        except (cv2.error, AttributeError):
            kept: list[int] = []
            order = sorted(range(len(boxes)), key=lambda i: boxes[i][4], reverse=True)
            used: list[tuple[float, float, float, float]] = []
            for i in order:
                x1, y1, x2, y2 = boxes[i][:4]
                if any(_iou((x1, y1, x2, y2), u) > iou_thr for u in used):
                    continue
                kept.append(i)
                used.append((x1, y1, x2, y2))
            return kept


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / union
