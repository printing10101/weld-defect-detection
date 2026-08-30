"""用例编排。

校验 → 预处理 → 检测 → 量化 → 判定 → 落库 → 报告 的全链路编排。
只做编排不写算法；领域接口经 Registry 装配，存储走 infra repository。

熔断语义：标准数值未授权时 grader 抛 GradingAmbiguousError，
编排捕获后落库 joint_level=None + need_review=True（不输出级别），
报告生成"需人工复核"版本——不违反"禁止输出级别"。
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from backend.app.dependencies import Registry
from backend.domain.density import check_density, estimate_density
from backend.domain.dto import BBox, DefectClass, DefectShape, Detection, ImageMeta, Modality
from backend.domain.errors import GradingAmbiguousError, IQIFailError
from backend.domain.film_region import FilmRegionCfg as DomainFilmRegionCfg
from backend.domain.film_region import detect_film_region
from backend.domain.iqi import IqiConfig, enrich_grade, verify_iqi
from backend.domain.preprocess.metrics import QualityCfg as DomainQualityCfg
from backend.domain.preprocess.metrics import assess_quality
from backend.domain.pseudo_defect import PseudoDefectCfg, screen_pseudo_defects
from backend.domain.quantify import get_quantifier
from backend.domain.recommend import recommend
from backend.domain.review import ReviewDecision, ReviewRole, resolve_review
from backend.domain.spacing import resolve_spacing as _resolve_spacing  # 单一真源（§T8/§6）
from backend.domain.standards.tables.loader import disclaimer_for
from backend.evaluation.gate_rejects import GateRejectStore
from backend.infra.config import resolve_config_path
from backend.infra.image_loader import load_image

_LOG = logging.getLogger("scandetection.pipeline")


def _shape_of(detection, geometry, round_aspect_max: float) -> DefectShape:
    """缺陷形状归类：优先检测器给出的 shape，否则按长宽比阈值（与 grader 同源）。"""
    if detection.shape is not None:
        return detection.shape
    return DefectShape.ROUND if geometry.aspect_ratio <= round_aspect_max else DefectShape.LINEAR


# 深孔判定阈值（启发式，待真实底片标定）：缺陷内部光学黑度超过母材黑度的比例。
# 黑度（光学密度 D）越高=透射越多=缺陷越深。
# 设为 1.2 表示"内部黑度比母材高 20% 以上"才判深孔，避免把普通气孔一律错判 IV。
# 8bit 底片无 density_array 时跳过（deep_hole 保持 False，由人工/其它信号兜底）。
DEEP_HOLE_DENSITY_RATIO = 1.2


def _derive_deep_hole(
    detections: list[Detection],
    meta: ImageMeta,
    base_density: float,
    bit_depth: int | None,
) -> list[Detection]:
    """从 density_array 推导 deep_hole 标记。

    Detection 为冻结 dataclass，故返回带 deep_hole 标记的新实例列表。
    density_array 缺失或 base_density 无效时原样返回。
    """
    if meta.density_array is None or base_density <= 0:
        return detections
    da = meta.density_array
    if da.ndim < 2:
        return detections
    h_img, w_img = da.shape[:2]
    out: list[Detection] = []
    for d in detections:
        flag = False
        try:
            x0 = max(0, int(d.bbox.x))
            y0 = max(0, int(d.bbox.y))
            x1 = min(w_img, int(d.bbox.x + d.bbox.w))
            y1 = min(h_img, int(d.bbox.y + d.bbox.h))
            if x1 > x0 and y1 > y0:
                crop = da[y0:y1, x0:x1]
                interior = estimate_density(crop, bit_depth)
                if interior > base_density * DEEP_HOLE_DENSITY_RATIO:
                    flag = True
        except Exception:  # noqa: BLE001 - 任意异常（越界/空裁切）都不应阻断评片
            flag = False
        out.append(
            Detection(
                id=d.id,
                bbox=d.bbox,
                class_id=d.class_id,
                score=d.score,
                uncertainty=d.uncertainty,
                shape=d.shape,
                mask_ref=d.mask_ref,
                deep_hole=flag,
            )
        )
    return out


# ---------------------------------------------------------------------------
# 扫描参数门禁（DB50/T 1807-2025 §5：dpi / 位深，评片硬前置）
# ---------------------------------------------------------------------------


def _estimate_dpi(meta: ImageMeta) -> float | None:
    """从影像元数据推算扫描分辨率 dpi；无法确定返回 None。

    DICOM 提供 PixelSpacing（mm/px）→ dpi = 25.4 / spacing；通用图像的
    文件密度元数据（PNG pHYs / JPEG JFIF）加载器未解析，只能交由
    gate.require_dpi 配置决定处置（true=从严拦截，false=放行并告警留档）。
    """
    if meta.pixel_spacing_mm and meta.pixel_spacing_mm > 0:
        return 25.4 / meta.pixel_spacing_mm
    return None


def _check_dpi(dpi: float | None, gate) -> tuple[bool, str]:
    """dpi 门禁：已知且低于下限 → 拦截；未知按 require_dpi 处置。

    返回 (是否通过, 不通过时的原因描述)。通过时原因串为空。
    """
    if dpi is not None:
        if dpi < gate.min_dpi:
            return False, f"扫描分辨率 {dpi:.0f} dpi 低于下限 {gate.min_dpi} dpi"
        return True, ""
    if gate.require_dpi:
        return False, "无法确定扫描分辨率（无 PixelSpacing/文件元数据），门禁要求提供"
    return True, ""


def _check_bit_depth(bit_depth: int | None, gate) -> tuple[bool, str]:
    """位深门禁：低于下限默认硬拦截（8bit 底片灰度精度不足），allow_8bit 可降级。"""
    if bit_depth is not None and bit_depth < gate.min_bit_depth:
        if gate.allow_8bit:
            return True, ""
        return False, f"位深 {bit_depth}bit 低于 {gate.min_bit_depth}bit 硬门禁"
    return True, ""


class InspectionPipeline:
    """一次完整评片的用例编排。"""

    def __init__(self, reg: Registry) -> None:
        self._reg = reg
        # 拦截留档台账懒建：仅在不合格底片归档时连接 DB（见 _gate_reject_store）
        self._reject_store_cache: GateRejectStore | None = None

    def run_inspection(
        self,
        image_path: Path,
        *,
        pixel_spacing_mm: float | None,
        base_metal_thickness_mm: float | None,
        standard_id: str | None = None,
        iqi_roi: tuple[int, int, int, int] | None = None,
        workpiece_no: str | None = None,
        weld_no: str | None = None,
        signer: str | None = None,
        actor: str | None = None,
        template: str = "standard",
        force: bool = False,
        witness: str | None = None,
    ) -> dict:
        """执行全链路并落库+生成报告，返回结果 dict。

        force=False（默认）时，底片不可评（黑度越界或 IQI 不达标）直接抛
        IQIFailError（409），符合设计文档"不通过则阻断评片并提示重拍"的硬前置；
        force=True 时仍出片，但**不输出级别**（need_review=True），因为
        不合格底片不构成评定依据。

        翻拍影像例外（film_region 判定 is_photo 且 density.photo_policy=warn）：
        相机拍灯箱的 8bit 照片绝对黑度不可测、IQI 识别不可靠，黑度/IQI/质量
        门禁不阻断，降级为告警（warnings）+ 强制人工复核（need_review=True），
        缺陷检测/量化/报告链路照常执行；级别输出语义与 force 路径一致。
        """
        reg = self._reg
        image_id = uuid.uuid4().hex
        report_id = uuid.uuid4().hex
        _LOG.info("inspection start image_id=%s src=%s", image_id, Path(image_path).name)
        t0 = time.perf_counter()

        # 1. 影像加载（infra）
        gray, meta = load_image(image_path)

        # 1.5 底片区域分割（翻拍影像前置）：黑度按胶片掩膜计算，IQI/伪缺陷/
        # 质量门禁在胶片区上评估，检测时屏蔽胶片区外背景（坐标系不变）。
        # 翻拍影像（相机拍灯箱，8bit、无原始密度数组）绝对黑度不可测且 IQI
        # 识别不可靠 → photo_mode 下门禁按 density.photo_policy 处置。
        fr = reg.config.film_region
        film = None
        if fr.enabled:
            film = detect_film_region(
                gray,
                DomainFilmRegionCfg(
                    min_area_frac=fr.min_area_frac,
                    max_photo_area_frac=fr.max_photo_area_frac,
                    surround_bright_gray=fr.surround_bright_gray,
                    surround_min_frac=fr.surround_min_frac,
                ),
            )
        if film is not None and not (film.is_photo or film.area_frac >= 0.7):
            # 分割结果既非翻拍、也不占大幅面（如 Otsu 只锁到焊缝条带）→ 不可信。
            # 必须按整图处理：据此掩膜会把条带外的真实缺陷一并屏蔽掉。
            film = None
        photo_mode = bool(
            film is not None
            and film.is_photo
            and meta.bit_depth == 8
            and meta.density_array is None
        )
        eval_gray = (
            gray[film.y : film.y + film.h, film.x : film.x + film.w]
            if film is not None
            else gray
        )

        # 2. 影像质量校验：黑度 + IQI（复用 领域逻辑）
        # 黑度须基于原始存储灰阶 + 位深，避免显示用的 min-max 拉伸破坏绝对光学密度；
        # 翻拍影像限定胶片掩膜，排除灯箱亮背景对平均灰阶的稀释。
        density_src = meta.density_array if meta.density_array is not None else gray
        density = float(
            estimate_density(
                density_src,
                bit_depth=meta.bit_depth,
                mask=film.mask if film is not None else None,
            )
        )
        density_ok = bool(check_density(density, reg.config.density.low, reg.config.density.high))
        iqi_cfg = IqiConfig(
            type=reg.config.iqi.type,
            wire_diameters_mm=tuple(reg.config.iqi.wire_diameters_mm),
            required_wire_no=reg.config.iqi.required_wire_no,
            hole_diameters_mm=tuple(reg.config.iqi.hole_diameters_mm),
            required_hole_no=reg.config.iqi.required_hole_no,
            min_contrast_ratio=reg.config.iqi.min_contrast_ratio,
            auto_locate=reg.config.iqi.auto_locate,
            locate_threshold=reg.config.iqi.locate_threshold,
            sensitivity=tuple(reg.config.iqi.sensitivity),
        )
        # IQI 验证：用户给 ROI 时按原图坐标在全图上验证（ROI 已隔离像质计，
        # 且胶片裁剪可能把像质计切掉）；无 ROI 时在胶片区上自动定位。
        iqi_target = gray if iqi_roi is not None else eval_gray
        iqi = verify_iqi(iqi_target, iqi_cfg, roi=iqi_roi, iqi_type=reg.config.iqi.type)
        # 用透照厚度 + 参考表补全 A/AB/B 等级（厚度缺失则 grade=None）。
        iqi = enrich_grade(iqi, base_metal_thickness_mm, iqi_cfg.sensitivity)
        # 伪缺陷筛查，仅严重项默认阻断。
        # 将 infra 配置适配为 domain 类型（与 IqiConfig 同模式，隔离 pydantic）。
        pd_cfg = reg.config.pseudo_defect
        pd_domain = PseudoDefectCfg(
            hough_threshold=pd_cfg.hough_threshold,
            scratch_min_ratio=pd_cfg.scratch_min_ratio,
            scratch_grating_min_lines=pd_cfg.scratch_grating_min_lines,
            canny_lo=pd_cfg.canny_lo,
            canny_hi=pd_cfg.canny_hi,
            uniformity_low_freq=pd_cfg.uniformity_low_freq,
            uniformity_max_ratio=pd_cfg.uniformity_max_ratio,
            dust_tophat_k=pd_cfg.dust_tophat_k,
            dust_min_area=pd_cfg.dust_min_area,
            dust_max_count=pd_cfg.dust_max_count,
            block_on_scratch=pd_cfg.block_on_scratch,
            block_on_uniformity=pd_cfg.block_on_uniformity,
            block_on_dust=pd_cfg.block_on_dust,
        )
        pd = screen_pseudo_defects(eval_gray, pd_domain)
        # 质量度量门禁：在胶片区上评估（反映底片本身质量，排除翻拍边框/灯箱背景）。
        q_cfg = reg.config.quality
        quality = assess_quality(eval_gray, DomainQualityCfg(**q_cfg.model_dump()))
        quality_fail_block = bool(q_cfg.block_on_quality and not quality.passed)
        quality_warn = bool((not quality.passed) and not q_cfg.block_on_quality)
        # 扫描参数门禁（§5）：dpi 与位深。dpi 无法确定时按 gate.require_dpi
        # 处置（默认放行+告警留档）；8bit 底片默认硬拦截（allow_8bit 可降级放行）。
        dpi = _estimate_dpi(meta)
        dpi_ok, dpi_reason = _check_dpi(dpi, reg.config.gate)
        bit_depth_ok, bit_depth_reason = _check_bit_depth(meta.bit_depth, reg.config.gate)
        evaluable = bool(
            density_ok
            and iqi.passed
            and pd.passed
            and not quality_fail_block
            and dpi_ok
            and bit_depth_ok
        )
        # 翻拍影像降级：photo_policy=warn 时黑度/IQI/质量门禁不阻断，
        # 转为告警 + 强制人工复核（绝对黑度不可测，未验证 ≠ 不合格）。
        photo_advisory = bool(photo_mode and reg.config.density.photo_policy == "warn")
        if not evaluable:
            reasons = []
            if not density_ok:
                reasons.append(
                    f"黑度 {density:.2f} 超出 [{reg.config.density.low}, {reg.config.density.high}]"
                )
            if not iqi.passed:
                reasons.append(f"IQI 未达要求（要求 {iqi.required}，实测 {iqi.achieved}）")
            if not pd.passed:
                reasons.append("存在严重伪缺陷（" + "；".join(pd.notes) + "）")
            if quality_fail_block:
                reasons.append(f"底片质量不达标（RQI={quality.score:.1f} < {q_cfg.min_score:.0f}）")
            if not dpi_ok:
                reasons.append(dpi_reason)
            if not bit_depth_ok:
                reasons.append(bit_depth_reason)
            if not force and not photo_advisory:
                # 不合格底片留档（E-05）：拦截发生在原图落盘之前，此前被拒
                # 底片无任何记录；先归档（密文副本+台账+审计）再阻断。
                self._archive_gate_reject(
                    image_path=image_path,
                    image_id=image_id,
                    reasons=reasons,
                    dpi=dpi,
                    bit_depth=meta.bit_depth,
                    operator=actor,
                )
                raise IQIFailError("底片质量不合格，阻断评片并提示重拍：" + "；".join(reasons))
        # 门禁降级放行的告警（require_dpi=false / allow_8bit=true 路径）：
        # 不阻断评片，但必须在结果与日志中留痕，避免"未验证"被静默当作合格。
        gate_warnings: list[str] = []
        if dpi is None and not reg.config.gate.require_dpi:
            gate_warnings.append(
                "无法确定扫描分辨率（无 PixelSpacing/文件元数据），已按配置放行并留档告警"
            )
        if (
            meta.bit_depth is not None
            and meta.bit_depth < reg.config.gate.min_bit_depth
            and reg.config.gate.allow_8bit
        ):
            gate_warnings.append(
                f"{meta.bit_depth}bit 底片按配置降级放行（灰度精度不足，建议 16bit 重新扫描）"
            )
        if gate_warnings:
            _LOG.warning(
                "gate advisory image_id=%s dpi=%s bit_depth=%s warnings=%s",
                image_id,
                dpi,
                meta.bit_depth,
                gate_warnings,
            )
        photo_warnings: list[str] = []
        if photo_mode:
            photo_warnings.append(
                f"翻拍影像：绝对黑度不可测（8bit 照片黑度上限 2.41），"
                f"胶片区估算 D={density:.2f} 仅供参考"
            )
            if not evaluable:
                photo_warnings.append(
                    "翻拍影像质量门禁未通过（" + "；".join(reasons) + "），已降级为人工复核"
                )
            _LOG.warning(
                "photo-mode advisory image density=%.2f iqi_pass=%s quality_pass=%s",
                density,
                bool(iqi.passed),
                quality.passed,
            )

        # 3. 原图副本落盘（报告缺陷图谱数据源；勿删）
        suffix = image_path.suffix or ".png"
        saved = self._persist_image(image_path, image_id, suffix)

        # 4. 预处理 + 检测 + 量化（ 基线 /  训练模型，同一接口）
        # 预处理（保边去噪+增强）后送检测；黑度/IQI/伪缺陷/质量门禁已在原始
        # 影像上完成，不受增强影响。检测源保持整图尺寸（缺陷框坐标系与
        # 原图/落库一致），仅把胶片区外背景填充为胶片中位灰阶，防止灯箱
        # 亮背景/翻拍边框被误检为缺陷。
        detect_src = gray
        if film is not None and not bool(film.mask.all()):
            detect_src = gray.copy()
            detect_src[~film.mask] = int(np.median(eval_gray))
        dc = reg.config.detect
        pp_cfg = reg.config.preprocess
        enhanced = detect_src
        preprocess_params: dict = {"enabled": False, "gamma": None}
        if pp_cfg.enabled:
            pp = reg.preprocessor
            denoised = pp.denoise(detect_src)
            gamma_v = pp_cfg.gamma
            enhanced = pp.enhance(denoised, gamma_v)
            preprocess_params = {
                "enabled": True,
                "gamma": gamma_v,
                "bilateral_d": pp_cfg.bilateral_d,
                "median_k": pp_cfg.median_k,
                "clahe_clip": pp_cfg.clahe_clip,
                "clahe_grid": pp_cfg.clahe_grid,
            }
        detections = reg.detector.infer(
            enhanced, conf=dc.infer_conf, iou=dc.infer_iou, class_conf=dc.class_conf
        )
        # 深孔推导：缺陷内部黑度显著高于母材 → 标 deep_hole（直判 IV 的前置信号）。
        # 必须在判定/落库前完成，使 grader 与 defect_rows 都能消费到该标记。
        detections = _derive_deep_hole(detections, meta, density, meta.bit_depth)
        # 经量化器注册表装配（去除 app 层 new 实现）；/report 历史用包围盒近似，
        # 显式请求 bbox 保持行为不变，量化参数（掩膜 cfg）无图时忽略。
        quantifier = get_quantifier("bbox")
        # 像素标定缺失时不再伪造 1.0 mm/px 后照常定级——几何量纲无据即触发熔断。
        spacing, spacing_known = _resolve_spacing(pixel_spacing_mm, meta.pixel_spacing_mm)
        quantified = [
            (d, quantifier.quantify(d, spacing, image=enhanced, cfg=None)) for d in detections
        ]

        # 5. 标准判定（，未授权/信息不足熔断 → 不输出级别）
        context = ImageMeta(
            modality=meta.modality,
            pixel_spacing_mm=spacing if spacing_known else None,
            base_metal_thickness_mm=base_metal_thickness_mm,
        )
        try:
            if not evaluable:  # force 出片：底片不合格不得作为评定依据
                raise GradingAmbiguousError("底片不可评（黑度/IQI 不合格），不输出级别")
            grade = reg.grader.grade(detections, context)
            joint_level: str | None = grade.joint_level.value
            per_grade = [g.value for g in grade.per_defect_grade]
            basis = list(grade.basis)
            need_review = bool(grade.need_review)
            std_version = grade.standard_version
        except GradingAmbiguousError as exc:
            joint_level = None
            per_grade = []
            # 保留熔断原因，报告与审计需可追溯（原实现丢弃后无从解释为何无级别）
            basis = [str(exc)] if str(exc) else ["判定信息不足，需人工复核"]
            need_review = True
            std_version = ""
        # 翻拍影像告警并入报告依据与人工复核标记：门禁未验证的结果不得静默当作合格片。
        if photo_warnings:
            basis = [*basis, *photo_warnings]
            need_review = True
        # 质量门禁未达阈值且非阻断模式（block_on_quality=False）时仅告警，并入人工复核标记。
        need_review = bool(need_review or quality_warn)

        std_id = standard_id or reg.config.standard.default_id
        # 工业过渡路径：免责声明只依赖标准表（standard-level），与判定结果无关，
        # 故无论评级成功或熔断均统一生成（authorized_copy=false 时为强声明）。
        disclaimer = disclaimer_for(reg.grader.tables)  # type: ignore[attr-defined]

        # 合规处置建议：消费评级输出，独立适配器（domain/recommend），
        # 不参与判定；熔断时降级为「需人工复核」，永不阻塞出片。
        rec = recommend(
            joint_level,
            detections,
            need_review=need_review,
            standard_id=std_id,
            disclaimer=disclaimer,
        )
        disposition = rec.disposition
        disposition_label = rec.disposition_label
        disposition_actions = list(rec.actions)

        # 6. 落库（images + defects + reports，一个事务）
        image_row = {
            "id": image_id,
            "path": str(saved),
            "source_type": "dicom" if meta.modality is Modality.DICOM else "image",
            "modality": meta.modality.value,
            "workpiece_no": workpiece_no,
            "weld_no": weld_no,
            "pixel_spacing_mm": spacing,
            "base_metal_thickness_mm": base_metal_thickness_mm,
            "iqi_pass": iqi.passed,
            "iqi_detail": {
                "type": iqi.iqi_type,
                "achieved": iqi.achieved,
                "required": iqi.required,
                "grade": iqi.grade,
            },
            "pseudo_defect_pass": pd.passed,
            "pseudo_defect_notes": list(pd.notes),
            "quality_pass": quality.passed,
            "quality_metrics": dict(quality.metrics),
            "preprocess_params": preprocess_params,
            "density": density,
            "density_ok": density_ok,
            "evaluable": evaluable,
            "joint_level": joint_level,
            "need_review": need_review,
            "standard_id": std_id,
            "standard_version": std_version,
        }
        # per_defect_grade 与 detections 按序对齐；长度不符说明 grader 契约被破坏，
        # 与其把级别错配到别的缺陷上（安全事故），不如整体退化为"无级别+需复核"。
        if per_grade and len(per_grade) != len(quantified):
            per_grade = []
            need_review = True
            basis = [*basis, "逐缺陷级别与检测数量不一致，已退化为人工复核"]

        defect_rows = [
            {
                "id": f"{image_id}:{d.id}",  # 全局唯一主键（同一影像内由 d.id 区分）
                "image_id": image_id,
                "class_id": d.class_id.value,
                "bbox_px": [d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h],
                "shape": _shape_of(d, g, dc.round_aspect_max).value,
                "length_mm": g.length_mm,
                "width_mm": g.width_mm,
                "area_mm2": g.area_mm2,
                "perimeter_mm": g.perimeter_mm,
                "position_x": g.position_x_mm,
                "position_y": g.position_y_mm,
                "confidence": d.score,
                "uncertainty": d.uncertainty,
                "joint_level": per_grade[i] if i < len(per_grade) else None,
                "need_review": need_review,
                "standard_id": std_id,
                "standard_version": std_version,
            }
            for i, (d, g) in enumerate(quantified)
        ]
        # 报告行先占位（pdf_path 待生成后回填）
        report_row = {
            "id": report_id,
            "image_id": image_id,
            "joint_level": joint_level,
            "pdf_path": "",
            "standard_ref": f"{std_id} {std_version}".strip(),
            "signer": signer,
            "basis": basis,
        }
        reg.repository.create_inspection(image_row, defect_rows, report_row)

        # 不可变审计日志：评片创建即记一笔，工业合规追溯。
        # actor = 请求头操作员（X-Operator-Name）；未携带时回退 "system"。
        reg.repository.append_audit(
            actor=actor or "system",
            action="inspect",
            object_type="image",
            object_id=image_id,
            before=None,
            after={
                "joint_level": joint_level,
                "need_review": need_review,
                "evaluable": evaluable,
                "defect_count": len(quantified),
            },
            note="force" if force else None,
        )

        # 7. 报告 PDF（Reporter 契约；读库拿数据 → 渲染 → 回填路径）
        # 复用已加载的灰度底片 gray，避免 pdf_reporter 对整张大底片二次解码
        # S-22 见证：witness 可选透传到报告签字栏（生成时生效，不落库）。
        pdf_path = reg.reporter.build(image_id, template, gray=gray, witness=witness)
        reg.repository.update_report(report_id, pdf_path=pdf_path)

        dt = time.perf_counter() - t0
        _LOG.info(
            "inspection done image_id=%s level=%s defects=%d density_ok=%s iqi_pass=%s "
            "evaluable=%s need_review=%s photo_mode=%s (%.1f ms)",
            image_id,
            joint_level,
            len(quantified),
            density_ok,
            bool(iqi.passed),
            evaluable,
            need_review,
            photo_mode,
            dt * 1000,
        )
        return {
            "image_id": image_id,
            "report_id": report_id,
            "joint_level": joint_level,
            "need_review": need_review,
            "evaluable": evaluable,
            "density": round(density, 3),
            "density_ok": density_ok,
            "iqi_pass": bool(iqi.passed),
            "defect_count": len(quantified),
            "photo_mode": photo_mode,
            "warnings": [*gate_warnings, *photo_warnings],
            "disclaimer": disclaimer,
            "disposition": disposition,
            "disposition_label": disposition_label,
            "disposition_actions": disposition_actions,
            "pdf_path": pdf_path,
        }

    def _gate_reject_store(self) -> GateRejectStore:
        """拦截留档台账（懒建）：只在首次归档时连接 DB，避免无关评片多开引擎。"""
        if self._reject_store_cache is None:
            self._reject_store_cache = GateRejectStore(
                str(resolve_config_path(self._reg.config.paths.db_path))
            )
        return self._reject_store_cache

    def _archive_gate_reject(
        self,
        *,
        image_path: Path,
        image_id: str,
        reasons: list[str],
        dpi: float | None,
        bit_depth: int | None,
        operator: str | None,
    ) -> str:
        """不合格底片留档（E-05）：密文归档 + gate_rejects 台账 + 审计哈希链。

        复用影像加密落盘路径模式（security.encrypt 开启时 AES-GCM 密文，
        密钥缺失降级明文并告警），保证留档不旁路加密；归档失败不吞掉门禁
        拦截本身，降级为仅记台账原因并在日志报错。
        """
        reg = self._reg
        reject_id = uuid.uuid4().hex
        detail: dict = {"reasons": list(reasons), "source": str(image_path)}
        try:
            dest = self._persist_reject(image_path, reject_id, Path(image_path).suffix or ".png")
            detail["archived"] = str(dest)
        except Exception as exc:  # noqa: BLE001 - 归档故障不得阻断门禁拦截
            _LOG.error("不合格底片归档失败 reject_id=%s: %s", reject_id, exc)
            detail["archive_error"] = str(exc)
        self._gate_reject_store().add(
            reject_id=reject_id,
            image_id=image_id,
            reject_reason="；".join(reasons)[:256],
            detail=detail,
            dpi=dpi,
            bit_depth=bit_depth,
            operator=operator or "system",
        )
        reg.repository.append_audit(
            actor=operator or "system",
            action="gate_reject",
            object_type="image",
            object_id=reject_id,
            before=None,
            after={"reasons": list(reasons), "dpi": dpi, "bit_depth": bit_depth},
        )
        return reject_id

    def _persist_reject(self, src: Path, reject_id: str, suffix: str) -> Path:
        """不合格底片副本归档到 gate.rejects_dir（密文，模式同 _persist_image）。"""
        rejects_dir = Path(resolve_config_path(self._reg.config.gate.rejects_dir))
        rejects_dir.mkdir(parents=True, exist_ok=True)
        dest = rejects_dir / f"{reject_id}{suffix}"
        shutil.copyfile(src, dest)
        if self._reg.config.security.encrypt:
            from backend.infra.crypto import AesCrypto, CryptoKeyError

            try:
                cipher = AesCrypto()
            except CryptoKeyError as exc:
                _LOG.warning("静态加密未生效（%s）：不合格底片归档以明文落盘", exc)
                return dest
            plaintext = dest.read_bytes()
            dest.write_bytes(cipher.encrypt(plaintext))
        return dest

    def regenerate_report(self, image_id: str, template: str = "standard") -> dict:
        """对已入库检查重新生成报告（不重跑检测/判定）。"""
        repo = self._reg.repository
        image = repo.get_image(image_id)
        if image is None:
            raise KeyError(f"image not found: {image_id}")
        pdf_path = self._reg.reporter.build(image_id, template)
        report_id = (image.get("report") or {}).get("report_id")
        if report_id:
            repo.update_report(report_id, pdf_path=pdf_path)
        else:
            # 影像尚无报告行：原实现直接 uuid4 后 update_report，报告行不存在
            # → KeyError 或凭空返回一个查不到的 report_id（下载必 404）。这里先建行。
            report_id = uuid.uuid4().hex
            repo.create_report_row(report_id, image_id)
            repo.update_report(report_id, pdf_path=pdf_path)
        stored_defects = image.get("defects") or []
        # 合规处置建议：由库里存的级别+缺陷类别重算（不重跑判定）。
        # 重建最小 Detection 仅需 class_id（零容忍判定），bbox 用占位零框。
        rec_defects = [
            Detection(
                id=str(d.get("id", "")),
                bbox=BBox(0.0, 0.0, 1.0, 1.0),
                class_id=DefectClass(int(d["class_id"])),
                score=float(d.get("confidence", 0.0)),
                uncertainty=float(d.get("uncertainty", 1.0)),
            )
            for d in stored_defects
            if "class_id" in d
        ]
        rec = recommend(
            image.get("joint_level"),
            rec_defects,
            need_review=bool(image.get("need_review", False)),
            standard_id=str(image.get("standard_id") or "NB/T47013.2-2015"),
            disclaimer=disclaimer_for(self._reg.grader.tables),  # type: ignore[attr-defined]
        )
        return {
            "image_id": image_id,
            "report_id": report_id,
            "joint_level": image.get("joint_level"),
            "need_review": bool(image.get("need_review", False)),
            "evaluable": bool(image.get("evaluable", True)),
            "defect_count": len(stored_defects),
            "disclaimer": disclaimer_for(self._reg.grader.tables),  # type: ignore[attr-defined]
            "disposition": rec.disposition,
            "disposition_label": rec.disposition_label,
            "disposition_actions": list(rec.actions),
            "pdf_path": pdf_path,
        }

    def _persist_image(self, src: Path, image_id: str, suffix: str) -> Path:
        images_dir = Path(self._reg.config.paths.images_dir)
        images_dir.mkdir(parents=True, exist_ok=True)
        dest = images_dir / f"{image_id}{suffix}"
        shutil.copyfile(src, dest)
        # 静态加密：encrypt=True 且密钥可用时，落盘影像副本加密
        # （AES-256-GCM，魔数 SDC1 前缀）。密钥缺失时降级明文并告警——
        # 桌面单机默认无密钥仍可运行，但日志明确提示未加密。
        if self._reg.config.security.encrypt:
            from backend.infra.crypto import AesCrypto, CryptoKeyError

            try:
                cipher = AesCrypto()
            except CryptoKeyError as exc:
                _LOG.warning("静态加密未生效（%s）：影像副本以明文落盘", exc)
                return dest
            plaintext = dest.read_bytes()
            dest.write_bytes(cipher.encrypt(plaintext))
        return dest

    # ---- 人工复核缺陷增删改（DB50/T 1807-2025 ）----
    # 增/改/删后自动重评级并重生成报告（不重跑检测器）；每次变更由仓储层写审计哈希链。

    def add_defect(
        self, *, image_id: str, class_id: int, bbox_px: list[float], operator: str, reason: str
    ) -> dict:
        """人工添加缺陷框 → 重评级 → 重出报告。"""
        if not 0 <= int(class_id) < len(DefectClass):
            raise ValueError(f"class_id out of range: {class_id}")
        if len(bbox_px) != 4 or any(v < 0 for v in bbox_px):
            raise ValueError("bbox_px must be [x,y,w,h] with non-negative values")
        row = self._reg.repository.add_manual_defect(
            image_id=image_id,
            class_id=int(class_id),
            bbox_px=bbox_px,
            operator=operator,
            reason=reason,
        )
        result = self._regrade_and_report(image_id)
        return {"defect": row, **result}

    def edit_defect(
        self,
        *,
        defect_id: str,
        operator: str,
        reason: str,
        class_id: int | None = None,
        bbox_px: list[float] | None = None,
    ) -> dict:
        """人工修改缺陷类型/位置 → 重评级 → 重出报告。"""
        if class_id is not None and not 0 <= int(class_id) < len(DefectClass):
            raise ValueError(f"class_id out of range: {class_id}")
        if bbox_px is not None and (len(bbox_px) != 4 or any(v < 0 for v in bbox_px)):
            raise ValueError("bbox_px must be [x,y,w,h] with non-negative values")
        row = self._reg.repository.edit_defect(
            defect_id=defect_id,
            operator=operator,
            reason=reason,
            class_id=class_id,
            bbox_px=bbox_px,
        )
        result = self._regrade_and_report(row["image_id"])
        return {"defect": row, **result}

    def delete_defect(self, *, defect_id: str, operator: str, reason: str) -> dict:
        """人工删除缺陷（软删除）→ 重评级 → 重出报告。"""
        row = self._reg.repository.delete_defect(
            defect_id=defect_id, operator=operator, reason=reason
        )
        result = self._regrade_and_report(row["image_id"])
        return {"defect": row, **result}

    def _regrade_and_report(self, image_id: str) -> dict:
        """按库内现存缺陷重评级（不重跑检测器）→ 重生成报告 PDF。

        熔断语义与 run_inspection 一致：缺标定/厚度/表格未授权 → joint_level=None
        + need_review=True（不输出级别，人工兜底）。
        """
        repo = self._reg.repository
        image = repo.get_image(image_id)
        if image is None:
            raise KeyError(f"image not found: {image_id}")
        defects = []
        for d in image.get("defects") or []:
            bbox = d.get("bbox_px") or [0.0, 0.0, 1.0, 1.0]
            defects.append(
                Detection(
                    id=str(d.get("id", "")),
                    bbox=BBox(*(float(v) for v in bbox[:4])),
                    class_id=DefectClass(int(d["class_id"])),
                    score=float(d.get("confidence", 0.0)),
                    uncertainty=float(d.get("uncertainty", 1.0)),
                )
            )
        try:
            modality = Modality(image.get("modality") or "GENERIC")
        except ValueError:
            modality = Modality.GENERIC
        spacing = image.get("pixel_spacing_mm")
        context = ImageMeta(
            modality=modality,
            pixel_spacing_mm=spacing if spacing and spacing > 0 else None,
            base_metal_thickness_mm=image.get("base_metal_thickness_mm"),
        )
        quantifier = get_quantifier("bbox")
        spacing_eff = spacing if spacing and spacing > 0 else 1.0
        try:
            grade = self._reg.grader.grade(defects, context)
            joint_level = grade.joint_level.value
            per = {str(d.id): g.value for d, g in zip(defects, grade.per_defect_grade)}
            need_review = bool(grade.need_review)
        except GradingAmbiguousError as exc:
            joint_level = None
            per = {}
            need_review = True
            _LOG.info("regrade fused image_id=%s reason=%s", image_id, exc)
        geometry = {
            str(d.id): {
                "shape": _shape_of(d, g, self._reg.config.detect.round_aspect_max).value,
                "length_mm": g.length_mm,
                "width_mm": g.width_mm,
                "area_mm2": g.area_mm2,
                "perimeter_mm": g.perimeter_mm,
                "position_x": g.position_x_mm,
                "position_y": g.position_y_mm,
            }
            for d, g in ((d, quantifier.quantify(d, spacing_eff)) for d in defects)
        }
        repo.store_regrade(
            image_id,
            joint_level=joint_level,
            per_defect_levels=per,
            need_review=need_review,
            geometry=geometry,
        )
        # 报告级联失效：重出 PDF（内容指纹随缺陷/级别变化而更新）
        self.regenerate_report(image_id)
        return {
            "image_id": image_id,
            "joint_level": joint_level,
            "need_review": need_review,
            "defect_count": len(defects),
        }

    def apply_review(
        self,
        *,
        image_id: str,
        reviewer: str,
        role: str,
        defect_grades: list[dict[str, str]],
        overall_level: str | None = None,
        note: str | None = None,
        actor: str | None = None,
    ) -> dict:
        """人工复核闭环：聚合自动级别 → 计算 κ → 落库 → 重生成 PDF/A → 审计。

        参数：
        - defect_grades: [{defect_id, joint_level}] 复核对部分缺陷的级别覆盖；
        - overall_level: 复核显式综合级别（可选，优先于按缺陷推算）；
        - role: initial / secondary / arbitrator（仲裁为最终权威）。
        返回复核响应 dict（consensus/kappa/needs_arbitration/joint_level/...）。
        """
        reg = self._reg
        image = reg.repository.get_image(image_id)
        if image is None:
            raise KeyError(f"image not found: {image_id}")

        defects = image.get("defects") or []
        auto_grades = [d.get("joint_level") for d in defects]
        defect_ids = [d["id"] for d in defects]
        reviewer_map = {g["defect_id"]: g["joint_level"] for g in defect_grades}

        try:
            role_enum = ReviewRole(role)
        except ValueError:
            raise ValueError(f"invalid role: {role}") from None

        decision: ReviewDecision = resolve_review(
            auto_grades=auto_grades,
            defect_ids=defect_ids,
            reviewer_grades=reviewer_map,
            overall_level=overall_level,
            reviewer=reviewer,
            role=role_enum,
            kappa_threshold=reg.config.review.kappa_threshold,
        )

        # 落库：更新 images/defects/reports + 写复核行
        summary = reg.repository.apply_review(
            image_id=image_id,
            reviewer=reviewer,
            role=role_enum.value,
            final_level=decision.final_level,
            per_defect_level=decision.per_defect_level,
            consensus=decision.consensus,
            kappa=decision.kappa,
            needs_arbitration=decision.needs_arbitration,
            note=note,
        )

        # 复核达成一致/仲裁结案 → 重生成 PDF/A 报告（含最终级别）
        if decision.final_level is not None:
            pdf_path = reg.reporter.build(image_id, "standard")
            report_id = (image.get("report") or {}).get("report_id")
            if report_id:
                reg.repository.update_report(report_id, pdf_path=pdf_path)

        # 不可变审计日志：记录级别/复核标记前后值
        # actor = 提交复核的操作员（X-Operator-Name）；缺省回退 reviewer。
        reg.repository.append_audit(
            actor=actor or reviewer,
            action="review",
            object_type="image",
            object_id=image_id,
            before={
                "joint_level": image.get("joint_level"),
                "need_review": image.get("need_review"),
            },
            after={
                "joint_level": decision.final_level,
                "need_review": decision.need_review,
                "consensus": decision.consensus,
                "needs_arbitration": decision.needs_arbitration,
            },
            note=f"role={role_enum.value}",
        )

        return {
            "image_id": image_id,
            "reviewer": reviewer,
            "role": role_enum.value,
            "consensus": decision.consensus,
            "kappa": decision.kappa,
            "needs_arbitration": decision.needs_arbitration,
            "joint_level": decision.final_level,
            "reviewed_by": decision.reviewed_by,
            "stage": decision.stage.value,
            "need_review": decision.need_review,
            "review_count": summary["review_count"],
        }
