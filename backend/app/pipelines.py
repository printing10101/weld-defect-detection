"""用例编排。

校验 → 预处理 → 检测 → 量化 → 判定 → 落库 → 报告 的全链路编排。
只做编排不写算法；领域接口经 Registry 装配，存储走 infra repository。

熔断语义：标准数值未授权时 grader 抛 GradingAmbiguousError，
编排捕获后落库 joint_level=None + need_review=True（不输出级别），
报告生成"需人工复核"版本——不违反"禁止输出级别"。
"""

from __future__ import annotations

import logging
import re
import shutil
import time
import uuid
from pathlib import Path

from backend.app.dependencies import Registry
from backend.domain.active_learning import (
    export_training_labels,
    save_pool_manifest,
    training_pool_manifest,
)
from backend.domain.density import check_density, estimate_density
from backend.domain.detect.thresholds import resolve_class_conf
from backend.domain.dto import BBox, DefectClass, DefectShape, Detection, ImageMeta, Modality
from backend.domain.errors import GradingAmbiguousError, IQIFailError
from backend.domain.film_region import FilmRegionCfg as DomainFilmRegionCfg
from backend.domain.film_region import detect_film_region_trusted, film_background_fill
from backend.domain.gate_adapters import iqi_cfg_from_settings, pseudo_cfg_from_settings
from backend.domain.iqi import enrich_grade, verify_iqi
from backend.domain.preprocess.metrics import QualityCfg as DomainQualityCfg
from backend.domain.preprocess.metrics import assess_quality
from backend.domain.pseudo_defect import screen_pseudo_defects
from backend.domain.quantify import MaskRefineCfg, get_quantifier
from backend.domain.recommend import recommend
from backend.domain.review import ReviewDecision, ReviewRole, resolve_review
from backend.domain.spacing import resolve_spacing as _resolve_spacing  # 单一真源（§T8/§6）
from backend.domain.stamp import (
    StampCfg,
    extract_film_no,
    filter_stamp_zone,
    read_stamp_aligned,
)
from backend.domain.standards.tables.loader import disclaimer_for
from backend.evaluation.gate_rejects import GateRejectStore
from backend.infra.config import resolve_config_path
from backend.infra.image_loader import load_image, read_gray
from backend.infra.pool_store import FilePoolStore

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
        except Exception as exc:  # noqa: BLE001 - 任意异常（越界/空裁切）都不应阻断评片
            # 静默吞掉会让"深孔判定整体失效"无从排查（如密度数组形状异常），
            # 至少留 debug 级日志。
            _LOG.debug("deep hole derivation failed for %s: %s", d.id, exc)
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


def _preliminary_basis(reasons: list[str]) -> str:
    """AI 预筛级别的强声明（置于 basis 首条）。

    预筛通道的合规前提是**不静默**：级别虽然输出了，但降级原因必须与级别同时
    出现在报告依据里，评片员一眼能看出"这不是正式级别、为什么不是"。
    """
    detail = "；".join(reasons) if reasons else "底片质量门禁未通过"
    return (
        f"注意：AI 预筛级别（非正式级别）：底片质量未达标准要求——{detail}。"
        "该级别仅供 AI 辅助预筛参考，不得作为验收/合格判定依据，"
        "须由持证人依标准原文重新评定"
    )


class InspectionPipeline:
    """一次完整评片的用例编排。"""

    def __init__(self, reg: Registry) -> None:
        self._reg = reg

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
        content_sha256: str | None = None,
        batch_no: str | None = None,
        report_meta: dict[str, str] | None = None,
        allow_preliminary_grade: bool = False,
        spacing_note: str | None = None,
    ) -> dict:
        """执行全链路并落库+生成报告，返回结果 dict。

        force=False（默认）时，底片不可评（黑度越界或 IQI 不达标）直接抛
        IQIFailError（409），符合设计文档"不通过则阻断评片并提示重拍"的硬前置；
        force=True 时仍出片，但**不输出级别**（need_review=True），因为
        不合格底片不构成评定依据。

        翻拍影像例外（film_region 判定 is_photo 且 density.photo_policy=warn）：
        相机拍灯箱的 8bit 照片绝对黑度不可测、IQI 识别不可靠，黑度/IQI/质量/
        位深/扫描参数门禁不阻断，evaluable 按翻拍口径重算（仅严重伪缺陷否决，
        与 /verify 同源）；级别经 AI 预筛通道输出（grade_preliminary=True，
        basis 首条警示声明）并强制人工复核（need_review=True），检测/量化/
        报告链路照常执行。
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
        film = (
            detect_film_region_trusted(
                gray,
                DomainFilmRegionCfg(
                    min_area_frac=fr.min_area_frac,
                    max_photo_area_frac=fr.max_photo_area_frac,
                    surround_bright_gray=fr.surround_bright_gray,
                    surround_min_frac=fr.surround_min_frac,
                ),
            )
            if fr.enabled
            else None
        )
        photo_mode = bool(
            film is not None
            and film.is_photo
            and meta.bit_depth == 8
            and meta.density_array is None
        )
        eval_gray = (
            gray[film.y : film.y + film.h, film.x : film.x + film.w] if film is not None else gray
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
        iqi_cfg = iqi_cfg_from_settings(reg.config.iqi)
        # IQI 验证：用户给 ROI 时按原图坐标在全图上验证（ROI 已隔离像质计，
        # 且胶片裁剪可能把像质计切掉）；无 ROI 时在胶片区上自动定位。
        iqi_target = gray if iqi_roi is not None else eval_gray
        iqi = verify_iqi(iqi_target, iqi_cfg, roi=iqi_roi, iqi_type=reg.config.iqi.type)
        # 用透照厚度 + 参考表补全 A/AB/B 等级（厚度缺失则 grade=None）。
        iqi = enrich_grade(iqi, base_metal_thickness_mm, iqi_cfg.sensitivity)
        # 伪缺陷筛查，仅严重项默认阻断。
        pd_domain = pseudo_cfg_from_settings(reg.config.pseudo_defect)
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
        gate_evaluable = bool(
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
        reasons: list[str] = []
        if not gate_evaluable:
            if not density_ok:
                reasons.append(
                    f"黑度 {density:.2f} 超出 [{reg.config.density.low}, {reg.config.density.high}]"
                )
            if not iqi.passed:
                achieved_txt = "未测得" if iqi.achieved is None else str(iqi.achieved)
                reasons.append(f"IQI 未达要求（要求 {iqi.required}，实测 {achieved_txt}）")
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
        # 翻拍口径下"能否评片"与门禁结论分离（与 /verify 同源）：黑度/IQI/质量/
        # 位深/扫描参数对翻拍照片不可验证或必然不成立（照片必为 8bit、无扫描
        # 元数据），不构成"不可评片"，只剩严重伪缺陷一票。此前直接用
        # gate_evaluable 落库/出响应，照片必被位深门禁判死，界面永远"不可评片"、
        # 评级被熔断。降级痕迹不丢：预筛声明 + photo_warnings 并入 basis，
        # need_review 强制兜底，未验证 ≠ 合格。
        evaluable = bool(pd.passed) if photo_advisory else gate_evaluable
        # 门禁降级放行的告警（require_dpi=false / allow_8bit=true 路径）：
        # 不阻断评片，但必须在结果与日志中留痕，避免"未验证"被静默当作合格。
        gate_warnings: list[str] = []
        # 标定注入留痕（G23）：像素标定来自设备档案或设备无标定的结论，
        # 评片员须能在结果里看到标定来源/缺失（响应 warnings → 告警面板）。
        if spacing_note:
            gate_warnings.append(spacing_note)
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
            if not gate_evaluable:
                photo_warnings.append(
                    "翻拍影像质量门禁未通过（" + "；".join(reasons) + "），已降级为人工复核"
                )
            _LOG.warning(
                "photo-mode advisory image density=%.2f iqi_pass=%s quality_pass=%s",
                density,
                bool(iqi.passed),
                quality.passed,
            )

        # （原图副本落盘移至第 6 步落库前一刻：副本先行落盘时，其后的检测/
        # 判定/落库任何一步失败都会留下无台账的孤儿密文副本，批量长跑下
        # 静默占满磁盘——磁盘看门狗只报警不定位。）

        # 4. 预处理 + 检测 + 量化（ 基线 /  训练模型，同一接口）
        # 预处理（保边去噪+增强）后送检测；黑度/IQI/伪缺陷/质量门禁已在原始
        # 影像上完成，不受增强影响。检测源保持整图尺寸（缺陷框坐标系与
        # 原图/落库一致），胶片区外背景填充为胶片中位灰阶，防止灯箱
        # 亮背景/翻拍边框被误检为缺陷。
        dc = reg.config.detect
        detect_src = film_background_fill(gray, film)
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
        # 底片印字识别须先于检测：命中印字框供下方 stamp-zone 误检过滤使用
        # （框经 read_stamp_aligned 映射回整图坐标，与检测框同系）；
        # 落库/缺印字复核语义与原 5.5 步完全一致（读数前移不改变任何判定）。
        _sc = reg.config.stamp
        stamp = read_stamp_aligned(
            gray,
            StampCfg(enabled=_sc.enabled, min_conf=_sc.min_conf, max_side=_sc.max_side),
            film=film,
        )
        # 工作模式阈值解析（balanced 恒等；recall/precision 档对逐类表统一缩放，
        # 类间相对关系不变）。缩放后的逐类表传给检测器，未列类回落缩放后的
        # infer_conf——回落语义与标定表保持一致。
        infer_conf, class_conf_eff = resolve_class_conf(
            dc.infer_conf,
            dc.class_conf,
            dc.mode,
            recall_scale=dc.recall_conf_scale,
            precision_scale=dc.precision_conf_scale,
        )
        detections = reg.detector.infer(
            enhanced, conf=infer_conf, iou=dc.infer_iou, class_conf=class_conf_eff
        )
        # 印字区误检过滤（detect.mask_stamp_zone）：真实底片的编号/日期铅字是
        # 首要误检源（实测 27/27 检出全落印字带）。OCR 读到的全部文本框（含
        # 中心标/片号等非日期编号铅字）外扩后吞掉检测中心的检出判为印字误检；
        # 屏蔽不静默——数量进响应 warnings 与审计留痕，绝不静默丢弃。
        stamp_masked = 0
        if dc.mask_stamp_zone:
            zones = stamp.text_boxes or stamp.boxes
            if zones:
                detections, masked_list = filter_stamp_zone(detections, zones)
                stamp_masked = len(masked_list)
                if stamp_masked:
                    gate_warnings.append(
                        f"已屏蔽印字区误检 {stamp_masked} 个（印字/铅字文本框外扩匹配）"
                    )
        # 深孔推导：缺陷内部黑度显著高于母材 → 标 deep_hole（直判 IV 的前置信号）。
        # 必须在判定/落库前完成，使 grader 与 defect_rows 都能消费到该标记。
        detections = _derive_deep_hole(detections, meta, density, meta.bit_depth)
        # 经量化器注册表装配，种类由 detect.quantifier_kind 配置驱动（与 /detect 同源）。
        # 手头有增强图，掩膜精修在主链路真实生效；精修失效/被禁用时量化器内部
        # 自动退化为包围盒近似（与旧行为等价）。
        mrc = MaskRefineCfg(**reg.config.mask_refine.model_dump())
        quantifier = get_quantifier(reg.config.detect.quantifier_kind)
        # 像素标定缺失时不再伪造 1.0 mm/px 后照常定级——几何量纲无据即触发熔断。
        spacing, spacing_known = _resolve_spacing(pixel_spacing_mm, meta.pixel_spacing_mm)
        quantified = [
            (d, quantifier.quantify(d, spacing, image=enhanced, cfg=mrc)) for d in detections
        ]

        # 5. 标准判定（，未授权/信息不足熔断 → 不输出级别）
        context = ImageMeta(
            modality=meta.modality,
            pixel_spacing_mm=spacing if spacing_known else None,
            base_metal_thickness_mm=base_metal_thickness_mm,
        )
        try:
            grade_preliminary = False
            if not evaluable:
                if not allow_preliminary_grade:
                    # force 出片：底片不合格不得作为评定依据
                    raise GradingAmbiguousError("底片不可评（黑度/IQI 不合格），不输出级别")
                # AI 预筛通道（调用方显式请求）：底片质量未达标仍计算级别，但
                # **必须**强标记 + 强声明——级别照常输出，可它不构成验收依据。
                # 这是"可见的降级"而非"静默错判"：basis 首条与 need_review 都会
                # 告诉评片员"这不是正式级别"。
                grade_preliminary = True
            elif photo_advisory:
                # 翻拍影像同走预筛通道：黑度/IQI 未经验证，级别不构成正式
                # 评定——必须强标记，否则报告会把未验证底片的级别当正式结论。
                grade_preliminary = True
            grade = reg.grader.grade(detections, context)
            joint_level: str | None = grade.joint_level.value
            per_grade = [g.value for g in grade.per_defect_grade]
            basis = list(grade.basis)
            need_review = bool(grade.need_review)
            std_version = grade.standard_version
            if grade_preliminary:
                basis = [_preliminary_basis(reasons), *basis]
                need_review = True
        except GradingAmbiguousError as exc:
            grade_preliminary = False
            joint_level = None
            per_grade = []
            # 保留熔断原因：报告与审计需可追溯（丢弃将无从解释为何无级别）
            basis = [str(exc)] if str(exc) else ["判定信息不足，需人工复核"]
            need_review = True
            std_version = ""
        # 翻拍影像告警并入报告依据与人工复核标记：门禁未验证的结果不得静默当作合格片。
        if photo_warnings:
            basis = [*basis, *photo_warnings]
            need_review = True
        # 质量门禁未达阈值且非阻断模式（block_on_quality=False）时仅告警，并入人工复核标记。
        need_review = bool(need_review or quality_warn)

        # 5.5 底片印字裁决（read_stamp 已前移到检测前，见第 4 步）：缺印字的
        # 处置分两种语义：
        # - 单图评片（batch_no=None）：缺印字即转人工复核（确认底片身份）；
        # - 批量评片（batch_no 非空）：缺印字**不**立即转复核，延迟到批次收尾
        #   按批内印字占比裁决（Registry._apply_batch_stamp_policy）——大批底片
        #   普遍无印字时豁免（缺印字是批次常态而非异常），占比达标才补标记。
        stamp_deferred = batch_no is not None
        stamp_need_review = bool(stamp.status == "missing" and not stamp_deferred)
        if stamp_need_review:
            basis = [
                *basis,
                "底片未识别到扫描日期/编号印字（正/镜像均未命中），需人工复核确认底片身份",
            ]
            need_review = True

        std_id = standard_id or reg.config.standard.default_id
        # 工业过渡路径：免责声明只依赖标准表（standard-level），与判定结果无关，
        # 故无论评级成功或熔断均统一生成（authorized_copy=false 时为强声明）。
        disclaimer = disclaimer_for(reg.grader.tables)  # type: ignore[attr-defined]

        # 合规处置建议：消费评级输出，独立适配器（domain/recommend），
        # 不参与判定；熔断时降级为「需人工复核」，永不阻塞出片。
        # 报告"合格级别"栏作为验收等级参与合格性判定（G20）。
        rec = recommend(
            joint_level,
            detections,
            need_review=need_review,
            standard_id=std_id,
            disclaimer=disclaimer,
            accept_level=(report_meta or {}).get("accept_level"),
        )
        disposition = rec.disposition
        disposition_label = rec.disposition_label
        disposition_actions = list(rec.actions)

        # 6. 原图副本落盘（报告缺陷图谱数据源；勿删）+ 落库（一个事务）：
        # 两者紧邻执行，落库失败即回收副本，不留无台账的孤儿文件。
        suffix = image_path.suffix or ".png"
        saved = self._persist_image(image_path, image_id, suffix)
        image_row = {
            "id": image_id,
            "path": str(saved),
            "source_type": "dicom" if meta.modality is Modality.DICOM else "image",
            "modality": meta.modality.value,
            "workpiece_no": workpiece_no,
            "weld_no": weld_no,
            # 只落可信标定：未标定时 spacing 是"1.0 mm/px 伪值"（几何量纲无据），
            # 落库会让人工复核重评级把它当真标定读回、绕过熔断输出综合级别。
            "pixel_spacing_mm": spacing if spacing_known else None,
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
            # 文件内容摘要（批量查重/历史比对索引；单图路径未计算时为 None）
            "content_hash": content_sha256,
            # 批量追溯归属 + 底片印字（扫描日期/编号）性质快照
            "batch_no": batch_no,
            # 报告补充信息（《射线检测报告》汇总表字段，路由层已白名单清洗）
            "report_meta": dict(report_meta or {}),
            "stamp_status": stamp.status,
            "stamp_text": stamp.text,
            "stamp_orientation": stamp.orientation,
            "stamp_confidence": stamp.confidence,
            "stamp_need_review": stamp_need_review,
            # 片号（G05）：印字文本结构化抽取；未识别到片号落 NULL
            "film_no": extract_film_no(stamp.text),
        }
        # per_defect_grade 与 detections 按序对齐；长度不符说明 grader 契约被破坏，
        # 与其把级别错配到别的缺陷上（安全事故），不如整体退化为"无级别+需复核"。
        if per_grade and len(per_grade) != len(quantified):
            per_grade = []
            need_review = True
            basis = [*basis, "逐缺陷级别与检测数量不一致，已退化为人工复核"]

        # 未标定（spacing_known=False）时量化走伪值 1.0 mm/px，mm 字段是
        # "像素=毫米"的伪物理量——落库置 None（与 /detect 的 calibrated=False
        # 语义对齐），禁止下游把伪尺寸当真实几何量消费。
        defect_rows = [
            {
                "id": f"{image_id}:{d.id}",  # 全局唯一主键（同一影像内由 d.id 区分）
                "image_id": image_id,
                "class_id": d.class_id.value,
                "bbox_px": [d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h],
                "shape": _shape_of(d, g, dc.round_aspect_max).value,
                "length_mm": g.length_mm if spacing_known else None,
                "width_mm": g.width_mm if spacing_known else None,
                "area_mm2": g.area_mm2 if spacing_known else None,
                "perimeter_mm": g.perimeter_mm if spacing_known else None,
                "position_x": g.position_x_mm if spacing_known else None,
                "position_y": g.position_y_mm if spacing_known else None,
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
            "standard_ref": (
                std_id
                if std_version and std_id.endswith(std_version)
                else f"{std_id} {std_version}".strip()
            ),
            "signer": signer,
            "basis": basis,
        }
        try:
            reg.repository.create_inspection(image_row, defect_rows, report_row)
        except Exception:
            # 落库失败：回收刚落盘的副本，不留孤儿密文（无台账文件无法经
            # API 定位清理，只能永久占盘）。
            saved.unlink(missing_ok=True)
            raise

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
                "stamp_zone_masked": stamp_masked,
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
            "evaluable=%s need_review=%s photo_mode=%s detect_mode=%s stamp=%s/%s (%.1f ms)",
            image_id,
            joint_level,
            len(quantified),
            density_ok,
            bool(iqi.passed),
            evaluable,
            need_review,
            photo_mode,
            dc.mode,
            stamp.status,
            stamp.orientation,
            dt * 1000,
        )
        return {
            "image_id": image_id,
            "report_id": report_id,
            "joint_level": joint_level,
            # AI 预筛级别标记：True=底片质量未达标但用户显式请求了预筛级别，
            # 级别不具合规效力（basis 首条已给出降级原因，前端须显著标识）。
            # 只出现在响应里，不落库（ImageRecord 无此列，语义由 joint_level + basis 承载）。
            "grade_preliminary": grade_preliminary,
            "need_review": need_review,
            "evaluable": evaluable,
            "density": round(density, 3),
            "density_ok": density_ok,
            "iqi_pass": bool(iqi.passed),
            "iqi_detail": {
                "type": iqi.iqi_type,
                "achieved": iqi.achieved,
                "required": iqi.required,
                "grade": iqi.grade,
            },
            "defect_count": len(quantified),
            "photo_mode": photo_mode,
            "detect_mode": dc.mode,
            # 门禁降级原因透出：客户端须能看到"为什么转人工复核"
            # （黑度越界/dpi 未定/印字区屏蔽等），而非只拿到 need_review 布尔
            "warnings": [*gate_warnings, *photo_warnings],
            "basis": basis,
            "stamp_zone_masked": stamp_masked,
            "disclaimer": disclaimer,
            "disposition": disposition,
            "disposition_label": disposition_label,
            "disposition_actions": disposition_actions,
            "pdf_path": pdf_path,
            "report_meta": dict(report_meta or {}),
            # 报告页样张式首页预览所需的表单回显字段
            "workpiece_no": workpiece_no,
            "weld_no": weld_no,
            "signer": signer,
            "standard_ref": (
                std_id
                if std_version and std_id.endswith(std_version)
                else f"{std_id} {std_version}".strip()
            ),
            "stamp": stamp.summary(need_review=stamp_need_review),
        }

    def _gate_reject_store(self) -> GateRejectStore:
        """拦截留档台账（E-05）：代理到 Registry 懒建单例（全进程一个引擎）。

        此前缓存在 pipeline 实例上——BatchManager 每任务新建 pipeline，
        每个触发过拦截的 pipeline 各建一个 SQLAlchemy engine（+create_all）
        且永不 dispose，长跑批量下连接池随任务累积。
        """
        return self._reg.gate_reject_store()

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
        try:
            self._gate_reject_store().add(
                reject_id=reject_id,
                image_id=image_id,
                reject_reason="；".join(reasons)[:256],
                detail=detail,
                dpi=dpi,
                bit_depth=bit_depth,
                operator=operator or "system",
            )
        except Exception as exc:  # noqa: BLE001 - 台账故障不得掩盖门禁拦截本身
            # 调用方在本方法返回后抛 IQIFailError(409)（重拍语义）；台账/审计
            # 异常若上抛会把设计好的 409 变 500，客户端无从判断该重拍。
            _LOG.error("gate_rejects 台账写入失败 reject_id=%s: %s", reject_id, exc)
        try:
            reg.repository.append_audit(
                actor=operator or "system",
                action="gate_reject",
                object_type="image",
                object_id=reject_id,
                before=None,
                after={"reasons": list(reasons), "dpi": dpi, "bit_depth": bit_depth},
            )
        except Exception as exc:  # noqa: BLE001 - 审计故障不得掩盖门禁拦截本身
            _LOG.error("gate_reject 审计写入失败 reject_id=%s: %s", reject_id, exc)
        return reject_id

    def _persist_reject(self, src: Path, reject_id: str, suffix: str) -> Path:
        """不合格底片副本归档到 gate.rejects_dir（密文，模式同 _persist_image）。"""
        return self._persist_copy(
            src,
            Path(resolve_config_path(self._reg.config.gate.rejects_dir)),
            reject_id,
            suffix,
            what="不合格底片",
        )

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
            # 影像尚无报告行：直接 uuid4 会凭空给出查不到的 report_id
            # （下载必 404），须先建报告行再回填路径。
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
            # 复核后重出报告同样按报告"合格级别"栏判定（与首评同口径）
            accept_level=(image.get("report_meta") or {}).get("accept_level"),
        )
        return {
            "image_id": image_id,
            "report_id": report_id,
            "joint_level": image.get("joint_level"),
            "need_review": bool(image.get("need_review", False)),
            "evaluable": bool(image.get("evaluable", True)),
            "defect_count": len(stored_defects),
            "density": image.get("density"),
            "density_ok": image.get("density_ok"),
            "iqi_pass": image.get("iqi_pass"),
            "iqi_detail": image.get("iqi_detail"),
            "photo_mode": False,
            "detect_mode": self._reg.config.detect.mode,
            "warnings": [],
            "basis": list((image.get("report") or {}).get("basis") or []),
            "stamp_zone_masked": 0,
            "disclaimer": disclaimer_for(self._reg.grader.tables),  # type: ignore[attr-defined]
            "disposition": rec.disposition,
            "disposition_label": rec.disposition_label,
            "disposition_actions": list(rec.actions),
            "pdf_path": pdf_path,
            "report_meta": dict(image.get("report_meta") or {}),
            "workpiece_no": image.get("workpiece_no"),
            "weld_no": image.get("weld_no"),
            "signer": (image.get("report") or {}).get("signer"),
            "standard_ref": (image.get("report") or {}).get("standard_ref"),
        }

    def _write_encrypted_copy(self, src: Path, dest: Path, *, what: str) -> None:
        """影像副本加密落盘（_persist_image/_persist_reject 共用）。

        encrypt=True（默认）时以 SDC2 国密信封写密文，密钥来自 env
        SCAN_CRYPTO_KEY 或本地持久密钥文件 data/.crypto_key（首启自动生成，
        见 crypto.py）；密钥不可用（env 与本地密钥文件均失败）时拒绝明文
        落盘并留痕——静态加密失效宁可阻断归档，不可静默降级（GB/T 28452
        用户数据保密性口径）。流式分块读写（crypto.encrypt_stream）：
        大底片不再整文件进内存（原内存峰值 ≈2×文件大小×并发 worker 数）。
        写失败回收半截密文，不留无台账的孤儿文件。
        """
        if not self._reg.config.security.encrypt:
            shutil.copyfile(src, dest)
            return
        from backend.infra.crypto import CryptoKeyError, default_crypto_provider

        try:
            # 进程内共享 provider（密钥来源不变时复用）：免除每张影像重复
            # 的 SM2 点乘/KDF 开销（gmssl 纯 Python，单次数十 ms）。
            cipher = default_crypto_provider()
        except CryptoKeyError as exc:
            _LOG.error("静态加密密钥不可用（%s）：拒绝将%s以明文落盘", exc, what)
            raise
        try:
            stream = getattr(cipher, "encrypt_stream", None)
            with open(src, "rb") as fin, open(dest, "wb") as fout:
                if stream is not None:
                    stream(fin, fout)  # 软国密：流式（与一次性 encrypt 同信封）
                else:
                    fout.write(cipher.encrypt(fin.read()))  # 硬件 provider 无流式接口
        except BaseException:
            dest.unlink(missing_ok=True)
            raise

    def _persist_copy(
        self, src: Path, directory: Path, stem: str, suffix: str, *, what: str
    ) -> Path:
        """影像类副本归档公共路径：suffix 白名单清洗 + 目录包含校验 + 密文落盘。

        suffix 源自上传文件名（外部输入）：白名单限定 ".字母数字(≤8)"，
        不合规回退 ".png"；stem 为内部生成的 UUID。dest 解析后必须仍在
        目标目录内（纵深防御，防拼接逃逸）。
        """
        directory.mkdir(parents=True, exist_ok=True)
        safe = suffix if re.fullmatch(r"\.[A-Za-z0-9]{1,8}", suffix or "") else ".png"
        dest = directory / f"{stem}{safe}"
        if not dest.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"{what}归档路径越界: {dest}")
        self._write_encrypted_copy(src, dest, what=what)
        return dest

    def _persist_image(self, src: Path, image_id: str, suffix: str) -> Path:
        return self._persist_copy(
            src,
            Path(resolve_config_path(self._reg.config.paths.images_dir)),
            image_id,
            suffix,
            what="影像副本",
        )

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
        # 种类随 detect.quantifier_kind 配置（与主链路同源）。复核重评级不重跑
        # 检测器，但可从落盘影像副本（密文）解出灰度图喂给量化器——否则掩膜
        # 量化静默退化为包围盒，"首评 vs 复评"几何口径跳变。解图失败才退化。
        quantifier = get_quantifier(self._reg.config.detect.quantifier_kind)
        regrade_gray = read_gray(str(image.get("path") or "")) if image.get("path") else None
        if regrade_gray is None:
            _LOG.info("regrade: 影像副本不可读 image_id=%s，量化退化为包围盒近似", image_id)
        spacing_known = bool(spacing and spacing > 0)
        # 内联条件而非 spacing_known：让类型检查器把 None 收窄掉
        spacing_eff = spacing if spacing is not None and spacing > 0 else 1.0
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
        if spacing_known:
            # 与主链路同口径回写几何（含配置驱动的掩膜精修参数）；未标定时
            # 1.0 mm/px 是伪值——回写会让人工复核把"像素=毫米"的伪物理量
            # 覆盖进 DB 并渲染进正式报告（与首轮落库置 None 的熔断语义相抵），
            # 故 geometry=None 只更新级别。
            mrc = MaskRefineCfg(**self._reg.config.mask_refine.model_dump())
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
                for d, g in (
                    (d, quantifier.quantify(d, spacing_eff, image=regrade_gray, cfg=mrc))
                    for d in defects
                )
            }
        else:
            geometry = None
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

    def _sync_review_to_training_pool(
        self, reg: Registry, image_id: str, image: dict, defects: list[dict]
    ) -> str:
        """复核结论落定后把人工确认缺陷自动回流训练池（G21）。

        此前回流依赖人工补调 POST /active/export，专家改判的类别/边界长期
        滞留业务库进不了训练数据。按 DB 现状（复核后的值）导出 YOLO 标注并
        刷新 manifest。返回失败原因（空串=成功）：影像不可读/训练池写失败
        不阻断复核主流程，但状态随响应暴露，不静默吞掉。
        """
        try:
            gray = read_gray(str(image.get("path") or "")) if image.get("path") else None
            if gray is None:
                return "image_unreadable"
            dets: list[Detection] = []
            for d in defects:
                bbox = d.get("bbox_px")
                if not bbox or "class_id" not in d:
                    continue
                dets.append(
                    Detection(
                        id=str(d.get("id", "")),
                        bbox=BBox(*(float(v) for v in bbox[:4])),
                        class_id=DefectClass(int(d["class_id"])),
                        score=float(d.get("confidence", 0.0)),
                        uncertainty=float(d.get("uncertainty", 1.0)),
                    )
                )
            pool_dir = Path(
                resolve_config_path(
                    str(Path(reg.config.paths.data_dir) / "active" / "training_pool")
                )
            )
            store = FilePoolStore(pool_dir)
            export_training_labels(
                image_id, dets, float(gray.shape[1]), float(gray.shape[0]), store=store
            )
            save_pool_manifest(store, training_pool_manifest(store))
            return ""
        except Exception as exc:  # noqa: BLE001 —— 回流是旁路动作，任何失败都不回滚复核
            _LOG.error("训练池自动回流失败 image_id=%s: %s", image_id, exc)
            return f"export_failed: {exc}"

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

        # 复核结论落定（级别确认/仲裁）→ 人工确认缺陷自动回流训练池（G21）。
        # 与重出报告同条件：结论未定时缺陷集还不是"人工确认标注"。
        training_pool_synced = True
        if decision.final_level is not None:
            training_pool_synced = (
                self._sync_review_to_training_pool(reg, image_id, image, defects) == ""
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
            "training_pool_synced": training_pool_synced,
        }
