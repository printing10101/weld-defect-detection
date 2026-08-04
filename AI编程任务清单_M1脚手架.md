# M1 脚手架 — AI 编程任务清单（冻结契约版）

> 目的：在写任何功能之前，先把"工程空壳 + 护栏 + 冻结契约"立起来，关闭技术债务窗口（见规格书 §19）。
> 使用方法：每个任务严格按 §19.5 模板执行；合并前逐条过 §19.9 检查清单；CI（T7）必须在第一个功能任务开始前全绿。
> 铁律：本清单中的接口契约（§T2 代码、§T3 变换、§T8 配置/标准 schema）为**冻结真源**，后续功能任务只能调用/实现，不得修改签名或另起；改动须先写 ADR（§T9）再动手。

---

## 执行纪律（必须先读）
1. **顺序**：T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 → T9。
2. **T7 的 CI 红线必须在任何"功能任务"（M2+）开始前全绿**；在此之前不得开始 M2。
3. **冻结契约不可被任何后续任务破坏**：`domain/interfaces.py`、`domain/dto.py`、`domain/preprocess/transform.py`、`configs/schema`、标准表加载器的签名/字段是合同。
4. 每个任务交付物须含：代码 + 单测（含契约测试）+ 文档/ADR 同步，否则按 §19.9 不予合并。

---

## T1 — 仓库与工程初始化（前后端空壳）
- **【模块】** 仓库根 / `src/` / `src-tauri/` / `backend/`
- **【接口】** 无（纯脚手架）
- **【入参/出参】** 空仓库；目录结构严格对齐规格书 §19.2
- **【约束】** 目录与包名严格按 §19.2；建好空包 `app/ app/routers/ domain/ domain/preprocess/ domain/grade/ domain/standards/ infra/ configs/ tests/ data/ docs/adr/ migrations/`；前端 Vue3+Vite+TS(`strict:true`)，后端 Python(3.12, uv/venv)；Tauri2 配置 + `externalBin` 占位（sidecar 待 T4 填）。
- **【验收】** `pnpm build` 空壳可起；`python -c "import backend"` 不报错；目录树与 §19.2 一致。
- **【禁止】** 不写任何算法/业务逻辑；不提前实现功能端点。

---

## T2 — 冻结领域接口与 DTO（真实可编译代码）【关键：堵缺口#2】
- **【模块】** `backend/domain/interfaces.py` + `backend/domain/dto.py` + `backend/domain/errors.py`
- **【接口】** `DefectDetector / StandardGrader / Preprocessor / IQIVerifier / Quantifier / Reporter / Syncer`（签名见下，为合同，禁止改）
- **【入参/出参】** 下述代码**原样落地**，不得增删字段或参数
- **【约束】** 全部类型标注 + docstring；`pyright --strict` 必须通过；`image: np.ndarray` 约定：单通道灰度，shape `(H, W)` 或 `(H, W, 1)`，dtype `uint8`(0–255) 或 `float32`(0–1)；`mask_ref` 为掩膜资源 URI 或 `None`。
- **【验收】** `pyright --strict` 绿；后续 T4/T5 引用这些接口编译通过。
- **【禁止】** 不在接口里写实现；不新增未声明参数；不改枚举值。

**`backend/domain/dto.py`（冻结）**
```python
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class DefectClass(Enum):
    POROSITY = 0
    SLAG = 1
    INCOMPLETE_PENETRATION = 2
    LACK_OF_FUSION = 3
    CRACK = 4


class DefectShape(Enum):
    ROUND = "round"    # L/W <= 3
    LINEAR = "linear"  # L/W > 3


class JointLevel(Enum):
    I = "I"; II = "II"; III = "III"; IV = "IV"


class Modality(Enum):
    CR = "CR"; DR = "DR"; DICOM = "DICOM"; GENERIC = "GENERIC"


@dataclass(frozen=True)
class BBox:
    x: float; y: float; w: float; h: float


@dataclass(frozen=True)
class Detection:
    id: str
    bbox: BBox
    class_id: DefectClass
    score: float
    uncertainty: float
    shape: Optional[DefectShape] = None
    mask_ref: Optional[str] = None


@dataclass(frozen=True)
class Geometry:
    length_mm: float
    width_mm: float
    area_mm2: float
    perimeter_mm: float
    aspect_ratio: float
    position_x_mm: float
    position_y_mm: float


@dataclass(frozen=True)
class GradeResult:
    joint_level: JointLevel
    per_defect_grade: tuple[JointLevel, ...]
    basis: tuple[str, ...]
    need_review: bool
    standard_id: str
    standard_version: str


@dataclass(frozen=True)
class IQIResult:
    iqi_type: str
    achieved: Optional[str]
    required: str
    passed: bool


@dataclass(frozen=True)
class ImageMeta:
    modality: Modality
    pixel_spacing_mm: Optional[float] = None
    base_metal_thickness_mm: Optional[float] = None
```

**`backend/domain/interfaces.py`（冻结）**
```python
from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np
from .dto import (Detection, Geometry, GradeResult, IQIResult, ImageMeta)


@runtime_checkable
class DefectDetector(Protocol):
    def load(self, model_uri: str, backend: str = "onnx") -> None: ...
    def infer(self, image: np.ndarray, conf: float, iou: float) -> list[Detection]: ...


@runtime_checkable
class StandardGrader(Protocol):
    def grade(self, defects: list[Detection], context: ImageMeta) -> GradeResult: ...


@runtime_checkable
class Preprocessor(Protocol):
    def denoise(self, image: np.ndarray) -> np.ndarray: ...
    def enhance(self, image: np.ndarray, gamma: float) -> np.ndarray: ...
    def edges(self, image: np.ndarray, roi) -> np.ndarray: ...
    def morph(self, edges: np.ndarray, k_open: int, k_close: int) -> np.ndarray: ...


@runtime_checkable
class IQIVerifier(Protocol):
    def verify(self, image: np.ndarray) -> IQIResult: ...


@runtime_checkable
class Quantifier(Protocol):
    def measure(self, detection: Detection, pixel_spacing_mm: float) -> Geometry: ...


@runtime_checkable
class Reporter(Protocol):
    def build(self, image_id: str, template: str) -> str: ...


@runtime_checkable
class Syncer(Protocol):
    def push(self, record) -> None: ...
    def pull(self) -> list: ...
    def federate(self, weights) -> None: ...
```

**`backend/domain/errors.py`（冻结骨架）**
```python
class AppError(Exception):
    code: str = "UNKNOWN"
    http_status: int = 500


class ImageUnreadableError(AppError):
    code = "IMG_UNREADABLE"; http_status = 400


class IQIFailError(AppError):
    code = "IQI_FAIL"; http_status = 409


class ModelUnavailableError(AppError):
    code = "MODEL_UNAVAILABLE"; http_status = 503


class GradingAmbiguousError(AppError):
    code = "GRADING_AMBIGUOUS"; http_status = 422
```

---

## T3 — 共享预处理变换契约（train/serve 一致）【关键：堵缺口#3】
- **【模块】** `backend/domain/preprocess/transform.py`（纯函数，无 I/O）
- **【接口】** `to_model_input(raw: np.ndarray) -> np.ndarray`
- **【入参/出参】** 入：原始灰度 `np.ndarray`（§T2 约定）；出：模型输入张量/数组，约定 **RGB/BGR 顺序、letterbox resize 到 640×640、归一化均值/方差（写死并注释来源）、dtype float32**。训练（§17）与推理（§5）必须共用同一函数。
- **【约束】** 纯函数、可单测；与 §5/§17 完全一致；变更须 ADR-007。
- **【验收】** 单测：同一图经 train 路径与 serve 路径输出**逐元素一致**；ADR-007 记录变换参数。
- **【禁止】** 在 `detect/` 或 `models/train.py` 各自写一份预处理。

---

## T4 — 基础设施空实现 + 共享状态 registry【关键：堵缺口#4】
- **【模块】** `backend/infra/`（`model_store.py` `db.py` `crypto.py` `fs.py`）+ `backend/app/dependencies.py`（DI/registry）
- **【接口】** 各 infra 实现对应 T2 的 Protocol 的"最小可用/桩"：`LocalModelStore`（load 桩/空）、`SqliteStore`（建空表，schema 按 §7.1）、`AesCrypto`（桩）、`SecureFs`（tempfile 安全目录）。`dependencies.py` 定义 **registry 单例**，管理模型常驻与批量队列状态（线程安全，用 `asyncio.Lock`/进程池）。
- **【约束】** infra 禁业务；共享状态只走 registry；推理调用经线程池包裹（§13.11），禁止在 async handler 直接调同步 `infer`。
- **【验收】** 应用可起；`/api/v1/health` 返回 `model_version`/`gpu`/`status`；registry 单例单测。
- **【禁止】** 在 router 里 `new` 模型；绕过 registry。

---

## T5 — API 路由骨架 + OpenAPI 优先【含缺口#5 后端侧】
- **【模块】** `backend/app/main.py` + `backend/app/routers/`（按 §14 每资源一 router）
- **【接口】** §14 全部端点建好，request/response 用 Pydantic（字段严格对齐 §14 / §T2 DTO）；`/health` 200，其余端点暂返 `501 Not Implemented`；主程序挂载 `/api/v1`，CORS 限 `127.0.0.1`。
- **【约束】** Pydantic 严格；路径/方法/错误码严格按 §14；router 不写算法。
- **【验收】** 启动后生成 `openapi.json`；`/api/v1/health` 200；其余 501。
- **【禁止】** 在 router 内写算法或直连 DB（须经 infra/registry）。

---

## T6 — 前端类型同步（防漂移）【关键：堵缺口#5】
- **【模块】** `src/services/api.ts` + `src/types/`
- **【接口】** 由 T5 的 `openapi.json` 经 `openapi-typescript`/`orval` 生成（或手写镜像）TS 类型；所有前端调用经此单点。
- **【约束】** 字段命名/类型与 openapi 一致；单点维护，禁止在组件里手写裸 fetch 结构。
- **【验收】** 前端类型与 `openapi.json` 字段一致（可用生成器校验）；空壳页面可调 `/health`。
- **【禁止】** 组件内另写一份请求结构。

---

## T7 — 护栏落地（import-linter / pre-commit / CI）【关键：堵缺口#1】
- **【模块】** 仓库根配置
- **【内容】** ① `import-linter` 配置：`layers: presentation(src), application(backend/app), domain(backend/domain), infrastructure(backend/infra)`，`forbidden: infrastructure -> domain(业务符号)`、`presentation -> domain(绕过API)`、任意反向依赖；② `ruff`+`black`、`eslint`+`prettier`+`vue-tsc`、`pre-commit`；③ CI yaml 顺序严格 `ruff+black → pyright --strict → eslint+vue-tsc → import-linter → pytest(单测+契约) → eval-harness(占位) → build`；④ 空 `pytest` + **契约测试骨架**（对 T2 各 Protocol 写 conformance 测试模板）。
- **【约束】** CI 任一失败阻断合并；`import-linter` 红灯即失败。
- **【验收】** 故意写一处跨层 import → CI 红灯；空测跑通；pre-commit 生效。
- **【禁止】** 留 TODO 不接 CI；临时关闭 lint。

---

## T8 — 配置与标准表 schema（防蔓延/错填）【关键：堵缺口#6】
- **【模块】** `backend/configs/schema.yaml` + `backend/configs/*.yaml` + `backend/domain/standards/tables/loader.py`
- **【接口】** `load_config() -> AppConfig`（pydantic-settings）；`load_standard_tables(standard_id, version) -> dict`（带 schema 校验）
- **【入参/出参】** 配置键清单冻结：`server.port`、`server.host(=127.0.0.1)`、`model.default_uri`、`security.encrypt(bool)`、`paths.*`；标准表 YAML 须含字段（点数表/不计点数/条形限值）并校验。
- **【约束】** 全配置走 YAML+环境变量，**禁硬编码**；标准数值仍为 `{{STD_TABLE_*}}` 占位；**熔断规则**：标准数值未授权时，`StandardGrader.grade` 必须 `need_review=True` 且不输出级别（防静默错判）。
- **【验收】** 配置加载测试；占位值下 judge 安全降级（need_review=true，joint_level 不填）；schema 校验失败启动即报错。
- **【禁止】** 在代码里写端口/路径/密钥/数值表。

---

## T9 — ADR 落盘
- **【模块】** `docs/adr/`
- **【内容】** 把规格书 §19.8 的 ADR-001~006 写成正式 ADR（背景/决策/后果/备选）；新增 **ADR-007 共享预处理变换契约**（对应 T3）。
- **【约束】** 新架构决策须先写 ADR 再动手。
- **【验收】** 7 篇 ADR 入仓；后续任务引用对应 ADR 编号。

---

## M1 完成判据（Definition of Done 汇总）
- [ ] 前后端空壳可构建；目录严格 §19.2
- [ ] `interfaces.py`/`dto.py`/`errors.py` 冻结且 `pyright --strict` 绿
- [ ] 共享预处理变换单测证明 train/serve 一致
- [ ] registry 单例 + `/health` 正常；推理经线程池
- [ ] §14 全部端点建好，`openapi.json` 生成，非 health 返 501
- [ ] 前端类型由 openapi 生成，可调 `/health`
- [ ] import-linter / pre-commit / CI 全绿，跨层 import 红灯
- [ ] 配置+标准表 schema 落地，占位值安全降级
- [ ] ADR-001~007 入仓
- [ ] **CI 全绿后，方可开始 M2（功能任务）**

> 到此，所有"前 2–3 个任务定结构"的债务窗口已关闭：后续任何 M2+ 任务都在接口、文件树、lint、CI 四类护栏内施工。
