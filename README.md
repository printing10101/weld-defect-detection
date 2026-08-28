# ScanDetection — 射线焊缝缺陷智能检测系统

工程级、抗过时、可机械护栏的 AI 编程项目。

规格书与任务清单（唯一事实源）：
- `技术规格书_射线焊缝缺陷智能检测系统.md` — 架构/模块/标准/编程规范/API/评测/护栏
- `AI编程任务清单_M1脚手架.md` — M1 九项任务（T1–T9）

## 当前状态（已超出 M1，勿再按"M1 未完成"理解）

后端里程碑 M1–M5 已全部落地，并在其基础上完成规格书 §12 的增强项：

| 能力 | 落地点 |
|---|---|
| 影像接入/黑度/IQI/伪缺陷（M2） | `backend/domain/iqi.py`、`density.py`、`pseudo_defect.py` |
| 预处理 + 质量门禁（M3） | `backend/domain/preprocess/` |
| 检测 + 量化 + 掩膜精修（M4a/M4b） | `backend/domain/detect/`、`quantify.py` |
| NB/T47013 评级 + 多标准适配（M5） | `backend/domain/grade/`、`standards/` |
| 报告 PDF/A + 数字签名（M6/M7） | `backend/infra/reporting/` |
| 批量任务队列（§12.1） | `backend/app/batch_queue.py` |
| 双人评片 + Cohen's κ（§12.2） | `backend/domain/review.py` |
| 设备标定档案（§12.4） | `backend/infra/device_store.py` |
| 不可变审计哈希链（§12.5） | `backend/infra/repository.py::append_audit` |
| 模型注册/热切换/自动评估（§7.4） | `backend/infra/model_registry.py`、`backend/evaluation/` |
| 主动学习闭环 + 标注器 | `backend/domain/active_learning.py`、`backend/annotator/` |

- **17 个 API 路由全部真实实现**（无 501 桩）：health / verify / preprocess / standards /
  detect / devices / judge / batch / review / explain / report / records / models /
  audit / active / evaluation / recommend。
- **后端单测 320 通过**（门禁 `--cov-fail-under=70` 强制）。
- 前端：Vue3 + TS 旅程式评片 + 批量进度 + 档案检索（`src/src/`）。单机科研自用、无用户系统，可在设置中填写操作员姓名（`X-Operator-Name`，用于报告签名与审计留痕）。

## 目录结构（§19.2，含已扩展部分）
```
src/                  Tauri + Vue3/TS 前端（表现层）
src/src-tauri/        Tauri 2 配置（sidecar + NSIS 打包）
backend/
  app/                应用层：FastAPI 路由 + registry + 鉴权 + 批量队列
  domain/             领域层：interfaces.py（冻结契约）/ dto.py / detect / preprocess /
                      grade / standards / quantify / review / iqi / explain / sync
  infra/              基础设施层：config / db / model_store / model_registry /
                      repository / device_store / crypto / fs / reporting
  evaluation/         评估 harness + Golden Set 门禁（§7.4）
  annotator/          人工标注器（随主进程可选启动，主动学习闭环）
  training/           训练/数据工具脚本（非运行时）
  models/             训练入口 + weights/（打包随 backend 资源分发）
  configs/            schema.yaml + default.yaml（禁硬编码）
  tests/              单测 / 契约 / API / 评估门禁（320 通过）
migrations/           Alembic
data/                不入版本库（.gitignore 排除各子目录，本地管理；未启用 DVC）
docs/adr/             ADR-001..010（架构决策）
scripts/              工具脚本（make_golden_set.py 等）
```

## 快速开始（后端）
```bash
python -m venv backend/.venv
backend/.venv/Scripts/python -m pip install -e ".[dev]"
backend/.venv/Scripts/uvicorn backend.app.main:app --host 127.0.0.1 --port 18773
curl http://127.0.0.1:18773/api/v1/health
```

## 快速开始（前端，无需 Rust）
```bash
cd src
yarn install
yarn dev          # http://127.0.0.1:5173（/api 代理到后端 18773）
yarn gen-api      # 由后端 openapi.json 重生成 src/types/generated.ts（前端实际类型以手写 src/src/types/api.ts 为准，generated.ts 为可选产物）
```

## 护栏与校验（合并前必过）
```bash
backend/.venv/Scripts/ruff check backend
backend/.venv/Scripts/ruff format --check backend
backend/.venv/Scripts/pyright backend        # 类型门禁（[tool.pyright] 已指向 backend/.venv）
backend/.venv/Scripts/lint-imports           # 分层契约（import-linter）
backend/.venv/Scripts/pytest backend/tests   # 320 通过，覆盖率门禁 70%
```
CI（`.github/workflows/ci.yml`）：ruff → ruff format → import-linter → **pyright** →
pytest（含覆盖率门禁）→ Golden Set 评估门禁；前端 vue-tsc → vite build。

## 模型升级流水线（一键，含打包）
```bash
# 用带 CUDA 的解释器（如 gpu venv）运行；新样本放 data/real_label/new_samples/{images,labels}
python scripts/model_tools/_upgrade_model.py                          # 收集→同步标注→重训→评估→导出ONNX→打包→复验
python scripts/model_tools/_upgrade_model.py --skip-train             # 跳过训练，仅评估+打包
```
发布前必过：`python scripts/model_tools/_verify_model_pkg.py`（确认安装包 `models/weights/best.onnx`
被真实加载而非静默回退基线）。

## 铁律（§19，AI 编程必须遵守）
- 接口契约以 `backend/domain/interfaces.py` 为准，禁止改签名（改前先 ADR）。
- 分层禁跨层调用（import-linter 强制）；配置禁硬编码（入 configs/*.yaml）。
- 标准数值未授权前 `StandardGrader.grade` 必须熔断（need_review=True，不输出级别）。

## 尚未完成（按优先级，均需人工/数据，非代码能解）
1. **真实标注数据**：真实稀有类仅 28 张人工标注（`_upgrade_model.py::MANUAL_STEMS`），
   其余为 AI 预标注；需补足 ~100 张真实标注（尤其真实裂纹）才能把检测从"召回优先预筛"
   提升到"可定级"。
2. **授权标准数值表**：`nb47013.yaml` 数值来自公开解读（`authorized=false`），
   评级输出在法理上不具合规效力；拿到授权原文替换后置 `authorized=true`。
3. **模型泛化**：气孔过检、稀有类精度低，依赖更多真实数据与难例挖掘。
