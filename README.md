# ScanDetection — 射线焊缝缺陷智能检测系统

工程级、抗过时、可机械护栏的 AI 编程项目。当前状态：**M1 脚手架已就位**。

规格书与任务清单（唯一事实源）：
- `技术规格书_射线焊缝缺陷智能检测系统.md` — 架构/模块/标准/编程规范/API/评测/护栏
- `AI编程任务清单_M1脚手架.md` — M1 九项任务（T1–T9）与本仓库一一对应

## 目录结构（§19.2）
```
src/                  Tauri + Vue3/TS 前端（表现层）
src/src-tauri/        Tauri 配置（M6 打包，需 Rust 工具链）
backend/
  app/                应用层：FastAPI 路由 + registry
  domain/             领域层：interfaces.py（冻结契约）/ dto.py / preprocess / standards
  infra/              基础设施层：config / model_store / db / crypto / fs
  models/             训练入口（M4 实现）
  configs/            schema.yaml + default.yaml（禁硬编码）
  tests/              单测 / 契约测试 / API 集成
data/                 DVC 追踪（M4 启用）
docs/adr/             ADR-001..007（架构决策）
migrations/           Alembic（M5 启用）
```

## 快速开始（后端）
```bash
# 仓库根
python -m venv backend/.venv
backend/.venv/Scripts/python -m pip install -e ".[dev]"
backend/.venv/Scripts/uvicorn backend.app.main:app --host 127.0.0.1 --port 18773
# 健康检查
curl http://127.0.0.1:18773/api/v1/health
```

## 快速开始（前端，无需 Rust）
```bash
cd src
yarn install      # 或 npm install（本机选用 yarn 1：pnpm 在中文路径下链接阶段挂起）
yarn dev          # http://127.0.0.1:5173（/api 代理到后端 18773）
# 由后端 openapi.json 重新生成前端类型（后端运行中执行）：
yarn gen-api      # -> src/types/generated.ts（T6 已生成，见 src/types/）
```

## 护栏与校验（§19.6，合并前必过）
```bash
backend/.venv/Scripts/ruff check backend
backend/.venv/Scripts/pyright
backend/.venv/Scripts/lint-imports
backend/.venv/Scripts/pytest backend/tests
```
CI：`.github/workflows/ci.yml`（ruff → pyright → eslint/vue-tsc → import-linter → pytest → build）。

## 铁律（§19，AI 编程必须遵守）
- 本仓库是实现层；接口契约以 `backend/domain/interfaces.py` 为准，禁止改签名。
- 新增代码必须落入 §19.2 目录；禁止跨层调用（import-linter 强制）。
- 配置禁硬编码（入 `backend/configs/*.yaml`）；标准数值未授权前 judge 必须熔断。
- 任何架构演进先写 `docs/adr/` 再动手。

## 待办
- [x] T6 收尾：`src/types/generated.ts` 已由 openapi.json 生成（`yarn gen-api` 可随时重生成）
- [ ] M2 起：按规格书 §4–§7 派发功能任务（T7 CI 全绿后方可开始）
- [x] Rust 工具链：本机已装 cargo 1.97.1 + cargo-tauri（Tauri 打包 M6 直接可用）
