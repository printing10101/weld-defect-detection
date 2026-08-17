# 优化清单 · E（工程卫生）+ D（前端健壮性）

> 日期：2026-08-17　|　前置批次：A1+B1–B4（发布/安全）、A2/C2/A3/C1（配置/CI/打包）
> 验证：前端 `vue-tsc --noEmit` 通过、生产 `vite build` 通过；后端 pytest **320 passed / 5 skipped**，覆盖率 **76.11%**（≥70% 门禁）。

## E · 工程卫生

| 项 | 文件 | 修复 |
|---|---|---|
| **E1** | `src/package.json` | `build`/`build:pkg` 改为 `vue-tsc --noEmit && vite build`，类型错误在打包前暴露（实测 typecheck 已绿，零风险）。 |
| **E2** | `src/eslint.config.js`（新增）、`src/.prettierrc.json`（新增） | 补 ESLint 9 扁平配置（typescript-eslint + eslint-plugin-vue）+ Prettier 配置；`package.json` 增 `lint`/`format` 脚本，与后端 ruff 同级。说明：`yarn lint` 为可选接线，全量启用前建议先跑一次处理存量风格告警。 |
| **E3** | `src/src/types/generated.ts`（删除） | 孤儿文件（全工程零 import，仅 `gen-api` 可重生成）。`git rm` 移除，根目录减一散落产物。 |
| **E4** | 7 个根目录脚本 → `scripts/model_tools/` | `_export_model.py` `_onnx_inspect.py` `_probe_newmodel.py` `_read_docx.py` `_upgrade_model.py` `_verify_model_pkg.py` `_verify_planA.py` 迁至 `scripts/model_tools/`（git 历史保留）；README 引用同步更新。 |
| **E5** | `README.md` | 路由数 17→**18**（补 `recommend`）；测试数 262→**320**；删除失实的「覆盖率 88.35%」「DVC 追踪」（实际 `.dvc` 不存在、data 各子目录被 .gitignore 排除）；`gen-api` 注明 generated.ts 为可选产物；模型升级脚本路径指向 `scripts/model_tools/`。 |
| **E6** | `src/vite.shared.ts`（新增）、`src/vite.config.ts`、`src/vite.tauri.config.ts` | 抽取共享 Vite 基础配置（去重 `plugins`）；打包配置加 `build.sourcemap: false`，避免桌面包泄露前端源码。 |

## D · 前端健壮性

| 项 | 文件 | 修复 |
|---|---|---|
| **D1** | `src/src/services/api.ts` | `request()` 新增 AbortController 超时（30s）+ 仅对「后端不可达（连接被拒）」做指数退避重试（超时/HTTP 错误不重试，避免重复提交）；连接失败时返回可操作的 `BACKEND_UNREACHABLE` 提示。 |
| **D2** | `src/src/views/BatchView.vue` | 提交加 `submitting` 忙态 + 重入守卫，按钮 `:disabled` 防双击重复提交。 |
| **D3** | `src/src/views/BatchView.vue` | 轮询连续失败指数退避（2s→4s→8s），累计 3 次判定后端离线并停止空转，展示「重试连接」入口。 |
| **D4** | `src/src/views/ArchiveView.vue` | 档案列表接分页（`page`/`pageSize`/`total` + 翻页控件 + 每页 50/100/200 切换），突破原 `size:50` 写死导致的 >50 条不可见。 |
| **D5** | `src/src/components/ReportView.vue` | 报告数字签名校验（`verifyReport`）接线：新增「验证数字签名」按钮 + 有效/无效/未签名三态徽标。 |

## 已知阻塞（需后续后端任务，非前端能解）

- **主动学习闭环回流 UI 未接线**：`active_export` 必须传 `defects` 列表，但 `ReportOut` 不含缺陷明细（仅 `defect_count`）。前端结果视图无法取得 defects → 强行导出会写空标注污染训练池。建议后端在 `ReportOut` 暴露 `defects`（或新增 `GET /report/{id}/detections`），再补「回流训练池」按钮。当前 ArchiveView 已只读展示训练池状态。

## 未自动运行（如实标注）

- `yarn lint`（ESLint）：配置已接线但未实跑，全量启用前可能需一次存量清理。
