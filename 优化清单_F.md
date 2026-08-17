# 优化清单 F（第二轮审计 · 2026-08-17）

> 基于已提交代码（含主动学习闭环、ESLint/README 收尾）重新审计。标记【真bug】= 当前会导致功能失败或明确安全缺口；其余为改进项。
> 门禁现状：import-linter 2 kept/0 broken；pytest 324 passed/4 skipped（76.49%）；vue-tsc 通过；vite build 通过；eslint 0 problems（ruff 本机未装，前端 yarn.lock 已提交）。

---

## P0 — 真 bug（建议立刻修）

| ID | 位置 | 问题 | 建议 | 改动 |
|----|------|------|------|------|
| F1【真bug】 | `_pkg/build_installer.py:21` ↔ `_pkg/assemble_final.py:26` | 前者产出 `ScanDetection_Setup.exe`，后者却要求 `安装程序.exe` → 跑 assemble 必 FAIL「缺少安装程序.exe」 | 统一文件名常量（建议在 `_pkg/__init__` 或共享 `common.py` 定义 `EXE_NAME`），两脚本 import 同一常量 | 小 |
| F2【真bug·安全】 | `backend/app/routers/report.py:152` | `pdf = Path(str(rep["pdf_path"]))` 直接用 DB 存储路径，未用 `safe_resolve` 约束到 reports_dir；若存了绝对路径可越界读文件 | 复用 `auth.py` 的 `safe_resolve(reports_dir, rep["pdf_path"])`；DB 改存相对名 | 小 |
| F3【真bug·CI】 | `.github/workflows/ci.yml:49-51` | `ml-tests` 设 `continue-on-error:true`，真实检测器零强制覆盖，门禁"虚假信心" | 拆 `ml-smoke`（仅 onnx 加载，必过）+ `ml-heavy`（需权重，可失败）；或后端 job 直接装 onnxruntime 纳入覆盖率 | 小 |

## P1 — 安全与配置（中高价值）

| ID | 位置 | 问题 | 建议 | 改动 |
|----|------|------|------|------|
| F4 | `requirements.txt:7-23` | 依赖全 `>=` 未锁版本，供应链不可复现 / 投毒风险 | 锁 `==` 版本 + `--require-hashes`（`pip freeze --require-hashes`） | 小 |
| F5 | `backend/app/routers/auth.py:88` | 登录无暴力破解防护（无失败锁定、无登录专用限流；全局限流可被 `SCAN_RATE_LIMIT=0` 关） | 登录专用限流 + 连续失败锁定账户/延迟 | 中 |
| F6 | `backend/app/auth.py:123-166` | token 无刷新 / 无服务端吊销，24h 内无法主动失效（不能登出/撤销） | 引入 jti 黑名单或短期 token + refresh | 中 |
| F7 | `backend/app/auth.py:80-101` | `os.chmod(0o600)` 在 Windows 是 no-op，`.auth_secret` 对其他本机用户可读 | 生产强制 `SCAN_AUTH_SECRET` 注入；Windows 用 `icacls` 设 ACL | 小 |
| F8 | `main.py` / `evaluation.py:31` / `active.py:113` | 仅 multipart 上传有体积上限；JSON 端点（drift/export 列表）无上限 → DoS 面 | 加请求体总大小中间件 + 列表长度/字段上限校验 | 小–中 |
| F9 | `main.py:126-136,218-236` | CORS 硬编码开发源 `localhost:5173` 进生产；SPA 回退可服务 DIST 任意文件（含 .map/源码） | CORS 白名单按环境读取；SPA 回退仅放行白名单扩展名 | 小 |

## P2 — 性能与可靠性（高影响）

| ID | 位置 | 问题 | 建议 | 改动 |
|----|------|------|------|------|
| F10【高】 | `backend/app/batch_queue.py:241-265,307` | 持锁同步全量写盘 + O(N²) I/O：每次状态变更把整批 JSON 写盘，N 张图≈2N 次全量写，写盘期间全局锁卡住所有 worker | `_persist` 移出锁（仅序列化快照再写）/ 仅 finish+retry 落盘 + 增量 | 中 |
| F11 | `backend/app/batch_queue.py` + `main.py` lifespan | `shutdown()` 未接 lifespan；`_load_existing` 恢复后非终态任务永久假 running | lifespan 退出调 `batch_manager.shutdown()`；启动对非终态标 failed+retry 提示 | 小 |
| F12 | `backend/app/pipelines.py:376` + `pdf_reporter.py:150,164` | 同图重复解码：pipeline 已 `load_image` 得 gray，reporter 又从磁盘重解码整张大底片 | `build()` 增可选 `gray` 参数，pipeline 直接传已加载灰度图 | 小 |
| F13 | `backend/infra/yolo_detector.py:64` | ONNX 会话未调优：无 `SessionOptions`（线程数/并行）、多 worker 并发 CPU 争用 | 配 `SessionOptions`（按核数设 intra_op）+ 推理信号量限并发 | 小 |
| F14 | `backend/infra/db.py:232` + `repository.py` | SQLite 单写者瓶颈，批量多 worker 写被串行 | 批量写走单写队列 / 串行写线程（或迁移 PG） | 中 |
| F15 | `backend/main.py:188-191` | 缺统一 `HTTPException` 处理器，router 抛的 500/404 走 Starlette 默认体（如 `image not found: /abs/path` 泄露路径） | 补 `@app.exception_handler(HTTPException)` 统一信封 + 脱敏 | 小 |
| F16 | `backend/main.py:107-112` | 非结构化文本日志，不利集中采集 | 换 structlog/JSON formatter（保留关键路径日志） | 小 |
| F17 | `batch_queue.py:100` / `dependencies.py:262` / `pipelines.py:539` | `_batches` 内存无上限；模型热切换并发竞争；复核定案同步重生成 PDF 时延高 | 内存仅保留最近 N；热切换加读锁/原子替换；PDF 改后台线程 | 小×3 |

## P3 — 前端健壮性 / UX（高价值）

| ID | 位置 | 问题 | 建议 | 改动 |
|----|------|------|------|------|
| F18【高】 | `src/src/services/api.ts:80` + `App.vue` | 仅 BatchView 局部有离线提示，Archive/Device/登录失败无全局离线横幅 | App.vue 监听网络错误事件派发全局 toast | 中 |
| F19【高】 | `src/src/services/api.ts:55,166,221` | 上传/批量提交同为 30s 超时且连接拒才重试，大文件/批量易超时不可重试 | 上传请求单独超时（120s）或流式进度 | 中 |
| F20 | `src/src/views/ArchiveView.vue:37-70` | 档案分页切回 `active` 时旧请求后到覆盖新响应（竞态） | 加 `reqId` 守卫，只接受最新响应 | 小 |
| F21 | `src/src/components/UploadPanel.vue:42-44` | objectURL 仅重选时 revoke，`reset()` 不清理 → 内存泄漏 | `onUnmounted` / reset 中 `revokeObjectURL` | 小 |
| F22 | `src/src/views/ArchiveView.vue:26-35` | 训练池加载失败静默置 null，无重试入口 | 补重试按钮 | 小 |
| F23 | `src/src/components/ReportView.vue:117-119` | 回流导出中关闭弹窗丢结果 | 导出中禁用关闭 / 二次确认 | 小 |
| F24 | `BatchView.vue:115` / `UploadPanel.vue:72` / `DeviceView.vue:83` | 母材厚度/像素标定仅判非空，非数字/负值到后端才 422 | 前端加 `Number.isFinite`+范围校验即时反馈 | 小 |
| F25 | `ReportView.vue:307-323` | 导出弹窗无 `role="dialog"`/`aria-modal`/ESC/焦点陷阱 | 补 a11y | 小 |
| F26 | `ReportView.vue:47-49` | `window.open(pdf_url)` 在 Tauri 中被拦截静默失败 | try/catch + 失败提示 / 改用 Tauri opener | 小 |
| F27 | `main.ts` / `App.vue:13` | 缺全局 `unhandledrejection` 兜底；AUTH 监听未 `removeEventListener` | 加兜底 + 单例管理监听 | 小 |

## P4 — 工程化 / CI

| ID | 位置 | 问题 | 建议 | 改动 |
|----|------|------|------|------|
| F28 | `.github/workflows/ci.yml:32-44` | 前端 eslint 从未进 CI，TS/JS 无护栏 | frontend 加 `yarn lint`，改为 `vue-tsc && eslint && vite build` | 小 |
| F29 | `pyproject.toml` / `requirements.txt` | 后端无 uv.lock/constraints，CI 每次解析最新，不可复现 | 引入 uv.lock + `--require-hashes` | 中 |
| F30 | `.pre-commit-config.yaml` | 仅 ruff+通用钩子，缺 lint-imports/eslint/prettier | 补 import-linter + 前端 eslint/prettier 钩子 | 小 |
| F31 | `backend/app/pipelines.py`（576 行） | 无专门单测，检测→量化→评级→报告编排层回归难捕获 | 补 `test_pipelines.py`（mock 检测器） | 中 |
| F32 | `backend/tests/conftest.py:63-81` | 全局 admin 覆盖 `get_current_user`，非 admin 403 路径仅局部验证 | 对 audit/models/records 补「非 admin→403」用例 | 中 |
| F33 | `README.md:31,28` | 数字过期（320→实测 326；"18 路由"实为 39 端点） | 改为 326 + 注明端点数 | 极小 |
| F34 | `README.md:74-78` ↔ 本机 | README 指示 `backend/.venv/Scripts/ruff`，但本机 venv 仅装 import-linter | 确保 `.[dev]` 含 ruff 或用 pre-commit(ruff-pre-commit 自带) | 小 |

---

## 建议优先顺序（若让我"按优先级修"）
1. **P0 三连**：F1（打包真 bug）、F2（PDF 越权读）、F3（CI 虚假门禁）—— 都是小改动、高确定收益。
2. **P2 高影响**：F10（批量写盘放大）、F11（假 running）、F12（重复解码）—— 批量场景体验与吞吐直接受益。
3. **P3 前端高价值**：F18（全局离线）、F19（上传超时）、F20/F21（竞态+内存泄漏）。
4. **P1 安全**：F4（锁依赖）、F5/F6（登录+token 吊销）、F7（Windows 密钥权限）。
5. **P4 工程化**：F28（eslint 进 CI）、F29（uv.lock）、F31（pipelines 单测）。

> 注：F1/F2/F3/F10/F11/F12/F18/F19/F20/F21 为"改动小且收益明确"的快赢；F5/F6/F14/F29/F31/F32 改动量中，建议单独成批。
