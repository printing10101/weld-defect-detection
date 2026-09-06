# ScanDetection — 射线焊缝缺陷智能检测系统

针对承压设备焊缝 X 射线底片的缺陷自动识别与评片辅助系统：上传/扫描底片后，
本地完成预处理、缺陷检测、几何量化、按 NB/T 47013.2 自动评级，并生成 PDF/A
评片报告。支持批量评片、人工复核（初评/复评/仲裁）、审计留痕与设备标定档案。

同时内置一套按重庆市地方标准 DB50/T 1807-2025《承压设备射线检测缺陷自动识别
系统评价方法》设计的系统评价能力：标准测试集指标（TDRn/FDRn/MDRn/KDR/WDR/
TDR/底片误报率）、混淆矩阵、L1–L4 系统分级、漏检/误检/误报风险分析与附录 A
评价记录表导出。

## 功能

- **影像接入**：DICOM/DICONDE（含 16bit）/ PNG / JPG / BMP / TIFF，黑度、像质计（IQI）、
  伪缺陷筛查等底片质量校验；SNRn 归一化信噪比与双丝像质计空间分辨率测量；
  透照工艺/设备元数据读取（`GET /api/v1/images/{id}/diconde`）
- **数据脱敏**：DICOM 患者标签与 JPEG EXIF 清理 + 隐私残留审计
  （`python -m backend.training.anonymize_images`，DB50/T 1807 §8.3.2）
- **检测与评级**：YOLO/ONNX 推理，7 类缺陷（气孔/夹渣/未焊透/未熔合/裂纹/咬边/内凹），
  圆形缺陷点数法、条形缺陷限值、综合评级，多标准可扩展
- **人工复核**：逐缺陷/综合级别复核 + κ 一致性 + 仲裁；缺陷增删、类型修改、
  位置调整全程审计留痕
- **报告**：PDF/A 长期归档 + 内容数字签名 + 防篡改校验
- **批量评片**：线程池并行、进度/取消/断点续跑
- **训练侧**：数据集构建（分层划分 + 互斥校验）、三人标注一致性仲裁、
  主动学习、伪标签回流
- **评价体系**：DB50/T 1807-2025 全套指标与记录表（`python -m backend.evaluation.run_std_eval`）；
  规格专项指标——量化一致性（Bland–Altman + 相对误差≤5%）、评级一致率（≥95% 且 κ≥0.8）、
  置信度校准（ECE≤0.05）（`python -m backend.evaluation.run_spec_eval`）
- **底片查看器**：缩放/平移/旋转/镜像/正反片转换/窗位窗宽/锐化/浮雕/双片对比

## 目录结构

```
src/                  Tauri + Vue3/TS 前端
backend/
  app/                FastAPI 路由 + registry + 批量队列
  domain/             领域层：interfaces.py（接口契约）/ detect / preprocess /
                      grade / standards / quantify / review / measure / labeling
  infra/              config / db / repository / crypto / reporting
  evaluation/         评估 harness、标准评价（std501807）与门禁
  annotator/          人工标注器（可选启动）
  training/           训练与数据工具脚本
  configs/            schema.yaml + default.yaml
  tests/              单测 / 契约 / API / 评估门禁
migrations/           Alembic
scripts/              工具脚本
```

## 快速开始（后端）

```bash
python -m venv backend/.venv
backend/.venv/Scripts/python -m pip install -e ".[dev]"
backend/.venv/Scripts/uvicorn backend.app.main:app --host 127.0.0.1 --port 18773
curl http://127.0.0.1:18773/api/v1/health
```

## 快速开始（前端）

```bash
cd src
pnpm install
pnpm dev          # http://127.0.0.1:5173（/api 代理到后端 18773）
```

## 检查与测试

```bash
backend/.venv/Scripts/ruff check backend
backend/.venv/Scripts/ruff format --check backend
backend/.venv/Scripts/lint-imports           # 分层依赖检查
backend/.venv/Scripts/pytest backend/tests   # 覆盖率门禁 70%
cd src && npx vue-tsc --noEmit && npm run test:run
```

CI（`.github/workflows/ci.yml`）在每次推送时执行上述检查。

## 构建安装包

```powershell
# 一键打包（推荐）：裁剪嵌入运行时 → 前端构建 → Tauri 打包 → 输出安装包路径
powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1
# 产物：src/src-tauri/target/release/bundle/nsis/ScanDetection_0.1.0_x64-setup.exe
```

- 安装包**离线自足**：内嵌 Python 运行时与全部后端依赖（fastapi/onnxruntime/
  opencv/国密库等）、WebView2 离线安装器；目标机无需联网、无需管理员权限。
- **模型权重**：打包前把训练产物放到 `backend/models/weights/best.onnx`
  （缺失时应用仍可安装运行，但自动退化为基线检测器并在界面标注降级）。
- 裁剪后的 `src/python_embed` 不能再用于跑后端单测（pytest 已剔除），
  后端测试请使用 `backend/.venv` 开发环境。

## 文档

- [用户手册](docs/用户手册.md) — 面向评片/管理人员的操作说明（GB/T 25000.51 用户文档集）
- [安装与卸载指南](docs/安装与卸载指南.md) — 运行环境要求、安装/升级/卸载
- [部署基线](docs/deployment-baseline.md) — 三员账号引导与运行基线
- [国产化适配矩阵](docs/国产化适配矩阵.md) — 已验证/待真机适配状态
- API 契约：`docs/api/openapi.json`（`python scripts/gen_openapi.py` 重新生成）

## 已知限制

1. **标准数值表**：`nb47013.yaml` 数值转录自公开解读资料（`authorized_copy=false`），
   评级输出仅供参考；取得授权原文复核后置 `authorized=true`。
2. **真实标注数据**：稀有类（裂纹等）真实标注量少，模型在稀有类上的精度
   依赖数据补充与难例挖掘。
3. 单机部署设计（科研自用），操作员仅以姓名标识（`X-Operator-Name`），
   用于报告签名与审计。

## 许可证

本项目采用自定义**非商用许可证**（见 [LICENSE](LICENSE)）：允许个人学习、
科研、教学等非商业目的的使用、复制与修改；**未经版权所有者书面授权，
禁止任何商业用途**（销售、集成入商业产品、对外提供收费检测服务等）。
评片/评级/评价输出仅为辅助参考，不构成法定无损检测结论。
