# 更新日志

本项目的所有重要变更记录于此。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本遵循语义化版本。当前处于 0.x 阶段，接口仍可能有破坏性调整。

## [Unreleased]

### 新增

- **批量上传查重与人工复核**：批量提交时对每个文件流式计算 SHA256，先批内比对
  （同内容首次出现视为原版）、再与历史已检影像（`images.content_hash` 索引，
  Alembic 迁移 `0010_image_content_hash`）一次 IN 批量比对；命中重复时整批转入
  `awaiting_review` 暂缓态（不消耗推理算力），由人工逐项复核——**跳过**（不重复
  检测不出报告，缺省）或**仍检测**（确认后照常评片归档），确认后批次继续执行；
  全跳过批次直接完成并清理暂存。单张评片链路同样落内容摘要，保证历史比对覆盖
  单图来源。复核动作入不可变审计链（`batch_dedup_resolve`）。可经
  `batch.dedup: false` 关闭；重启后暂缓批保持待复核状态不丢失。
- **逐类温度校准闭环**（§15.4 ECE 修复）：`training/fit_calibration` 在训练留出
  划分（val+test 合并）上以部署同参推理收集校准对，**图像级 2 折交叉验证**
  估计泛化、全量留出数据拟合最终表；分层策略为逐类拟合 → 稀有类池化共享
  温度 → 池化仍不足则诚实恒等。校准表随权重落盘并绑定 model_id 指纹，
  换权重自动失效防污染。**工作点保持设计**：温度只改变输出置信度，类别指派、
  阈值筛选（同变换）与 NMS 排序（原始分）全部保持基线行为——检出集合逐位
  一致，杜绝"校准悄悄改变查全率"（实测发现跨类 NMS 重排序会致 mAP -7.8 点、
  召回 -6.7 点，已修正）。实测（合成域）：部署口径 ECE 0.273→0.185，泛化
  ECE（CV）0.272→0.217；类 4/5/6（裂纹/咬边/内凹）为过自信需软化、类 0-3
  欠自信需锐化——逐类校准方向相反，全局单一温度不可行。受留出数据量限制
  尚未达 0.05 门槛，真实标注恢复后重拟合。
- **端到端稳定性资产**（`scripts/e2e_api_smoke.py`）：真实 HTTP 进程（非
  TestClient）全链冒烟——进程启停/就绪等待、SM2 挑战-响应双角色登录
  （sysadmin+secadmin）、检测（含坏图拒绝）、标准判定、报告生成 + **C-14
  受控导出完整审批流**（申请→保密员批准→一次性令牌→凭令下载 PDF 魔数
  校验）、批量提交/轮询/失败隔离（质量门禁拦截为预期行为）/取消、**循环
  推理内存盯测**（进程树 RSS 泄漏启发式，实测 120 次推理漂移 3MB/1.01 倍）。
- **部署后评估闭环**（`backend/training/post_deploy_eval.py`）：每次部署自动在带标注
  评估集上以**与部署一致的推理路径**产出实测指标（mAP50 / 逐类 AP / 召回 / 精确 /
  ECE 校准）+ 与上次评估的回归对比，并落盘三件套——`data/eval/deployed_eval_<域>.json`
  评估报告、`data/model_cards/<model_id>.json` 模型卡、`data/experiments/experiments.jsonl`
  实验记录（规格书 §7.4/§15：harness 必须留下数字）。评估域按数据路径如实标注
  synthetic/real，禁止把同源合成域指标冒充真实域性能。
- **增广集成不确定性**（`domain/detect/uncertainty.py` + `infer_tta`）：多尺度 TTA 下
  统计每条检出的跨视角检出比例与得分标准差，与单视角启发式 max 融合——"仅单个
  视角冒出"的候选自动获得更高不确定性、更早触发人工复核（Deep Ensemble 认知
  不确定性的测试时增广近似）。
- **真 Grad-CAM**（`domain/detect/gradcam.py`）：torch 后端（训练/验证工作站）下对
  目标缺陷类别回传梯度生成类激活图；多层 hook + 自动选"梯度非零的最深层"，
  兼容 YOLO 多尺度分支（P3/P4/P5）。ONNX 部署路径无梯度，自动回退原 Sobel
  显著性近似——可解释性为辅助功能，任何失败不阻断主链路。
- **嵌入运行时可复现供给**（`scripts/provision_python_embed.ps1`）：python.org 嵌入包
  + 锁定 requirements.txt 一键构建 `src/python_embed`（解开 `import site`、跨解释器
  pip 安装、导入冒烟），打包机与 CI 共用同一条可重复命令；`build_installer.ps1`
  接入为第 0 步，缺失自动构建。
- **Release 流水线**（`.github/workflows/release.yml`）：打 tag 自动构建安装包 →
  端到端冒烟 → 附加到 GitHub Release；无权重时诚实构建"基线降级版"并在产物中
  标注（不签名，SmartScreen 提示见 SECURITY.md）。
- 新增测试：增广集成不确定性（纯函数 + TTA 端到端桩）、Grad-CAM（回退 + ml 真实
  权重定位）、部署评估闭环纯函数，共 30+ 用例。

### 变更

- **产品名统一为中文正名「射线焊缝缺陷智能检测系统」**：窗口标题、网页标题、
  登录页（原误写"射线评片智能检测系统"）、NSIS 安装器/开始菜单/卸载列表显示名
  （原英文 "ScanDetection"）全部对齐；程序文件名经 `mainBinaryName` 保持
  `ScanDetection.exe` 不变，标识符 `com.scandetection.sd` 不变（用户数据目录、
  升级链路不受影响）。安装目录随之变为 `%LOCALAPPDATA%\射线焊缝缺陷智能检测系统`。
- **交付口径收敛为安装包单入口**：`scripts/launch_app.vbs`（打开系统默认浏览器的
  开发调试启动器）与 `stop_app.vbs` 明确标注"仅开发机自用、非交付物"，README 与
  打包脚本写明对外只交付桌面安装包；安装包资源清单本就不含这两个脚本。
- **AI 权重与校准表纳入打包路径**：`best.onnx` + `best.calibration.json` 复制到
  `backend/models/weights/`（打包注入目录，gitignore 内），安装包不再是基线降级版；
  新增《底片扫描与数字化要求》文档（`docs/底片扫描与数字化要求.md`），把评片门禁
  （位深/分辨率/黑度/IQI/伪缺陷/质量分）翻译成扫描作业规范。

### 修复

- `infer_tta` 跨视角集成统计的坐标系错位：views 中误存缩放后坐标系的框，与还原
  到原图系的保留检出做 IoU 匹配会错位、集成不确定性系统性虚高。

### 已知问题（实测数字，见模型卡）

- 当前部署权重（合成域）ECE 校准后 0.216，仍超 0.05 门槛——瓶颈是验证划分仅
  50 张图，稀有类校准对不足；逐类温度校准闭环已就位（`training/fit_calibration`），
  真实标注恢复后重拟合即可收敛。校准表与权重指纹绑定，换权重自动失效防污染。

## [0.1.0] — 2026-09

首个内部交付版本。射线焊缝缺陷智能检测系统：DICOM/DICONDE/PNG/JPG/BMP/TIFF
影像接入、底片质量校验（IQI/黑度/SNRn/双丝）、YOLO/ONNX 检测（tiling/TTA/逐类
阈值）、NB/T 47013.2 评级 + 多标准熔断适配、初评/复评/仲裁复核、PDF/A 报告
（SM2 签名 + 防篡改校验）、批量评片、DB50/T 1807-2025 系统评价、审计双链
（SM3/SHA-256 混合哈希链）、国密静态加密、三员分岗、Tauri 桌面壳（sidecar
自愈/看门狗/单实例/数据迁移）、离线自足 NSIS 安装包 + 端到端冒烟。

### 工程基线

- 分层架构合约（import-linter 强制 app→infra→domain）+ pyright 类型门禁
- 后端 642 用例 / 70% 覆盖率门禁 / Golden Set 检测回归门禁（CI 阻断）
- 前端 vue-tsc strict + ESLint 9 + vitest 逻辑单测
- 用户手册 / 安装卸载指南（GB/T 25000.51）、部署基线、国产化适配矩阵、8 篇 ADR
