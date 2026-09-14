# X 光域视觉基座调研（"预训练先验 + 薄适配器"路线的基座候选）

> 调研日期：2026-09-14
> 目的：为 `docs/adr/ADR-011.md` 中**被推迟的选支②**（是否改用 X 光/医学域基座替代/补充现
> 有 YOLO 路线）提供可核查的事实依据。
> 边界：本 memo **只是调研**，不含任何代码改动，也不构成采纳决定。采纳与否须先经拍板。

---

## 0. 结论先行

| 问题 | 结论 |
|---|---|
| 有没有"工业焊缝 RT 域"的公开基座可直接下载？ | **没有公开权重**。最对口的 WRT-SAM（中国特种设备检测研究院）**未发现开源 checkpoint**；焊缝 RT 自监督方向有代码仓库（Wdsf-ssl）但用的是自建数据。 |
| 有没有"近域"基座可用？ | **有，但都是医学域**。RAD-DINO（胸片 X 光，DINOv2 系）是当前最接近"灰度透射 X 光"的公开基座。 |
| 直接换基座能解决本项目痛点吗？ | **不能单独解决**。本项目痛点是"标注少/边界模糊/稀有类缺样"，而公开基座解决的是"通用表征"。且已有实证：**SAM 在工业表面缺陷上表现不佳**（见 §4.1）。 |
| 有没有比"换基座"更对的借鉴点？ | **有，且已在本项目落地**：WRT-SAM 的**频域 Prompt（DCT）**手法与本项目"灰度 RT + 多尺度缺陷"的处境高度同构，可移植性远高于换基座本身。 |
| 建议 | **先做可行性探针，不要直接上基座**。路径见 §5。 |

**一句话**：基座候选存在（RAD-DINO / SAM 系），但**没有一个是"焊缝 RT 域"的**；而
WRT-SAM 用一个反例证明了**朴素用 SAM 是失败的（IoU 1.13）**，真正起效的是"Adapter +
频域 Prompt + 多尺度 Prompt"这套**轻量适配配方**——配方比基座更值得偷。

---

## 1. 为什么值得看"基座"这条路

本项目现状（`docs/adr/ADR-011.md` 记录）：
- 默认检测器 YOLO 系，监督信号来自 144 张真实底片的预标注 + 合成数据；
- 痛点在**标注质量与稀有类样本量**，不在模型容量；
- 稀有类（裂纹/未熔合/未焊透）在 §15.1 有明确 AP 指标要求。

"预训练先验 + 薄适配器"之所以诱人：如果基座在"灰度透射 X 光"上见过足够多的结构，
那么下游只需要极少标注即可适配。**但这个前提必须被验证，不能假定。**

---

## 2. 候选基座逐项

| 基座 | 出处 | 预训练数据/规模 | 模态相近度 | 与本项目匹配度 | 权重可得性 |
|---|---|---|---|---|---|
| **RAD-DINO** | Microsoft Health Futures，arXiv 2024（Pérez-García 等） | DINOv2（ViT-B/14）在 Multi-CXR 上继续预训练；DINOv2 的 MIM + 实例判别 + KoLeo | **中**：同为灰度透射 X 光，但**人体胸片 vs 工业焊缝**，解剖结构 vs 材料缺陷 | ⭐⭐⭐ 最值得探针 | HF `microsoft/rad-dino`（**许可证冲突，见 §4.3**） |
| **SAM / SAM2 + Adapter** | Meta SAM；WRT-SAM 为焊缝 RT 适配版 | SAM：11M 图 / 1B mask（自然图像） | **低**：自然 RGB，灰度 RT 上零样本很差 | ⭐⭐ 只作为**适配配方的载体** | SAM 权重公开；**WRT-SAM 未发现开源** |
| **MedSAM** | bowang-lab，1.57M 图像-掩膜对 / 10 模态 | 医学多模态（CT/MR/US/X 光…） | **低-中**：模态多样，含 X 光但以解剖为主 | ⭐⭐ 适配配方参考 | 公开，ViT-B：`medsam_20230423_vit_b_0.0.1.pth` |
| **Med-SA（Medical SAM Adapter）** | 参数高效微调，**仅更新约 2% 参数** | 基于 SAM | 低 | ⭐⭐⭐ **方法学参考**（薄适配器范式） | 公开 |
| **S-SAM（SVD 调优）** | 只调权重 SVD 奇异值，**0.4% 参数**，支持 X 光与文本提示 | 基于 SAM | 低 | ⭐⭐ 极低参数预算下的备选 | 公开 |
| **Wdsf-ssl（焊缝 RT 自监督分割）** | NDT&E 2025，Cheng 等 | 自建焊缝 RT 无标注数据 + 对象级掩码策略 + 多头解码分割器 | **高**（同为焊缝 RT） | ⭐⭐⭐⭐ **最同构** | 代码公开：`github.com/longteng-coder/Wdsf-ssl`（数据需自行获取） |
| **WRT-SAM** | arXiv 2502.11338，**中国特种设备检测研究院**（Zhou / Shi / Hao），2025-02，2025-06 更新 | 基于 SAM，Adapter + 频域 Prompt + 多尺度 Prompt | **高**（焊缝 RT 缺陷分割） | ⭐⭐⭐⭐ 方法学最对口 | 论文公开，**未发现 checkpoint** |

---

## 3. WRT-SAM 详解（最对口的一篇，值得逐条拆）

**出处**：arXiv 2502.11338，单位为中国特种设备检测研究院（CSIRI）——即压力容器/焊缝
RT 的行业主管研究机构，与本项目 §1 的应用场景（承压设备）**同一体系**。

**做法**：冻结 SAM 的 mask decoder，通过 adapter 注入任务知识；两个自研 prompt 生成器：
- **FPG（Frequency Prompt Generator）**：用类 FcaNet 的 **2D DCT** 提取频域通道信息，
  再编码成 frequency prompt。动机是**灰度图像缺乏颜色/纹理线索**，频域能区分"低频主体 vs
  高频噪声/细节"，从而把缺陷信号凸显出来；
- **MSPG（Multi-Scale Prompt Generator）**：用深度可分离条形卷积（核 **7 / 11 / 21**）近似大
  核深度卷积，多分支捕获多尺度上下文（对应焊缝缺陷的尺寸跨度）。

**训练**：IoU 损失；AdamW，lr 2e-4；**20 epoch**；输入保持原高、按宽裁剪到 **640px**。

**结果（GDXray，10 张有官方标注的焊缝图）**：

| 方法 | Recall | Precision | AUC | IoU |
|---|---|---|---|---|
| U-Net | 74.34 | 75.97 | — | 66.72 |
| DeepLabv3 | 78.36 | 79.52 | — | 73.54 |
| **原版 SAM（Everything 模式）** | **39.79** | **49.18** | — | **1.13** |
| SAM-Adapter（基线） | 78.87 | 78.39 | 0.9596 | 49.25 |
| **WRT-SAM（本文）** | 78.87 | **84.04** | **0.9746** | 51.36 |

**私有集 WRTD（115 张）**：U-Net++ IoU 63.05 / SAM-Adapter 57.96 / WRT-SAM **59.34**，
recall 79.61（比 U-Net++ 高 11.13 点）。

### 3.1 从这张表里读出的三条硬事实

1. **原版 SAM 在焊缝 RT 上几乎完全失效**——IoU **1.13**。与"SAM 在自然图像上无所不能"
   的直觉相反。**"零样本通用基座直接拿来用"这条路，已被反例打死。**
2. **起效的是适配配方，不是基座本身**：SAM-Adapter 把 IoU 从 1.13 拉到 49.25，
   FPG/MSPG 再推到 51.36、precision +5.65 点。
3. **传统 U-Net 在 IoU 上仍能打赢 SAM 系（66.72 vs 51.36）**，只在 recall/precision/AUC 上
   落后。**"换基座"不是免费升级，在部分指标上会倒退。**

---

## 4. 三类风险（必须提前摆明）

### 4.1 SAM 在工业缺陷上泛化差的实证

《SAM Era: Can It Segment Any Industrial Surface Defects?》在三个工业场景（带钢表面、
瓷砖表面、钢轨表面）系统评测 SAM，结论是 **SAM 在复杂工业场景下达不到令人满意的性能**，
与 13 个 SOTA 方法相比存在明显差距（代码：`github.com/VDT-2048/SAM-IS`）。
→ **不要假定"通用分割基座 + 少量标注"就能解决工业缺陷。**

### 4.2 模态鸿沟**依然存在**（只是比自然图像小）

Marigold V2 的教训（见 ADR-011）：自然图像生成模型的几何先验**不迁移**到射线底片。
医学 X 光基座往前走了一步（同为灰度透射成像），但：
- 胸片是**人体解剖**：肋骨/肺野/纵隔是稳定的结构先验；
- 焊缝 RT 是**材料内部不连续**：气孔/夹渣/裂纹没有"解剖学"意义上的稳定形态，
  不同材料、工艺、透照参数下分布差异极大。
- 医学基座能否提供有用先验，**属于经验问题，必须实测，不能推理**。

### 4.3 许可证与合规（**当前未核实清楚，是硬阻塞**）

- **RAD-DINO**：Microsoft 官方目录标注 **MSRLA**，且明示"仅限研究用途，
  不得用于临床实践"；但部分第三方镜像站标注为 **MIT**。**两者冲突。**
  本项目是**承压设备检测**，若进入商用交付，许可证是硬门禁。
  → **采用前必须核对原始许可文件，不能采信二手标注。**
- **WRT-SAM**：未发现 checkpoint 发布，规避了许可问题，但也意味着**无法直接复用权重**。
- **MedSAM / SAM**：Apache-2.0 / 各自条款，需按实际版本核对。

---

## 5. 建议路径（按 ROI，**待拍板后执行**）

**不建议**：直接下载 RAD-DINO 当骨干、替换 YOLO。

**建议按序做三件事**：

1. **① 频域 Prompt 手法移植（最高 ROI，不依赖任何基座）**
   WRT-SAM 的 FPG 用的是**标准 2D DCT + 通道分组**，本质是一个即插即用的通道注意力模块。
   本项目已有 §5.x 预处理链路与 YOLO 检测器，把 DCT 通道注意力加进骨干是**低风险、可量化**
   的改动，且动机（灰度图缺色彩线索 → 频域补信息）与本项目处境完全一致。
   → 产出：可插拔模块 + Golden Set 回归门禁对比。

2. **② 1 页可行性探针（用 RAD-DINO 冻结特征做线性探测）**
   不训练、不微调：拿冻结的 RAD-DINO patch 特征，在本项目 144 张底片上跑一个**线性探测 /
   最近邻**，看缺陷区域与正常区域是否可分。**如果线性探测都不行，后面全都不用谈。**
   这一步成本极低，却能一次性回答"医学域先验到底有没有用"。
   → 产出：探针脚本 + 可分性数值（AUROC / 线性探测准确率）。**这是决定性的负向筛子。**

3. **③ 仅当 ② 为正，再考虑"冻结骨干 + 薄适配头"**
   参照 Med-SA（~2% 参数）或 S-SAM（0.4% 参数）的参数预算，避免全量微调。

**明确不做**：以"跑得动"为由跳过 ②，直接上基座——那正是 ADR-011 记的坑。

---

## 6. 与本项目直接相关的两条旁证（值得单独跟踪）

- **焊缝 RT 自监督预训练**（NDT&E 2025）：用无标注 RT 数据做自监督预训练 + 对象级掩码 +
  多头解码分割器，证明"无标注数据能提升缺陷分割"。代码 `github.com/longteng-coder/Wdsf-ssl`。
  → 对应本项目可做的**自有数据自监督预训练**（无需外部基座，也无需许可）。
- **扩散模型做焊缝 RT 无监督检测**（J. Manufacturing Processes, Vol 160, 2026-02-28, pp 429-441）：
  残差学习的扩散重建（H-DiffuM）+ 多尺度频域注意力（MFDAFM），报告像素级 **AUROC 97.80% /
  AP 93.34%**，称超越当时无监督 SOTA。
  → 这是 Marigold V2 扩散谱系在**焊缝 RT 上的同域落地**，比借鉴自然图像扩散模型更靠谱；
  但需 GPU 且训练成本高，列入跟踪、不作为近期动作。
- **少样本焊缝提取 + 无监督缺陷检测**（IEEE Access, Vol 14, 2026, pp 32395-32406，
  DOI 10.1109/ACCESS.2026.3667521）：用 **183 张正常焊缝图**训练无监督检测，
  报告 94.5% 准确率 / 95.7% 召回，且**无需像素级标注**。
  → 对"稀有类缺样本"是本项目最现实的旁路：正常样本远多于缺陷样本。

---

## 7. 未核实事项（不猜，如实列出）

| 事项 | 状态 |
|---|---|
| RAD-DINO 确切许可证 | **冲突未决**（官方目录 MSRLA vs 镜像站 MIT），采用前须核对原始许可 |
| RAD-DINO 训练集规模 | 两处口径不一致：HF 模型卡列公有 5 集合计 **882,775** 张；论文口径为 Multi-CXR **838,336** 张。引用前须核对原论文 |
| WRT-SAM checkpoint | 未发现发布 |
| 上述论文指标的可复现性 | 未复现（GDXray 焊缝类仅 68 张、其中 10 张有官方标注，样本量极小，指标波动区间未知） |
| 本项目 144 张底片上的实际收益 | **完全未知**，须由 §5 的②探针回答 |

---

## 8. 参考来源（可核查）

1. RAD-DINO 模型卡（Microsoft / HF）：https://huggingface.co/microsoft/rad-dino
2. RAD-DINO 论文条目：Pérez-García F. 等，*RAD-DINO: Exploring Scalable Medical Image Encoders Beyond Text Supervision*, arXiv 2024
3. Microsoft Azure AI 目录 RAD-DINO 卡（许可证 MSRLA 出处）：http://ai.azure.com/catalog/models/microsoft-rad-dino
4. WRT-SAM：Zhou Y., Shi K., Hao G.（中国特种设备检测研究院）*WRT-SAM: Foundation Model-Driven Segmentation for Generalized Weld Radiographic Testing*, arXiv **2502.11338**：https://arxiv.org/abs/2502.11338
5. 工业表面缺陷上 SAM 的评测：*SAM Era: Can It Segment Any Industrial Surface Defects?*（代码 https://github.com/VDT-2048/SAM-IS）
6. MedSAM：https://github.com/bowang-lab/MedSAM ；微调工具 https://github.com/mazurowski-lab/finetune-SAM
7. 焊缝 RT 自监督分割：Cheng H., Jiang H., Zhang Y. 等，*A weld defect segmentation framework based on visual self-supervised learning*, **Nondestructive Testing and Evaluation**, 2025, DOI 10.1080/10589759.2025.2595520 ；代码 https://github.com/longteng-coder/Wdsf-ssl
8. 扩散重建无监督焊缝检测：Chen X., Zi B. 等，*An unsupervised welding quality detection method based on high-quality condition-guided diffusion reconstruction*, **J. Manufacturing Processes**, Vol 160, 2026-02-28, pp 429-441
9. 少样本 + 无监督焊缝检测：Lee S.-H. 等，**IEEE Access**, Vol 14, 2026, pp 32395-32406, DOI 10.1109/ACCESS.2026.3667521
