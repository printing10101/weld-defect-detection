# 焊缝射线缺陷检测 — 数据资产与许可证台账

> 用途：记录用户自有数据、可合法用于训练/比赛的外部公开数据集，以及合规使用边界。  
> 更新日期：2026-08-05  
> 重大更新：本机磁盘装不下 SWRD（115.86 GB），已确认 **Roboflow 上的两个小体积、合规 X 光焊缝集** 可作为实际可落地的主训练集补充（见 §2.1）。SWRD 降级为「有大磁盘时的增强选项」。

---

## 0. 数据集横向对比（一表速览）

| 数据集 | 许可 | 体积 | 真 X 光 | 格式 | 本机可落地 | 可用于比赛 |
|--------|------|------|---------|------|-----------|-----------|
| **Roboflow Danila「X-ray Weld Defect」** | **Public Domain** | 416 张(~几十 MB) | 是 | YOLOv8 直下 | ✅ 直接 | ✅ 无限制 |
| **Roboflow Cassius Fro「XrayWeld」** | **CC BY 4.0** | 619 张 | 是 | 实例分割(转 bbox) | ✅ 直接 | ✅ 需署名 |
| SWRD（北理工） | CC BY 4.0 | **115.86 GB** | 是 | 多边形/VOC/YOLO | ❌ C/D 盘装不下 | ✅ 需署名 |
| GDXray+ Welds | NC(仅研究/教育) | 3.5–4.5 GB 全集 | 是 | bbox 文本 | ⚠️ 可下 | ⚠️ 仅教育/无奖 |
| **HF rikkarth（Kaggle CC0 镜像）** | **CC0** | ~2.0k 张 | **否(外观)** | YOLOv8 直下 | ✅ **已下载至 data/external/hf/rikkarth/** | ✅ 仅预训练 |
| RIAWELC | 不明(无 LICENSE) | 24k 张 | 是 | 分类 | ✅ | ❓ 待确认 |

**结论**：实际可执行的主训练集组合 = **Roboflow(Public Domain/CC BY 4.0) + 用户 165 张精标 + 合成增强**，三者都小体积、合规、可比赛，完全绕开 SWRD 115GB 死穴。

---

## 1. 用户自有数据审计

### 1.1 文件位置
`C:\Users\Lenovo\Desktop\扫描检测软件\图片\`

### 1.2 实际内容
| 子集 | 数量 | 状态 | 说明 |
|------|------|------|------|
| `微信图片_*.jpg` | 13 张 | 视觉上已标注 | 微信导出的小图（17–67 KB），部分图片上有红色箭头 + 缺陷名称（如"未熔合"）等 burned-in 标注 |
| `定检.rar` → 解压为 `定检\*.jpg` | 165 张 | **无机器可读标注** | 原始射线焊缝底片，文件名如 `PG101-1-1.jpg`、`PL118-30-4.jpg`，无任何 `.xml`/`.txt`/`.json` 标注文件 |
| **合计** | **178 张** | — | 与用户所称"约 200 张总共"基本吻合 |

### 1.3 类别/结构
- 无机器可读类别分布；无法自动统计 A–G 类。
- `定检` 图像按接头前缀分为 9 组：PG101、PG102、PG103、PG12、PG120、PG121、PG132、PL117、PL118。

### 1.4 精度可行性结论
**仅依靠这些图片，无法保证目标检测任务达到 ≥90% 精度。**
- 若人工标注 165 张后配合外部 X 光集（Roboflow 416+619 张）预训练/联合训练 + 合成增强，检测 mAP@0.5 有望达到 75–88%（取决于标注质量与增强强度）；≥95% 需更多真实样本或更强域适应，属高难目标。

---

## 2. 外部公开数据集清单（按竞赛合法性 + 本机可落地性排序）

### 2.1 首选小体积主训练集：Roboflow X 光焊缝集（推荐实际落地）

#### 2.1.1 Danila「X-ray Weld Defect」— Public Domain（最优先）
| 字段 | 内容 |
|------|------|
| 名称 | X-ray Weld Defect Object Detection (by Danila) |
| 链接 | https://universe.roboflow.com/danila-wjnju/x-ray-weld-defect |
| 许可证 | **Public Domain**（无署名/商用限制，可放心比赛） |
| 规模 | 416 张（train 333 / valid 56 / test 27） |
| 图像类型 | **X 射线焊缝底片**（真 X 光） |
| 预处理 | Auto-Orient + Resize 640×640；增强 3×/flip |
| 格式 | **YOLOv8 txt 直下**（含 data.yaml，开箱即用） |
| 本机可落地 | ✅ 几十 MB，C/D 盘随便装 |
| 比赛可用性 | ✅ **无任何限制** |
| 类别 | 原始名需看其 data.yaml（通常为 porosity/slag/crack 等），接入脚本自动映射到本项目 6 类 |
| 注意事项 | ① Roboflow 有 Cloudflare 防护，**脚本直链拿不到 zip**，需用户侧浏览器/API key 下载（见附录 A）；② 社区标注质量参差，建议训练后抽样核对 |

#### 2.1.2 Cassius Fro「XrayWeld」— CC BY 4.0（补充域预训练）
| 字段 | 内容 |
|------|------|
| 名称 | XrayWeld Instance Segmentation (by Cassius Fro) |
| 链接 | https://universe.roboflow.com/cassius-fro-9ykox/xrayweld |
| 许可证 | **CC BY 4.0**（可比赛，须署名） |
| 规模 | 619 张 |
| 图像类型 | X 射线焊缝底片 |
| 格式 | 实例分割；类别为**匿名 0–4**，需用户提供语义映射才能接入检测训练 |
| 本机可落地 | ✅ 几十~百 MB |
| 比赛可用性 | ✅ 需署名 |
| 用途建议 | 类别匿名 → **助手决定仅作域预训练/特征提取**（不强行并入检测训练，避免错误标注）；推测 0–4 = 气孔/夹渣/裂纹/未焊透/未熔合（顺序待核） |

### 2.2 原首选（本机暂不可用）：SWRD
| 字段 | 内容 |
|------|------|
| 名称 | SWRD — Seam Weld Radiographic Dataset |
| 官方下载 | http://www.tz-ndt.com/#/download |
| 许可证 | **CC BY 4.0**（可训练 + 可比赛，须署名） |
| 规模 | **115.86 GB**（3,675 原始 + 4,930 滑窗） |
| 缺陷类别 | 6 类（气孔/夹杂/裂纹/咬边/未熔合/未焊透） |
| 基准 | YOLOv8m mAP@0.5 = 0.663 |
| 本机可落地 | ❌ C 盘剩 39.6 GB、D 盘剩 119.7 GB（解压需 200 GB+），装不下 |
| 比赛可用性 | ✅ 合规 |
| 处置 | **降级为「日后有大磁盘/分卷下载时的增强选项」**。若未来要启用：可只取原图 3,675 张（约 30–50 GB）而非全量滑窗，或下到 D:\SWRD\ 后流式解压 + 降采样至 640px 再训练 |

### 2.3 已拉取：HuggingFace 镜像 rikkarth（Kaggle Weld Quality 的 CC0 镜像）
| 字段 | 内容 |
|------|------|
| HF 链接 | https://huggingface.co/datasets/rikkarth/welding-defect-object-detection （已通过 **hf-mirror.com** 镜像拉取到 `data/external/hf/rikkarth/`） |
| 原始来源 | Kaggle `sukmaadhiwijaya/weld-quality-inspection-instance-segmentation`（**CC0: Public Domain**） |
| 许可证 | **CC0**（无署名/商用限制，可放心比赛） |
| 规模 | ~2,028 张（train 1619 / valid 283 / test 126） |
| 类别 | `['Bad Weld','Good Weld','Defect']`（3 类粗粒度） |
| 图像类型 | **焊缝外观照片（可见光表面）**，**不是 X 射线** |
| 格式 | YOLOv8 txt 直下（含 data.yaml + COCO 冗余标注） |
| 本机可落地 | ✅ **已下载**（经 HF 镜像 HTTP 直链 + Python 线程池并行，绕开 LFS 协议与 huggingface_hub 库 bug） |
| 比赛可用性 | ✅ 无限制 |
| 用途建议 | **仅作 backbone 预训练 / 域适应辅助**（可见光表面 vs X 光内部，域 gap 大，不直接混入 X 光检测训练集）；阶段 C 训练脚本需支持两阶段（先 rikkarth 预热 backbone，再 X 光数据微调） |

### 2.4 研究/基准（谨慎使用）：GDXray / GDXray+
| 字段 | 内容 |
|------|------|
| 链接 | http://dmery.ing.puc.cl/material/gdxray/ |
| 许可证 | **Research / Educational only（非商业 NC）；禁止再分发** |
| 规模 | GDXray+ 共约 21,100 张，Welds 子集更小；总包 3.5–4.5 GB |
| 比赛可用性 | ⚠️ **仅建议用于无奖/教育性质比赛或纯研究基准**。有奖/商用比赛存在 license 风险 |
| 用途建议 | 预训练 backbone 或方法对比；保留 NOTICE；**不打包进提交物**（redistribution  prohibited → 仅放下载脚本+引用） |
| 决策待确认 | 取决于**用户比赛性质**（见 §5.1）。若比赛为校内课程/无奖教育竞赛，可 argue 为 educational 使用 |

### 2.5 备选分类（需确认 license）：RIAWELC
| 字段 | 内容 |
|------|------|
| 链接 | https://github.com/stefyste/RIAWELC |
| 许可证 | README 写 "released freely"，**仓库无 LICENSE 文件**，授权不明 |
| 规模 | 24,407 张 224×224 PNG |
| 比赛可用性 | ❓ 待定，建议避免直接用于比赛提交 |

### 2.6 不建议/待核实
| 数据集 | 问题 |
|--------|------|
| WDXI | 下载渠道与授权不明 |
| Kaggle yolov5-weld (`aadityadhiwijaya`) | X 射线但**许可证缺失** |
| Roboflow `weld-defect-detection-5hwzo` | 标 CC BY 4.0，但源自 license 不明的 yolov5-weld，上游风险未解除 |
| NEU-DET / MVTec AD / KolektorSDD | CC BY-NC-SA 或学术申请；且为钢板/表面缺陷，非焊缝 X 光 |
| Severstal Steel Defect | Kaggle 竞赛锁定数据，不可挪作他用 |
| CSDN/魔乐社区「3056 张 / 999 张 X 光焊缝」 | 声称 CC 4.0 BY-SA，但**原始来源不明、多为搬运、许可声明不可靠**，作为比赛训练集有合规风险，**不建议**作主训练集 |

---

## 3. 零 License 风险：合成与自生成数据
| 方案 | 优点 | 缺点 |
|------|------|------|
| Copy-Paste 增强 | 把用户已标注小图缺陷抠出粘贴到正常底片；版权归用户 | 需先做少量精确标注 |
| Blender 焊接仿真 | 完全可控缺陷几何；零 license 风险 | 与真实胶片有域 gap |
| CycleGAN / 扩散模型 | 风格迁移到竞赛片 | 需算力；可能生成伪影 |

**建议**：合成数据作为「Roboflow + 用户自有数据」之上的补充，而非唯一来源。

---

## 4. 推荐数据策略（分阶段，已适配本机约束）

### 阶段 A：立即落地 Roboflow 小体积集（当天可跑）
1. 用户侧下载 Roboflow 导出（见附录 A）：优先 Danila（Public Domain），可选 XrayWeld（CC BY 4.0）。
2. 解压到 `data/external/roboflow/<name>/`（含 data.yaml + {train,valid,test}）。
3. 运行 `python -m backend.training.roboflow_ingest --name danila` → 自动映射类别并装配训练集。
4. 保留 LICENSE/NOTICE 到 `data/external/roboflow/<name>/`。

### 阶段 B：用户数据利用（与阶段 A 并行）
1. 用 Label Studio 对 `定检\*.jpg` 进行 bbox 标注（优先危险缺陷：裂纹/未熔合/未焊透/咬边）。
2. 13 张微信小图作标注参考；用户 165 张作目标域 fine-tune（最重要，决定竞赛片上的真实表现）。
3. 与 Roboflow 分开验证，评估域 gap。

### 阶段 C：训练策略
1. **Backbone 预训练**：CC0 Kaggle Weld Quality（外观）或 ImageNet。
2. **主训练**：Roboflow X 光集（Public Domain + CC BY 4.0）+ 用户 165 张。
3. **域适应/微调**：用户标注 + 强增强（Mosaic、MixUp、Copy-Paste、CLAHE、随机黑度）。
4. **合成增强**：copy-paste / Blender 补足危险缺陷与缺类（如咬边可能 Roboflow 不含）。
5. **人工兜底**：保留 `need_review` 机制，低置信/危险缺陷漏检/域外样本强制人工复核。

### 阶段 D：合规提交
- 在代码/报告中保留所有外部数据 NOTICE。
- Danila(Public Domain) 无需署名；XrayWeld(CC BY 4.0) 须署名（见附录 B BibTeX 占位）。
- SWRD（若日后启用）引用格式：
  ```bibtex
  @article{zhao2025swrd,
    title={SWRD: A Dataset of Radiographic Image of Seam Weld for Defect Detection},
    author={Zhao, Xuefeng and Wu, Juntao and Zhang, Baoxin and Wen, Haoyu and Wang, Xiaopeng and Li, Yan and Yu, Xinghua},
    journal={Journal of Nondestructive Evaluation}, volume={44}, number={2}, pages={50}, year={2025},
    publisher={Springer}, doi={10.1007/s10921-025-01186-w}
  }
  ```

---

## 5. 关键决策与风险提示
1. **仅凭 178 张图无法保证 ≥90% 检测精度**；引入 Roboflow(416+619) + 合成后可望 75–88%，≥95% 为高难目标。
2. **Roboflow Danila(Public Domain) 是当前本机最优选**：体积几 MB、真 X 光、YOLOv8 直下、比赛无限制。
3. **SWRD 虽合规但 115GB 本机装不下**，降级为增强选项；切勿为它重买磁盘前先卡住进度。
4. **GDXray(NC) 仅可用于教育/无奖比赛**；有奖比赛避免作主训练集（见 §5.1）。
5. **CSDN/魔乐社区搬运集许可不可靠**，不作主训练集。
6. **竞赛规则未禁止外部公开数据**，但建议最终向组办方确认 Roboflow/CC0 的使用与署名方式。
7. **国内网络现实（2026-08-05 追加）**：Roboflow 被 Cloudflare 拦截，用户侧浏览器卡验证界面，真·X 光合规集（Danila/XrayWeld）**暂时无法下载**；GitHub 直连被墙；百度网盘 `admin1523` 是真 X 光但 **non-commercial**（按最严口径比赛禁用）。当前实际可用合规数据 = **rikkarth(CC0 外观, 已拉) + 用户 165 张精标 + 合成**。若用户能提供梯子/换网络打开 Roboflow，则 Danila(真X光 CC0) 仍是最优解；否则建议直接用「165 张精标 + 合成 X 光」跑通 M4b 全流程，精度目标务实下调（≥95% 难达，先定位小样本基线）。

### 5.1 助手自主判断（用户未知，2026-08-05）
- **比赛性质未知 → 按最严合规处理**：视为"可能有奖 / 企业主办 / 商业竞赛"。据此 **GDXray(NC) 禁用作主训练集**（仅保留为研究参考，不打包进提交物）；主训练集严格限定为 **Public Domain(Danila) + CC BY 4.0(XrayWeld) + 用户自有数据 + 合成增强**。
- **缺陷类别 → 由数据自带语义 + 自动映射决定，无需用户判断**：
  - Danila 自带语义类别名，`roboflow_ingest` 用 `map_source_label` 自动映射到本项目 6 类。
  - XrayWeld 类别匿名 0–4，**助手决定仅作图像域预训练/特征提取**（不强行并入检测训练，避免错误标注）；推测 0–4 = 气孔/夹渣/裂纹/未焊透/未熔合（顺序待核，不保证正确）。

---

## 附录 A：Roboflow 下载与接入命令

> Roboflow 有 Cloudflare 防护，**本机脚本直链会被拦**（返回 "Just a moment..." 挑战页），需用户侧用浏览器或 API key 下载。

**方式一（浏览器，最简单）**：
1. 打开 https://universe.roboflow.com/danila-wjnju/x-ray-weld-defect
2. 点 `Export` → 选 `YOLOv8` → `Download`
3. 把下载的 zip 解压到 `data/external/roboflow/danila/`（保持 data.yaml + train/valid/test 结构）

**方式二（CLI + API key，需免费注册 Roboflow 拿 key）**：
```bash
pip install roboflow
roboflow download --dataset https://universe.roboflow.com/danila-wjnju/x-ray-weld-defect \
  --model 1 --format yolov8 --api_key <YOUR_ROBOFLOW_API_KEY>
```

**接入（本助手执行）**：
```bash
python -m backend.training.roboflow_ingest --name danila
# 可选 XrayWeld（需先补类别映射）：
python -m backend.training.roboflow_ingest --name xrayweld
```

---

## 附录 B：Roboflow 数据集引用占位（CC BY 4.0 须署名）
```bibtex
@misc{xrayweld_dataset,
  title={XrayWeld Dataset}, author={Cassius Fro},
  howpublished={\url{https://universe.roboflow.com/cassius-fro-9ykox/xrayweld}},
  publisher={Roboflow}, year={2024}, month={mar}
}
@misc{xray_weld_defect_danila,
  title={X-ray Weld Defect Dataset}, author={Danila},
  howpublished={\url{https://universe.roboflow.com/danila-wjnju/x-ray-weld-defect}},
  publisher={Roboflow}, year={2024}, month={may}
}
```
> Danila 集为 Public Domain，署名非强制但建议保留以尊重作者。

---

## 附录 C：文件落地结构
```
data/
  external/hf/
    rikkarth/   # HF 镜像 CC0 焊缝外观集（已拉，仅 backbone 预热）
  external/roboflow/
    danila/                 # Roboflow 导出解压于此（data.yaml + train/valid/test）
    xrayweld/               # 可选
  training/raw/
    roboflow/{images,labels}   # roboflow_ingest 写出
    user/{images,labels}       # 用户 165 张精标（Label Studio 导出）
    synthetic/{images,labels}  # 可选合成
    swrd/                      # 仅当 SWRD 启用时
  training/{train,val,test}/{images,labels} + data.yaml  # dataset_builder 最终装配
```
