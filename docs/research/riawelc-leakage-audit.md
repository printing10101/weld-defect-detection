# RIAWELC 官方划分同源泄漏审计报告

> 日期：2026-09-18
> 执行：`scripts/audit_riawelc_leakage.py`（md5 字节簇 + dHash 64bit 感知对 + 源底片分组，口径与 `backend/domain/labeling/dataset_guard.py` 一致）
> 数据：RIAWELC 官方 `Dataset_partitioned`（github.com/stefyste/RIAWELC，19 卷 RAR，逐卷 git blob SHA-1 校验通过）
> 报告：`data/eval/riawelc_leakage_audit.json`（全部数字的可机读事实源）
> 动机：`docs/论文发表可行性评估_三维度_2026-09-15.md` 方向 2-A；2026-09-18 检索确认此前无公开量化审计

## 0. 结论先行

**官方 RIAWELC 测试集（2,443 张）100% 逐字节包含于训练集之中。** 全部 4 个类别的每一张测试图都在训练集里有同名同字节的副本——在该官方划分上报告的任何"测试精度"，实质是**训练集内评估**。此结论不需要任何建模即可证明（md5 逐字节相等），且一条命令可复现。

## 1. 数据完整性

- 下载：19 卷 RAR 分卷（`raw.githubusercontent.com` 断点续传，单连接约 9 MB 即被掐断，需 `curl -C -` 多轮续传）。
- 校验：19 卷全部与 GitHub git blob SHA-1 逐字节一致（0 坏卷）。
- 解压：**24,407 张** PNG（train 15,863 / valid 6,101 / test 2,443），与 RIAWELC 论文宣称总数一致。
- 目录：`DB - Copy/{training,validation,testing}/{Difetto1,Difetto2,Difetto4,NoDifetto}`；本地路径 `data/external/riawelc_extract/`。

## 2. 审计结果（三个维度）

### 2.1 字节重复（硬泄漏）

| 指标 | 数值 |
|---|---|
| 跨 split 字节重复簇 | **2,443** |
| 簇构成 | 全部恰为 `{test/X.png, train/X.png}`（同名同字节） |
| test 集被 train 覆盖 | **2,443 / 2,443 = 100.0%** |
| valid 集被重复 | 0（valid 与 train/test 无字节重复） |
| split 内重复簇 | 0（train 内部 15,863 张全唯一） |

分类别（test 被 train 字节重复）：

| 类别 | test 总数 | 被重复 | 占比 |
|---|---|---|---|
| Difetto1 | 765 | 765 | 100.0% |
| Difetto2 | 632 | 632 | 100.0% |
| Difetto4 | 446 | 446 | 100.0% |
| NoDifetto | 600 | 600 | 100.0% |

注：类名 Difetto1/2/4 与 RIAWELC 论文的 CR/LP/PO 对应关系以论文原文为准，不影响泄漏判定。

### 2.2 源底片分组（结构性泄漏）

文件名编码了来源（如 `RRT-101R_Img2_A80_S5_[3][10].png` = 底片前缀 `_S{段号}_[行][列]` 网格坐标）：

| 分组口径 | 组数 | 跨 split 组 | 占比 | 受污染图像 |
|---|---|---|---|---|
| 底片级（剥掉段号与网格坐标） | 137 | **110** | 80.3% | **24,377 / 24,407 = 99.9%** |
| 分段级（保留段号） | 506 | 457 | 90.3% | — |

注意：137 是"文件名可解析出的同源组"数（前缀含工艺参数如 A80，可能为 底片×参数 组合），不等于文献口径的"约 72 张物理底片"。论文写作时按"137 个文件名同源组"表述，两种口径的差异如实说明。

### 2.3 感知重复（dHash 64bit，汉明 ≤4）

跨 split 感知疑似对 **8,531**，距离分布：d=0: 3,104（含上述字节重复）、d=1: 758、d=2: 1,243、d=3: 1,456、d=4: 1,970。近邻重复主要来自同底片相邻网格 patch（重叠裁切）。

## 3. 影响与解读

1. **官方口径不可用于泛化评估**：test ⊂ train（字节级），任何基于官方划分的 test 指标都是训练集内复述。已发表的 RIAWELC 官方划分结果（含被引 80+ 的下游工作）需要按本审计重新解读。
2. **valid 集是干净的**（无字节重复），但 valid/test 之间与 train 的同源组污染仍达 99.9%（组级口径）——同底片 patch 的纹理/噪声/缺陷形态高度相关，组级泄漏依然虚高性能。
3. **修正路径明确且可行**：按 137 个底片组做组级重划分（如 7:1.5:1.5，分层按类构成），即可构建"干净版 RIAWELC"（可命名 RIAWELC-C）作为论文的数据集贡献；在干净划分上重训代表性模型，即可量化"泄漏虚高了多少个点"——这是方向 2-A 论文的第二块硬实验。

## 4. 复现

```bash
# 1) 获取并校验数据（19 卷 RAR，git blob SHA-1 对齐后解压）
#    见本报告 §1；解压目标 data/external/riawelc_extract/
# 2) 审计（约 2.5 分钟，单进程 CPU）
backend/.venv/Scripts/python scripts/audit_riawelc_leakage.py \
    --root data/external/riawelc_extract \
    --json data/eval/riawelc_leakage_audit.json
# 存在泄漏 → 退出码 1
```

## 5. 局限

- 组键来自文件名解析（`_S{n}_[r][c]` 前缀），无外部 provenance 佐证"同组=同物理底片"；但字节重复维度不依赖该假设。
- 本审计只证明泄漏存在与规模，未量化精度虚高点数（需干净重划 + 重训，见 §3.3）。
- 感知维度使用 dHash（结构敏感、对平滑渐变不敏感）；作为辅助维度，主结论建立在 md5 上。

## 6. 后续工作（论文 2-A 的剩余实验）

1. 组级重划分脚本 + 干净版基准构建（分层按类构成、固定种子、随论文发布划分清单）。
2. 在官方划分 vs 干净划分上重训同一轻量模型（如 YOLOv8n-cls 或 ResNet18），报告逐类精度差 = 虚高量化。
3. 对 SWRD 做同口径审计（需人工下载，见 `backend/training/download_swrd.py` 指引）。
