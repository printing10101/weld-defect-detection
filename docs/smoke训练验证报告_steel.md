# Smoke 训练验证报告：steel + synthetic 数据可用性与裂纹检出

**日期**：2026 ｜ **目的**：验证公开数据（steel-tube）接入后训练管线可用，且裂纹（最稀缺类）可被检出。

---

## 一、数据

| 源 | 图像 | 说明 |
|---|---|---|
| steel-tube（GitHub，gh-proxy 镜像下载） | 2699 | 真实灰度 X 光钢管焊缝底片（958×653），8 类→6 类映射（跳过断弧/夹珠/焊瘤） |
| synthetic（既有） | 300 | 合成缺陷图 |
| **合计** | **2999** | train 2403 / val 303 / test 365 |

训练集类别分布（train labels）：气孔 4758 / 未熔合 413 / 裂纹 171 / 夹渣 157 / 咬边 94 / 未焊透 79。

---

## 二、训练配置（smoke，非调优）

| 项 | 值 |
|---|---|
| 模型 | yolo11n.pt（预训练起点） |
| epoch | 30 |
| imgsz | 640 |
| batch | 16 |
| 设备 | GPU（torch 2.6.0+cu124，RTX 系列） |
| 单 epoch 耗时 | ~25s |
| 产物 | runs/smoke_steel/steel_synth/（best.pt 5.2MB） |

---

## 三、结果

### 训练收敛
- loss 全程下降（cls_loss 4.80→1.02）；val mAP50 随 epoch 升至 0.988（30 epochs 末）。

### test 集评估（365 张，1173 框）

| 指标 | 值 |
|---|---|
| mAP50 | **0.976** |
| mAP50-95 | 0.777 |
| Precision | 0.966 |
| Recall | 0.964 |

### 裂纹（class 4）专项检出 —— 用户最关心

test 集裂纹 GT：**51 框 / 45 图**；conf 阈值扫描：

| conf | 裂纹召回 |
|---|---|
| 0.05 | **51/51 = 100%** |
| 0.10 | **51/51 = 100%** |
| 0.25 | **51/51 = 100%** |
| 0.50 | **51/51 = 100%** |

→ **裂纹在所有阈值下 100% 检出**，且总体 Precision=0.966（误检低）。

---

## 四、结论与边界（务必如实理解）

**结论**：
1. ✅ **数据可用性验证通过**：steel 2699 张接入后训练管线完全正常（无数据错误/标签错乱），30 epoch 即收敛到 mAP50 0.976。
2. ✅ **裂纹检出验证通过**：51 个裂纹 GT 框全部检出。
3. ✅ 复现路径完整：gh-proxy 镜像下载 → steel_tube_ingest.py 转换 → dataset_builder 装配 → 训练 → 评估。

**边界（诚实说明）**：
1. **训练/测试同源**：裂纹 100% 召回是『模型见过同分布数据』的自洽结果；真实价值是证明数据与管线可用，
   **不代表在用户现场定检底片上也能 100% 检出**——域偏移（钢管焊缝 vs 现场底片）仍需用户现场标注数据验证。
2. **smoke 非调优**：30 epoch、无数据增强调优、无类别权重；正式训练需更多 epoch + 均衡采样（气孔 4758 vs 咬边 94 不均衡）。
3. **许可**：steel-tube GPL-3.0（代码），仅限内部/研究使用，不商用（已与负责人确认）。

---

## 五、复现命令

```bash
# 1. 转换（已做）：python -m backend.training.steel_tube_ingest
# 2. 装配（已做）：python -c "from backend.training.dataset_builder import build_dataset; build_dataset()"
# 3. 训练：gpu-venv python scripts/train_smoke_steel.py
# 4. 评估：gpu-venv python scripts/eval_smoke_steel.py
```
