# 正式训练验证报告：steel + synthetic + 稀有类均衡

**日期**：2026-08 ｜ **目的**：在 smoke 验证基础上，用稀有类 Copy-Paste 均衡训练集，产出正式模型，
验证均衡对稀有类（尤其未焊透/咬边/裂纹）召回的效果。

---

## 一、与 smoke 的区别

| 维度 | smoke | 正式（本报告） |
|---|---|---|
| 数据 | steel 2699 + synthetic 300 = 2999 张 | + 稀有类 Copy-Paste 均衡 → **train 4641 张** |
| 类别均衡 | 无（气孔 4758 vs 咬边 94 = 50:1） | 稀有类×8（未焊透 626/咬边 766/裂纹 1357/夹渣 1268）→ 12:1 |
| epoch | 30 | 120（**patience 25 早停，101 停止**） |
| 增强 | 默认 | 方案A 验证配置（mosaic 0.5/mixup 0.1/翻转/hsv） |
| 优化器 | 默认 | lr0=1e-3, lrf=1e-2 |

均衡效果（train 集框数）：气孔 9126 / 裂纹 1357 / 夹渣 1268 / 未熔合 815 / 咬边 766 / 未焊透 626。

---

## 二、结果（test 集 365 张 / 1173 框）

| 指标 | smoke | 正式 | 变化 |
|---|---|---|---|
| mAP50 | 0.976 | **0.978** | +0.002 |
| **mAP50-95** | 0.777 | **0.803** | **+0.026**（定位精度提升） |
| Precision | 0.966 | **0.976** | +0.010 |
| Recall | 0.964 | **0.969** | +0.005 |

### 稀有类召回（test 集，conf 0.05–0.5 全阈值一致）

| 类 | test GT | 召回 |
|---|---|---|
| 夹渣 | 53 | **100%** |
| 未焊透（最缺） | 37 | **100%** |
| 未熔合 | 83 | 98% |
| 裂纹 | 51 | **100%** |
| 咬边（最缺） | 51 | 96% |

→ 均衡后**全部 5 个稀有类召回 ≥96%**，且高 conf（0.5）下召回不降（模型对稀有类置信度高）。

---

## 三、训练过程

- 模型 yolo11n（预训练起点），GPU（torch 2.6.0+cu124），单 epoch ~58s。
- **101/120 epochs 早停**（patience=25：best mAP50=0.98979 在 ~76 epoch 达到，之后 25 epochs 未突破）。
- 最终产物：runs/train_steel_v1/steel_balanced/weights/best.pt（5.3MB）。
- 数据卫生：val(303)/test(365) 零增强图污染，评估可信。

---

## 四、结论与边界

**结论**：
1. ✅ 均衡有效：最缺类（未焊透 37→train 626、咬边 51→766）从『几乎不学』变为 **96–100% 召回**。
2. ✅ 定位精度提升：mAP50-95 0.777→0.803（+0.026）。
3. ✅ 整体无回退：mAP50/P/R 全部 ≥ smoke，未因均衡牺牲气孔等大类。
4. ✅ 复现：scripts/train_formal.py（训练）+ scripts/train_formal_step1_rare_aug.py（均衡）+ scripts/eval_formal.py（评估）。

**边界（诚实）**：
1. **同源自洽**：训练/测试同为 steel+synthetic 分布，100%/96% 召回是『模型见过同分布』的自洽结果；
   现场定检底片（域偏移）的最终效果需用户现场标注验证。
2. **augment.py 修复**：本轮发现并修复 cv2.imwrite 在中文绝对路径下静默失败的 bug（改为 imencode+write_bytes），
   该修复对项目所有调用 augment 的路径生效。
3. 类别仍不完美均衡（气孔 9126 仍为未焊透 626 的 14.6 倍）；若要进一步均衡可加重采样或损失加权。
4. 训练在 101/120 早停，未跑满 120；如需更强模型可提高 patience 或加大模型（yolo11s）。

---

## 五、复现命令

```bash
# 1. 均衡增强（已做）：gpu-venv 不需要，backend/.venv 即可
#    backend/.venv python scripts/train_formal_step1_rare_aug.py
# 2. 训练：gpu-venv python scripts/train_formal.py
# 3. 评估：gpu-venv python scripts/eval_formal.py
# 产物：runs/train_steel_v1/steel_balanced/weights/best.pt
```
