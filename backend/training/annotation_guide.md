# 165 张 定检 图标注工作流（域适应微调集）

目标：把用户 `图片/定检/` 下的 165 张焊缝 X 光底片标注为 6 类 YOLO，
作为 SWRD 主训练之后的**域适应微调集**（见 DATA_LICENSE.md 训练策略）。

## 类别（顺序固定，对应 DefectClass / data.yaml）
| idx | 中文 | 英文 | 备注 |
|----|------|------|------|
| 0 | 气孔 | porosity | 圆形暗斑 |
| 1 | 夹渣 | slag/inclusion | 不规则暗团 |
| 2 | 未焊透 | lack_of_penetration | 零容忍 |
| 3 | 未熔合 | lack_of_fusion | 零容忍 |
| 4 | 裂纹 | crack | 零容忍，线状 |
| 5 | 咬边 | undercut | 边缘凹槽（ADR-010 新增） |

## 步骤
1. **预标注（省时）**：运行
   `python -m backend.training.prelabel_user`
   用基线检测器给出缺陷**位置初稿**（类别占位 0），写到 `data/training/raw/user/`。
2. **人工校正**：用 Label Studio 打开 `backend/training/label_studio_config.xml`，
   导入 `data/training/raw/user/images`；逐张校正类别、补漏、修框。
3. **导出**：Label Studio 导出为 **YOLO** 格式，将得到的 `images/` + `labels/`
   覆盖到 `data/training/raw/user/`。
4. **纳入训练**：`python -m backend.training.download_swrd --ingest`（SWRD 在时）
   或 `python -m backend.training.dataset_builder` 会把 swrd + user 合并分层划分。
5. **微调**：见 `backend/models/train.py` 的域适应阶段（freeze backbone + 小学习率）。

## 合规
- 用户自有 定检 图权属归用户，仅用于训练自用模型；不随仓库公开分发。
- SWRD 须按 DATA_LICENSE.md 的 BibTeX 署名（CC BY 4.0）。
- 评测集（test split）须与训练严格隔离（§15.6 Golden Set）。
