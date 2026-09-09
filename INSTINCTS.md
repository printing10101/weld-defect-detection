# INSTINCTS

## 公开仓本机路径（2026-09-09）

- **触发**：写入校准表、评估 JSON、model card、DATA_LICENSE 等会提交的产物。
- **正确做法**：路径用相对路径或 `<user>` / `<project>` 占位符；提交前 `git grep 'C:\\Users'`。
- **证据**：`best.calibration.json`、`real_baseline.json` 等曾含 `D:\扫描检测软件\...`，清洗后 grep 为空。

## 权重与数据

- **触发**：训练产出 `best.onnx` / 真实底片。
- **正确做法**：权重与真实工业数据不入库；合成评估图可入库。
- **证据**：`backend/models/weights/` 仅有 README 与 calibration 元数据。
