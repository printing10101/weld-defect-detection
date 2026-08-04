# migrations

数据库迁移目录（Alembic，§7.1 / §19.7）。

- M5 里程碑启用：`alembic init` 后，表结构（images/defects/reports/models）以迁移脚本管理。
- 约束：升级可回滚；schema 变更须与 §7.1 数据模型保持一致。
