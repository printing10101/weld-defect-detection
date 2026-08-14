# migrations

数据库迁移目录（Alembic，§7.1 / §19.7）。

## 用法（P2-7 已启用）

```bash
# 应用全部迁移（SCAN_PATHS__DB_PATH 指定目标库，缺省 data/scan.db）
python -m alembic -c migrations/alembic.ini upgrade head

# schema 变更后生成新迁移（须在**空库或已迁移库**上，避免对旧表生成增量碎片）
export SCAN_PATHS__DB_PATH=/path/to/fresh.db
python -m alembic -c migrations/alembic.ini revision --autogenerate -m "describe change"

# 回滚一级
python -m alembic -c migrations/alembic.ini downgrade -1
```

## 约束

- 升级可回滚；schema 变更须与 §7.1 数据模型保持一致。
- `env.py` 以 `backend.infra.db.Base.metadata` 为 autogenerate 基准，与应用模型同源，
  避免"迁移脚本与模型分叉"。
- 应用启动仍以 `Base.metadata.create_all` 兜底（对已存在旧库幂等，不重复建表）；
  新库建议先跑 `alembic upgrade head` 以获得版本化 schema。

## 现有迁移

- `aea14cec7369_initial_schema.py`：全量初始 schema（images/defects/reports/reviews/audit_log + 索引）。
