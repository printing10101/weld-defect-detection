# 更新日志

本项目的所有重要变更记录于此。格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本遵循语义化版本。当前已发布 1.0.0。

## [1.0.0] - 2026-09-17

### 修复

- **翻拍照片永远显示「不可评片」（定检照片事故）**：`run_inspection` 的
  `evaluable` 直接取全部门禁的与结果，而照片必为 8bit（位深硬门禁
  `allow_8bit=false` 必拒）且黑度/IQI 不可测必挂——翻拍降级只豁免了
  "阻断"，没重算 `evaluable`，于是任何翻拍照片都 `evaluable=false`：
  前端横幅显示「不可评片」、评级被熔断，翻拍可评特性形同虚设。
  修复：翻拍口径（`photo_policy=warn`）下 `evaluable` 重算为仅严重伪缺陷
  否决（与 `/verify` 端点既有口径对齐）；级别经 AI 预筛通道输出
  （`grade_preliminary=true`，basis 首条警示强声明）并强制人工复核；
  前端横幅/报告页对 `photo_mode` 显示翻拍降级文案，PDF 结论对
  "有级别+需复核"追加人工复核限定语，不以正式口吻裸判合格。
  测试长期未暴露的帮凶：conftest 注入 `SCAN_GATE__ALLOW_8BIT=true`，
  位深门禁在测试环境恒放行。回归锚定 `test_run_inspection_photo_film_advisory`
  新增断言（evaluable/预筛级别/警示声明/落库口径一致）。
- **CORS 移到中间件最外层（同类误报的收口）**：CORS 只装饰"流经它"的响应——
  原先它在内层，外层中间件直接返回的响应（IPC 令牌 401、限流 429）同样没有
  `Access-Control-Allow-*`，跨源页面一律拦成 TypeError。典型场景：后端重启
  令牌刷新的窗口期里，页面旧令牌的每个 401 都被前端误报成"无法连接本地推理
  服务"，用户以为断连实际该重新登录。现在中间件顺序为 UnhandledException →
  SecurityHeaders → RateLimit → IpcToken → Metrics → CORS（最外），一切错误
  响应（500/401/429）对跨源页面可见、可被前端按状态码正确处置；代价仅为
  CORS 短路的 OPTIONS 预检不再带安全头/进指标（无业务内容，可接受）。
  回归测试锚定 IPC 401 的 CORS 可见性（`test_ipc_401_response_carries_cors_headers`）。
- **迁移自愈补"版本已到 head 但物理列缺失"场景**：安装版事故的精确状态
  （历史版本被 stamp head 跳过 DDL、版本号与物理 schema 脱节）此前只有
  "无版本表"路径被测试覆盖；补 `test_migrate_versioned_db_missing_column_healed`
  锚定该状态下列自愈仍然生效。
- **全库 schema 漂移体检（无残留）**：以 ORM 元数据为真源对开发库与安装版
  scan.db 做全表全列对照，均无缺表/缺列/多余列；各 Store（Security/Carrier/
  Export/Device/GateReject/Atlas）共用同一 `paths.db_path`，单库调和即全覆盖。

### 修复（第一批）

- **安装版评定归档必炸（images.film_no 列缺失）**：遗留 create_all 库
  （无 alembic_version）启动时被 `stamp head` 直接到 0014——版本号推进
  而不执行任何 DDL，`create_all` 只建缺失表、永不补列；单幅评定在归档步
  INSERT `film_no` 即 `OperationalError`（用户可见症状：评定任务失败，
  六步全「已中断」）。修复分两层：`migrate.py` 新增**列级自愈**
  `_reconcile_missing_columns`（每次迁移收尾以 ORM 元数据为真源，对
  "表在、列缺"幂等 ADD COLUMN；upgrade 中途撞表失败也在 finally 补齐），
  并补三条回归测试（遗留库缺列 / 升级失败仍自愈 / 幂等）。
- **服务端 500 被前端误报「无法连接本地推理服务」**：未处理异常由
  Starlette `ServerErrorMiddleware`（固定最外层）回裸 500，不经过 CORS
  中间件、无 `Access-Control-Allow-Origin`，跨源页面把响应拦成 TypeError。
  新增 `UnhandledExceptionMiddleware`（CORS 内层）把异常转统一错误包
  `INTERNAL_ERROR` 500 后照常流经 CORS 出栈，真实故障不再伪装成断连。
- **前端耗时提示与实测脱节**：评定进行页「预计耗时 15–30 秒」改为
  「5–15 秒（超大底片略久）」，与优化后实测对齐。

### 性能

- **单幅评定端到端压入 15 秒内**（实测：2048×2600 ≈4.4s、3000×8000
  ≈10.4s、1200 万像素照片 ≈6.8s；此前同口径分别为 63.8s/258s/78.8s）：
  - **静态加密换原生后端**：SDC2 信封的 SM4-CTR 与 HMAC-SM3 从 gmssl
    纯 Python（~100-200 KB/s，大底片单张归档加密数十秒，是端到端最大
    瓶颈）迁到 `cryptography` 的 OpenSSL 后端；SM2 签名仍用 gmssl。
    信封格式与密钥流逐字节不变（存量密文互解），新增 gmssl 参考实现
    逐字节对照的兼容锚点测试。落盘加密 157s → 0.35s（3000×8000）。
  - **BRISQUE 特征评估降本**：MSCN 高斯卷积从 scipy `convolve2d`
    （单线程 float64）改 `cv2.filter2D`（SIMD+多线程）；特征评估长边
    上限 2048 等比降采样（特征为信息性输出，门禁判定走 RQI 不受影响）。
    质量门禁 16.8s → ~3s（3000×8000）。
  - 新增 `scripts/profile_single_eval.py` 分段耗时画像（生产口径：YOLO
    ONNX 分块 + 印字 OCR，按 run_inspection 真实阶段计时）。

### 修复

- **代码审查修复第一批（审查发现的可靠性/资源/可观测缺陷）**：
  - **/detect 印字区屏蔽留痕**：预检链路此前 `detections, _ =
    filter_stamp_zone(...)` 静默丢弃被屏蔽列表，违反 stamp.py「屏蔽不
    静默」契约；`DetectResponse` 新增 `stamp_zone_masked`/`warnings`
    （文案与评片主链路同口径）。
  - **Registry 懒建单例加锁**：`gate_reject_store()`/`atlas_store()` 无锁
    check-then-act，批量 worker 并发首访可各建一个 store（多出的
    SQLAlchemy engine 永不释放）——同一问题此前已在 pipelines 显式修复，
    这是同形状残留；现持 `_lazy_store_lock` 双检。
  - **评片孤儿副本治理**：原图副本落盘从第 3 步移至第 6 步落库前一刻，
    落库失败即回收副本；`_write_encrypted_copy` 写失败回收半截密文——
    消除「检测/判定/落库任一失败留下无台账孤儿密文，批量长跑静默占满
    磁盘」的累积路径。
  - **影像加密落盘/备份改流式**：`crypto.encrypt_stream`（SDC2 信封与
    一次性 encrypt 同 nonce 逐字节一致，增量 HMAC-SM3 基于 gmssl 压缩
    函数 `sm3_cf` 搭建并经等值测试锚定）；大底片不再整文件进内存（原
    峰值 ≈2×文件大小×并发 worker 数）。备份暂存改流式哈希+写盘，
    mkstemp 临时 zip 加 try/except 清理兜底。
  - **训练标注原子写**：pool_store tmp 名含 pid+线程 id——复核自动回流
    与 `/active/export` 并发写同一 stem 时，固定 `.tmp` 名会互相交错写，
    最后一次 replace 把损坏标注原子转正被训练静默消费。
  - **静默失败补告警**：报告免责声明表加载失败（恰是最需强声明的场景）
    从静默空串改为 warning 留痕；yolo 越界类别索引从静默折算成气孔改为
    丢弃+告警（防权重/类别表版本错配被掩盖，同时按类别预算阈值表替代
    ~8400 锚框逐个 Python 循环）；llama-server 日志轮转失败、
    `sync_io.count()` 句柄显式关闭；std_eval 记录 JSON 损坏从 500 改
    422 错误信封。
  - **前端消费补齐**：评片结果页新增告警面板 `ReportNotices`（门禁降级/
    印字区屏蔽/单张查重命中——后端已返回但 UI 此前零展示）；复核面板
    展示 `training_pool_synced=false` 回流失败告警（此前只进后端日志，
    专家改判滞留训练池无人察觉）；`ExportRequestOut.status` 建模为
    `pending|approved|rejected|consumed` 联合类型（此前裸 string+魔法
    字符串比较，安全审批链零编译期保护）。

- **代码审查修复第二批（性能/数据完整性/前端消费）**：
  - **/detect 掩膜单遍化**：`quantify.refine_and_quantify`——refine 与量化共用一次掩膜流水线（此前同一缺陷的高斯+双重自适应阈值+形态学+轮廓每缺陷跑两遍，/detect CPU 直接翻倍），且框与几何同源同一轮廓；共享判据由拆分路径逐行提取，bbox 输出与 `refine_detections` 逐值一致。
  - **批次快照轻量投影**：`_result_projection` 只保留 status 接口与收尾钩子消费的字段（完整结果已落业务库）——消除「每任务完成全量重写快照」的 O(N²) 落盘与轮询深拷贝随批规模线性膨胀；存量快照整包结果为超集，向后兼容。
  - **缺陷图谱数据完整性**：409 重复发布回收刚落盘的局部图（不再累积孤儿密文）；DELETE 先删文件再删台账，文件删失败中止撤销（行删后 crop_path 不可查，孤儿文件永久无法定位）；解密失败与样本缺失区分（密钥丢失/信封损坏 → 422 CROP_DECRYPT_FAILED，不再混同 404 整库静默变砖）。
  - **前端**：批量状态建模为 `awaiting_review|running|paused|finished` 联合类型；LlmView 状态轮询加在途守卫（后端卡顿时请求不再逐层堆叠）。

### 新增

- **设备标定自动注入（G23）**：`POST /report` 新增 `device_id` 表单字段——显式未给 `pixel_spacing_mm` 时自动取该设备档案最近一次标定值注入，注入原因/无标定结论进 gate_warnings（前端告警面板可见）；设备不存在 404 显式失败；显式标定优先（调用方口径与 /detect 一致）；无标定照常走未标定熔断语义。
- **独立片号字段（G05）**：迁移 0014 新增 `images.film_no`；`stamp.extract_film_no` 从印字文本保守抽取首个编号样 token（日期不算、两位序号不冒认，宁可 NULL 人工补录）；评片管线落库、《底片评定表》片号列优先消费（未识别到回退原影像短号口径）。

- **真实定检底片端到端暴露的一批问题（8bit JPEG 扫描件批量实测）**：
  - **印字区误检过滤（detect.mask_stamp_zone，默认开）**：真实底片的编号/
    日期铅字与中心标是首要误检源（实测某真实底片 27/27 检出全落印字带，
    焊缝本体零检出——合成训练域未见过边缘印字模式）。印字 OCR 读到的全部
    文本框外扩后，中心落入的检出判为印字误检并屏蔽；屏蔽不静默——数量进
    响应 `stamp_zone_masked`/`warnings` 与审计。实测 PG101-1-1 屏蔽 8 个
    （27→19）。纯函数 `filter_stamp_zone` + `read_stamp_aligned`（印字框
    经胶片区偏移映射回整图坐标，与检测框同系，消除裁剪/整图坐标错位）。
  - **印字识别支持四方向**：此前仅正向/水平镜像；背面装反（180°倒置）或
    翻面扫描的底片印字全部 missing。现补试 rotated/flipped 取最高置信度，
    并把日期模式扩展到真实底片常见的两位年份中文格式（23年1月8日）。
    实测 PG102 系（倒置扫描）从全部 missing 恢复为 present/rotated。
  - **/detect 预检接口补齐胶片区屏蔽**：/report 链路会把胶片区外背景填充
    为胶片中位灰阶，/detect 此前直接整图推理，预览误检多于报告链路（亮
    背景/翻拍边框被误检）。两链路收敛到共享实现
    `detect_film_region_trusted` + `film_background_fill`（domain/film_region），
    消除分叉；/detect 同样应用印字区过滤。
  - **/report 响应透出门禁降级原因**：ReportOut 新增 `warnings/basis/
    density/density_ok/iqi_pass/iqi_detail/photo_mode/detect_mode`——客户端
    此前只拿到 need_review 布尔，看不到"黑度越界/翻拍降级/dpi 未定"这些
    原因（只写在 PDF 里）。前端 types/api.ts 契约同步（前后端契约测试锚定）。
  - **单张评片查重提示**：批量查重的单张对应物——管线落库前按内容哈希查
    历史，命中时响应 `duplicates` 携带记录摘要 + warnings 提示"与 N 张历史
    影像内容完全相同"（仅提示不阻断；实测同一文件字节级重复的两个文件名
    各评一次即命中）。

### 新增

- **技术路线差距修复第一批（对照 2026-09 项目评审 PPT，详见 docs/技术路线差距清单.md，
  本批落地 G01/G07/G13/G14/G17/G18/G20/G21）**：
  - **鲁棒性扰动验证（G07，evaluation/robustness.py）**：亮度增益/偏移、Gamma、
    对比度四族灰度扰动模拟扫描/曝光差异，输出逐条件检出保持率、长边量化偏差、
    新增误检三项稳定性指标与阈值判定（技术路线任务1"质量与鲁棒性验证"）；
    空 GT 判不通过（保守口径）。
  - **条形缺陷中心线长度（G13）**：PCA 主轴分 bin 质心折线测弧长，`length_mm`
    对条形缺陷（长宽比>3）改用中心线口径——矩形长边量"弦"系统性低估弯曲裂纹，
    条形限值评级偏松；圆形/退化回退矩形口径，`Geometry.centerline_mm` 同步输出。
  - **钟点位/轴向位置/最近邻间距（G14/G17）**：/detect 响应新增
    `clock_position`（"H:MM" 半小时精度，请求提供焊缝圆心 `weld_cx/weld_cy`
    时输出，不从底片反推几何）、`axial_position_mm`（胶片长边走向投影，走向由
    胶片区外接框判定）、`nearest_defect_id/nearest_gap_mm`（最近邻边缘间距，
    47013 同线合并 gap 的 2D 推广）；未标定时物理量一律 None。
  - **验收级别参与合格性判定（G20）**：`recommend()` 新增 `accept_level`，
    报告"合格级别"栏（设计/合同要求）现真实参与判定（级别 ≤ 验收级别 → 合格）；
    零容忍/深孔/复核兜底不因验收等级放宽；无法识别的自由文本回退默认口径不
    阻塞出片。评片与重出报告两处接线。
  - **管径上下文通道（G18）**：`ImageMeta`/judge API 新增
    `pipe_outer_diameter_mm`；当前 47013.2 规则库无管径条款，存在时在判定依据
    中记录"Φxx 记录备查"（证明输入被接收且未被使用），为小径管规则留缝。
  - **复核结论自动回流训练池（G21）**：`apply_review` 级别落定后自动把人工
    确认缺陷导出 YOLO 标注并刷新 manifest（此前依赖人工补调 /active/export，
    专家改判滞留业务库）；失败不回滚复核，`training_pool_synced` 字段 +
    ERROR 日志显式暴露。
  - **报告二维码追溯（G01）**：报告每页页脚右下角嵌入追溯二维码
    （`RT-TRACE|report_id|指纹前16位`，不含业务数据），新增
    `GET /report/trace/{code}` 扫码核验端点（档案存在性 + 指纹一致性）；
    生成失败 fail-soft 不阻断出片；依赖 qrcode==8.2。

- **检测工作模式（高检出/高准确双档）**：对标商业评片软件的"双模式"
  实践，`detect.mode` 三档——`balanced`（标定阈值原样）/`recall_first`
  （整体放宽，初筛/复核兜底，漏检代价高的场景）/`precision_first`
  （整体收紧，终审定级压误报）。实现为对 `infer_conf`/`class_conf` 统一
  缩放（`domain/detect/thresholds.py` 纯函数，裁剪到 [0.01, 0.95]），类间
  相对次序保持 ADR-010 标定不变（裂纹最低/气孔最高），避免另维护两份
  逐类表漂移失步。配置键 `mode/recall_conf_scale/precision_conf_scale`
  三处同步（schema.yaml/default.yaml/`DetectCfg`）；评片主链路与
  `POST /detect`（新增 `mode` 表单字段，显式给 `conf` 时以调用方为准）
  均生效，响应与日志带 `detect_mode` 可观测。
- **POD 曲线（按缺陷尺寸的检出概率）**：`evaluation/harness.py` 新增
  `pod_curve`（工作点 POD：真值在操作阈值下存在同类 IoU≥阈值的预测即记
  检出；特征长度 sqrt(面积) 分位数等频分箱，Wilson 95% 置信区间，NDT
  可靠性口径 MIL-HDBK-1823A）。`training/post_deploy_eval` 评估报告新增
  `pod` 节并在模型卡记录 `pod_overall/pod_bins`；最小尺寸箱 POD<0.8
  （n≥10）时模型卡如实声明小缺陷漏检风险。此前 mAP/召回是聚合量，
  "多大的缺陷开始漏"落到尺寸轴上才可回答。
- **缺陷图谱样本库（defect_atlas）**：人工筛选沉淀的典型缺陷样本库
  （培训/比对/复核参考，对标商业评片软件的"缺陷图谱"产品线），与
  defects 事实记录分离、显式发布/撤销并留主审计链。API：
  `POST /atlas`（按源缺陷发布：从落盘影像裁缺陷局部图——支持静态加密
  副本，随 `security.encrypt` 密文落盘到 `data/atlas/`，重复发布 409）、
  `GET /atlas`（类/级别/工件号过滤分页检索）、`GET /atlas/{id}`、
  `GET /atlas/{id}/crop.png`（密文自动解密）、`DELETE /atlas/{id}`
  （必须留撤销原因）。迁移 0013；`AtlasStore`/裁图工具在
  `infra/atlas_store.py`，Registry 懒建单例（模式同 gate_reject_store）。

- **本地大模型随软件启停（llama.cpp / Qwen3-4B）**：新增
  `backend/infra/llm_server.py`（`LlamaServerManager`）把 llama-server
  纳入主应用生命周期——lifespan 装配期后台拉起（独立线程，不阻塞端口
  绑定与 registry 装配，实测 ~9s 就绪），就绪判定 `GET /health` 200；
  应用退出时 terminate→kill 回收。Windows 以 **Job Object
  （KILL_ON_JOB_CLOSE）** 兜底：桌面壳对后端是硬杀、Python 退出钩子
  不执行，由 OS 保证"后端死 → llama-server 同死"（已端到端验证：硬杀
  后端，18780 端口随之释放），Linux 用 PR_SET_PDEATHSIG 同语义；看门狗
  线程按 `max_restart` 上限复活意外退出的进程，`/health` 新增 `llm`
  字段（state/endpoint/error，进程死未及巡检时如实降格 starting）。
  服务收敛于回环：`llm.host` 仅接受 127.0.0.0/8、::1、localhost（非回环
  拒绝拉起/探测），健康探测先 getaddrinfo 解析并校验全部结果为回环再以
  IP 字面量建连（封死 DNS rebinding），子进程 argv 列表 + shell=False，
  关闭 llama-server 内置 Web UI。二进制（`tools/llama/`，CUDA 版
  llama.cpp）与权重（`models/llm/Qwen3-4B-Q4_K_M.gguf`）均不入库
  （.gitignore），随安装包分发；`server_exe`/`model_file`/`n_ctx`/
  `n_gpu_layers` 等全部配置化（default.yaml + schema.yaml + `LlmCfg`，
  env `SCAN_LLM__ENABLED` 可覆盖），二进制/模型缺失、启动超时一律降级
  不阻断主应用启动；测试默认 `SCAN_LLM__ENABLED=false`，生命周期由
  `test_llm_server.py` 以测试替身进程专项覆盖（9 例：状态机/启停回收/
  看门狗重启/超时回收/回环校验）。OpenAI 兼容端点
  `http://127.0.0.1:18780/v1`（对话/补全），供评片辅助判读等上层能力调用。

- **报告补充信息（report_meta）全链路贯通 + 前端按样张分区录入**：新增
  `backend/domain/report/meta_fields.py` 作为字段白名单唯一事实源（工程信息/
  工件概况/技术要求/检测器材及工艺参数 33 键，`sanitize_report_meta` 清洗：
  未知键/空值丢弃、超长截断、宽容不阻断）。API `POST /report` 新增
  `report_meta` 表单字段（JSON 串，非法 JSON 422），落库 `images.report_meta`
  （迁移 0012），`ReportOut` 回显 `report_meta/workpiece_no/weld_no/signer/
  standard_ref`（standard_ref 去除版本号尾注重复）。PDF 汇总表对应空格自动
  填充；表单提供合格级别（验收要求）时按级别序判定合格（Ⅰ<Ⅱ<Ⅲ<Ⅳ），
  未提供沿用 NB/T47013 惯例（Ⅰ/Ⅱ 合格）。前端：`types/api.ts` 新增
  `REPORT_META_GROUPS` 分组字段模式；评片表单新增可折叠《射线检测报告》
  补充信息区（4 组 fieldset 双列栅格，全选填）；报告页新增**样张式首页
  预览表**（与打印 PDF 同源同款 21 列合并网格：委托单位/工程/工件概况/
  技术要求/器材参数/检测情况/结论及说明/签字专用章），标题改为《射线
  检测报告》；pro.css 新增 fset/fgrid 表单组与 rt 样张预览表样式。

### 变更

- **报告版式对齐正式 RT 样张**（`backend/infra/reporting/pdf_reporter.py`）：
  按传统《射线检测报告》纸质样张 1:1 重排 PDF 报告——第 1 页为大标题 +
  `NO:` 编号 + 21 列合并网格汇总表（委托单位/工程名称/工件概况/技术要求/
  检测器材及工艺参数/检测情况分级张数统计/检测结论及说明/检测·审核签字 +
  检测单位检测专用章），软件已知字段（工件/标准/黑度/丝号/级别/缺陷统计/
  签字日期）自动填入，未知工艺字段留空供机构打印后手工补填；第 2 页起为
  《射线检测底片评定表》（序号/焊缝管口编号/片号/黑度/识别丝号/缺陷性质
  与缺陷尺寸/缺陷部位/评定等级/备注），一行一缺陷、同焊缝同片号纵向合并、
  空行补满整页，缺陷尺寸采用样张代号记法（`D:Φ1.4`/`E:L=6.2`，类别映射
  A裂纹/B未焊透/C未熔合/D圆形/E条形/F内凹/G咬边）；末页附图（标注影像 +
  送检底片 + 判定依据/免责声明/防伪指纹，PDF/A 转写与 SM2 签名 sidecar
  链路不变）。页脚改为样张同款『共 N 页　第 M 页』（数字带下划线，含首
  页）；字体由黑体改为宋体（simsun.ttc 优先，与样张同款），正文 12pt、
  大标题 18pt；原封面/注意事项页取消，注意事项中的 AI 辅助声明并入结论
  栏第 4 条。

### 新增

- **训练侧数据泄漏审计**（`backend/domain/labeling/leakage.py` +
  `backend/training/audit_dataset_leakage.py`）：在既有互斥校验（字节 md5 +
  感知哈希）之上引入**同源底片等价类**——过采样副本（`os{N}_`）、跨源同名
  去重副本（新引入 `dup{N}_` 写盘前缀）与 copy-paste 合成图
  （`cp_/rcp_{序号}_{src}_x_{tgt}`）经并查集与亲本底片归并成组（组代表取
  类内最小编号底片名）。评估图只要与训练图同源，指标即被乐观污染
  （RIAWELC 基准 24k 图整体虚高的教训：patch 级随机划分令同一底片的衍生图
  跨 split）。落地三件套：①`build_dataset` 层内**整组划分**
  （`_split_stratum`，同组图像永不跨 split，字节/感知重复因此天然同
  split）；②划分后自动跑泄漏审计，报告落盘
  `data/training/leakage_audit.json`（跨 split 重复簇/感知疑似对/跨 split
  同源组三维结论，构建失败现场也留痕），`enforce_groups=True` 时分组越界
  升级为阻断；③独立 CLI
  `python -m backend.training.audit_dataset_leakage --train ... --test ...`
  （可审计任意目录组，含外部基准复现 RIAWELC 式审计；存在泄漏退出码 1，
  供训练前 CI 拦截）。配套：配置漂移校验将空映射视为已声明叶子
  （`class_review_conf: {}` 不再误报缺失）。
- **逐类复核阈值路由**（`detect.class_review_conf`，键=DefectClass.value）：
  Nb47013Grader 的人工兜底由全局 `review_conf` 升级为**逐类灰区门槛**
  （未列出的类回落全局），`get_grader` 新增 `class_review_uncertainty`
  贯穿装配链（dependencies → registry → grader）。依据：逐类温度校准后
  各类置信度尺度系统分化（§15.4 实测过自信/欠自信方向相反，全局单一阈值
  不可行），u_score 已落校准尺度，逐类阈值才与校准成果对齐。路由触发时
  依据落文本（"检出置信度处于复核灰区，转人工复核：咬边(不确定度0.45>
  阈值0.40)"），报告/审计可解释"为何转人工"；默认空映射 = 行为与历史
  完全一致。

### 变更

- **数据集写盘去重前缀**：跨源同名图（同一张图多源摄入，如 user 与
  synthetic 各有一份 `rare1`）在写盘唯一化时改用 `dup{N}_` 前缀
  （原复用过采样的 `os{N}_`，语义混淆）；`os` 专属过采样副本语义。两者均被
  同源分组剥离。行为影响：跨源同名图因同组必然落入同一 split（历史行为
  为随机散落，构成字节级跨 split 泄漏，只是小样本时仅告警未阻断）。

- **底片印字识别（扫描日期/编号，正向/镜像）**：评片链路新增"底片印字"阶段
  （`backend/domain/stamp.py`，RapidOCR-ONNX 引擎，纯 pip 依赖随包分发），
  在胶片区域上识别透照日期与底片编号印字，作为底片性质落库
  （迁移 `0011_film_stamp`：`images.stamp_status/text/orientation/confidence/
  stamp_need_review` + 批次归属 `batch_no`）供档案检索与追溯。**镜像判定**：
  背面扫描的镜像印字，正向 OCR 常读出"编号样"乱码，单纯阈值挡不住——
  正/镜像双向各跑一次 OCR，按置信度+余量裁决，翻转后命中即记 `mirrored`
  （印字文本取翻转后读数）；日期模式含 `-`/`.`/`/`/年月日与 8 位紧凑格式
  （后两者带年月日合理性校验）。**缺印字处置（两档语义）**：单图评片缺印字
  直接转人工复核（依据写明"未识别到日期/编号印字"）；批量评片**延迟到批次
  收尾按印字占比裁决**（`BatchManager.on_finished` 钩子 +
  `stamp.batch_flag_ratio/batch_flag_min`）——批内有效片 ≥`batch_flag_min`
  且有印字占比 <`batch_flag_ratio` 时视为该批普遍无印字，**豁免缺印字复核**
  （已识别到的印字仍照常落库），占比达标或小批量才补标转复核（need_review
  延迟合并只增不减，不撤销其它来源的复核标记），裁决摘要随批次状态下发并
  入审计链（`batch_stamp_policy`）。功能可经 `stamp.enabled: false` 整体关闭
  （落库 `off`，不参与复核）；引擎缺失/异常降级 `unavailable`，绝不阻断评片。
  前端查阅：检测档案"底片印字"列（正向/镜像绿标、无印字待复核红标、豁免
  灰标）、批量结果逐任务徽标+批次裁决摘要行、底片观察主片/缩略图徽标、
  单图评片结论横幅印字行。
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
  温度 → 池化仍不足则回退恒等映射。校准表随权重落盘并绑定 model_id 指纹，
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
  端到端冒烟 → 附加到 GitHub Release；无权重时构建"基线降级版"并在产物中
  标注（不签名，SmartScreen 提示见 SECURITY.md）。
- 新增测试：增广集成不确定性（纯函数 + TTA 端到端桩）、Grad-CAM（回退 + ml 真实
  权重定位）、部署评估闭环纯函数，共 30+ 用例。

### 变更

- **界面文案全面专业化（NDT 行业术语对齐）**：六大工作区更名——单张检测→
  **单幅评定**、批量检测→**批量评定**、档案检索→**检测档案**、底片查看→
  **底片观察**（设备标定/系统评价不变），菜单/快捷键表/工具栏/页内标题同步；
  术语规范化——母材厚度 T→**母材公称厚度 T**、像素标定→**空间像素标定**、
  黑度→**黑度 D**、IQI→**像质计（IQI）**、综合级别→**质量级别**、焊口编号→
  **焊缝编号**、操作员→**检测人员**、后端→**推理服务/本地推理服务**、训练池→
  **主动学习训练样本库**；流水线阶段名对齐工艺语义（缺陷检出与当量测定、
  标准符合性判定 NB/T 47013.2、评片报告签发 PDF/A 等）；口语文案改写为规范
  表述（「接下来可以做什么」→「后续操作」、「仍检测」→「强制评定」、登出→
  注销等）。仅改显示文案，路由名/ViewId/接口字段/密级映射均未变动；同步更新
  `BatchProgress.spec.ts` 两处文案断言。
- **应用图标重新设计**：桌面/任务栏/安装器图标由通用蓝底"±"占位图更换为领域
  语义设计——深钢蓝底 + X 光底片（齿孔）+ 焊缝亮带鱼鳞纹 + AI 扫描线 + 琥珀色
  缺陷标记环。矢量源文件 `src-tauri/icons/icon.svg`（完整设计）与
  `icon-small.svg`（16/20px 简化变体，小尺寸可读性优化），`icon.ico` 内嵌
  16–256 八档尺寸（16/20 取简化变体，24 起取完整设计渲染），`icon.png` 512px；
  经 `tauri build --no-bundle` 验证嵌入。安装器/桌面快捷方式图标随 exe 资源更新。
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

- **打开应用后长时间"后端未启动"**（冷启动可达五六分钟且前端永远 401）的
  双重根因：① 壳侧以 `PYTHONDONTWRITEBYTECODE=1` 运行后端，每次启动都为全部
  重依赖重新源码编译字节码，冷开机叠加杀毒实时扫描后导入阶段被放大到分钟级；
  ② 就绪等待有 180s 硬截止，后端超时后仍在导入、最终就绪时**没有任何机制补
  注入 IPC 令牌**，`ipc.enforce=true` 下前端全部业务请求永远 401。修复：字节码
  缓存经 `PYTHONPYCACHEPREFIX` 重定向到 `%TEMP%\ScanDetection\pycache`（安装
  期 NSIS 钩子预编译一次，首启即免全量重编译；缓存不进安装目录，卸载清单
  依旧干净；实测热启动 ~3s 完全就绪）；壳侧监督循环新增**令牌补注入**（子进程
  存活但未注入时持续探测 /health，就绪即注入，后端多晚就绪都能恢复）与**早死
  快速重启**（子进程先行退出立即重启，不空等满超时；秒退加退避防重生风暴）；
  就绪等待窗超时语义改为"转监督循环继续探测"而非放弃。登录页现在同样显示
  启动横幅（此前冷启动期间用户停在无反馈的登录页，误以为后端没启动）。
- 打包布局下模型注册表扫到旧版残留的空权重目录：`model_registry._resolve`
  锚点选择从"存在即可"改为**优先含权重文件的目录**（`<安装根>/models/weights`
  空目录恒真存在，导致 `mark_active_by_uri` 反复告警"权重目录中未找到"、模型
  管理页权重清单为空）；状态文件等非目录路径语义不变。
- 模型注册表状态文件卸载残留（冒烟测试"卸载零残留"断言抓出）：
  `model_registry.json` 此前按安装根直锚、不跟随 `SCANDETECTION_USER_DATA_DIR`
  重定向，打包版把它写进安装目录 `data\`（卸载器清单之外）——现改走
  `resolve_data_path` 与 db/影像/IPC 令牌同源落用户数据目录；`_save_state`
  写失败降级为告警（活跃指针可由启动期 `mark_active_by_uri` 重算），不再可能
  中断装配。
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
