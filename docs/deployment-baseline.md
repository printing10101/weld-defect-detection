# 部署基线：身份认证与外设管控（C-06/C-07/C-14/C-15/C-16/C-17/C-18 补充）

本文说明三员身份体系的部署步骤与操作系统层（OS 层）的外设管控基线要求。
应用层无法完全替代 OS 管控，USB/外设管控须在部署主机的操作系统上落实。

## 一、三员账号引导（C-06/C-07）

### 引导窗口

系统首次启动时 `accounts` 表为空，允许调用 `POST /api/v1/auth/bootstrap`
创建**第一个账号**（建议角色 `sysadmin`，系统管理员）；存在任意账号后该
端点永久关闭（返回 409）。前端登录页提供同语义的"首次启动引导"入口。

```
curl -X POST http://127.0.0.1:18773/api/v1/auth/bootstrap \
  -H "Content-Type: application/json" \
  -d '{"username":"sysadmin-01","role":"sysadmin"}'
```

响应中的 `private_key` 为系统现场签发的 SM2 软证书私钥（**仅此一次返回**，
系统不留存），应立即保存为私钥文件（文本，64 hex）交本人保管；公钥已登记
在账号上，用于登录验签。

### 三员岗位与权限矩阵（C-06）

| 能力 | 系统管理员 sysadmin | 安全保密管理员 secadmin | 安全审计员 auditor |
| --- | --- | --- | --- |
| 账号增删/启停/软证书签发 | ✔ | ✘ | ✘ |
| 模型激活/评估 | ✔ | ✘ | ✘ |
| 密级设定/变更（C-10） | ✘ | ✔ | ✘ |
| 载体登记/销毁发起（C-12） | ✘（仅销毁确认） | ✔（销毁发起） | ✘ |
| 导出审批（C-14） | ✘ | ✔ | ✘ |
| 告警处置/账号解锁 | 仅解锁 | ✔ | ✘ |
| 主审计链读取 | ✔ | ✔ | ✔（只读） |
| **独立安全审计链**读取（C-19） | ✘ | ✘ | ✔（只读） |
| 评片/复核/检索等业务 | ✔（登录即可） | ✔ | ✔ |

一人一岗：一个账号只绑定一个角色；关键操作（账号增删、密级变更、载体
登记/销毁、导出审批、告警处置）入独立安全审计链（SM3 哈希链，仅审计员
可读），与主审计链互为备份（C-19 双链）。

### 登录方式

1. **软件模式（默认）**：管理员经 `POST /auth/accounts/{id}/keypair` 为账号
   签发 SM2 软证书（私钥一次性下发交本人），登录流程：`GET /auth/challenge`
   取随机数（一次一用，60s 有效）→ 客户端以私钥对 nonce 做 SM3withSM2 签名
   → `POST /auth/login` 提交 `signature`；或提交 `private_key` 由后端代签后
   验签（前端不碰密码学的简化，单机本地软件可接受，私钥仅在本机进程内存
   中出现、不落日志/审计）。
2. **UKey 硬件模式（接口预留，未真机验证）**：账号 `auth_mode=ukey` 时登录
   走 `infra.crypto.Pkcs11Provider` 骨架（PKCS#11 商密硬件），需配置
   `SCAN_PKCS11_LIBRARY` 等并完成厂商联调；未对接时返回 501 显式提示。

### 会话策略（configs/default.yaml → auth 节）

- `idle_timeout_min`（默认 15）：会话空闲超时（滑动过期），前端同步登出；
- `session_ttl_min`（默认 720）：会话绝对有效期；
- `max_sessions`（默认 1）：单账号并发会话上限，超限吊销最旧会话（单点登录）；
- `max_failed_attempts`（默认 5）：连续挑战失败锁定账号，并落安全告警
  （alerts 表，保密员处置）+ 安全审计链留痕；
- `lockout_min`（默认 30）：锁定时长，保密员/系统管理员可提前解锁。

## 二、USB/外设管控（OS 层部署基线，C-14 补充）

导出管控在应用层覆盖 API 出口（报告 PDF、清单导出需保密员预授权或一次性
令牌）；**物理出口须在 OS 层按以下基线配置**（以 Windows 10/11 为例）：

1. **USB 存储禁用**：组策略 `计算机配置 → 管理模板 → 系统 → 可移动存储访问`
   全部设为"拒绝"；或注册表
   `HKLM\SYSTEM\CurrentControlSet\Services\USBSTOR` 的 `Start` 值设为 `4`；
   有外带需求时改用"只读+审批登记"策略并与载体台账（C-12）对应。
2. **外设白名单**：仅放行扫描仪/采集卡等必要设备（设备管理器 + 组策略
   "设备安装限制"按硬件 ID 白名单）；禁用蓝牙、无线网卡（单机部署）。
3. **打印/截屏管控**：报告打印走系统内 PDF 导出（已纳入导出审批）；
   部署环境如需打印纸版，打印机须受控登记，并在载体台账中以"报告"类
   载体登记去向。
4. **磁盘与备份**：`data/`（影像/报告/DB）所在分区建议 BitLocker 等全盘
   加密；备份归档（system/backup）同样按载体台账管理（kind=backup）。
5. **网络**：单机部署仅监听 127.0.0.1；如启用 http 同步（sync.kind=http），
   须在保密委员会审批的网络边界内进行并留存审批记录。
6. **审计**：OS 层登录/USB 插拔事件日志保留期与应用审计数据一致
   （建议 ≥3 年，按单位保密要求），定期由安全审计员核查。

## 三、导出管控开关（C-14）

`configs/default.yaml → export 节`：

- `require_approval`（默认 **true**）：报告 PDF/记录表 PDF/误报清单导出需
  "申请 → 保密员批准 → 一次性令牌 → 凭令下载"，或保密员本人预授权导出；
- `token_ttl_sec`（默认 600）：一次性导出令牌有效期；令牌一次一用，
  库中只存 SM3 哈希。

单机调试可临时 `SCAN_EXPORT__REQUIRE_APPROVAL=false` 关闭（生产禁止）。

## 四、纯离线部署验证（C-15）

系统软件侧自证"无外网依赖"：启动时执行静态配置自检并记录离线模式结论
（`GET /api/v1/system/network-status` 可随时查询）：

```json
{
  "offline_mode": true,          // sync.kind=local 即离线模式（数据不出本机）
  "sync_kind": "local",
  "egress_guard_enabled": true,  // 进程级外联防护在岗（C-16）
  "egress_blocked_events": 0     // 外联拦截事件计数（alerts 表持久统计）
}
```

`sync.kind` 非 local（http/cloud）属"数据出本机"的显式配置选择：启动日志
出现 WARNING 且落 high 级安全告警（kind=sync_nonlocal），`offline_mode=false`。
生产部署验收步骤：

1. 确认 `configs/default.yaml → sync.kind: local`（默认）；
2. 确认 `egress.enabled: true`（默认），`allow_cidrs` 保持空（除回环外全拦截）；
3. 启动后请求 `GET /api/v1/system/network-status`，核对 `offline_mode=true`、
   `egress_blocked_events` 数值；
4. **OS 层兜底（必须）**：主机防火墙出站方向默认拒绝（仅放行 127.0.0.1），
   或物理断网/禁用无线网卡——软件层外联防护不能替代网络边界管控；
5. 现场保密检查时留存 network-status 响应截图与启动日志（含离线自检结论行）。

## 五、进程级外联防护（C-16）

`configs/default.yaml → egress 节`：

- `enabled`（默认 **true**）：启动时装配进程级外联拦截
  （monkeypatch `socket.socket.connect` 与 `urllib.request.OpenerDirector.open`）；
  非白名单目的连接 → 阻断（抛错，连接不发生）+ high 级安全告警
  （kind=egress_blocked）+ 主审计链留痕（action=egress_blocked）；
- `allow_cidrs`（默认 **空**）：额外放行的目的网段（CIDR）。本机回环
  127.0.0.0/8 与 ::1/128 由代码恒放行（前后端/标注器通信必需），不在此列。
  仅当确需 `sync.kind=http` 推送到内网服务器时，把端点网段显式加入，
  如 `allow_cidrs: ["192.168.1.0/24"]`。

诚实边界：主机名目的地址需先 DNS 解析才能判定，解析本身可能产生 DNS 查询
外发——离线部署的同步端点应配置 **IP 字面量**；本防护只覆盖本软件进程，
不能替代 OS 层出站防火墙（两者互补，见第四节第 4 步）。

## 六、IPC 一次性启动令牌（C-17）

前端（Tauri WebView）与本地后端的进程间通信加固：

- 后端每次启动生成一次性令牌（`secrets.token_urlsafe`，32 字节熵），写入
  `data/ipc_token` 文件，**有效期 = 进程生命周期**（重启即换）；
- `ipc.enforce`（默认 **true**）时：除 `/health`、`/metrics`、`/auth/*`
  （登录引导需先于令牌分发）与静态资源外，所有请求须携带 `X-IPC-Token` 头
  或已带会话凭据（Authorization Bearer / ?access_token），否则 401；
- Tauri 外壳在端口就绪后读取令牌文件并注入 WebView
  （`window.__IPC_TOKEN__`），前端 `services/api.ts` 统一携带该头。

威胁模型（诚实声明）：本机前后端为回环明文 HTTP，令牌**不解决传输加密**——
它防的是"其他本机进程误调 / 浏览器网页 CSRF 式调用本机 API"。需要传输加密
时应挂本机证书启用 TLS（不在本次范围）。令牌文件权限为尽力而为：POSIX
chmod 600；Windows 下依赖数据目录继承的用户级 ACL。

单机调试可临时 `SCAN_IPC__ENFORCE=false`（生产保持 true）。令牌文件属敏感
凭据，按载体台账管理，不得拷贝出部署主机。

## 七、远程运维受控（C-18）

当前**无任何远程运维通道**（无 SSH/远程桌面/远程 API 管理面）。系统管理类
操作（备份/恢复、模型激活、密级变更、门禁拦截、账号锁定/解锁、告警处置、
导出管控、载体借用/销毁、外联拦截等）全部经主审计链留痕，软件侧提供
"运维操作清单 + 回放"能力：

```
GET /api/v1/audit/operations    # 仅安全审计员（auditor），只读
```

返回结构化时间线：时间 / actor / 动作 / 参数摘要 / 结果，动作白名单见
`backend/app/routers/audit.py → OPERATION_ACTIONS`；底层为主审计链（SM3
哈希链防篡改），可离线回放每一次运维操作的"谁/何时/做了什么/结果"。

**物理/流程基线（必须，OS 与制度层落实）**：

1. 远程维护（如确需）必须**经堡垒机**接入，严禁维护终端直连部署主机；
2. 远程维护操作**全程录屏**，录屏归档保留期与应用审计数据一致（建议 ≥3 年，
   按单位保密要求），并由安全审计员定期核查录屏与 /audit/operations 时间线
   的一致性；
3. 维护会话使用一次性授权：维护前后由系统管理员与保密员双人在场确认，
   操作窗口计入载体台账备注；
4. 堡垒机账号、部署主机本地账号均纳入三员管理，禁止共享账号。

## 八、静态加密密钥管理（GB/T 28452 用户数据保密性）

影像副本/报告等落盘数据默认启用国密静态加密（SM4-CTR + HMAC-SM3，信封
SDC2）。主密钥来源优先级：

1. **加固部署（推荐涉密环境）**：环境变量 `SCAN_CRYPTO_KEY`（base64/hex
   32 字节，可用 `SoftSmProvider.generate_key()` 生成后注入）；
2. **桌面单机默认**：本地持久密钥文件 `data/.crypto_key`——首启自动生成
   一次并复用，生成/加载事件均留日志。

部署须知（两种模式同）：

- `data/.crypto_key` 必须随 `data` 数据目录一并备份；**密钥文件与密文二者
  丢失其一站，另一者即不可解**；
- 密钥文件严禁入库/外传（已在 .gitignore 列为"运行时密钥，绝不入库"）；
- 更高保密要求请对接商密硬件（`SCAN_CRYPTO_PROVIDER=pkcs11`，接口就绪、
  待真机验证）。

密钥完全不可用时系统**拒绝明文落盘**（报错阻断而非降级），属安全设计。

## 九、交付前合规自检门槛

正式交付/验收前必须完成一轮合规自检并确认**全部检查项通过**：

```
GET /api/v1/compliance/selfcheck      # 生成 JSON+PDF 自检报告（data/compliance/）
```

- 检查项按 GB/T 22239 / GB/T 17859 二级口径（软件侧自动检查部分）设计，
  覆盖身份鉴别、访问控制、安全审计、边界防护等 18 项；
- 历史上出现过的 fail 均为"生产加固开关未打开"（`export.require_approval`、
  `ipc.enforce` 被临时关闭用于调试）——交付前必须恢复出厂默认（均为 true）
  并重跑至 `overall=pass`；
- 自检报告随交付物归档（PDF 版本可直接纳入验收材料）。
