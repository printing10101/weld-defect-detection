# 前端设计规范 · 极简 ZINE 设计语言

> 由 GitHub 开源 skill **gc-minimal-zine-poster-v0-1**（MIT, LiamGvchi）的视觉规则
> 转译而来，用于「射线焊缝缺陷智能检测评片系统」的桌面前端（Vue3 + Tauri）。
> 可查看原型：`index.html`（浏览器直接打开）。

## 1. 设计哲学（来自 skill 的稳定共性）
- **大量留白**：每屏 70%–90% 为空白/纸张，单一焦点。桌面应用里 = 大边距、稀疏布局、每屏一个主任务。
- **单一高饱和色锚**：全站只允许一个主色（钴蓝）。缺陷红仅作数据标注，不计入"装饰色"。
- **仿旧纸张，非纯白**：底色调暖、哑光，避免 "clean digital UI white"（skill 硬禁）。
- **衬线标题 + 等宽数据**：标题用衬线（中文宋体/思源宋体），编号/测量/ID 用等宽体——技术档案感。
- **印刷瑕疵作为质感**：纸张颗粒噪点 + 标题轻微错版(misregistration)，但不损可读性。
- **安静、档案、克制的情绪**：避免 3D、霓虹、商业广告排版、密集仪表盘。

## 2. 设计令牌（CSS 变量，移植到 Vue 时放进 :root 或 Tailwind theme）

| Token | 值 | 用途 |
|------|-----|------|
| `--paper` | `#F3EFE6` | 主背景（仿旧纸，非纯白） |
| `--paper-2` | `#EAE4D6` | 侧栏 / 卡片面板 |
| `--paper-3` | `#E2DBC9` | 输入 / hover 态 |
| `--ink` | `#2B2620` | 主文字（暖近黑） |
| `--ink-soft` | `#6B6458` | 次级文字 |
| `--ink-faint` | `#9A9286` | 微文本 / 占位 |
| `--accent` | `#1F4E8C` | **唯一主色锚**（钴蓝 risograph ink）：激活态、主按钮、链接、级别 |
| `--signal` | `#C0392B` | 仅缺陷标注 / 告警（数据色，非装饰） |
| `--line` | `rgba(43,38,32,.14)` | 分隔线 |
| `--serif` | Noto Serif SC / Source Han Serif SC / Songti SC / SimSun / Georgia | 标题、签字 |
| `--mono` | JetBrains Mono / Cascadia Code / Consolas / monospace | 数据、ID、表格、导航 |

## 3. 组件映射（→ Vue3 组件建议）
- `AppShell.vue`：左 `RailNav`（窄 226px，含 14px 钴蓝方块 mark + 衬线应用名 + 等宽导航）、右 `main`（padding 54/60）。
- `TitleZine.vue`：`<h1>` 衬线 + `::before` 错版重影（accent @ opacity .16，偏移 2px）。
- `DropZone.vue`：虚线边框，hover 变 accent 边。
- `ComparePlate.vue`：双图并排（送检原图 / 标注图），图下等宽图注；射线图保持清晰，不叠纸张纹理。
- `KvTable.vue` / `DataTable.vue`：等宽、细线、稀疏；级别用 accent，待复核用 signal。
- 纸张颗粒：用 `feTurbulence` SVG data-URI 作 `body` 背景，opacity ~0.04。

## 4. 移植注意
- 射线底片/标注图区域**不要**加纸张纹理与错版，保持像素清晰（功能优先）。
- 主色严格唯一；若需第二语义色（如"通过/警告/危险"），应来自数据状态而非新增装饰色，且仍克制。
- 字体优先系统字体栈，离线可用；如需思源宋体/等宽可随 Tauri 一并打包。

## 5. 可进一步用 skill 生成的素材
- 启动闪屏 / About 海报：用 `gc-minimal-zine-poster-v0-1` 生成一张主题海报（如"焊缝·记忆"），
  作为应用关于页或空状态插画（skill 本职就是出图）。
- 加载/空状态的小图：同风格单色剪影。
