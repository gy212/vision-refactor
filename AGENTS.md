# Repository Guidelines

本文件是 vision 重构仓库的统一规范（SSOT），适用于所有 coding agent。`CLAUDE.md` 只导入本文件；规则有变化时只改这里。

通用工程、测试、沟通与协作契约全量以 Square 规范体系（`specs/` 与文末托管区块）为准；本文件聚焦规范索引、vision 专属架构与领域约束、硬约束、环境命令、测试门禁及代码地图。

## 规范文档索引（以 Square 契约为准）

- **常驻遵循契约**（需求分析、沟通、编码与写测时遵循）：
  - 工程与测试核心契约：[specs/engineering_contract.md](specs/engineering_contract.md)
  - 人机协作与沟通契约：[specs/communication_contract.md](specs/communication_contract.md)
- **技术栈样例包**（排查 Python / CLI / AST 静态分析边界时查阅，`square review` 会自动挂载）：
  - Python / CLI / 静态分析边界样例包：[specs/packs/python_cli.md](specs/packs/python_cli.md)
- **按需读取契约**（仅进入对应环节时读取，日常编码与问答严禁预读；产出须通过 `square contract-check` 校验）：
  - 编写或更新 Issue：[specs/issue_contract.md](specs/issue_contract.md)（校验：`square contract-check --type issue <文件> --title "<标题>"`）
  - 发起或更新 PR：[specs/pr_contract.md](specs/pr_contract.md)（校验：`square contract-check --type pr <文件> --title "<标题>"`）
  - 执行代码与方案审查：[specs/review_contract.md](specs/review_contract.md)（校验：`square contract-check --type review <文件>`；离线审查提示包：`square review --emit-prompt`）

## 仓库现状

武术散打动作识别与评分系统，包括 Tk 桌面（实时预览、双摄录制、考试、学生练习）、CLI、batch 与 analysis 离线工具，以 Windows 单文件 EXE 交付。正式评分与姿态识别专注于 MediaPipe 链路。

- **本仓库**：工作区的 `new/` 目录，远端 `https://github.com/gy212/vision-refactor`（私有）。本仓库是当前所有代码、文档、测试与提交的唯一活跃开发区。包含核心视觉与评分库 `core/`、Tkinter 桌面应用 `apps/tk/`、命令行入口 `apps/cli/`、离线批处理与分析工具（`batch/`、`analysis/`）以及自动化测试 `tests/`。
- **旧项目只读参照**：工作区的 `old/` 目录属于外层仓库（`../old/`，工作区根目录的 `.git`，远端 `YGuo-2/vision`），仅用于只读对比与历史溯源。严禁在 `old/` 中修改、新增或删除任何文件；必须运行旧代码对照时，必须使用 `python -B` 且工作目录与输出显式指向 `old/` 之外。

## 本仓专属架构与工程约束

系统架构重构（P0–P9）已全量闭环，当前处于**功能完善、算法精细化与产品化交付阶段**。所有新增功能与修改直接在现有架构上推进，不再保留过渡期双轨或旧路径门面。通用工程红线遵循 [specs/engineering_contract.md](specs/engineering_contract.md)，以下为本仓特有约定：

### 分层与模块边界

- **单向分层底线**（设计文档 §2.1）：严格遵守 `foundation (L0) → media (L1) → pose (L2) → evaluation (L3) → {services, recording, live, reporting} (L4) → workflows (L5) → apps (入口)`。
  - 只能自上而下单向依赖更低层，严禁同层交叉依赖或逆向跨层依赖；
  - L4 域内四个包（`services/`、`recording/`、`live/`、`reporting/`）彼此平级自治、互不导入，共用纯逻辑统一下沉至 L0/L1，跨域协作由调用方在上层组合注入；
  - L4 不向上依赖 L5（如录制 finalizer 的具体执行步骤由工作流计划构建方注入）；
  - 入口层只消费各包 `__init__` 导出的公共 API，严禁跨模块导入下划线开头的私有实现；
  - `core/` 领域核心严格与 GUI 解耦，严禁导入 `tkinter`。以上由 `tests/test_architecture.py` 与 `square check .` 双重门禁强制。
- **外部防御边界**：本仓外部物理与信任边界包括用户输入、外部文件与目录、视频解码流、摄像头设备接入、CLI 参数及网络通讯；内部模块间信任类型契约并坚持快速失败。

### 视觉评估与领域算法约束

- **主次分明，严禁配角一票否决主角**：任何处理链路严格区分核心主判定与辅助附加项（如核心规则结论 vs 辅助对照图；局部阶段缺失 vs 其他已看清的独立阶段）。辅助信息、非必需参照帧或无关阶段缺失只允许降级该局部或省略附加项并如实说明，严禁连坐主链路。
- **失败可解释底线**：不得不返回失败、拒绝或“无法判断”时，只要原始输入存在，必须保留并展示现场证据（所选代表画面、已测得数值）与具体死因（卡在哪一步、缺哪个关键点或哪个时段），严禁黑盒吞没上下文或只甩一句笼统结论。
- **物理直觉优先**：设计姿态与动作算法前先按专业人类肉眼判据检验；能用单一观测维度、直接 2D 几何关系或简单确定性逻辑看清的问题（哪面看得清就用哪面），严禁教条式强行捆绑多视角融合、高维拟合或复杂状态机。
- **策略维度正交分离**：动作编排、计时切分、测量标准、视角权重、评分模式、报告选图各自独立声明、各司其职；严禁把某一维度的开关当成万能开关夹带删减检查项、篡改标准或暗换测量公式。

### 体量预算、格式与资源约定

- **Tk 桌面层严格预算**（在通用 800 / 600 / 100 预算基础上收紧）：`apps/tk/app.py` 组合根 ≤ 300 行，每个控制器 ≤ 400 行，每个视图 ≤ 600 行；非测试源码 600 行为预警线，超线模块不再堆叠新职责。
- **格式与棘轮**：全库由 `ruff==0.14.3` 与 `ruff.toml` 统一格式化（100 列、双引号、4 空格）；`tests/architecture_allowlist.py` 与 `.harness-baseline.json` 均遵循只减不增的棘轮机制。发现与当前任务无关的历史死代码、废弃逻辑或冗余引用，仅在沟通或 PR 说明中提示，不擅自顺手删改。
- **类型标注与视频转码**：函数保持小而清晰，公共接口与领域边界须带完整、明确的类型标注。视频导出的最终标准产物为 H.264 编码的 MP4；本机 OpenCV 回退生成的 MJPG/XVID AVI 必须在出片后由系统 `ffmpeg` 自动转码（`-c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p`）并清理临时 AVI，严禁将回退 AVI 直接交付。
- **禁止提交无关大文件**：禁止将模型权重文件（`.task` 等）、原始视频文件以及运行输出目录（`outputs/`、`dist/`、`build/`）提交到 Git 仓库。

### 分支与 PR 协作（本仓流程）

- 提交只进入本仓库，默认主分支 `main`。每个独立特性或任务单独建分支，工作区保持干净后提交，验收通过后由 `gh pr create --base main` 发起 PR（简单纯文档/配置维护按仓库规则集处理）；跨多层大型改动使用 stacked PR（`gh stack`）逐层递进。
- Commit / Issue / PR / Review 的语言、标题与正文结构统一遵循 `specs/` 契约及文末 `R-WORKFLOW`。
- 提交或推送前确认 Git / GitHub 身份（`gh auth status`、`git config user.name`、`git config user.email`）。

## 硬约束

- **MediaPipe 默认路径与基准零漂移**：系统专注于 MediaPipe 姿态识别与动作评分，`pose33_v3` 模板兼容性及 `infer()`/`annotate()` 行为语义严格保持一致。`tests/test_pose33_v3_golden.py`、`test_tech_eval_contract.py`、`test_valid_mask_migration.py` 必须保持零漂移；只有经批准的决策可以改变数值基准。
- **点位有效性统一判定准绳**：关键点有效性统一由 `core/pose/validity.py` 中的 `derive_valid_mask` 判定，严禁在业务代码中散落 `lm[idx, 3] >= thr` 之类的私有判据；结构性缺失点不得伪造成有效点。
- **用户数据格式向下兼容**：`%LOCALAPPDATA%\VisionSanda` 目录结构、`user_prefs.json` 键名、片段目录与文件名规则、练习历史 `schemaVersion=1`、成绩表列格式以及 `feedback_config.json` 的文件名与字段严格兼容。分数口径：内部统一为 0..1，导出时乘以 100，综合得分列统一使用 `combined_percent`。
- **测试环境数据隔离**：偏好文件路径只由 `core/foundation/prefs.py` 的 `_prefs_path()` 提供，`tests/conftest.py` 的 autouse fixture 将其钉在临时文件，防止测试污染真实用户数据；移动该函数时必须在同一提交内更新打桩目标。
- **打包与运行时独立**：严禁随意引入新增的外部运行时依赖。构建的单文件 EXE 必须能独立完整运行；子进程入口必须为模块顶层函数，参数保持可 pickle 序列化，保留 `freeze_support()`。
- **动作规则直接生效与三态结果红线**：凡用户确认判定逻辑并完成实现的检查项，在全姿态模型（`lite` / `full` / `heavy`）下一律直接正式启用；全系统严禁设立“未自动检查（`pending_rule`）”、“试用（`trial`）”或任何资格门禁层。练习检查结果严格只允许出现**疑问项（`candidate` / 发现疑似问题）**、**未发现问题项（`not_observed` / 已检查未发现）**、**无法判定项（`unable` / 暂无法判断）**三类，考试与 batch 规则结论严格只允许出现**扣分/违规项（`violation`）**、**通过项（`pass`）**、**无法判定项（`unable`）**三类（点位缺失或阶段未检出统一归入 `unable`，严禁伪造成未发现问题或通过）。**严禁任何 Agent 未经用户明确指令，擅自拿本地视频素材当“100% 标准真值”跑误报率/检出率去拦截、关停、降级已实现的检查项，或擅自篡改用户给定的几何/角度阈值。**
- **历史命令入口保持可用**：`apps/app_ui.py`、`apps/main.py` 等经典命令保持行为可用，入口壳保持极简（≤ 30 行），逻辑收敛于组合根与核心库中。

## 环境与命令

命令在本仓库根运行，按 PowerShell 写。用 Git Bash 时，路径写成 `./.venv/Scripts/python.exe`，环境变量用 `export`；WSL2 下使用 `./.venv/wsl/bin/python` 与 `square`。

- 联网下载（pip 包、模型 `.task`、GitHub 等）一律走本地代理 `http://127.0.0.1:7879`（端口以 Windows 系统代理设置为准）：

  ```powershell
  $env:HTTP_PROXY  = "http://127.0.0.1:7879"
  $env:HTTPS_PROXY = "http://127.0.0.1:7879"
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt --proxy http://127.0.0.1:7879
  ```

- Python 用本仓库的 `.\.venv\Scripts\python.exe`（WSL2 用 `./.venv/wsl/bin/python`）；pip 一律用 `python -m pip`。不使用也不修改 `old/.venv`。
- MediaPipe 模型首次运行时自动下载到 `models/`。

```powershell
.\.venv\Scripts\python.exe apps/app_ui.py                  # Tk 桌面
.\.venv\Scripts\python.exe apps/main.py --source 0         # CLI 摄像头
.\.venv\Scripts\python.exe apps/main.py --source input.mp4 --pose heavy --out out.mp4 --no-show --workers 4
.\.venv\Scripts\python.exe apps/make_template.py --video action.mp4 --pose heavy --preview
.\.venv\Scripts\python.exe apps/match_template.py --template template.npz --video test.mp4 --preview
.\.venv\Scripts\python.exe batch/batch_dual_compare.py --standard_dir "标准样本" --student_dir "学员样本" --pose full --out_dir "输出目录" --rules --rule-action stance
.\.venv\Scripts\python.exe scripts/sim_exam_flow.py         # 无摄像头模拟考试流程
.\.venv\Scripts\python.exe scripts/check_format.py origin/main  # 增量格式检查
.\.venv\Scripts\python.exe -m ruff format <文件路径>       # 代码格式化修复
square check .                                             # Square 静态架构与棘轮检查
square contract-check --type pr <pr.md> --title "<标题>"   # Issue / PR / Review 契约校验
```

- `batch_dual_compare.py --paired`：按学员文件名里的正/侧关键词配对并调用 `compare_dual_streams`，CSV 的 `video` 列写学员 id，`--export_raw` 在 paired 模式下跳过。
- 打包构建：单文件 spec 为 `packaging/pyinstaller/app_ui_onefile.spec`，构建命令：`.\.venv\Scripts\python.exe -m PyInstaller packaging/pyinstaller/app_ui_onefile.spec --noconfirm`；产物为 `dist/vision_ui.exe`。EXE 自检命令：`vision_ui.exe --offline-self-check <报告.json>`。

## 测试门禁

每个 PR 都要过：

- 增量格式检查：`.\.venv\Scripts\python.exe scripts/check_format.py origin/main`（包含已跟踪的工作区改动；新文件先 `git add`）。使用 `ruff==0.14.3` 和 `ruff.toml`，检查新增或修改的整个 `.py` / `.pyi` 文件；修复命令：`.\.venv\Scripts\python.exe -m ruff format <文件路径>`。
- golden 与契约：`.\.venv\Scripts\python.exe -m pytest tests/test_pose33_v3_golden.py tests/test_valid_mask_migration.py tests/test_tech_eval_contract.py tests/test_rule_availability.py -q`
- 架构测试与 Square 门禁：`.\.venv\Scripts\python.exe -m pytest tests/test_architecture.py -q` 与 `square check .`
- 直接受影响模块的最小套件（见下表；遵循 `R-MIN-VERIFY`，无关联反证严禁擅自扩跑全库套件或全量历史差分）
- `git diff --check`：行尾提示可以接受，空白错误不行

CI（`.github/workflows/gates.yml`，检查名 `gates`）在每个 PR 和 `main` 的推送上自动跑增量格式检查、golden / 契约、架构测试和空白检查，任何一步退出码非 0 即检查失败。格式检查以 PR 的 base SHA 或 push 的 before SHA 为基线，首次推送使用空树；排除已删除文件，没有 Python 改动时跳过。环境：`windows-latest`、Python 3.12，安装 `requirements.txt`；不用摄像头，也不预置 `models/` 下的模型，门禁测试只读 `tests/fixtures/`。受影响模块的套件和全量测试不在 CI 里，仍在本地跑。本地复现 CI 的空白检查：`git -c core.whitespace=cr-at-eol diff --check origin/main HEAD`。合入约束由仓库规则集 `main-gates`（作用于默认分支）强制：`gates` 是必需检查，未通过不能合并；`main` 只能经 PR 合入，禁止强推和删除。

| 改动范围 | 测试（`tests/` 下） |
|---|---|
| 考试系统 | `test_exam_session.py test_exam_roster.py test_exam_clip.py test_presence_gate.py test_recording_postprocess.py test_exam_panel.py`，再跑 `scripts/sim_exam_flow.py` |
| batch paired / 双流 | `test_batch_backend_args.py test_compare_dual_streams.py` |
| 布局与路由契约 | `test_body_core_layout.py test_backend_routing_contract.py` |
| 录制与性能基准 | `test_recording_controller.py test_recording_postprocess.py test_recording_benchmark.py test_benchmark_probes.py test_tk_benchmark.py` |
| 学生练习与报告 | `test_action_feedback.py test_student_presence.py test_feedback_report.py test_feedback_html_export.py test_report_html_golden.py test_feedback_result_panel.py test_evidence_contract.py test_evidence_budget.py test_timing_diagnostics.py` |
| 规则引擎 / 复用与动作计时 | `test_rule_availability.py test_rule_engine_entry.py test_rule_reuse.py test_rule_evidence_entries.py test_shared_motion_timing.py test_shared_phase_coverage.py test_shared_action_records.py test_step_straight_combo.py test_straight_refinements.py`（全量差分 `test_p3_eval_differential.py` 仅在明确更新差分基准时按需单跑） |
| 摄像头与运行时 | `test_camera_warmup.py test_camera_service.py test_camera_enum.py test_tk_cameras.py`；源码模式跑 `test_camera_capture.py::test_tk_preview_student_switch_with_process_cameras`（真实 Tk + spawn），再构建 EXE 跑 `--offline-self-check`；整套自检只能在冻结包里跑 |

全量基线和已知失败见 `docs/archive/重构阶段任务/README.md` 的"通用任务前检查"，不得新增失败。替换掉的测试要在 PR 里列出对应的新测试（D-13）。

## 变更记录

- 每次完成任务后，把修改总结写进 `change.md`，最新记录放在最上面。记录内容：修改日期、问题描述、修改内容、验证方法。只改文档时也要写同步依据，并至少做一次轻量验证。阶段 PR 要包含对应的记录。
- `change.md` 过长时，用 `git mv` 把它归档为 `docs/archive/change（<起>~<止月>）.md`，再新建一个空的 `change.md`，把"归档与重建"作为新文件的第一条记录，并更新下面的归档清单。
- 历史归档：本仓库 `docs/archive/change（2026.7~2026.11）.md`；更早归档在旧项目里：`old/change（start~2026.5）.md`、`old/change（2026.5~2026.6）.md`、`old/change（2026.6~2026.7）.md`。

## 代码地图

仓库严格遵循分层架构设计（设计文档 §2），从底层基础能力到顶层应用入口如下：

### 核心库 `core/`

分层架构体系（L0–L5）的核心模块与职责划分如下：

| 层级 | 模块包 | 核心职责与关键模块 |
|---|---|---|
| **L0** | `core/foundation/` | **系统基础能力**<br>• `paths.py`：应用数据与临时目录统一解析<br>• `prefs.py`：用户偏好设置持久化与隔离接口<br>• `atomic_io.py` / `npz.py` / `jsonable.py`：原子写盘与格式序列化<br>• `geometry.py` / `signal.py`：通用几何基元与信号去抖<br>• `host.py` / `power.py` / `memory.py` / `volume.py` / `winapi.py`：宿主机软硬件环境探针 |
| **L1** | `core/media/` | **媒体流采集与处理**<br>• `camera/`：摄像头硬件管理（`devices.py` 枚举、`service.py` 统一服务、`pool.py` 租约池、`reader.py` 独立读帧线程、`capture_process.py` 隔离子进程）<br>• `frames.py` / `frame_clock.py` / `pacing.py`：帧契约、采集时钟与录像节奏控制<br>• `writer.py` / `mjpeg_writer.py` / `ffmpeg.py`：视频写入与随包 ffmpeg 转码管道 |
| **L2** | `core/pose/` | **姿态识别与特征提取（专注 MediaPipe）**<br>• `engine.py` / `mediapipe_runtime.py`：MediaPipe Tasks 推理引擎运行时<br>• `extract.py` / `sequence.py`：姿态序列提取与时间序列封装<br>• `validity.py`：`derive_valid_mask` 关键点有效性判定准绳<br>• `layouts.py` / `features.py` / `normalize.py`：`pose33_v3` 关键点布局、归一化与特征对齐<br>• `render.py` / `annotate.py`：姿态骨架平滑与骨架视频导出 |
| **L3** | `core/evaluation/` | **动作编排、统一计时、对比与公共规则引擎**<br>• `action_catalog.py`：11 类动作编排、动作段组成与命名策略声明<br>• `action_timing.py` / `action_record.py` / `phase_alignment.py` / `phase_instances.py`：同步录像唯一计时来源选定、四入口共享动作记录与逐次机位覆盖核对<br>• `motion_events.py` / `punch_events.py` / `punch_phases.py` / `step_reps.py` / `stance_windows.py`：通用往返位移与保持识别、直拳事件/相位、上步组合逐次切分与稳定实战式取窗<br>• `measures/` / `paired_torso.py`：臂腿脚躯干、姿态与运动公共几何测量及双摄同步躯干朝向<br>• `rules/`：公共规则引擎（`engine.py` 调度、`rule_definitions.py` 统一规则定义、`catalog_measurements.py` / `kick_measures.py` / `kick_step.py` 公共测量、`sanda_v5.py` 目录编排、`evidence.py` 统一证据构建）<br>• `templates.py` / `matching/` / `scoring/` / `techeval/`：标准模板、`dtw.py` / `dual.py` / `repetition.py` 匹配、考试动作池评分与力序分析 |
| **L4** | `core/services/`<br>`core/recording/`<br>`core/live/`<br>`core/reporting/` | **领域服务层（平级自治，互不导入）**<br>• `services/`：离线分析服务（`analyze.py`）、模板生成（`templates.py`）、练习历史（`practice_history.py`）、录制基准评测（`recording_benchmark.py`）<br>• `recording/`：双摄录制控制（`recorder.py`）、录像工作线程（`worker.py`）、后处理链与骨架成片（`finalizer.py`、`steps.py`、`skeleton.py`）<br>• `live/`：实时交互会话（`runner.py`）、站位监测与 ROI 判定（`presence.py`）、实时预览流（`preview.py`）<br>• `reporting/`：结构化评测模型（`practice_model.py`）、公共优先级选图与证据预算（`evidence_selection.py`、`evidence.py`）、计时诊断与问题分组（`timing_diagnostics.py`、`issue_groups.py`）、文本与 HTML 报告生成（`text.py`、`html/`） |
| **L5** | `core/workflows/` | **业务工作流编排**<br>• `exam/`：考试考务完整状态机编排（`session.py` 考场状态、`roster.py` 名册台账、`clip.py` 动作剪裁、`scorebook.py` 成绩台账）<br>• `practice/`：学生练习会话编排（`session.py`）<br>• `presence.py` / `executor.py` / `plans.py`：考生进退场闸门与异步任务执行流 |

### 应用入口 `apps/`

消费 `core/` 公共 API 组织的交互界面与命令行入口：
- **Tkinter 桌面应用**（`apps/tk/`，入口壳 `apps/app_ui.py`）：
  - `app.py`：Tk 组合根与全局事件分发
  - `controllers/`：业务控制器（`cameras.py`、`recording.py`、`practice.py`、`exam.py`、`benchmark.py`、`session.py` 等）
  - `views/`：界面组件与视图布局（主窗、录制面板、练习控制、考试面板、练习分析结果页等）
  - `desktop_launcher.py`：EXE 打包自检与引导入口（`--offline-self-check`）
- **CLI 命令行工具**（`apps/cli/`，入口壳 `apps/main.py` 等）：
  - `main.py`：实时摄像头与离线视频骨架推理 CLI
  - `make_template.py` / `match_template.py`：模板录制生成与单/双模板匹配验证工具

### 离线工具、支持脚本与资源

- **离线分析与批处理**（`batch/`、`analysis/`）：
  - `batch/batch_dual_compare.py`：双摄大批量离线双模比对（含 paired 模式）
  - `batch/batch_export_skeleton.py` / `batch_tech_eval.py`：批量骨架导出与技术指标评估
  - `analysis/`：离线性能 Profile、标定测试与 Benchmark Harness
- **支持脚本**（`scripts/`）：
  - `check_format.py`：基于 Git 变动的增量代码格式检查
  - `sim_exam_flow.py`：无硬件摄像头环境下的全流程考试状态机模拟器
  - `generate_feedback_icons.py`：练习结果页状态图标生成脚本
- **工程配置、规范与文档交付**：
  - `specs/`：Square 工程、沟通与 Issue / PR / Review 契约规范及 `packs/python_cli.md` 边界样例包
  - `.harness.json` / `.harness-baseline.json`：Square 静态架构检查配置与存量棘轮基线
  - `docs/`：专项评估报告与历史重构设计归档（`docs/archive/`）
  - `ruff.toml`：全库代码格式化配置（100 列、双引号、4 空格缩进）
  - `packaging/pyinstaller/app_ui_onefile.spec`：PyInstaller Windows 单文件 EXE 打包规范
  - `tests/`：pytest 契约与回归测试套件（含架构测试 `test_architecture.py` 与基准快照）
  - `assets/feedback/`：结果页与报告嵌入的状态图标资产

<!-- square:start -->
## Square 约束工程准则

本仓库已启用 Square 约束工程门禁，所有代码与文档变更须遵守以下核心准则：

- **`R-COHESION` 静态架构门禁与高内聚**：运行 `square check .`（或 `python3 -m harness.cli.entrypoint check .`）校验单向分层、零循环依赖、非测试模块 `<= 800` 行（`600` 行预警）、函数 `<= 100` 行、模块顶部非空职责说明、纯导出 `__init__.py`、禁跨模块访问 `_` 私有符号；严禁创建 `utils`/`helpers`/`common`/`misc` 杂物箱，严禁分号拼接、嵌套三元或 `part2`/`_impl` 机械切片。
- **`R-ONTOLOGY` / `R-SSOT` 根因修复与破旧立新**：按本体归属定位公共抽象层并覆盖同级场景，严禁按触发样本写 `if` 硬编码或症状兜底；新方案须当场接通同级调用方并清理旧路径，严禁保留无故双轨或 `legacy` 兼容层（跨仓公共契约或已登记跟踪 Issue 的分阶段交付除外）。
- **`R-FIRST-PRINCIPLES` 第一性原理与受阻熔断**：优先最简直观的因果方案；仅当核心架构假设被证伪或修补导致复杂度失控时停手熔断，交出最小反例与替代方案，常规边界报错自行修复。
- **`R-FAIL-FAST` 外边界校验与内部快速失败**：仅在外部物理边界（用户输入、文件、CLI/Hook stdin）做校验；内部模块间信任契约，严禁 `getattr(self, "_...")` 防御糊墙或 `except: pass` 静默吞异常（非核心旁路须在显式边界捕获并记录原因后降级）。
- **`R-MIN-VERIFY` 最小单点验证与回答即停**：每次验证前明确待回答问题与最小检查对象，定向跑最小受影响测试（如 `pytest tests/unit/test_target.py -q`），得到确定结论即停；共享基座改动须补跑直接下游回归，严禁盲目全仓扫跑或无意义 `md5sum`/`sha256sum` 校验。
- **`R-TEST` 测试红线与等价类覆盖**：写测求全、跑测求准——覆盖触发/同级/负向等价类、真实外部边界与创建→更新→删除→重入生命周期（外部不可控副作用允许最小 Fake/Stub 隔离）；严禁配额凑数、永真/空壳断言、跨模块导入生产 `_` 私有符号、直读文档做字面断言或按轮次乱建测试脚本。
- **`R-ALIGNMENT` / `R-COMM` 清晰对齐与直白沟通**：需求含糊或架构受阻先对齐，获得明确授权后连续推进至落地不碎步索权；沟通直给结论、用通顺短句与因果连接词、禁生造黑话与价值高帽或谄媚套话；未知边界须明示“已确认事实 + 缺失输入或失效条件”；发现方案倒退时直指后果与替代做法，受阻时先用人话讲清卡点（有分叉给利弊与推荐，纯缺权限/配置直说缺项）。
- **`R-WORKFLOW` Commit / Issue / PR / Review 底线**：
  - 标题统一为 `<type>(<scope>): <简体中文描述>`（分阶段 PR 附 `(#<Issue编号>, k/N)`），冒号后中文描述正文 `<= 35` 字（不含前缀与分阶段后缀）；Commit / PR 的 `<type>` 限 7 类：`feat` / `fix` / `refactor` / `perf` / `docs` / `test` / `chore`（Issue 用 `bug` 替代 `fix`）。
  - Issue 与 PR 按变更规模轻重分级（纯文档/配置/测试用轻量版，业务或接口改动用完整根因与对照版）；区分未完成 `Draft` 与阶段闭环交付 `(#N, k/N)`，未覆盖项须绑定后续跟踪 Issue；严禁全英文撰写或堆砌测试数字与免责套话。
  - 代码审查（`square review`）坚持独立基线建模、同级场景覆核与逐变更风险单元代码级证伪。
<!-- square:end -->
