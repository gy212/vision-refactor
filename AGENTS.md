# Repository Guidelines

本文件是 vision 重构仓库的统一规范，适用于所有 coding agent。`CLAUDE.md` 只导入本文件；规则有变化时只改这里。

## 仓库现状

武术散打动作识别与评分系统，包括 Tk 桌面（实时预览、双摄录制、考试、学生练习）、CLI、batch 与 analysis 离线工具，以 Windows 单文件 EXE 交付。正式评分以 MediaPipe 为准。本仓库按 `整体架构重构设计.md`（下称设计文档）重构了这个系统，P0–P9 已全部完成（各阶段记录见 `重构阶段任务/README.md`）；之后的改动按设计文档的分层与本文件的规则在新结构上进行。

- 本仓库：工作区的 `new/` 目录，远端 `https://github.com/gy212/vision-refactor`（私有）。现有内容是设计文档、`重构阶段任务/`（P0–P9 执行清单）、摄像头设计 `camera_session_architecture.md` 和 `摄像头架构设计交接说明.md`、变更记录 `change.md`。代码由 P0 从旧仓库导出。
- 旧项目：工作区的 `old/` 目录（相对本仓库是 `../old/`），内部结构就是整理前的仓库根：代码、测试、`.venv`、`models/`、`dist/`，以及旧文档（`old/README.md`、`old/docs/`、change 归档）。它属于外层仓库（工作区根目录的 `.git`，远端 `YGuo-2/vision`）。
- 路径约定：本仓库文档中的路径相对于本仓库根，`old/` 开头的指旧项目。现有文件的路径（如 `apps/app_ui.py`、`docs/…`）指旧项目中的同名文件；P0 导入后，导入的部分以本仓库为准。

`old/` 只读，只作参照：不修改、不新增、不删除其中的任何文件（包括 `.venv`、缓存和输出）；除非用户要求，也不在外层仓库提交。优先阅读源码；必须运行旧代码对照时，用 `python -B`（pytest 再加 `-p no:cacheprovider`），工作目录放在 `old/` 之外，输出路径显式指到 `old/` 之外。

已弃用，不参与重构：web 端，包括 `apps/ui_backend.py`、`frontend/`、`ui_backend_sidecar.spec`、`package.json`（`npm run *` 脚本）、`start-desktop-dev.cmd`、`scripts/` 下 3 个 Tauri 脚本，以及相关测试（`tests/test_ui_backend_*.py`、`test_vue_tauri_acceptance_gaps.py`、`test_windows_packaging_smoke.py`）。建议 P0 不把它们导入本仓库，留在 `old/` 即归档（P0-4）。门禁测试 `tests/test_backend_routing_contract.py`、`tests/test_batch_backend_args.py` 在模块顶层导入 `apps.ui_backend`：导入时去掉顶层导入，依赖它的 6 个用例留在 `old/`，文件名不变。如果导入，就不改代码、不加功能、不作门禁，迁移模块时为它保留旧路径门面，web 测试的失败不得多于基线，去留由 D-10 决定（P0-4 不导入即了结，否则 P9 处置）。web bridge 的接口约定在工作区根目录用 `git show 5a6d99c:AGENTS.md` 查看（Desktop Bridge Contracts 一节）。

## 重构工作方式

- 先读设计文档 §0–§1 和 `重构阶段任务/README.md`（通用任务前检查与通用验收标准），再读当前阶段的文件。细节以设计文档为准；清单与设计文档不一致时，先改设计文档，再改清单。
- 按清单顺序推进。P0a（基线导入）是阻断项，P0a 验收前 P1 起各阶段都不开工；P0b（打包与交付）阻断交付，最晚在 P5 验收前完成。开工前逐项完成"任务前检查"，收尾时逐项核对"验收标准"，并更新设计文档 §12 和阶段文件里的状态。
- 代码位置：重构的代码全部在本仓库。P0 按 P0-4 从旧仓库导出参与重构的部分作为基线，之后按设计文档在本仓库内迁移。文档里的"旧实现""门面""删除"都指本仓库里的副本，`old/` 始终不动。
- 行为不变（设计文档 §1.3）：每项能力只留一个实现。同一能力在各入口的现有差异改写为命名策略，默认值复现当前输出；用差分测试证明新旧输出一致后，删除旧实现。改变行为（合并策略、修正数值）属于 §11 的决策项，单独提 PR、单独重新生成 golden，不和重构混在一起。需要新行为时新增策略和测试，不要复制一份旧代码再改。
- 待决策项没有结论时，按清单总表"未决时的做法"保持现状，不要自行决定。需要拍板的问用户，结论记入设计文档 §11，已决的条目划线。
- 分层（设计文档 §2.1）：`foundation → media → pose → evaluation → {services, recording, live, reporting} → workflows → 入口`。只能依赖更低层；L4 的四个包互不导入（要共用的纯逻辑下沉到 L0/L1，要调另一个 L4 包的函数时由调用方注入）；L4 不向上调用 L5（finalizer 的处理步骤由构建计划的一方注入）；入口层只用各包 `__init__` 导出的公共 API；不跨模块导入下划线开头的符号（`tests/` 除外）；`core/` 不导入 tkinter。由 `tests/test_architecture.py` 强制；P9 起 allowlist 已清零，只保留设计文档 §13.2 记载的线程例外。
- 门面：迁移期间的旧导入路径门面已在 P9 全部删除。以后搬迁模块时，调用方在同一 PR 内改用新路径，不再新增门面；被取代的旧实现同一 PR 内删除，不长期双轨。
- 迁移期间的新功能一律写在新结构上，不往旧结构加代码，并遵守下一节的职责与体量规则。

## 模块职责与体量

不再出现 `App`（4855 行、135 个方法）这样的文件：一个文件只做一件事，体量有硬上限，超限项只许变小。能机器检查的部分由 `tests/test_architecture.py` 强制（设计文档 §13.2），其余在评审时把关。

- 单一职责：`core/`、`apps/`、`batch/`、`analysis/`、`scripts/` 下的每个模块都有模块 docstring，第一句写明它负责什么。要用"和""以及"连接两件不相关的事才说得清，就该拆。新代码放进设计文档 §2.2 指定的模块；不属于现有模块职责的，按 §2.2 新建模块，不顺手塞进手边的文件。
- 不建杂物模块：不新增 `utils`、`helpers`、`common`、`misc` 这类名字的模块（含 `pose_utils.py` 这样的后缀形式），共用代码按用途放进具体模块，如 `foundation/geometry.py`、`foundation/signal.py`。`__init__.py` 只做导出，只含 docstring、导入和 `__all__`。
- 体量上限（不含 `tests/`）：模块不超过 800 行，函数不超过 100 行，按物理行计数（含注释、docstring 和空行）。P8 起 Tk 层更严：`apps/tk/app.py` 不超过 300 行，每个控制器不超过 400 行，每个视图不超过 600 行。
- 预警线 600 行：超过 600 行的模块不再接收新职责。改动使模块越过 600 行时，同一 PR 内按职责拆分，或在 PR 说明里写明暂不拆的原因和拆分时点。
- 拆分按职责：拆出的每个模块、每个函数都能用一句话说清职责，并据此命名，如 `eval_force_sequence` 按"事件检测 → 时序 → 判定"拆。不按行数机械切片（`xxx_part2.py`、`xxx_impl.py`、`_step1()`），也不靠删注释、多条语句并成一行、写超长表达式来压行数。
- 体量 allowlist（`tests/architecture_allowlist.py`）已于 P9 清零，不再新增条目：超限的模块或函数在同一 PR 内按职责拆分。allowlist 的棘轮机制保留（登记值即上限，低于上限也判失败），只作为历史机制，不用于新代码。

## 硬约束

- MediaPipe 默认路径、`pose33_v3` 模板兼容和 `infer()`/`annotate()` 语义不变。`tests/test_pose33_v3_golden.py`、`test_tech_eval_contract.py`、`test_valid_mask_migration.py` 必须零漂移。只有已批准的决策可以改变数值。
- YOLO 不进入正式评分、规则评分、full tech_eval，也不进入 Tk 与 CLI 的默认路径：`apps/main.py` 不加 `--backend yolo`，`apps/app_ui.py` 不加 YOLO 后端选项，Hybrid（`yolo_body_mp_pose_supplement`）不实现。凡有后端选择的入口都经过 router（`core/pose/backends/router.py`，经 `core.pose.backends` 导出），入口层不复制分叉判断。YOLO 适配在 `core/pose/backends/yolo/`，桌面 EXE 的 spec 把它排除在外。
- 点位有效性只由 `derive_valid_mask`（`core/pose/validity.py`）判定，不要散落 `lm[idx, 3] >= thr`。缺失点不得伪造成有效点（COCO17 缺嘴角、脚跟脚尖和手指）。现有一处违反按 D-2 处理。
- YOLO 与 `body_core_v1` 的产物必须带 `backend`、`raw_layout`、`feature_layout`、`capability`、`score_authorized`、`calibration_status`、`display_scope`。`body_core_v1` 的 baseline `1.2826`、YOLO `valid_conf_thr=0.6` 已落库，但仍是 `calibration_status=unvalidated`、`score_authorized=False`，未标定的分数不得进入对外报告。这些结论出自旧仓库的 `docs/yolo_*.md`（已从工作区删除），需要原文时在工作区根目录运行 `git show 5a6d99c:docs/yolo_default_switch_decision.md` 之类的命令。
- 用户数据兼容：`%LOCALAPPDATA%\VisionSanda` 目录结构、`user_prefs.json` 键名、片段目录与文件名、练习历史 `schemaVersion=1`、成绩表列格式，以及 `feedback_config.json` 的文件名与字段。分数口径：内部 0..1，导出时 ×100，综合列用 `combined_percent`。
- 测试隔离：偏好文件路径只由 `core/foundation/prefs.py` 的 `_prefs_path()` 提供，`tests/conftest.py` 的 autouse fixture 把它钉到临时文件。测试不得读写真实的 `user_prefs.json`；以后移动该函数时，在同一个提交内把补丁目标改到新位置。
- 不新增运行时依赖。冻结 EXE 必须能运行：子进程入口是模块顶层函数，参数可 pickle，保留 `freeze_support()`。
- `old/README.md` 和本文件记载的命令在本仓库中保持可用，必要时保留为不超过 30 行的入口壳。

## 环境与命令

命令在本仓库根运行，按 PowerShell 写。用 Git Bash 时，路径写成 `./.venv/Scripts/python.exe`，环境变量用 `export`。

- 联网下载（pip 包、模型 `.task`/`.pt`、GitHub、HuggingFace）一律走本地代理 `http://127.0.0.1:7879`（端口以 Windows 系统代理设置为准）：

  ```powershell
  $env:HTTP_PROXY  = "http://127.0.0.1:7879"
  $env:HTTPS_PROXY = "http://127.0.0.1:7879"
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt --proxy http://127.0.0.1:7879
  ```

- Python 用本仓库的 `.\.venv\Scripts\python.exe`，由 P0 新建（P0-1）；pip 一律用 `python -m pip`。不使用也不修改 `old/.venv`。本仓库的 `.venv` 建好之前，按只读规则用 `D:\DevTools\venvs\vision\Scripts\python.exe` 运行旧代码作参照。
- MediaPipe 模型首次运行时自动下载到 `models/`；YOLO 权重要手动放进 `models/`。

```powershell
.\.venv\Scripts\python.exe apps/app_ui.py                  # Tk 桌面
.\.venv\Scripts\python.exe apps/main.py --source 0         # CLI 摄像头
.\.venv\Scripts\python.exe apps/main.py --source input.mp4 --pose heavy --out out.mp4 --no-show --workers 4
.\.venv\Scripts\python.exe apps/make_template.py --video action.mp4 --pose heavy --preview
.\.venv\Scripts\python.exe apps/match_template.py --template template.npz --video test.mp4 --preview
.\.venv\Scripts\python.exe batch/batch_dual_compare.py --standard_dir "标准样本" --student_dir "学员样本" --pose full --out_dir "输出目录" --rules --rule-action stance
.\.venv\Scripts\python.exe scripts/sim_exam_flow.py         # 无摄像头模拟考试流程
```

- `batch_dual_compare.py --paired` 只用于默认的 MediaPipe `pose33_v3` 路径：按学员文件名里的正/侧关键词配对，再调用 `compare_dual_streams`，CSV 的 `video` 列写学员 id。`--export_raw` 在 paired 模式下跳过；`body_core_v1`/YOLO 调试批处理不支持 paired。
- 打包与交付按 P0 执行：合并 spec、写入 `build_info`、只从干净的 commit 构建。唯一的 spec 是 `app_ui_onefile.spec`，构建命令写在它顶部：`.\.venv\Scripts\python.exe -m PyInstaller app_ui_onefile.spec --noconfirm`；打包路径不干净时拒绝构建，产物是 `dist/vision_ui.exe` 和交付目录 `dist/vision_ui_<日期>_<commit>/`。EXE 自检命令：`vision_ui.exe --offline-self-check <报告.json>`。

## 测试门禁

每个 PR 都要过：

- golden 与契约：`.\.venv\Scripts\python.exe -m pytest tests/test_pose33_v3_golden.py tests/test_valid_mask_migration.py tests/test_tech_eval_contract.py tests/test_rule_availability.py -q`
- 架构测试：`.\.venv\Scripts\python.exe -m pytest tests/test_architecture.py -q`
- 受影响模块的套件（见下表）；改动面大时跑全量 `-m pytest tests -q`
- `git diff --check`：行尾提示可以接受，空白错误不行

CI（`.github/workflows/gates.yml`，检查名 `gates`）在每个 PR 和 `main` 的推送上自动跑上面前两条和空白检查，任何一步退出码非 0 即检查失败。环境：`windows-latest`、Python 3.12，只装 `requirements.txt`；不用摄像头，也不预置 `models/` 下的模型，门禁测试只读 `tests/fixtures/`。受影响模块的套件和全量测试不在 CI 里，仍在本地跑。本地复现 CI 的空白检查：`git -c core.whitespace=cr-at-eol diff --check origin/main HEAD`。合入约束由仓库规则集 `main-gates`（作用于默认分支）强制：`gates` 是必需检查，未通过不能合并；`main` 只能经 PR 合入，禁止强推和删除。

| 改动范围 | 测试（`tests/` 下） |
|---|---|
| 考试 | `test_exam_session.py test_exam_roster.py test_exam_clip.py test_presence_gate.py test_recording_postprocess.py test_exam_panel.py`，再跑 `scripts/sim_exam_flow.py` |
| batch paired / 双流 | `test_batch_backend_args.py test_compare_dual_streams.py` |
| YOLO / layout / router | `test_yolo_backend_contract.py test_yolo_landmark_mapping.py test_body_core_layout.py test_backend_routing_contract.py` |
| 录制 | `test_recording_controller.py test_recording_postprocess.py` |
| 学生练习 | `test_action_feedback.py test_student_presence.py test_rule_engine_entry.py` |
| 规则引擎 / 评估 | `test_rule_availability.py test_rule_engine_entry.py test_p3_eval_differential.py` |
| 摄像头与运行时 | `test_camera_warmup.py test_camera_service.py test_camera_enum.py test_tk_cameras.py`；源码模式跑 `test_camera_capture.py::test_tk_preview_student_switch_with_process_cameras`（真实 Tk + spawn），再构建 EXE 跑 `--offline-self-check`；整套自检只能在冻结包里跑 |

全量基线和已知失败见 `重构阶段任务/README.md` 的"通用任务前检查"，不得新增失败。替换掉的测试要在 PR 里列出对应的新测试（D-13）。

## 编码约定

- Python 用 4 空格缩进，函数小而清晰，公共边界带类型标注。
- 视频导出的最终产物是 H.264 MP4。本机 OpenCV 没有 H.264 编码器，`open_video_writer` 会回退成 MJPG/XVID AVI。出片后用系统 `ffmpeg` 转码（`-c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p`），并删除临时 AVI；不要把回退的 AVI 当最终产物。
- 不提交大文件和生成产物：模型、视频、`outputs/`、`dist/`、`build/`。

## 提交与 PR

- 重构的提交只进本仓库，默认分支 `main`，不直接推送 `main`。
- 一个阶段一个分支、一个 PR：前置阶段的 PR 合并后，从最新的 `main` 新建 `refactor/p<N>-<主题>`，开工时 `git status` 干净；阶段内可以分多次提交。验收标准全部满足后推送分支，用 `gh pr create --base main` 开 PR。用户已同意：阶段分支上的提交、推送和开 PR 不必逐次询问；PR 由用户确认后合并。
- 跨多层的大改动用 stacked PR（官方扩展 `gh stack`，安装：`gh extension install github/gh-stack`），不开一串互不相干的 PR：每层一个分支、一个 PR，基于下一层，最底层基于 `main`；分支名 `<决策编号>/<层名>`，如 `d18/capture`。每层单独过门禁、单独可审阅，不留半接线的旧路径；文档与决策放最底层。`gh stack add` 加层，`gh stack submit` 推送并开 PR，`gh stack rebase` 在下层变动后重排。整栈由用户确认后合入（`gh stack merge`）。
- Conventional Commits：`feat:`、`fix:`、`refactor:`、`docs:`、`test:`、`chore:`。PR 和 issue 用中文，写明问题、改动、验证命令和结果，以及被替换的测试和对应的新测试（D-13）；UI 改动附截图，或说明可见行为。
- 提交或推送前确认身份：`gh auth status`、`gh api user`、`git config user.name`、`git config user.email`。GitHub connector 不可用时，用本地的 `gh pr view`、`gh pr checks`、`gh issue view`。

## 变更记录

- 每次完成任务后，把修改总结写进 `change.md`，最新记录放在最上面。记录内容：修改日期、问题描述、修改内容、验证方法。只改文档时也要写同步依据，并至少做一次轻量验证。阶段 PR 要包含对应的记录。
- `change.md` 过长时，用 `git mv` 把它归档为 `change（<起>~<止月>）.md`，再新建一个空的 `change.md`，把"归档与重建"作为新文件的第一条记录，并更新下面的归档清单。
- 历史归档在旧项目里：`old/change（start~2026.5）.md`、`old/change（2026.5~2026.6）.md`、`old/change（2026.6~2026.7）.md`。

## 代码地图

路径相对于本仓库根。分层与各模块职责见设计文档 §2；旧项目文件的去向见设计文档附录 B，需要保留的能力清单见附录 A。

- `core/`（按层，从低到高；各包经 `__init__` 导出公共 API）：
  - `foundation/`（L0）：路径（`paths.py`）、用户偏好（`prefs.py`）、原子写、几何与信号基元、任务取消、JSON 化、非模板 npz 读取（`npz.py`）、诊断日志、本机环境查询（`host.py`：桌面目录、CPU 与内存）。
  - `media/`（L1）：视频读取（`video_source.py`）、视频写出与转码（`writer.py`）、画面旋转（`transforms.py`）、帧契约与采集时钟（`frames.py`）、录像时间轴（`pacing.py`）、到达时刻去抖（`frame_clock.py`）与采集记录 `capture.json`（`capture_record.py`）、片段布局；`media/camera/` 是摄像头：设备目录（`devices.py`）、采集子进程（`capture_process.py`）、会话期间每路的持续读帧线程、最新帧槽与录像用的有界帧队列（`reader.py`）、池（`pool.py` 占用表与租约，`slot.py`、`slot_lifecycle.py` reader 与释放重试，`pool_checks.py` 参数校验）与 `CameraService`（`service.py`）。
  - `pose/`（L2）：MediaPipe 引擎与模型、`PoseSequence` 提取、`derive_valid_mask`（`validity.py`）、特征与布局（`pose33_v3`、`body_core_v1`）、归一化、并行引擎、平滑与绘制、骨架视频导出（`annotate.py`）；`pose/backends/` 是 router、body_core 特征与 YOLO 适配（`yolo/`，不进 EXE）。
  - `evaluation/`（L3）：模板（`templates.py`）、匹配（`matching/`：DTW、单/双模板、双流、在线匹配、body_core）、切分与相位、测量（`measures/`）、规则（`rules/`：`spec.py`、唯一引擎 `engine.py` 的 `evaluate_rules`、唯一目录 `sanda_v5.py`、`adapters.py`；考试、练习、离线共用的单视角判定 `rule_checks.py`）、评分口径（`scoring/`）、学生练习分析（`practice.py`）、技术指标（`techeval/`）。
  - L4：`services/`（离线分析、模板生成、练习历史、录制性能测试 `recording_benchmark.py`）、`recording/`（录制控制、摄像头会话每路的录像线程 `worker.py`、开录确认与异步结束 `recorder.py`、录后 finalizer 与步骤、录后骨架成片 `skeleton.py`）、`live/`（实时会话 `SessionRunner`、预览与按时间采样的站位检测 `presence.py`）、`reporting/`（练习文本与 HTML 报告）。
  - `workflows/`（L5）：考试（`exam/`：会话、名册、片段、成绩表）、学生练习会话、命令执行器与工作流宿主、在场闸门、播报。
- `apps/`（入口）：
  - 桌面：`app_ui.py`（入口壳）→ `tk/`（`app.py` 组合根、`controllers/`、`views/`、`dispatcher.py`、`shutdown.py`、`self_check.py` 自检内容、`build_info.py`；录制性能测试是隐藏入口 Ctrl+Shift+B → `controllers/benchmark.py` 与 `views/benchmark_dialog.py`，报告存到桌面，甲方操作见 `docs/录制性能测试操作说明.md`）；`desktop_launcher.py` 是打包入口（`--offline-self-check`）。
  - CLI：`main.py`、`make_template.py`、`match_template.py`（入口壳）→ `cli/`。
- `batch/`、`analysis/`：离线批处理、tech_eval、标定和 benchmark，只用 `core` 各包的公共 API，不接入默认的生产评分路径。
- `scripts/`：`sim_exam_flow.py`（无摄像头模拟考试）、`match_segments_probe.py`、`generate_feedback_icons.py`。
- 其它：`tests/` 是 pytest 回归与 fixtures（架构测试 `test_architecture.py`、allowlist `architecture_allowlist.py`、差分基准 `eval_diff/`）；`assets/feedback/` 是结果页状态图标；`packaging/pyinstaller/` 是单文件 spec 也要用的 runtime hook。
