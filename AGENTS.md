# Repository Guidelines

本文件是 vision 重构仓库的统一规范，适用于所有 coding agent。`CLAUDE.md` 只导入本文件；规则有变化时只改这里。

## 仓库现状

武术散打动作识别与评分系统，包括 Tk 桌面（实时预览、双摄录制、考试、学生练习）、CLI、batch 与 analysis 离线工具，以 Windows 单文件 EXE 交付。正式评分以 MediaPipe 为准。本仓库按 `整体架构重构设计.md`（下称设计文档）重构这个系统，各阶段状态见 `重构阶段任务/README.md`。

- 本仓库：工作区的 `new/` 目录，远端 `https://github.com/gy212/vision-refactor`（私有）。现有内容是设计文档、`重构阶段任务/`（P0–P9 执行清单）、摄像头设计 `camera_session_architecture.md` 和 `摄像头架构设计交接说明.md`、变更记录 `change.md`。代码由 P0 从旧仓库导出。
- 旧项目：工作区的 `old/` 目录（相对本仓库是 `../old/`），内部结构就是整理前的仓库根：代码、测试、`.venv`、`models/`、`dist/`，以及旧文档（`old/README.md`、`old/docs/`、change 归档）。它属于外层仓库（工作区根目录的 `.git`，远端 `YGuo-2/vision`）。
- 路径约定：本仓库文档中的路径相对于本仓库根，`old/` 开头的指旧项目。现有文件的路径（如 `apps/app_ui.py`、`docs/…`）指旧项目中的同名文件；P0 导入后，导入的部分以本仓库为准。

`old/` 只读，只作参照：不修改、不新增、不删除其中的任何文件（包括 `.venv`、缓存和输出）；除非用户要求，也不在外层仓库提交。优先阅读源码；必须运行旧代码对照时，用 `python -B`（pytest 再加 `-p no:cacheprovider`），工作目录放在 `old/` 之外，输出路径显式指到 `old/` 之外。

已弃用，不参与重构：web 端，包括 `apps/ui_backend.py`、`frontend/`、`ui_backend_sidecar.spec`、`package.json`（`npm run *` 脚本）、`start-desktop-dev.cmd`、`scripts/` 下 3 个 Tauri 脚本，以及相关测试（`tests/test_ui_backend_*.py`、`test_vue_tauri_acceptance_gaps.py`、`test_windows_packaging_smoke.py`）。建议 P0 不把它们导入本仓库，留在 `old/` 即归档（P0-4）。如果导入，就不改代码、不加功能、不作门禁，迁移模块时为它保留旧路径门面，web 测试的失败不得多于基线，去留由 D-10 在 P9 决定。web bridge 的接口约定在工作区根目录用 `git show 5a6d99c:AGENTS.md` 查看（Desktop Bridge Contracts 一节）。

## 重构工作方式

- 先读设计文档 §0–§1 和 `重构阶段任务/README.md`（通用任务前检查与通用验收标准），再读当前阶段的文件。细节以设计文档为准；清单与设计文档不一致时，先改设计文档，再改清单。
- 按清单顺序推进。P0 是阻断项，P0 验收前 P1 起各阶段都不开工。开工前逐项完成"任务前检查"，收尾时逐项核对"验收标准"，并更新设计文档 §12 和阶段文件里的状态。
- 代码位置：重构的代码全部在本仓库。P0 按 P0-4 从旧仓库导出参与重构的部分作为基线，之后按设计文档在本仓库内迁移。文档里的"旧实现""门面""删除"都指本仓库里的副本，`old/` 始终不动。
- 行为不变（设计文档 §1.3）：每项能力只留一个实现。同一能力在各入口的现有差异改写为命名策略，默认值复现当前输出；用差分测试证明新旧输出一致后，删除旧实现。改变行为（合并策略、修正数值）属于 §11 的决策项，单独提 PR、单独重新生成 golden，不和重构混在一起。需要新行为时新增策略和测试，不要复制一份旧代码再改。
- 待决策项没有结论时，按清单总表"未决时的做法"保持现状，不要自行决定。需要拍板的问用户，结论记入设计文档 §11，已决的条目划线。
- 分层（设计文档 §2.1）：`foundation → media → pose → evaluation → {services, recording, live, reporting} → workflows → 入口`。只能依赖更低层；L4 的四个包互不导入；入口层只用各包 `__init__` 导出的公共 API；不跨模块导入下划线开头的符号（`tests/` 除外）；`core/` 不导入 tkinter。P1 起由 `tests/test_architecture.py` 强制，allowlist 只减不增。
- 门面：迁移期间，旧导入路径可以保留为只做转发的门面，门面里不能有逻辑；按设计文档 §12 和清单附录 B 在对应阶段删除，大部分在 P9。阶段结束时，被取代的旧实现必须删除，不长期双轨。
- 迁移期间的新功能一律写在新结构上，不往旧结构加代码。新代码模块不超过 800 行，函数不超过 100 行；现有超限项只许变小。

## 硬约束

- MediaPipe 默认路径、`pose33_v3` 模板兼容和 `infer()`/`annotate()` 语义不变。`tests/test_pose33_v3_golden.py`、`test_tech_eval_contract.py`、`test_valid_mask_migration.py` 必须零漂移。只有已批准的决策可以改变数值。
- YOLO 不进入正式评分、规则评分、full tech_eval，也不进入 Tk 与 CLI 的默认路径：`apps/main.py` 不加 `--backend yolo`，`apps/app_ui.py` 不加 YOLO 后端选项，Hybrid（`yolo_body_mp_pose_supplement`）不实现。凡有后端选择的入口都经过 router（现为 `core/backend_router.py`），入口层不复制分叉判断。
- 点位有效性只由 `derive_valid_mask`（`core/pose_features.py`）判定，不要散落 `lm[idx, 3] >= thr`。缺失点不得伪造成有效点（COCO17 缺嘴角、脚跟脚尖和手指）。现有一处违反按 D-2 处理。
- YOLO 与 `body_core_v1` 的产物必须带 `backend`、`raw_layout`、`feature_layout`、`capability`、`score_authorized`、`calibration_status`、`display_scope`。`body_core_v1` 的 baseline `1.2826`、YOLO `valid_conf_thr=0.6` 已落库，但仍是 `calibration_status=unvalidated`、`score_authorized=False`，未标定的分数不得进入对外报告。这些结论出自旧仓库的 `docs/yolo_*.md`（已从工作区删除），需要原文时在工作区根目录运行 `git show 5a6d99c:docs/yolo_default_switch_decision.md` 之类的命令。
- 用户数据兼容：`%LOCALAPPDATA%\VisionSanda` 目录结构、`user_prefs.json` 键名、片段目录与文件名、练习历史 `schemaVersion=1`、成绩表列格式，以及 `feedback_config.json` 的文件名与字段。分数口径：内部 0..1，导出时 ×100，综合列用 `combined_percent`。
- 测试隔离：偏好文件路径只由 `_prefs_path()` 提供，`tests/conftest.py` 的 autouse fixture 把它钉到临时文件。测试不得读写真实的 `user_prefs.json`。P1 迁移 `core/paths.py` 时，在同一个提交内把补丁目标改到新位置。
- 不新增运行时依赖。冻结 EXE 必须能运行：子进程入口是模块顶层函数，参数可 pickle，保留 `freeze_support()`。
- `old/README.md` 和本文件记载的命令在本仓库中保持可用，必要时保留为不超过 30 行的入口壳。

## 环境与命令

命令在本仓库根运行，按 PowerShell 写；P0 导出代码之前，本仓库还没有代码。用 Git Bash 时，路径写成 `./.venv/Scripts/python.exe`，环境变量用 `export`。

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
.\.venv\Scripts\python.exe batch/batch_dual_compare.py --standard_dir "标准样本" --student_dir "学员样本" --pose full --out_dir "输出目录" --rules --action both
.\.venv\Scripts\python.exe scripts/sim_exam_flow.py         # 无摄像头模拟考试流程
```

- `batch_dual_compare.py --paired` 只用于默认的 MediaPipe `pose33_v3` 路径：按学员文件名里的正/侧关键词配对，再调用 `compare_dual_streams`，CSV 的 `video` 列写学员 id。`--export_raw` 在 paired 模式下跳过；`body_core_v1`/YOLO 调试批处理不支持 paired。
- 打包与交付按 P0 执行：合并 spec、写入 `build_info`、只从干净的 commit 构建。构建命令写在 spec 顶部，如 `.\.venv\Scripts\python.exe -m PyInstaller app_ui_onefile.spec --noconfirm`。EXE 自检命令：`vision_ui.exe --offline-self-check <报告.json>`。

## 测试门禁

每个 PR 都要过：

- golden 与契约：`.\.venv\Scripts\python.exe -m pytest tests/test_pose33_v3_golden.py tests/test_valid_mask_migration.py tests/test_tech_eval_contract.py tests/test_rule_availability.py -q`
- 架构测试（P1 起）：`.\.venv\Scripts\python.exe -m pytest tests/test_architecture.py -q`
- 受影响模块的套件（见下表）；改动面大时跑全量 `-m pytest tests -q`
- `git diff --check`：行尾提示可以接受，空白错误不行

| 改动范围 | 测试（`tests/` 下） |
|---|---|
| 考试 | `test_exam_session.py test_exam_roster.py test_exam_clip.py test_presence_gate.py test_recording_postprocess.py test_exam_panel.py`，再跑 `scripts/sim_exam_flow.py` |
| batch paired / 双流 | `test_batch_backend_args.py test_compare_dual_streams.py` |
| YOLO / layout / router | `test_yolo_backend_contract.py test_yolo_landmark_mapping.py test_body_core_layout.py test_backend_routing_contract.py` |
| 录制 | `test_recording_controller.py test_recording_postprocess.py` |
| 学生练习 | `test_action_feedback.py test_student_presence.py` |
| 摄像头与运行时（P0、P5、P6、P8、P9） | 源码模式跑 `test_camera_capture.py::test_tk_preview_student_switch_with_process_cameras`（真实 Tk + spawn），再构建 EXE 跑 `--offline-self-check`；整套自检只能在冻结包里跑 |

全量基线和已知失败见 `重构阶段任务/README.md` 的"通用任务前检查"，不得新增失败。替换掉的测试要在 PR 里列出对应的新测试（D-13）。

## 编码约定

- Python 用 4 空格缩进，函数小而清晰，公共边界带类型标注。
- 视频导出的最终产物是 H.264 MP4。本机 OpenCV 没有 H.264 编码器，`open_video_writer` 会回退成 MJPG/XVID AVI。出片后用系统 `ffmpeg` 转码（`-c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p`），并删除临时 AVI；不要把回退的 AVI 当最终产物。
- 不提交大文件和生成产物：模型、视频、`outputs/`、`dist/`、`build/`。

## 提交与 PR

- 重构的提交只进本仓库，默认分支 `main`，不直接推送 `main`。
- 一个阶段一个分支、一个 PR：前置阶段的 PR 合并后，从最新的 `main` 新建 `refactor/p<N>-<主题>`，开工时 `git status` 干净；阶段内可以分多次提交。验收标准全部满足后推送分支，用 `gh pr create --base main` 开 PR。用户已同意：阶段分支上的提交、推送和开 PR 不必逐次询问；PR 由用户确认后合并。
- Conventional Commits：`feat:`、`fix:`、`refactor:`、`docs:`、`test:`、`chore:`。PR 和 issue 用中文，写明问题、改动、验证命令和结果，以及被替换的测试和对应的新测试（D-13）；UI 改动附截图，或说明可见行为。
- 提交或推送前确认身份：`gh auth status`、`gh api user`、`git config user.name`、`git config user.email`。GitHub connector 不可用时，用本地的 `gh pr view`、`gh pr checks`、`gh issue view`。

## 变更记录

- 每次完成任务后，把修改总结写进 `change.md`，最新记录放在最上面。记录内容：修改日期、问题描述、修改内容、验证方法。只改文档时也要写同步依据，并至少做一次轻量验证。阶段 PR 要包含对应的记录。
- `change.md` 过长时，用 `git mv` 把它归档为 `change（<起>~<止月>）.md`，再新建一个空的 `change.md`，把"归档与重建"作为新文件的第一条记录，并更新下面的归档清单。
- 历史归档在旧项目里：`old/change（start~2026.5）.md`、`old/change（2026.5~2026.6）.md`、`old/change（2026.6~2026.7）.md`。

## 现有代码地图

路径相对于项目根，`old/` 和 P0 导出后的本仓库相同。每个文件的去向见设计文档附录 B，需要保留的能力清单见附录 A。

- `core/`：
  - 姿态与匹配：`vision_pipeline.py`（MediaPipe Tasks 封装）、`pose_features.py`（归一化、DTW、`derive_valid_mask`）、`feature_layout.py`、`action_compare.py`（单/双模板对比、`compare_dual_streams`）、`online_matcher.py`、`parallel_pose_engine.py`、`preview_smoother.py`。
  - 评分与练习：`rule_scoring.py`、`kick_quality.py`、`action_feedback.py`（学生练习 v4，`RULE_VERSION`）、`feedback_geometry.py`、`feedback_phases.py`、`feedback_history.py`、`feedback_report.py`。
  - 后端：`backend_router.py`、`yolo_adapter.py`、`body_core_compare.py`、`model_manager.py`。
  - 录制与考试：`recording_controller.py`、`exam_session.py`、`exam_roster.py`、`presence_gate.py`、`exam_clip.py`、`exam_announcer.py`。
  - 其它：`paths.py`、`video_writer.py`、`qwen_coach.py`（去留见 D-12）。
- `apps/`：
  - 桌面：`app_ui.py`（Tk 主程序）、`desktop_launcher.py`（打包入口，`--offline-self-check`）、`desktop_smoke.py`（自检内容）、`exam_panel.py`、`feedback_panel.py`、`feedback_result_panel.py`、`recording_postprocess.py`（双摄录后处理与池级比对）。
  - 摄像头：`camera_enum.py`、`camera_capture.py`、`camera_warmup.py`（PR #75 的池 + 租约）。
  - CLI：`main.py`、`make_template.py`、`match_template.py`。`ui_backend.py` 已弃用。
- `batch/`、`analysis/`：离线批处理、tech_eval、标定和 benchmark，不接入默认的生产评分路径。
- `scripts/`：`sim_exam_flow.py`、`match_segments_probe.py`、`generate_feedback_icons.py`；其余是 web 端和 qwen 的脚本。
- 其它：`tests/` 是 pytest 回归与 fixtures；`assets/feedback/` 是结果页状态图标；`packaging/pyinstaller/` 是单文件 spec 也要用的 runtime hook。
