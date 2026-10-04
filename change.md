## 2026-10-03（America/Los_Angeles）: [feat] P0b 打包与交付：单一 spec、build_info、干净构建、自检补强、交付目录

- 问题描述：两份 Tk spec 并存且都没有收入 `assets/feedback`（9-29 包的结果页因此没有状态图标）；EXE 里没有源码版本信息，v4 曾从未跟踪的源码出包；交付附带文件不在仓库里；自检不检查图标和构建来源。
- 修改内容：
  - spec（P0-2）：删除目录版 `app_ui.spec`；`app_ui_onefile.spec` 收入 `assets/feedback`、`build_info.json`，顶部写明构建命令、产物和干净构建规则，EXE 生成后组织交付目录。
  - `packaging/pyinstaller/build_info.py`：记录 commit、UTC 构建时间、`RULE_VERSION`（从源码读取，不导入业务模块）、4 个模型的 SHA-256、Python 与 PyInstaller 版本、是否干净；`core apps analysis batch assets templates packaging app_ui_onefile.spec` 下有改动或未跟踪文件时拒绝构建（比任务 3 多了 `templates`、`packaging`，两者都打进 EXE）。顶层包名 `packaging` 与 PyPI 的 `packaging` 冲突，spec 与测试按文件路径加载。
  - 交付（P0-3）：`packaging/pyinstaller/delivery.py` 把 EXE、`build_info.json` 和 `packaging/delivery/` 复制到 `dist/vision_ui_<日期>_<commit>/`。`使用说明.txt` 由 9-20 包改写（EXE 名、v4 的分析模型与逐帧分析、图文报告、14 天历史）；`feedback_config.example.json` 按 v4 `FeedbackConfig` 默认值重新生成（9-20 包里是 v3 参数，去掉由界面决定的 `pose_variant`）；`第三方许可/` 按 `PYZ-00.toc` 实际收入的包从 `.venv` dist-info 重建，比 9-20 包多出 openpyxl、et_xmlfile、absl-py、flatbuffers（上游未附文件，放 Apache-2.0 全文）、sounddevice、Python，numpy 换成当前版本，新增 `清单.txt`；`.gitattributes` 让许可证原样入库（pillow 的上游文本有行尾空格）。
  - 自检（`apps/desktop_smoke.py`）：报告写入 `buildInfo`，冻结包内断言 `build_info.json` 存在、`clean=true`、规则版本与 `RULE_VERSION` 一致；断言 24 个结果页状态图标都能从包内解码且尺寸正确；新增 `build_info`、`feedback_status_icons` 两项检查。
  - 新测试 `tests/test_packaging_build_info.py`（11 项）：干净/改动/未跟踪/打包路径外改动、build_info 字段、真实规则版本、交付目录内容、配置示例与默认值同步、源码树 24 个图标、缺图标报错、只有一份 spec 且收入图标和 build_info、每个运行时依赖都有许可证。未替换任何旧测试（D-13）。
  - 文档：P0 文件、README 阶段表、设计文档 §12 的 P0 行与工作量段、AGENTS.md 的打包一条。设计文档风险表第一行要等实机验收后交付时再改。
  - 未完成：§13.4 实机清单要在甲方的目标双摄上由人执行。按用户要求，阶段 PR 等 P0 全部完成后再开。
- 同步依据：`apps/feedback_result_panel.py` 的 `_ASSETS_DIR` 与图标文件名；`core/paths.py` 的冻结模式解析；`core/feedback_geometry.py` 的 `FeedbackConfig`；`old/dist/散打动作练习_双视角版_20260920/` 的附带文件；`build/app_ui_onefile/PYZ-00.toc`；`imageio_ffmpeg.get_ffmpeg_version()` 为 7.1 essentials（与 `FFmpeg说明.txt` 一致）。
- 验证方法：工作区有未提交改动和未跟踪文件时运行 PyInstaller，输出"拒绝构建"、退出码 1、不生成 `dist/`；从干净的 `2219a8f` 构建成功，`vision_ui.exe --offline-self-check` 退出码 0，报告 `ok=true`、`frozen=true`、`feedbackVersion=sanda-feedback-mediapipe-2026-09-20-v4`、`feedbackIcons=24`、`buildInfo.commit=2219a8f…`、`clean=true`，`checks` 共 12 项，模型哈希与 `models/` 的 `sha256sum` 一致；报告留在 `outputs/p0b/`（不入库）。交付目录含 EXE、`build_info.json`、`使用说明.txt`、`feedback_config.example.json`、`第三方许可/`。全量 `pytest tests -q` 958 passed（947 + 新增 11）；golden 与契约门禁 49 passed；`git diff --check` 通过。

## 2026-10-03（America/Los_Angeles）: [chore] P0a 基线导入：从旧仓库 5a6d99c 导入代码，新建 .venv，测定基线

- 问题描述：P0a 阻断 P1。本仓库此前只有文档，需要按 P0-4 从旧仓库导入参与重构的代码、新建本仓库 `.venv` 并重新测定全量基线。开工前用户对 P0-1～P0-4 均采纳清单建议。
- 修改内容：
  - 导入（提交 `5574d95`，内容与 `5a6d99c` 一致）：`git archive 5a6d99c` 导出 `core/`、`apps/`、`analysis/`、`batch/`、`scripts/`、`tests/`、`assets/`、`templates/`、`packaging/`、三份 requirements、两份 Tk spec，以及按原路径的 6 份 YOLO 文档和 `docs/yolo_eval_samples.json`（`analysis/` 三个脚本的 `--samples` 默认值）。未导入：web 端（`apps/ui_backend.py`、`frontend/`、`ui_backend_sidecar.spec`、`package.json`、`start-desktop-dev.cmd`、3 个 Tauri 脚本、`test_ui_backend_*.py` 5 个、`test_vue_tauri_acceptance_gaps.py`、`test_windows_packaging_smoke.py`）、其余旧文档、`README.md`、`DUAL_TEMPLATE_COMPARE.md`、change 归档。D-10 随之了结。
  - 门禁测试拆分：`tests/test_backend_routing_contract.py` 去掉顶层 `from apps import ui_backend`，删除依赖它的 4 个用例（`test_ui_session_start_uses_yolo_preview_pipeline_and_surfaces_frame_meta`、`test_ui_session_start_requires_capabilities_prevents_yolo_preview_route`、`test_default_pipeline_factory_maps_yolo_realtime_profile_to_yolo26n_without_loading_ultralytics`、`..._yolo26s_...`）和只被它们用到的 `_wait_frame_events`、`time`、`numpy` 导入；`tests/test_batch_backend_args.py` 去掉顶层导入，删除 2 个用例（`test_ui_high_quality_analysis_runs_yolo26l_internal_body_core`、`test_ui_high_quality_missing_yolo26l_is_structured_error_without_job`）。这 6 个用例测的是已弃用的 web 桥，留在 `old/`，没有替换测试（D-13）；router 本身的契约仍由两文件其余 11 + 17 个用例覆盖。
  - 依赖（P0-1）：`requirements.txt` 的 pytest 固定为 9.1.1；`requirements-dev.txt` 的 pytest 由 8.4.2 改为 9.1.1，新增 `pyinstaller==6.22.3`、`imageio-ffmpeg==0.6.0`（`app_ui_onefile.spec` 顶层导入）。
  - 环境：`D:\DevTools\Python312\python.exe -m venv .venv`（`include-system-site-packages = false`），经代理 7879 安装两份 requirements；`old/models/` 的 4 个 MediaPipe 模型复制到 `models/`（不入库）。
  - 文档：设计文档 §11.2 新增 P0-1～P0-4 并划线、D-10 划线；§12 P0 行、基线段、`.venv` 段、工作量段改为 P0a 已完成；README 阶段表、决策表、通用任务前检查的基线；P0 文件的状态、任务前检查、决策结论与 P0a 验收。按用户要求，阶段 PR 等 P0b 完成后再开。
- 同步依据：`git ls-tree`/`git grep 5a6d99c` 确认导入范围与对 `docs/`、`templates/`、`ui_backend` 的引用；`old/` 与本仓库逐文件的用例收集数对比；`D:\DevToolsenvsision` 的 `pip freeze`。
- 验证方法：本仓库 `.venv` 全量 `pytest tests -q`：947 passed，0 failed，0 skipped。旧基线 1033 项（1026 + 6 + 1）减去未导入的 86 项（web 端 7 个文件 80 项，含 5 项失败与 1 项跳过；拆出的 6 项，含 1 项失败）正好 947，其余每个文件的用例数与 `old/` 相同。golden 与契约门禁 49 passed；`test_camera_capture.py::test_tk_preview_student_switch_with_process_cameras` passed；`.venv` 中 `pip freeze` 与 DevTools 环境只差 `comtypes`（1.4.17/1.4.16）和 `pyinstaller-hooks-contrib` 的补丁版本；`git diff --check` 通过。

## 2026-10-03（America/Los_Angeles）: [docs] AGENTS.md 新增"模块职责与体量"：单一职责、体量上限与棘轮 allowlist；D-15 已决

- 问题描述：用户要求防止重构后再长成"屎山"：不允许单文件承担过多职责，不允许单文件代码行数过度累积。原规则只有一句"新代码模块不超过 800 行，函数不超过 100 行；现有超限项只许变小"：没有职责要求；只管"新代码"，搬迁过来的旧文件不受约束；"只许变小"没有记录行数，架构测试无法校验，文件降下来后还能涨回去；没有预警，模块要到 799 行才会被注意。另外 `CameraWarmupPool` 是单个 969 行的类，P5"池不重写"原样迁入后 `pool.py` 仍超过 800 行，与 P9 的 allowlist 清零冲突，文档没有处置。用户随后拍板：D-15 采纳建议；`tests/` 不纳入体量上限。
- 修改内容：
  - AGENTS.md：新增"模块职责与体量"一节，取代"重构工作方式"末条里的体量一句：模块 docstring 第一句写明职责，新代码按设计文档 §2.2 落位；不建 `utils`/`helpers`/`common`/`misc` 类模块，`__init__.py` 只做导出；模块 ≤ 800 行、函数 ≤ 100 行，按物理行计数，不含 `tests/`，P8 起 Tk 层更严；600 行预警线，越线时同一 PR 拆分或写明原因；拆分按职责，不机械切片、不压行；现有超限项按棘轮处理（记下行数上限，高于或低于上限都判失败，修 bug 加行须等量移出，搬迁不重置上限，条数只减不增，P9 清零）。
  - 设计文档：§0 第 5 条、§1.1"可维护"补职责要求；§13.2"体量"行写明计数口径、不含 `tests/`、Tk 预算、600 行警告和棘轮校验，新增"职责"行（模块 docstring、杂物模块名、`__init__.py` 内容）；§11.2 新增 D-15 并按用户决定划线（P5 原样迁入，P9 按职责拆分、只搬不改）；§12 的 P5、P9 行与工作量段同步，P9 由"小"改为"中等"。
  - 阶段清单：README 已决行加 D-15，P9 规模改"中"，通用验收补"上限没有调高"与新建模块的职责、600 行说明；P1 补 2026-10-03 的体量与职责基线、先补 docstring 再记上限、验收时验证棘轮双向失败；P5 写明 D-15 已决及 `pool.py` 条目随迁；P8 的 Tk 预算写进架构测试长期生效；P9 新增按职责拆分剩余超限项（`analysis/`、`batch/`、`scripts/`）与按 D-15 拆分 `pool.py` 的任务。
  - 本次在本条之下那条记录（尚未提交的审阅改动）的基础上修改，两者合为一个提交；按用户要求不开 PR，直接推送 `main`。
- 同步依据：用 `ast` 只读统计 `old/`（`5a6d99c`，不含 web 端，不含 `tests/`，`utf-8-sig` 读取）：超过 800 行的模块 core 5、apps 5、analysis 3；超过 600 行的类 5 个，其中 `App` 4855 行、`CameraWarmupPool` 969 行；超过 100 行的函数 53 个（core 14、apps 22、batch 4、analysis 12、scripts 1）；没有模块 docstring 的 13 个；没有 `utils`/`helpers`/`common`/`misc` 类文件名；5 个 `__init__.py` 都只有一行 docstring。设计文档 §2.2 的目标文件名没有命中禁用名。
- 验证方法：`git diff --check` 通过（只有行尾提示）；表格列数检查脚本对 AGENTS.md、设计文档和全部阶段文件报 0 处不一致；搜索确认旧的体量一句只剩本条的引用，"P9 小"、"D-15 未决"、"`tests/` 同样适用"无残留；D-15 在 §11.2、§12、README、P5、P9 的引用一致且都按已决表述；"模块职责与体量"的引用都指向 AGENTS.md 中存在的章节。

## 2026-10-03（America/Los_Angeles）: [docs] 审阅设计文档与阶段清单：修正 4 处分层矛盾、拆分 P0、补门禁测试处置

- 问题描述：按用户要求审阅 `整体架构重构设计.md` 与 `重构阶段任务/`。文档里的计数与 `old/`（`5a6d99c`）逐项吻合，但有几处安排与自己定的分层规则冲突，P1 上线架构测试后会直接报错：（1）现状调用链 `annotate → draw → draw_pose_frame → _classify_pose_actions`（`core/vision_pipeline.py:178`），绘制放 L2 而实时标签放 L3，形成向上依赖；（2）`feedback_history`、`feedback_report` 迁入 `services`、`reporting` 后都要用 `SegmentLayout`，`core/feedback_history.py:198` 前后还直接调 `format_report`/`render_html_report`，与"L4 四包互不导入"冲突；（3）§8.4 把考试裁剪（L5 `workflows/exam/clip.py`）和台账回填（L5 `ScorebookRegistry`）列为 L4 finalizer 的步骤，§8.7 又写成事件驱动；（4）`apps/camera_warmup.py` 自建 reader 与回收线程（:868、:936），公开 API 用 `threading.Event`，而 §2.3、§13.2 只豁免 `ParallelPoseEngine` 与 `capture_process`，与 P5 的"池不重写"矛盾。另外 `RecordingService`、`WorkflowHost` 没在 §2.2 落位，`score_segment` 归属不清。清单方面：P0 把 §13.4 实机验收列为验收标准，而 P0 不完成 P1 不开工，硬件约不到就卡死全局，"规模：小"也不实（本仓库尚无代码）；P0-4 建议不导入 web 端，但 `tests/test_backend_routing_contract.py:34`、`tests/test_batch_backend_args.py:22` 在模块顶层 `from apps import ui_backend`，两者都是 AGENTS.md 的门禁测试，不导入即整文件 ImportError；D-10 在 README、P9、§11.2 仍按 P9 决策写；P3 遗留"拆 PR"与"一个阶段一个 PR"矛盾；两条并行线都改 `apps/recording_postprocess.py`、`apps/app_ui.py` 但没写冲突对策；D-13 设为 P1 前置但 P1 不依赖它；P5 排在 P4 之后没给理由；没有浮点容差政策。
- 修改内容：
  - 设计文档：`live_labels` 移到 `pose/`（§2.2、§3 第 20 项、§6.4、附录 B）；`SegmentLayout` 下沉到 `media/segment_layout.py`（§2.1、§2.2、§3 第 31 项、§8.4）；新增 `recording/service.py`、`recording/steps.py`（`Step` 协议，评分步骤直接用 L3），考试裁剪步骤由 L5 实现、由构建计划的一方注入，台账回填改为 `WorkflowHost` 收到终态事件后触发（§2.2、§3 第 33、38 项、§8.4、§8.6、§12 P4 行、附录 B）；新增 `workflows/host.py` 并写进组合根（§2.2、§8.6、§9.1.1）；`HistoryStore` 的报告渲染调用移出（§9.3、附录 B）；`CameraPool` 的线程与 `stop_event` 列入豁免，`CameraStream` 承担适配且长期保留（§2.3、§4.3、§13.2）；`annotate()` 注明是薄组合（§6.1）；`score_segment` 注明不供 finalizer（§2.2）；P0 拆为 P0a/P0b（文首、§0、§12 P0 行与工作量段）；§12 新增并行冲突与浮点容差两条风险，"独立回滚"改为连同后续阶段回滚；§12 基线段与 §11.2 D-10 写明两份门禁测试的拆分与 D-10 随 P0-4 了结。
  - 阶段清单：README 阶段表拆出 P0a/P0b，P1 前置改 P0a，P5 行补理由，两条线段落补合并顺序（P4 先于 P3），D-13 改到 P4 开工前，D-10 改到 P0a，附录 B 与通用验收同步；P0 文件按 3a/3b 拆任务与验收，P0-4 写明两份门禁测试的处置（去掉顶层导入，4 + 2 个依赖 `ui_backend` 的用例留在 `old/`）；P1 去掉 D-13 前置、适配器说明分长期与迁移期、补浮点容差政策；P2 `live_labels` 路径；P3 "拆 PR"改为"分批提交"、差分补容差说明；P4 补 L5 包骨架说明、`segment_layout`/`service`/`steps` 位置、步骤注入、终态事件回填、`HistoryStore` 渲染移出、D-13 成为本阶段决策、新增分层验收；P5 补池不重写的边界与排序理由；P7 `WorkflowHost` 落位与测试；P8 组合根加 `WorkflowHost`；P9 D-10 条件化、豁免清单加 `pool.py`。
  - AGENTS.md：P0 阻断改为 P0a；分层一句补 L4 下沉/注入与 L4 不向上调用；弃用段补两份门禁测试的处置，D-10 的时点改为随 P0-4。PR 粒度维持用户已定的"一个阶段一个 PR"不变。未改：`foundation/prefs.py` 集中定义业务键的安排（审阅时标为轻微，保留）。
- 同步依据：对 `old/`（`5a6d99c`）的核查：排除 web 端后 `MediaPipePipeline(` 24 处/14 文件、`cv2.VideoCapture(` 35/22、`np.load(` 22、`threading.Thread(` 27/10、`.after(` 29、App 4855 行/135 方法/162 属性、ExamPanel 11 个私有成员、半成品构造 32 + 16 处，与文档一致；`grep -ln ui_backend tests/*.py` 与 AST 统计两份门禁测试的用例数（15 个中 4 个、19 个中 2 个）；`apps/camera_warmup.py` 的线程与 `stop_event` 用法；`core/feedback_history.py:186-202` 的导出逻辑；本文件 2026-10-03 的记录里用户已决定"一个阶段一个 PR"。
- 验证方法：替换脚本逐条断言每处原文命中且只命中一次，按文件原有行尾（设计文档、AGENTS.md、P0、本文件为 CRLF，其余为 LF）写回；改后搜索本仓库文档：`evaluation/live_labels`、`recording/layout`、`recording.layout`、`拆 PR`、`P0 没完成` 无残留；`git diff --check` 通过。

## 2026-10-03（America/Los_Angeles）: [docs] 本地代理端口改为 7879

- 问题描述：本地代理已从 7890 换到 7879：Windows 系统代理为 `127.0.0.1:7879`，7890 拒绝连接。AGENTS.md"环境与命令"和 P0-基线 第 1 项任务里的代理地址过时。
- 修改内容：AGENTS.md 4 处、`重构阶段任务/P0-基线.md` 1 处的 `7890` 改为 `7879`；AGENTS.md 注明端口以 Windows 系统代理设置为准。
- 同步依据：用户本次指示；注册表 `HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings` 的 `ProxyServer`；本日经 7879 的 `git ls-remote`、`git push`、`gh api user` 均成功。
- 验证方法：除本条记录外，本仓库内不再有 `7890`；按 AGENTS.md 的写法只设置 `HTTP_PROXY`/`HTTPS_PROXY` 环境变量后，`git ls-remote origin` 和 `gh api user` 成功；`git diff --check` 通过。

## 2026-10-03（America/Los_Angeles）: [chore] 删除与本项目无关的记录，以当前内容重建初始提交

- 问题描述：初始提交 `422b324` 的 change.md 末尾有一条与本项目无关的 2026-10-02 记录（对应文件早已删除），另有三处提到它：2026-09-19 学生练习整理记录、P0 基线整合记录的"来源"一条、P0-基线 任务前检查第 1 项。用户要求全部删除，并以当前内容为基准，保证本仓库的文件和历史里都没有这些内容；外层仓库暂不处理。
- 修改内容：删除这条记录和三处提及，其余内容不变。改动用 `git commit --amend` 并入初始提交，提交说明不变；以 `--force-with-lease=main:422b324` 推送 `main` 替换远端的 `422b324`，再清空本地 reflog、回收不可达对象。
- 同步依据：用户本次指示；对本仓库工作区和 `422b324` 全文搜索文件名、关键词和间接提法，命中只有上述四处。
- 验证方法：工作区和新提交全文搜索均无命中；`git diff --check` 通过，change.md 仍全部为 CRLF 行尾；推送后用 `git ls-remote` 核对远端 `main` 与本地一致，`git status` 干净，`git fsck --unreachable --no-reflogs` 无输出。

## 2026-10-03（America/Los_Angeles）: [chore] new/ 独立为私有仓库 gy212/vision-refactor，old/ 改为只读

- 问题描述：用户决定重构在 `new/` 中进行，`old/` 一律只读参照；要求把 `new/` 初始化为 git 仓库，用 gh 新建私有库，之后每完成一个阶段就向该库提 PR。
- 修改内容：
  - `new/` 执行 `git init -b main`，新增 `.gitignore`（由 `old/.gitignore` 改写，去掉 web 端和旧目录的条目）。初始提交直接推到新建私有库的 `main`，作为 PR 的基线；之后每个阶段一个分支、一个 PR，PR 由用户确认后合并。
  - 规范：本仓库的 `AGENTS.md` 由上一条记录里的根目录版本改写为本仓库视角，增加 `old/` 只读规则、P0 导入代码、阶段分支与 PR 流程；`CLAUDE.md` 只导入它。工作区根目录的 `AGENTS.md` 改为 10 行的工作区说明并导入 `new/AGENTS.md`，根目录的 `CLAUDE.md` 导入根目录的 `AGENTS.md`。
  - 外层仓库：根目录 `.gitignore` 加入 `new/`；用 `git rm --cached` 取消外层索引对 `new/` 下 4 个文件的跟踪（只改索引，文件不动），改由本仓库管理。外层仓库的其余未提交改动保持原样。
  - 文档：仓库内路径去掉 `new/` 前缀（设计文档 4 处，摄像头设计说明、阶段 README、P5 各 1 处）；设计文档文首和阶段 README 改写仓库与路径说明。P0 改为在本仓库完成：新增待决策项 P0-4（导入范围），第 1 项任务改为导入基线并新建 `.venv`，EXE 改用干净的 commit 构建。"一个阶段一个 PR"写入设计文档 §12 和阶段 README 的通用验收；P1、P3、P6、P8 与设计文档里的"拆多个 PR"改为阶段内分多次提交，"同一 PR 内"改为"同一提交内"。
- 同步依据：用户本次指示；`gh auth status`（账号 gy212，token 有 `repo` 权限）；`git config --show-origin`（身份在用户级 `.gitconfig`；`core.autocrlf=true`；系统默认分支是 `master`，所以显式用 `-b main`）；`old/.gitignore`；`5a6d99c` 的已跟踪文件清单（259 个）；`old/.venv/pyvenv.cfg`。
- 验证方法：提交前，脚本逐条确认每处替换的命中次数符合预期，并检查本仓库文档里已没有带 `new/` 前缀的仓库内路径、`重构阶段任务/` 的相对链接都能解析；`git diff --cached --check` 通过。推送后用 `gh repo view` 核对可见性和默认分支，用 `git ls-remote` 核对远端 `main` 与本地一致；外层仓库用 `git status` 确认 `new/` 已被忽略。

## 2026-10-03（America/Los_Angeles）: [docs] 重写 AGENTS.md，CLAUDE.md 改为只导入 AGENTS.md

- 问题描述：项目整理后，AGENTS.md（227 行）和 CLAUDE.md（279 行）里的路径、命令工作目录和 change.md 位置都过时了；两份文件大量重复，仍以已弃用的 Vue/Tauri 为主线，引用的 `docs/yolo_*.md` 等文档已删除，也没有写重构的工作方式。用户要求修改优化这两份文件，为重构做准备。
- 修改内容：
  - AGENTS.md 重写为 121 行，分为：仓库现状（`old/`、`new/` 的分工；web 端弃用范围）、重构工作方式（阅读顺序、P0 阻断、代码位置待定、行为不变、待决策项、分层、门面、新代码体量）、硬约束、环境与命令（都在 `old/` 下运行）、测试门禁（按改动范围列表）、编码约定、提交与 PR、变更记录（`new/change.md` 与归档位置）、现有代码地图。
  - 删除内容：web bridge 契约、Vue/Tauri 命令与 `npm run verify:*` 门禁、只适用于 web 端的 YOLO 路由细则、2026-06-11 的验收记录。原文可用 `git show 5a6d99c:AGENTS.md`、`git show 5a6d99c:CLAUDE.md` 查看。
  - 新增一项待定事项：P1 起的新分层代码放在 `old/` 内原地迁移，还是放在 `new/` 下另建项目。确定之前不开始 P1。
  - CLAUDE.md 只剩一行 `@AGENTS.md` 导入和一句说明，规则只在 AGENTS.md 维护。
- 同步依据：设计文档 §0–§2.1、§11–§13.2；`new/重构阶段任务/README.md` 和 P0、P9 文件；`old/` 下的实际目录与文件；spec 顶部的构建命令；`apps/desktop_launcher.py` 的 `--offline-self-check` 参数；`test_windows_packaging_smoke.py` 全部 10 个测试都只针对 Tauri sidecar；`app_ui_onefile.spec` 用到 `packaging/pyinstaller/` 的 runtime hook；`git cat-file` 确认 `5a6d99c` 里有被引用的 `docs/yolo_*.md`；Claude Code 文档说明 `@path` 导入相对于 CLAUDE.md 所在目录解析，反引号里的路径不会被导入。
- 验证方法：脚本逐个检查 AGENTS.md 中的 95 个路径。除尚未创建的 `tests/test_architecture.py`、用户数据文件（`user_prefs.json`、`feedback_config.json`）、生成目录（`frontend/dist/`、`frontend/src-tauri/target/`）和根目录工具配置外，其余都存在。`new/` 文档中提到 AGENTS.md 的 6 处（硬约束、命令、change.md 记录、测试套件、D-10、P9）对应的内容仍在。`git diff --check` 通过。

## 2026-10-03（America/Los_Angeles）: [chore] 项目整理：旧项目整体移入 old/，重构文档移入 new/

- 问题描述：用户要求在仓库根新建 `old/`、`new/`：`new/` 放重构所需文档，其余全部移入 `old/`；`AGENTS.md`、`CLAUDE.md` 留在根目录，不处理。
- 修改内容：
  - `new/`：`整体架构重构设计.md`、`重构阶段任务/`（11 个文件）、`camera_session_architecture.md`、`摄像头架构设计交接说明.md`、`change.md`。
  - `old/`：其余内容按原目录结构整体移入，包括代码、测试、`docs/`、spec、requirements、`package.json`、`README.md`、`DUAL_TEMPLATE_COMPARE.md`、3 份 change 归档、原 `.gitignore`，以及未跟踪的 `.venv`、`models`、`outputs`、`dist`、`build` 和测试缓存。已跟踪文件用 `git mv`（223 处改名）。`assets`、`frontend`、`packaging`、`templates`、`tests` 的目录句柄被占用、整体改名被拒，改为逐个移动其中的条目，没有遗留。
  - 根目录只剩 `AGENTS.md`、`CLAUDE.md`、`new/`、`old/`、新的 `.gitignore`（只管根下的 `.venv/`、`.kiro/` 等；旧项目的规则随 `old/.gitignore` 生效），以及未移动的 `.git`、`.kiro`、`.agents`（版本库与 agent 工具配置）。
  - `new/` 文档：路径引用改为新位置；设计文档文首和阶段 README 注明现有代码路径相对于 `old/`、命令在 `old/` 下运行；P0 任务前检查更新了工作区现状。
  - 业务代码、测试和 `old/` 下文档的内容都未改。
- 同步依据：引用检索表明代码、测试、spec、脚本都不读取被移动的文档；测试和 `core/paths.py` 都按 `__file__` 定位仓库根，整体移动后指向 `old/`；`old/.venv/pyvenv.cfg` 只依赖 `D:\DevTools\Python312`。
- 验证方法：
  - 移动前后各跑一次全量测试（`D:\DevTools\venvs\vision`，移动后在 `old/` 下运行）：都是 1011 passed、21 failed、1 skipped，失败清单相同。21 项是 web 桥 6 项，加上读取 YOLO 文档的 15 项：工作区中原 `docs/` 下 33 个已跟踪文件在本次整理前已被删除（回收站和 HEAD 中都还在）。
  - `git check-ignore`：`old/` 下的模型、输出、dist/build、`.venv`、`user_prefs.json` 仍被忽略，在线模板仍被放行；已跟踪文件数与 HEAD 相同（另加 `old/.gitignore`）。
  - `old/.venv` 的 python 可运行，能导入 cv2、mediapipe、numpy；文档检查脚本 0 问题；`git diff --check` 通过。
- 遗留：`old/.venv/Scripts` 下 pip 等 exe 启动器和 activate 脚本里写死了旧路径，需改用 `python -m pip`。`AGENTS.md`、`CLAUDE.md` 中的路径（change.md、归档、`docs/*`、命令工作目录）按要求未改。

## 2026-10-03（America/Los_Angeles）: [docs] 新增分阶段任务清单目录，修正设计文档 8 处不准确的描述

- 问题描述：用户要求把每个阶段的任务前检查、待决策事项、具体任务、验收标准分别写成 md，放在一个文件夹里，防止以后遗忘。逐项对照代码时发现设计文档有 8 处与现状不符或前后矛盾。
- 修改内容：
  - 新增 `docs/重构阶段任务/`：README（阶段一览与依赖；待决策总表，含最晚在哪个阶段决定、未决时怎么做；附录 B 搬迁的阶段归属；通用检查与通用验收）和 P0～P9 共 10 个文件。各文件列出 2026-10-03 在 5a6d99c 上统计的代码位置（文件:行）、计数和验收用的搜索。新增 3 项交付决策 P0-1～P0-3：构建环境与固定版本、合并 spec 后保留单文件版、交付包附带文件（9-20 包有使用说明、配置示例和第三方许可，9-29 包只有 EXE）。
  - 设计文档修正：
    - §1.2 原写"conftest 无需修改"，与 P1 把 `core/paths.py` 迁入 foundation 矛盾（只给门面打补丁时，新模块里的调用不受隔离），改为同一 PR 内改打补丁目标。
    - §3 第 6 项删去已随 PR #75 删除的 `_preopen_generation`；§9.1.1 摄像头一行原列已删除的 `_release_preopen_cap`、`_warm_dual_cameras`，改为现有方法。
    - §3 第 34 项与 §12 的 P6 行：CLI 有 3 个逐帧循环（`run`、`run_realtime_parallel`、`run_realtime_latest_frame_smoke`），另有 `_process_video_multithread`，原写 2 个。
    - §9.2：offline_matching_profile 从 action_compare 导入 7 个私有函数，原写 5 个。
    - 附录 B：action_feedback 的规则去向改为 `evaluation/rules/practice_v4.py`，与 §2.2 一致。
    - §12 并行安排：P6 还依赖 P2 的 `create_engine`、`VideoSource`、`extract_pose_sequence`，原写两条线只在 P4 的评分步骤汇合。
    - §13.1：整套离线自检只能在冻结包里运行（它断言模型已播种到临时 LOCALAPPDATA），源码模式改为运行 `test_tk_preview_student_switch_with_process_cameras`。
    - §12 原则补一条：附录 B 中没写阶段的搬迁按目标层归入对应阶段，并指向新目录。
- 同步依据：AST 统计（定义位置与行数、App 方法、ExamPanel 访问的 App 私有成员、CLI 函数）；正则计数（`MediaPipePipeline(` 24 处/14 个文件，`cv2.VideoCapture(` 35/22 并按用途分类，`np.load(` 22/11，`threading.Thread(` 27/10，`.after(` 29/4，写死的 baseline 3.0 共 9 处）；conftest 与 3 个 `_prefs_path()`；`apps/desktop_smoke.py` 的 `run()` 与 `apps/desktop_launcher.py`；两份 requirements 与两个环境的包版本；两份 spec；dist 下 9-20、9-29 交付包的内容；`assets/feedback` 的 24 个图标。
- 验证方法：检查脚本对设计文档 0 问题（68 个章节，登记表 41 行，V/D 定义有序，表格列数一致，引用路径存在，过时措辞无命中）；对新目录 0 问题（11 个文件，表格与链接有效，§ 与 V/D 引用都能在设计文档中解析，89 处"文件:行"引用逐一定位到代码行，无法自动比对符号的 17 处由人工核对打印出的代码行）；git diff --check 通过。

## 2026-10-03（America/Los_Angeles）: [docs] 设计文档划掉已决的决策项与 P0 已完成部分

- 问题描述：用户要求梳理设计文档中仍待决策的事项，已完成的可删除或划线。D-1（v4 基线）和 D-14（摄像头沿用 PR #75 的"池 + 租约"）已决，P0 大部分已完成，但文档仍按待办排列；文首"未修改任何代码，也未运行测试"与 §12 的实测基线矛盾。
- 修改内容：
  - §11.2：D-1、D-14 整行划线，编号不重排，正文中的引用仍有效；表前注明划线含义。
  - §12：P0 行已完成部分划线，只留剩余项；测试基线删去 2 项 Tk 测试的修正过程（见 aecc6bd 的记录），web 端 6 项由"随 D-10 删除"改为"按 D-10 处置"（D-10 仍待决）。
  - 文首状态改为"P0 基线整合已基本完成（§12），P1 起的重构尚未开始"。
  - 仍待决：D-2～D-13 共 12 项；§11.1 的 V-1～V-8 均待批准，其中 V-2、V-8 即 D-8、D-9。
- 同步依据：main 与 origin/main 均为 5a6d99c；`.venv` 仍缺 hypothesis，两份 spec 都没有收入 `assets/`，仓库中没有 build_info，dist 最新仍是 camera-mode-fix-20260929，因此 P0 剩余 4 项与相应风险保持原文。
- 验证方法：文档检查脚本 0 问题（68 个章节，登记表 41 行，V-1～V-8、D-1～D-14 均有定义且有序，9 处划线的定界符都能正确闭合，表格列数一致，引用的现有文件路径均存在）；过时措辞扫描无命中；git diff --check 通过。

## 2026-10-03（America/Los_Angeles）: [docs] 整体架构重构设计同步推送后的基线，修正摄像头接口分层

- 问题描述：main 推送、PR #75 关闭、2 项 Tk 测试修正后，设计文档中"本地未推送""PR 仍为 open""8 项失败"等描述过时；复查时还发现摄像头门面的接口违反文档自己的分层规则。
- 修改内容：
  - 基线说明、§5.3.1、D-14、§12 的 P0 行、测试基线与风险表同步到当前状态；P0 剩余项补上"合并 spec 时收入 `assets/feedback` 结果页图标"（现有两份 spec 都没有收入，9-29 交付包的结果页因此不显示图标）。
  - 分层修正：`CameraService` 属 L1（media），原写 `open_input(...) -> CameraInput`，而 `CameraInput` 属 L4（live），构成反向依赖。改为 `open_stream(specs) -> CameraStream`（L1），由 `live/inputs.py` 的 `CameraInput` 包装后供 runner 使用；§2.2 包结构、§5.3.2、§8.2、P5 行同步。
  - 池接口清单补上 PR #75 已有的 `warm`、`release_all`；V-1、V-6 中"PR #75 目前……"改为"现状……"；§1.2 用户数据兼容补上可选的 `feedback_config.json`（§7.8 已承诺保留文件名与字段）。
- 同步依据：`git ls-remote`、`gh pr view 75` 与全量 pytest 结果；`apps/camera_warmup.py` 中 `CameraWarmupPool` 的公开方法签名（AST）；`apps/feedback_result_panel.py` 的图标目录 `_ASSETS_DIR`；`dist/` 下最新交付包为 camera-mode-fix-20260929，包内没有 png。
- 验证方法：文档检查脚本 0 问题（68 个章节，登记表 41 行，V-1～V-8、D-1～D-14 均有定义且有序，表格列数一致，引用的现有文件路径均存在）；过时措辞扫描无命中；git diff --check 通过。

## 2026-10-03（America/Los_Angeles）: [test] 推送 P0 基线、关闭 PR #75，修正两项过时的 Tk 测试

- 问题描述：用户确认推送当前 main，并同意按已查明的根因修正 2 项失败的 Tk 测试。两项都是产品行为先变、测试没有跟上，产品代码无需修改。
- 修改内容：
  - 推送：`main` 以 958dd2f..eb6e16c 推送到 origin。推送前核对了 git/gh 身份（gy212）、仓库权限（WRITE）、远端 main 仍为 958dd2f，新增对象中没有大文件和敏感文件名。
  - PR #75：留言说明内容已以 095aed2（`cherry-pick -x b667650`）合入，并附冲突处理与验证结果，随后关闭。`refactor/camera-session` 分支（本地与远端）和 worktree `D:/CodeProject/vision-camera-session` 保留未删。
  - tests/test_ui_controls.py：运行态联动测试先按单摄会话设置 `_dual_recording_ready=True`，与真实 `_start` 经 `_begin_preview_session` 的设置一致（a3a605f 起录制按钮由该标志把关）；新增"双摄首帧就绪前录制按钮保持禁用"的用例。
  - tests/test_s5_hands_toggle.py：AST 测试只豁免 `occupancy_pipe` 的站位检测管线（bf8c908 起固定 lite、`enable_hands=False`），并断言它保持 lite、不跑手部；其余 4 处 worker 管线仍须取 `state.enable_hands`，失败时报出行号。
- 验证方法：
  - 两个测试文件 12 passed。变异检查：worker 管线改为忽略开关、站位管线改为跟随开关或改用 full 时，AST 测试都失败；真实源码通过。
  - 全量 pytest（D:\DevTools\venvs\vision）：1026 passed / 6 failed / 1 skipped，6 项均为已弃用的 web 端。
  - 推送后 `git ls-remote` 确认远端 main 为 eb6e16c；`gh pr view 75` 为 CLOSED。git diff --check 通过。

## 2026-10-02（America/Los_Angeles）: [chore] P0 基线整合：学生练习 v4 等未提交工作入库，PR #75 rebase 合入（本地，未推送）

- 问题描述：用户确认 P0 决策：v4 先提交到 main；工作区其余未提交改动按功能拆分提交；PR #75 在新基线上 rebase 后合入，合入后删除 recover 分支；web 端失败不再处理；2 项 Tk 失败先查原因。整合前 main 为 958dd2f，v4 分析、图文报告、结果页、9-29 摄像头补丁都只在工作区或 Codex 快照里，从未入库。
- 修改内容：本地 main 958dd2f → fe4b62f，共 8 个提交，均未推送：
  - f7112f9 fix(student)：v4 逐帧分析与正侧面证据融合（含 feedback_phases、分析模型选择、同一视频不得当两路、播报隐藏窗口、v4 离线自检断言、.gitignore 的 feedback_config.json、实施文档），change.md 恢复当时的两条 09-20 记录。
  - 1548e95 feat(student)：HTML 图文报告（Milestone 1）；a5b1704 feat(student)：结果页与生命周期集成（Milestone 2，含 24 个状态图标及生成脚本），change.md 恢复当时的两条 09-26 记录。
  - 67861d1 fix(camera)：9-29 摄像头补丁；1a7934d docs(camera)：摄像头架构设计交接说明。
  - 095aed2 refactor(camera)：PR #75（cherry-pick -x b667650）。冲突处理：camera_warmup.py 取 PR #75 版本（它重写了 9-29 补丁，补丁的测试已在 PR #75 测试文件中）；desktop_smoke.py 取 PR #75 的模式切换检查并保留 v4 自检断言；change.md 中 PR #75 记录置顶；app_ui.py 自动合并结果正好是 PR #75 版本加结果页改动。
  - c3d63bb test：tests/conftest.py 新增会话级 fixture，在 pytest fd 捕获之外先建立 Tcl 标准通道，消除反复创建 Tk 时偶发的 "Can't find a usable init.tcl … No error"。
  - fe4b62f test(student)：修正 test_m2_adversarial_stress 中自创建起即失败的 3 个用例（导入不存在的 FeedbackPanel；半成品 App 缺 _record_pair_lock 等属性；_student_start 前置条件缺失、实例属性拦截不到类方法调用），产品代码不变。
  - 来源：v4 及 09-20 相关文件取自 Codex 快照（09-26 06:22、09-28 20:43），其余取自主工作区。两份 M2 测试仅去掉行尾空白。与本任务无关的 docs/动作识别需求调查问卷(2).docx、.agents/ 保持未提交。
  - 清理：删除本地分支 recover/practice-v4-snapshot（其树仍由 Codex checkpoint ref 引用）和临时 worktree。整合前主工作区的完整状态（含未跟踪文件）备份为 refs/kiro-backup/main-worktree-before-p0（4b60b4c）。
- 验证方法：
  - 每个提交都在干净的临时 worktree 中跑全量 pytest（D:\DevTools\venvs\vision）：958dd2f 为 8 failed / 777 passed；补丁提交处为 11 failed / 1020 passed；最终 fe4b62f 跑两次均为 8 failed / 1023 passed / 1 skipped。8 项 = web 桥 6 项 + 陈旧 Tk 测试 2 项，均为 958dd2f 上已有的失败。
  - 字节码：f7112f9 中 action_feedback、feedback_geometry、feedback_phases、feedback_panel、feedback_history、exam_announcer 与 dist/散打动作练习_精度优化版_20260920 EXE 完全一致（含行号表）；67861d1 的 core/apps/analysis/batch 源码与 dist/camera-mode-fix-20260929 EXE 全部 37 个项目模块一致（36 个含行号表一致，feedback_report 仅差一行注释）。
  - PR #75：摄像头、预热、生命周期、双摄、学生练习、控件、枚举测试 223 passed；真实 Tk + spawn 模式切换检查 test_tk_preview_student_switch_with_process_cameras 通过。
  - Tk 偶发失败：同一组新建解释器的对照中，fd 捕获下 16/360 次失败，加 fixture 后 0/360；test_action_feedback.py 连跑 12 次、test_m2_adversarial_stress.py 连跑 10 次均 0 失败。
  - 2 项 Tk 失败根因（未修，待用户确认后改测试）：test_ui_controls::test_set_running_controls_enable_disable_linkage 自 a3a605f（07-10）起失败，录制按钮改由 _dual_recording_ready 把关，测试未经 _begin_preview_session 就直接调用 _set_running_controls(True)；test_s5_hands_toggle::test_ui_worker_pipelines_use_state_enable_hands 自 bf8c908（07-11）起失败，考试/练习站位检测管线有意固定 enable_hands=False，而该 AST 测试要求全部取 state.enable_hands。两项均在引入提交的父提交上通过、在引入提交上失败。
  - 每个提交 git diff --check 通过。未构建 EXE，未推送，GitHub 上的 PR #75 仍为 open，未做甲方双摄实机验收。

## 2026-10-02（America/Los_Angeles）: [docs] 整体架构重构设计（待审阅）

- 问题描述：用户在全量盘查（单文件行数、模块耦合、总体与模块架构、可扩展性、"理应复用却各写各的"清单）之后，要求重新设计 Tk 侧整体架构：保持现有能力不变，优化可扩展性与可维护性，消除重复实现；web 端已弃用，不纳入设计。随后要求同步 main 并确保文档无误。
- 修改内容：新增 docs/整体架构重构设计.md：七层分层与依赖规则、目标包结构、41 项唯一实现登记表；各层设计（偏好、原子写、任务取消与 epoch、诊断日志；视频解码与摄像头设备所有权、状态与完成条件、竞争请求与恢复；PoseSequence 与提取策略；测量库、规则引擎与 profile、模板契约、统一比对与计分方案；冻结/可调会话配置、唯一逐帧循环、片段录制与录后处理、模式要求与工作流命令执行）；Tk 拆分与关窗协调；可扩展性对照；8 项可见行为变化、14 项决策（D-1、D-14 已完成）；P0–P9 迁移计划与门禁、当前测试基线；架构测试与验收场景；能力清单与现有文件去向。摄像头子系统以 PR #75 的"池 + 租约"为基础；学生练习按 v4 描述（默认 Full、逐帧、feedback_config.json、正侧证据融合）。未修改业务代码。
- 更正记录：核对代码后修正初稿的错误。D-6 原写"现状会提示尺寸不符"，实际只在状态栏显示驱动报告的尺寸。ExamPanel 访问的 11 个 App 私有成员原写全部经 _run_command，实际分布在 8 个方法中。cv2.VideoCapture（35 处/22 个文件）、写死的 3.0（9 处）等计数按代码重核。PR #75 合入后的 App 规模（4855 行、135 个方法、162 个实例属性）、threading.Thread（27 处）和两份 App 测试文件中的半成品构造（48 处）已按 fe4b62f 更新。
- 同步依据：AST 依赖图与行数统计；app_ui、camera_*、recording_postprocess、exam_panel、action_compare、rule_scoring、tech_eval、feedback_* 等源码；PR #75 源码与 docs/camera_session_architecture.md；docs/摄像头架构设计交接说明.md；AGENTS.md 硬约束；Codex 快照与交付 EXE 的字节码比对。
- 验证方法：文档检查脚本确认无行尾空白和制表符，代码块闭合，§ 交叉引用可解析，V-1～V-8、D-1～D-14 均有定义且编号有序，登记表 41 行与摘要一致，表格列数一致，引用的现有文件路径均存在（缺失的 9 个均为规划中的新文件）；git diff --check。

## 2026-09-30: [refactor] Tk 摄像头统一设备所有者与会话 lease

### 问题描述

用户连续遇到启动失败、学生练习切换失败，要求按 `docs/摄像头架构设计交接说明.md` 重构而非逐入口打补丁。根因是单摄预打开与双摄预热池两套所有者并存、打开锁不覆盖句柄生命周期、采集尺寸在打开途中读取可变模式标志、`cancel()` 返回被当作已释放、失败会话不分代次取消预热、退出学生练习过早恢复控件、会话自行终止不通知考试面板。

### 修改内容

- apps/camera_warmup.py：`CameraWarmupPool` 成为 Tk 桌面唯一设备所有者。新增冻结的 `CaptureSpec`（编号+请求尺寸，预热复用须 spec 一致）、`sync()` 声明式预热（支持交换正/侧）、`wait_one/claim_one` 单摄路径、`CameraLease`（会话期间设备仍记为占用，`release()` 成功才解除；失败则隔离并在下次同编号打开前按间隔重试）、按代次 `cancel`、`device_busy/wait_released`、结构化生命周期日志。取消/替换只发信号，释放由 reader/reaper 线程完成。
- apps/app_ui.py：删除单摄预打开整套状态、按编号打开锁、`_ExclusiveCameraCapture`、`_open_camera_serialized/_open_camera_exclusive`、`_warm_dual_cameras`、`_cancel_dual_warmup_roles`；单摄、并行单摄、双摄统一经池等待首帧并接管 lease。`UiState.capture_size` 在启动时冻结（普通 720p、学生 1080p）；学生冷启动不再盲目释放预热。失败会话只取消自己观察到的预热代次；单摄 `_post_done` 带会话代次；lease 释放失败显示原因；退出学生练习在 worker 结束前保持「开始」禁用；会话自行终止时通知考试面板；刷新枚举前等待采集进程释放；关窗有界等待进程退出。新增 `[camera]` JSON 诊断日志（会话、设备、请求/实际尺寸、阶段、耗时、错误）。
- apps/desktop_smoke.py、tests：迁移预打开/索引锁测试到新契约；新增 spec 重新协商、lease 占用阻止重开、释放失败隔离与重试、代次取消、角色交换、单摄停止等测试；真实 Tk+spawn 模式切换检查扩展为两轮 720p↔1080p、开始即停止再开始、双摄转单摄、关窗无残留采集进程。
- docs/camera_session_architecture.md：所有权、状态完成条件、并发与失败策略、删除迁移清单、验收映射、日志字段与基线说明。

### 验证方法

- 摄像头/预热/生命周期/双摄/学生练习/控件/打包契约/枚举/考试/录制回归（D:\DevTools\venvs\vision）：338 passed、1 skipped（可选 sidecar 运行检查）；含真实 Tk mainloop + spawn 子进程模式切换检查。
- 全量 tests：784 passed、1 skipped、8 failed。8 项失败（`test_backend_routing_contract` 1 项、`test_ui_backend_sessions` 5 项、`test_ui_controls` 1 项、`test_s5_hands_toggle` 1 项）在未改动的 958dd2f+补丁基线上同样失败，属 Vue/Tauri bridge 与既有 AST 契约，与本次无关。
- git diff --check 通过。
- 未重新打包 EXE，未做甲方双摄实机验收；本分支不含主工作区未提交的结果页工作与 v4 分析规则，发布前需先统一源码基线（见设计文档 §5）。

## 2026-09-30（UTC，本地 09-29）: [docs] 摄像头架构设计交接说明

- 问题描述：用户要求整理说明，交由 Claude 独立进行摄像头架构设计。
- 修改内容：新增 docs/摄像头架构设计交接说明.md，记录现场症状、确定性竞态、现有调用结构和阅读入口、已实施补丁的边界、设计待决问题、业务约束、模式切换验收场景及验证证据。明确 Git 源码/已交付 v4 EXE 的差异和工作区已有导入阻塞，不把前序建议当成既定架构方案。
- 同步依据：当前源码符号与 diff、958dd2f 基线、本次会话实际验证记录、冻结 EXE 自检及版本保持报告；未向 Claude 或其他会话发送消息，未修改业务代码。
- 验证方法：检查 Markdown 结构、关键文件/符号与证据路径可定位，执行本次文档差异的 git diff --check；设计说明不重新运行相机或业务测试。

## 2026-09-29: [fix] 学生练习切换时异步释放与重开竞态

### 问题描述与复现

- 阅读前次「修复摄像头回退并提升精度」会话（01a0e88e-10e0-7232-8030-05c731d99430）并核对当前 958dd2f。学生练习 `_student_start()` 为 1080p 先调用 `_release_camera_warmups()`，但 pool.cancel() 只请求异步取消，随后立即 wait_pair() 被预热池自己的 `camera 0 is still shutting down` 拒绝。普通模式不强制执行这次格式重建，故单摄/双摄普通预览正常不能排除此竞态。
- 新增可控释放屏障，修复前“直接重启”“先预热再重启”两条路径均稳定重现相同报错；前次单独启动/回收检查和仅记录 release/start 调用顺序的测试未覆盖这个真实交接窗口。

### 修改内容

- apps/camera_warmup.py：将同编号设备尚在退出的处理移入共用 reader 的 `_open_capture()`，等待旧 owner 完整释放后才调用摄像头工厂；warm()/wait_pair() 均复用，Tk 主线程不等待驱动。保留设备排他、代次隔离和 15 秒启动预算，取消或超时后的排队 reader 不得迟到重开设备。真实超时附带仍未释放的设备编号。
- tests/test_camera_warmup.py：覆盖直接/预热重试的释放完成、主动停止、等待超时六种组合，验证不重叠打开、取消后不复活；阻塞 open 测试继续验证设备隔离。
- apps/desktop_smoke.py、tests/test_camera_capture.py：加入真实 Tk mainloop、按钮回调、预热池、实际 spawn 子进程的两轮“普通 720p 预览→停止/自动预热→学生 1080p→退出→再开始”检查；仅物理摄像头由模拟驱动替代。

### 验证方法与交付

- 当前工作区摄像头/预热/生命周期/双摄/学生练习回归：199 passed。首次真实 Tk 检查被原有未提交结果页改动阻塞：core.feedback_report 导入当前 core.action_feedback 不存在的 format_fusion；保留这些原有工作，不混入摄像头修复。
- 在 958dd2f 的隔离 worktree 应用仅本次补丁后，摄像头、预热、Tk 生命周期、双摄、学生练习、控件及打包契约：220 passed、1 skipped（可选 sidecar 运行检查）；含真实 Tk/进程模式切换检查。目录：C:/Users/21240/.codex/worktrees/camera-mode-recovery/vision。
- 核对发现 Git 基线为分析规则 v2，而上次交付的 dist/camera-fix-20260928/vision_ui.exe 为 v4。最终交付因此基于上次 EXE，仅替换 apps.camera_warmup 和 apps.desktop_smoke 两个模块；895 个其他 Python 模块、1224 个其他归档资源条目逐项确认保持原字节，保留 v4 分析、模型、界面和资源。隔离源码构建的旧规则版本不作为最终交付。
- 最终 EXE：dist/camera-mode-fix-20260929/vision_ui.exe。实际运行 --offline-self-check 返回 ok=true、frozen=true，四个模型、Lite/Full/Heavy+手部 CPU 推理、录像转码解码、v4 双视角逐帧分析、历史导出、采集异常恢复和两轮 Tk 720p/1080p 切换通过。
- 证据：outputs/camera-mode-offline-check-20260929.json 及同名 .log；版本保持核验：outputs/camera-mode-v4-repackage-20260929.json；可复现重打包脚本：outputs/rebuild_camera_mode_v4_fix.py。
- git diff --check 通过。本机只枚举到一台 FHD Camera；双路验证使用模拟驱动，不声称已完成甲方 USB 双摄硬件验收。现场日志仍为 %LOCALAPPDATA%/VisionSanda/desktop.log。未提交或推送；原有未完成工作保留。

## 2026-09-28: [fix] Tk 双摄格式异常与启动失败后的进程级恢复

### 问题描述

最新版桌面程序出现双摄 OpenCV cv::Mat 的 step >= minstep 断言，以及点击开始后退回、必须重启应用才能恢复。根据前次调用链排查与本次用户报错，修复采集格式热切换和原生驱动阻塞无法回收的路径；未取得甲方现场日志，现场触发条件仍待实机确认。

### 修改内容

- 新增 apps/camera_capture.py，仅供 Tk 桌面入口使用。每路摄像头由独立 spawn 子进程完成 open/read/release，像素通过单槽共享内存传输，返回独立且连续的 uint8 BGR 数组。驱动打开、读帧、释放发生阻塞时有界终止该采集进程，清理完成后允许重新开始。
- 分辨率与格式在首次读帧前协商并验证；短暂空帧重试，首帧格式失败尝试其他格式。Media Foundation 回退 DirectShow 时按唯一设备名映射编号，避免同名双摄错配。
- 学生练习冷启动先释放旧预热资源，再按 1080p 建立采集；删除预热接管后的运行中分辨率切换，显示实际采集尺寸。
- 双摄预热等待覆盖子进程启动预算；单摄、并行单摄、双摄失败均保留错误并完成清理。失败退出不自动重新预热，避免错误被覆盖或立即重入失败路径；正常停止仍保留原有预热行为。
- 增补实际 spawn 的故障模拟和 EXE 离线自检：覆盖原生 open/read/release 卡死、stride 异常、独立帧内存、同编号重新打开、设备锁释放，以及双摄预热线程接管后继续读帧。打包自检使用模拟摄像头，不访问物理设备。

### 验证方法与交付

- Git 交付按用户要求仅纳入本次摄像头修复、回归测试和排查记录，目标 origin/main；其他本地开发改动与 EXE 构建产物不纳入此提交。
- 摄像头采集、预热、桌面生命周期、双摄、学生练习、控件定向回归：204 passed；新增双摄进程预热接管/再次打开集成检查：1 passed。
- Windows 打包契约检查：9 passed，1 skipped（现有可选 sidecar 运行检查）。
- PyInstaller 生成 dist/camera-fix-20260928/vision_ui.exe，保留旧版产物。
- 实际运行该 EXE 的 --offline-self-check：ok=true、frozen=true；四个内置模型、Lite/Full/Heavy+手部 CPU 推理、录像转码解码、分析历史导出、摄像头异常清理/重新打开和 Tk 学生练习 UI 均通过。报告：outputs/camera-fix-offline-check-20260928.json。
- git diff --check 通过。上述采集验证为真实子进程配合模拟驱动，不能代替甲方双摄硬件现场验收；故障日志位于 %LOCALAPPDATA%/VisionSanda/desktop.log。

## 2026-09-28: [docs] 桌面摄像头启动失败与分析精度只读排查

### 问题描述

用户反馈最新版桌面程序点击开始后自动退回未开始状态，反复重试无效，重启应用后恢复；同时咨询办公本环境下的视频分析精度提升路线。用户明确本轮不处理 Web 端。

### 排查结论

- 仅检查当前桌面源码及模拟驱动行为，未修改业务代码、采集设置或模型，未启动摄像头，也未取得甲方现场故障日志。不能将模拟结果直接认定为现场根因。
- 双摄循环任一路单次 read 失败即退出，并统一显示已停止；学生练习在预热接管后从默认 720p 再协商 1080p，未验证连续首帧就进入运行态。open_camera 仅以 isOpened 判断后端成功，首帧失败不会触发后端回退。
- 模拟不可中断的预热 read：首次等待超时，第二、三次等待均被 still shutting down 拒绝；本机 OpenCV VideoCapture 无 interrupt API。接管后的 release 异常也可能保留设备锁，而 worker 清理路径吞掉异常，需统一保留失败资源并提供有界恢复。
- 当前学生练习分析已默认 MediaPipe Full、CPU、VIDEO 模式、逐帧及最长边 1920；Heavy 已有选择入口。应先校准实际采集时间轴、双视角同步、阶段定位与教师规则，再用同批真人标注样本比较 Full/Heavy 或全身关键点模型。
- 双摄录像按固定 30fps 写入，分析时间按帧号除以文件帧率推导；若实际采集不足 30fps，存在动作时间失真风险。该风险尚未用甲方原视频量化。

### 验证方法

- 定向运行 tests/test_camera_warmup.py 中 blocked_open 隔离与 release_failure 保留重试两项测试：2 passed。
- 内存模拟阻塞采集与释放异常：确认连续重试被拒绝及设备锁保留；模拟结束主动解除阻塞并关闭 pool，未访问物理摄像头。
- 核对当前模型配置、录像写入、离线时间轴及官方 MediaPipe / Ultralytics / MMPose / MotionBERT 文档；以 change.md 定向 diff 检查作为本次文档验证。

## 2026-09-26: [feat] 学生练习分析结果页展示与生命周期集成 (Milestone 2)

### 问题描述

散打系统学生练习模式原有分析结果仅挤在左侧侧边栏底部仅9行高度的文本框内，无法直观呈现四分类状态、动作阶段时序、正侧双视角代表帧及结构化依据，查看历史或证据帧需要弹出多层独立子窗口，割裂教学体验。同时，异步分析工作线程缺少任务代次（generation token）防护，在快速切换学生、取消分析、点击再练或关闭窗口时容易发生晚到回调串页、覆写新界面或抛出 TclError 异常。

### 修改内容

1. **Tkinter 分析结果面板 (`apps/feedback_result_panel.py`)**：
   - 严格遵循 Style A「简洁教学风」视觉规范（页面底色 `#F5F7FB`、卡片白底 `#FFFFFF`、细边线 `#D8E0EA`、主要操作蓝 `#2563EB`）。
   - 头部卡片展示学号、姓名、动作、实战式、本地带时区分析时间及记录标识。
   - 四态概览卡片使用本地静态 PNG 图标（`candidate`、`not_observed`、`unable`、`pending_rule`），附带独立撤销项计数标签，严格遵守检查项总数守恒，禁止任何总分、扣分、合格率或评分进度条。
   - 核心结论横幅基于 7 级优先级结论逻辑展示主副标题与警示色调。
   - 动态动作阶段导航栏基于真实阶段动态生成（如开始实战式、前手直拳★等），支持按阶段筛选左侧问题列表。
   - 左侧问题列表（~40%）支持状态分类筛选、同原因合并与教师复核标签展示。
   - 右侧证据区（~60%）安全加载正侧代表帧（`extract_evidence_frame` 骨架高度归一化投影）、展示一句话检查依据、提供教师复核确认/撤销控件，以及可折叠的技术诊断与测量依据。
   - 底部操作栏支持“再练一次”、“导出图文报告”（支持 HTML/TXT）和“个人历史”。
2. **生命周期与主窗口平滑切换 (`apps/app_ui.py`)**：
   - `FeedbackResultPanel` 挂载在主容器 `outer` 跨列 0-1 展示。
   - 实现 `_show_student_result_view(record)` 与 `_show_student_preview_view()`，通过 `grid_remove()` / `grid()` 实现录制预览与分析结果两态无冲突切换，保持后台摄像头采集线程与渲染定时器 `_tick()` 持续安全运行，切回预览即刻恢复。
   - 实现 `_on_student_practice_again()`：安全切回预览，显式清空 `_student_presence_gate = None`、`_student_pending_record = False` 并作废任务代次，使学生开始按钮可用，绝不误触自动录制。
3. **任务代次并发安全防御 (`apps/app_ui.py`, `apps/feedback_panel.py`)**：
   - 引入单调递增任务代次 `_feedback_task_token` 与活跃代次 `_active_feedback_token`。
   - 在 `_on_student_analysis_saved(token, student_id, record)` 中严格核对代次、当前学号、窗口关闭态和录制状态，彻底拦截取消、切学生、再练一次及关闭后的晚到迟滞回调。

### 验证方法

- 运行全部练习反馈与结果面板测试集：`$env:PYTHONIOENCODING = "utf-8"; & "D:\DevTools\venvs\vision\Scripts\python.exe" -m pytest tests/test_action_feedback.py tests/test_app_controls.py tests/test_student_presence.py tests/test_feedback_report.py tests/test_feedback_html_export.py tests/test_feedback_result_panel.py -v`，96 项测试全部通过（100% pass）。
- 运行核心 golden 与考试回归测试：`& "D:\DevTools\venvs\vision\Scripts\python.exe" -m pytest tests/test_pose33_v3_golden.py tests/test_exam_session.py tests/test_exam_roster.py tests/test_exam_clip.py tests/test_presence_gate.py tests/test_recording_postprocess.py -q`，110 项测试全部通过。
- 格式检查：`git diff --check` 验证无空白与语法格式错误。
- 语法编译检查：`python -m py_compile apps/feedback_result_panel.py apps/app_ui.py apps/feedback_panel.py` 通过。

---

## 2026-09-26: [fix] 学生练习图文报告Milestone 1红蓝对抗缺陷修复

### 问题描述

在学生练习动作分析图文报告（`core/feedback_report.py`）Milestone 1阶段由Challenger对抗测试发现多处安全与健壮性缺陷：
1. 报告核心标题裁决逻辑（`get_headline_summary`）在极端组合下出现状态语义漂移：未发现问题被误判为无法判断、撤销项未从可测量基数排除导致全撤销时分母虚高、全项目无法判断时未正确映射至最高优先级错误提示，以及覆盖161,051种状态空间的边界case未严格遵循层级定义。
2. 证据帧提取（`extract_evidence_frame`）对格式异常的骨骼关键点结构缺少充分防御，遇到空列表嵌套（如 `[[]]`）、单维度或含非法浮点值（如 NaN、Inf）的关键点时会抛出 `IndexError` 或 `TypeError`。
3. 路径脱敏函数（`desensitize_paths`）正则表达式无法捕获含空格路径（如 `C:\Program Files\...`、`C:\Users\John Doe\...`）与 Windows UNC 网络共享路径（如 `\\server\share\...`），且仅脱敏了部分字段，学员姓名、学号、检查项名称及教师签名等用户输入/元数据字段缺少脱敏过滤，存在本地隐私泄露隐患。
4. HTML报告导出解析部分字段缺少空值（NoneType）安全校验及异常类型转换保护（`sourceFps`、`sideOffsetSeconds`、`revision` 遇到非数值字符串导致崩溃）。

### 修改内容

- 核心裁决修复：`get_headline_summary` 重构判定优先层级与统计基数，`measurable_count` 严格排除 `revoked_count`；优先响应全项无法判断（`level="error"`, `title="本次未能完成有效判断"`, `status_tone="unable"`），依次严格实现全待判断、部分待判断、部分无法判断与全部未发现问题层级，并对全撤销项与空记录提供兜底说明；补充前端对应的 `.banner-unable` 警示样式。
- 坐标解析健壮性：在 `extract_evidence_frame` 引入 `_is_valid_point` 校验，严格限制关键点为长度不小于2的序列且坐标均为有限数值（`math.isfinite`），自动跳过畸形关键点，避免 `IndexError` 崩溃。
- 全面脱敏与防注入：重构 `WIN_PATH_RE`、新增 `WIN_UNC_PATH_RE` 及优化 `UNIX_PATH_RE`，全面覆盖含空格路径与网络共享 UNC 路径，避免误吞合法标点；将脱敏函数应用到 HTML 渲染中的所有用户可见及元数据字段（包括 `student_name`、`student_id`、`rule_version`、`check.name`、`check.body_part`、`audit.teacher` 等）。
- 空值安全与容错转换：新增 `safe_clean(val, default="")` 与 `_safe_float(val, default=0.0)`，安全处理任意 `None` 或畸形输入；`try/except` 包裹帧提取与指标解析过程，确保报告导出始终优雅降级。

### 验证方法

- 运行全部 254 项测试套件：`Remove-Item Env:\TCL_LIBRARY, Env:\TK_LIBRARY; python -m pytest tests/test_student_presence.py tests/test_app_controls.py tests/test_feedback_report.py tests/test_feedback_html_export.py tests/test_adversarial_m1.py tests/test_feedback_report_adversarial.py tests/test_action_feedback.py -q`，254 项全部通过（100% pass）。
- 穷举验证：`tests/test_adversarial_m1.py` 覆盖 161,051 种状态空间排列组合，零违规（0 violations）。
- 对抗攻击与畸形数据验证：`tests/test_feedback_report_adversarial.py` 164 项用例覆盖 XSS 注入、含空格路径与 UNC 路径脱敏、离线零外链、畸形数据及文件损毁恢复，全部通过。
- 格式检查：`git diff --check` 验证无空白与语法格式错误。

---

## 2026-09-20: [fix] 学生练习v4逐帧分析、遮挡分段与播报隐藏窗口

### 问题描述

甲方双1080P摄像头实时录制后分析，Lite运行快但分析精度低。提供的6、7号文字报告均为v3，各25项卡在双视角阶段门、13项规则待确认；没有原始真人录像，不能仅凭文字报告区分具体机位和关键点失败。另开始/结束播报启动PowerShell会弹出控制台。

### 修改内容

- 学生双摄接管后请求1920×1080，按实际返回帧尺寸录制；离线最长边1920、默认原帧率逐帧推理，默认Full，界面可切Lite/Heavy比较，实时到位检测仍用Lite。三个人体模型和手部模型均随EXE内置。
- 新增局部 `core/feedback_phases.py`：在录像内找稳定护位、按秒设置起动/回收/短漏点窗口、保留多轮完整前后组合。长遮挡和错序不拼接；短漏点保留无效标记，不插补。
- 可见同侧肩髋链用于尺度，检查只要求相关关键点，远侧遮挡不否定无关近侧规则。两路多个峰值可估计小固定偏移；明确次数不一致不融合。同步且起始护位相符时允许可见机位提供阶段时间，但各路独立测量，不借点。
- 报告增加模型、尺寸、帧数、耗时、逐点有效比例、阶段来源/时段/对齐、测量值；无法判断具体指出机位与失败环节。历史可查看每路最多三张原证据帧及有效点。13项未标定或不支持规则维持待确认，不输出正式成绩。
- 共用 `ExamAnnouncer` 的PowerShell调用增加 `CREATE_NO_WINDOW`、`-WindowStyle Hidden`和关闭标准输入，修正学生/考试播报弹窗的同一根因。
- 同步实施文档。交付 `dist/散打动作练习_精度优化版_20260920.zip`，独立目录保留旧交付包；说明默认Full更耗时、旧配置可覆盖默认采样、旧历史需重分析且保留旧结果。

### 验证方法

- `test_action_feedback.py`、`test_pose33_v3_golden.py`、`test_app_controls.py`、`test_student_presence.py` 共72项通过；双摄1080P请求及驱动拒绝回退的针对性检查1项通过。覆盖短快动作、重复组合、侧面遮挡、阶段共享不借点、错序/错时拒绝、1080P每帧送入提取器、真实Tk证据帧/模型选择与隐藏播报参数。
- Windows打包检查9项通过、1项跳过（未构建另一入口Tauri sidecar）；PyInstaller构建成功。
- 实际EXE在仓库外中文空格路径、PATH仅保留系统目录、无Python环境变量并禁止Python网络连接时通过四模型CPU推理、双路Full逐帧分析、转码解码、历史/报告及Tk学生入口。结果 `outputs/offline-check-precision-20260920.json`，构建日志 `outputs/build-offline-precision-20260920.log`。
- `git diff --check`。算法行为用可控关键点验证，实际模型链路用空白视频验证；不把这些结果当作甲方人体动作准确率、1080P实测采集帧率或双摄时钟标定验收。

---

## 2026-09-20: [fix] 学生练习按动作与检查项融合正侧面证据

### 问题描述

用户现场试用反馈模特动作无法得到合理结果，要求正侧面联合分析并按不同动作分配权重。此前只要正侧结论冲突就无法判断，部分项目只依据一路，缺少可解释融合。

### 修改内容

- `core/feedback_geometry.py` 使用实战式/直拳/摆拳/低鞭腿四类权重，并覆盖肘膝、护手、脚背等检查项；实际权重按检查项有效帧比例调整。问题支持度默认达到70%列候选、不超过30%列未发现，中间保留复核，非成绩/概率。
- 缺一路或一路关键点不足不升级剩余权重；同一阶段时间差超过0.4秒不融合。保留两路支持与反对证据、基础/实际权重及质量信息。向后拉手、脚尖方向、躯干朝向保留其公式适用视角并标注 `primary_view`，不伪造双视角一致。
- 可从用户数据目录的 `feedback_config.json` 调整参数，校验配置并随结果留存。权重仍为未标定工程初值，正式评分路径保持不变。没有更换MediaPipe模型或增加视觉大模型。
- 历史详情和文字报告展示融合依据，并列出已检查未发现的问题项；旧报告兼容。禁止把同一文件当两路输入。更新实施文档及EXE内检查。
- 重新生成内置四个MediaPipe模型的交付包 `dist/散打动作练习_双视角版_20260920.zip`，包含当前9月19日到位/离场录制功能；不覆盖旧9月18日交付目录和压缩包。

### 验证方法

- 学生分析、姿态golden及桌面控件检查57项通过；Windows打包检查9项通过、1项跳过（未构建Tauri sidecar）。
- 实际EXE复制至仓库外中文空格路径，清空Python环境、PATH仅保留系统目录，检查入口禁止Python网络访问：四个模型CPU推理、双路分析、视频转码/解码、历史/文字导出、Tk学生页面均通过；证据 `outputs/offline-check-20260920.json`。
- `git diff --check`。权重行为使用可控关键点测试，实际推理使用空白视频；仍需用户标准正侧录像做误报率和权重标定，未宣称解决全部现场识别问题。

---

## 2026-09-19: [chore] 整理学生练习功能并同步远端

### 问题描述

将9月18日的MediaPipe学生练习、个人历史及离线运行支持，与9月19日的到位/离场自动录制播报一并提交。按用户要求，远端允许直接写入时提交到main，不创建PR。

### 修改内容

- 纳入学生练习相关源码、测试、需求与实施文档、离线构建配置；模型、EXE与录像不进入提交。
- 补齐身份绑定测试中的等待到位状态，增加等待期间不能对上一段录像发起评判的断言。

### 验证方法

- 本次学生分析、姿态golden、离线路径、视频转码、MediaPipe委托和Windows打包检查：66项通过、1项跳过（未生成Tauri sidecar实物）。
- 复用本任务此前通过的183项学生到位、考试及桌面生命周期回归；相关运行代码未发生后续修改。
- 提交前核对远端main与本地基线一致，并检查提交文件清单及空白错误。未重新构建包含9月19日优化的离线EXE；真实双摄和语音现场验收仍未进行。

---

## 2026-09-19: [feat] 学生练习到位播报开始、离场2秒播报结束

### 问题描述

学生练习原来需要手动开始和结束录制；甲方要求复用考试模式的站位检测与播报，到达指定位置报开始，离开2秒报结束。

### 修改内容

- “开始”改为开启双摄并等待到位；沿用考试模式保存的ROI和正面髋部中点检测，站稳0.8秒后播报“开始”并录制，离开区域连续2秒后播报“结束”并停录。不叠加考试的最短录制3秒限制。
- 复用现有MediaPipe Lite、`PresenceGate`及`ExamAnnouncer`，不新增依赖，不更改考试状态机或评分规则。
- 等待时锁定身份/动作并允许手动取消；保留手动结束。结束、退出、会话停止后解除检测，过滤上一轮排队样本；下一段需再点“开始”，避免覆盖待评判片段。
- 同步学生界面提示与 `docs/学生练习动作问题说明实施.md`；新增 `tests/test_student_presence.py`。

### 验证方法

- `D:\DevTools\venvs\vision\Scripts\python.exe -m pytest tests/test_student_presence.py tests/test_presence_gate.py tests/test_app_controls.py tests/test_app_ui_lifecycle.py tests/test_exam_session.py tests/test_exam_roster.py tests/test_exam_clip.py tests/test_recording_postprocess.py tests/test_exam_panel.py -q`：183项通过。
- 新检查使用真实站位闸门与UI控制方法，替换摄像头、录像落盘和语音设备，覆盖连续到位、短暂离场、精确2秒阈值、重复事件、取消等待、再次练习、停止后迟到事件与开录失败。
- `git diff --check` 通过；本次更新源码，未重新构建离线EXE，未进行真实双摄站位与扬声器现场验收。

---

## 2026-09-18: [build] 旧版桌面离线 EXE，内置全部 MediaPipe 模型

### 问题描述

甲方无法下载模型，需要不安装 Python、不依赖开发电脑路径、双击即可启动的旧版桌面程序。

### 修改内容

- `app_ui_onefile.spec` 打包 Lite / Full / Heavy 三个人体模型及手部模型、两个公开直拳在线模板、FFmpeg 录像转码工具，排除 Torch / YOLO 重依赖。
- 新增桌面启动入口及 EXE 内离线检查入口；普通启动日志保存至用户数据目录。内置模型包首次将模型与模板释放到 `%LOCALAPPDATA%\VisionSanda`，录制、历史和偏好可持久保存；源码和不内置模型的 sidecar 路径保持原行为。
- 实际 EXE 检查发现 MediaPipe 原生加载器不能读取中文模型路径，统一在 Windows 非 ASCII 模型路径改用字节缓冲加载，保留原 ASCII 路径和 CPU/GPU 委托选择。
- 录像转码使用内置 FFmpeg，隐藏子进程控制台；补充 `docs/离线桌面EXE交付.md`、交付说明及打包路径回归检查。
- 交付产物：`dist/散打动作练习_离线版/散打动作练习.exe` 和同目录使用说明；压缩包为 `dist/散打动作练习_离线版_20260918.zip`。

### 验证方法

- 针对打包、视频转码、路径、姿态 golden 和委托配置的检查通过；Tauri sidecar 实物检查因本次未构建该入口而跳过。无须构建另一套 Vue/Tauri 桌面。
- 实际 EXE 拷至仓库外中文/空格目录，PATH 仅含 Windows 目录，清空 Python 环境配置，禁用 Python 网络连接和下载：四个模型 CPU 推理、模板/偏好持久化、录像→H.264 MP4→解码、MediaPipe 学生问题说明、历史保存/读取/文字导出、真实 Tk 学生练习控件全部通过。
- 普通无参数启动：主窗口可见，正常关窗退出码为 0。证据位于 `outputs/离线交付 检查/offline-check.json` 和 `normal-launch.json`。
- `git diff --check` 通过。视频使用空白生成样本，仅验证分发与调用链，不代表识别准确率或甲方摄像头/性能验收。

---

## 2026-09-18: [feat] 学生练习使用MediaPipe规则检查及个人历史

### 问题描述

依据需求汇总和《武术散打得分点.docx》扩展旧版学生练习。用户明确要求复用MediaPipe、不得依赖甲方电脑带不动的本地视觉大模型；已替换此前未提交的Qwen实现。

### 修改内容

- 新增规则目录 `core/action_feedback.py`、MediaPipe几何/时序实现 `core/feedback_geometry.py`，只用现有Lite、CPU、pose-only；复用valid_mask、角度及屈膝/脚尖规则，低置信度与阶段不明不当合格，不改正式评分路径。
- 学生练习入口、错误提示、历史视频解析均不再导入或调用Qwen，无HTTP模型服务依赖。首尾实战式、前后侧动作规则分别检查，保留待标定/待教师确认项目。
- 新增 `core/feedback_history.py`、`apps/feedback_panel.py`：学号与录制动作绑定、个人历史、旧视频导入与补充分析、文字导出、教师确认/撤销、两周受管副本保留；外部原视频不删除。
- 同步 `docs/学生练习动作问题说明实施.md` 和针对性测试。测试环境在 `D:\DevTools\venvs\vision`；本机通过现有模型管理器安装了Lite供真实CPU运行验证。

### 验证方法

- `tests/test_action_feedback.py tests/test_pose33_v3_golden.py tests/test_app_controls.py`：49项通过。
- 真实加载MediaPipe Lite CPU，处理24帧空白视频，正确标记证据不足；确认未导入Qwen模块。没有把空白输入冒烟当作动作识别率/甲方性能验收。
- `git diff --check`；旧版真实Tk控件的任务/历史流程通过。厘米标定、规则冲突和教师样本准确率尚待验收。
- 既有UI录制按钮normal/disabled断言曾在修改前HEAD复现，未扩大修改。

---

## 2026-07-20: [fix] 学生点评：纯固定文案，禁止回显模型自由文本

### 问题描述

Codex 二次阻塞：unknown action 回显「模型标记：得分95分」；自由 summary/problem/suggestion
仍优先展示（如 guard_hand_low →「闭眼向后走十步」）。

### 修改内容

- **core/qwen_coach.py**：summary/problem/suggestion 仅由 code+severity+view_hint / 固定 reason 生成；
  未知 action 用固定 not_jab 文案，warning 只记 len，不回显原值；raw_text 恒空。
- **tests/test_qwen_coach.py**：得分95分 action、闭眼建议、roundhouse 不回显回归。

### 验证方法

`powershell
.\.venv\Scripts\python.exe -m pytest tests/test_qwen_coach.py -q
`

---

## 2026-07-20: [fix] 学生练习 Codex P1/P2：fail-closed、受控文案、后处理就绪、健康检查后台化

### 问题描述

Codex deep review：未知 action 误判为成功直拳；summary/problem 可泄漏分数与面部评价；
停录后立刻可评判与转码竞态；health_check 阻塞 Tk 主线程。

### 修改内容

- **core/qwen_coach.py**：未知/缺失 action fail-closed → unable_to_judge；白名单 issue 全滤掉也 fail-closed；
  sanitize + 受控模板；coach.json 不写 raw_text；支持显式 front/side 终态路径。
- **apps/recording_postprocess.py**：PostprocessUpdate 透传 front_video/side_video。
- **apps/app_ui.py**：后处理终态就绪后才启用「动作评判」；评判用终态视频；
  health+推理同后台线程；顺带修退出 pack 顺序与 online_match 恢复。
- **tests/test_qwen_coach.py**：roundhouse_kick / 分数文案 / invalid issues 回归。

### 验证方法

`powershell
.\.venv\Scripts\python.exe -m pytest tests/test_qwen_coach.py tests/test_app_controls.py tests/test_app_ui_dual_camera.py tests/test_recording_postprocess.py -q
`

---

## 2026-07-20: [feat] 学生练习模式（Qwen 直拳视觉粗评，无分数）

### 问题描述

教师端已有双摄录制 + 模板比对/考试编排；需要面向学生的极简练习页：开始、结束、
动作评判、预览。评判走本机 Qwen3-VL 粗粒度中文点评，不打分、不改正式评分链路。

### 修改内容

- **`core/qwen_coach.py`**（新增）：直拳检查项白名单、`unable_to_judge` 契约、正/侧
  抽帧 contact sheet、stdlib HTTP 调 `llama-server`、结果归一与展示文本（无分数）。
- **`apps/app_ui.py`**：主界面「学生练习…」模式；开始=双摄预览+开录、结束=停录、
  动作评判=手动 Qwen；强制 `auto_compare=False`、关骨架/手部；与考试模式互斥。
- **`tests/test_qwen_coach.py`**（新增）：契约过滤、mock HTTP、抽帧与 `coach.json`。
- **`scripts/start_qwen_server.ps1`**（新增）：启动 `E:\AI\qwen3-vl` 本机服务（:8091）。
- **`docs/error_analysis_plan.md` §C**：记录学生 UI v1（prompt-only）范围与白名单。

### 验证方法

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_qwen_coach.py tests/test_app_controls.py tests/test_app_ui_dual_camera.py -q
.\.venv\Scripts\python.exe -m py_compile core/qwen_coach.py apps/app_ui.py
```

手工：启动 `scripts/start_qwen_server.ps1` → 学生练习 → 双摄开始/结束 → 动作评判。

---

## 2026-07-14: [fix] 考试系统结构对照 + 重考可用性问题

### 问题描述

用户反馈重考「一次都没成功过」。对照设计文档与实现后，逻辑层与物理层大体对齐，但重考在面板层有多处易踩坑，导致现场表现为按钮无用/点了无反应。

### 结构对照结论

| 层次 | 职责 | 对应实现 | 对齐情况 |
|:---|:---|:---|:---|
| 状态机 | 叫号/进场/录制/收尾/中止 | `core/exam_session.py` | 对齐；命令由 panel 执行 |
| 占用闸门 | enter/empty 去抖 | `core/presence_gate.py` + App occupancy | 对齐 |
| 台账/补考行 | 队尾追加、最后成功计分 | `core/exam_roster.ExamScorebook` | 对齐 §7.6 |
| 执行层 | 播报/开停录/写盘 | `apps/exam_panel.py` + `app_ui` | 对齐 |
| 重考事件 | 设计有 `retest_request` | **未进 `session.handle()`**，由 panel 直接 `append_retest` + `queue_retest_row` | 行为等价，接口名不一致 |
| 持久化 | 设计 S5 仅已完成行不丢 | 只恢复名单路径，**不恢复成绩簿/阶段/重考行** | 有意局限，但 UI 易误解 |

重考真实契约（实现）：选中考生 → 台账队尾追加 attempt_n → 若 `completed`/`aborted` 则 phase→`ready` → **必须再点「开始考试」** 才叫号；考试进行中只排队尾、不插队。

### 修改内容（修踩坑，不改核心契约）

- **`apps/exam_panel.py`**：
  - 按钮文案改为「重考（选中考生）」；底部说明写清四步流程。
  - 无选中时回退到当前叫号行（进行中）。
  - `scorebook is None`（仅恢复名单未开考）给出完整操作顺序，不再含糊「导入后开考」。
  - 拒绝纯 pending / processing / recording 的无效重考。
  - 已有待考重考行时拒绝重复追加。
  - 终态入队后弹窗询问「是否现在开始考试」，确认则直接 `_start_exam()`（根治「追加了但不知道要再点开始」）。
  - 进行中入队则明确提示「排在队尾」。
- **`tests/test_exam_panel.py`**：补终态自动开考、无台账提示、纯 pending 拒绝、无选中回退当前行等用例。

### 验证方法

- `pytest tests/test_exam_panel.py tests/test_exam_session.py -q` → **22 passed**。

### 现场正确操作（重考）

1. 双摄主界面「开始」保持运行（中止主会话会 abort 考试且可能无双摄）。
2. 考试面板「开始考试」产生 `exam_*/成绩汇总.xlsx` 台账。
3. 考生出现 completed/failed/skipped 后，列表选中该人 →「重考」。
4. 若场次已 completed/aborted：在弹窗选「是」立即开考，或手动再点「开始考试」。
5. 进行中追加的重考在名单最末尾，不会打断当前人。

---

## 2026-07-14: [docs] 同步 CLAUDE.md / AGENTS.md 至最近改动

### 问题描述

CLAUDE.md 与 AGENTS.md 落后于最近几次改动：①「动作分析」窗口已改模板池逐动作评分、直拳技术评估已删，文档仍描述旧单模板形态；②`recording_postprocess.py` 新增的八段锦 opt-in 注册表（`_VIEW_WEIGHTS`/`_SCORE_SQUEEZE`）未记录，后续 agent 可能误改踩坏生产行为；③AGENTS.md 结构清单整块缺考试系统模块；④`tests/conftest.py` 的 prefs 隔离约定未成文，易被后续测试绕过再引入污染。

### 修改内容

- **`CLAUDE.md`**：`apps/app_ui.py` 条目改述「动作分析」为模板池→逐动作成绩（走 `compare_video_to_templates`、动作名取模板 stem、综合分=各动作平均），注明旧单模板 UI 与直拳技术评估已移除；`apps/recording_postprocess.py` 条目补池级评分口径 + `_VIEW_WEIGHTS`/`merge_view_weighted_actions()`（攀足 side0.75/front0.25、攒拳怒目 50/50）与 `_SCORE_SQUEEZE`（背后七颠 `[0.75,0.90]`）两个 opt-in 注册表及散打 no-op 零回归说明。
- **`AGENTS.md`**：结构清单补齐考试系统模块（`exam_session`/`exam_roster`/`presence_gate`/`exam_clip`/`exam_announcer`/`exam_panel`/`recording_postprocess`）；Testing Guidelines 加考试系统回归门（`test_exam_*` + `test_recording_postprocess.py`）与「测试隔离」约定（`tests/conftest.py` 把 `_prefs_path` 钉到临时文件，禁止后续测试绕过或直接读写真实 `user_prefs.json`）。

### 验证方法

- 文档同步依据：本轮已落地的三次代码改动（八段锦评分接入 `c4237b9`、动作分析池式重构 `c9ead24`、prefs 隔离 `5b5dd11`）。
- 轻量验证：脚本核对文档引用的 8 个模块/文件路径全部真实存在（`core/exam_*`、`apps/exam_panel.py`、`apps/recording_postprocess.py`、`tests/conftest.py`）。

---

## 2026-07-14: [fix] 录制保存目录持久化失效——测试污染真实 user_prefs.json

### 问题描述

用户反馈录制保存目录持久化「又失效了，总回退到奇怪地址」。排查发现仓库根真实 `user_prefs.json` 的 `record_dir` 被写成了一个 pytest 临时目录（`C:\Users\ny\AppData\Local\Temp\pytest-of-ny\pytest-595\test_second_dual_segment_can_s0`）。根因：测试 `test_second_dual_segment_can_start_before_first_result_finishes`（`tests/test_app_ui_lifecycle.py`）走真实录制路径，`App._begin_recording_segment`（`apps/app_ui.py:2181`）里的 `save_record_dir(next_base_dir)` 未被打桩，把 pytest `tmp_path` 写进了真实偏好文件。每跑一次测试就冲掉一次用户目录，启动时该临时目录不存在 → `load_record_dir` 回退 `outputs_dir()`，表现为「持久化失效」。

### 修改内容

- **`tests/conftest.py`**（新增）：autouse fixture `_isolate_user_prefs` 把 `core.paths._prefs_path` 钉到 `tmp_path_factory` 临时文件。任何测试都无法再读写真实 `user_prefs.json`——一劳永逸堵住污染源，不必逐个测试补 monkeypatch。
- **`user_prefs.json`**：删掉被污染的 `record_dir`（临时目录）键，让其回退正常默认，用户下次在 GUI 选一次即持久化回来。

### 验证方法

- `pytest tests/test_user_prefs.py tests/test_app_ui_lifecycle.py -q` → 63 passed。
- 跑完测试后 grep 真实 `user_prefs.json` 确认无 `record_dir` 键（不再被污染）。

---

## 2026-07-14: [refactor] 动作分析窗口改模板池逐动作评分 + 删除直拳技术评估

### 问题描述

Tkinter「动作分析」窗口（`CompareWindow`）是早期做的单模板→单相似度形态，没跟上考试系统的池式多动作口径：只能选一个模板出一个综合相似度，无法「模板里有什么动作就出什么动作的成绩」。同时窗口里嵌的「直拳技术评估」已随主链路调整废弃。

### 修改内容

- **`apps/app_ui.py`（`CompareWindow`）**：
  - 删除直拳技术评估 UI 及相关死代码（`_run_tech_eval`、`_set_detail` 等）。
  - 单模板选择框 → **模板池 Listbox**（添加/移除/清空，多选，`try_load_heavy_template` 校验 heavy+pose33_v3、去重）；「从视频生成模板」折叠区生成后自动入池。
  - 比对改调 `compare_video_to_templates`（视频姿态提取一次、池内各模板复用），结果由「单个大百分比」改为**综合分（各动作平均）+ 逐动作成绩列表**，动作名取模板文件 stem，每行按分数红黄绿着色。
  - 清理单模板/预览导出相关的死变量与方法、不再用的 `compare_video_to_template` 单数导入。
- **`tests/test_app_ui_lifecycle.py`**：双摄提交测试桩补 `_current_template_lists`（生产 `_end_recording_segment` 读模板池，旧桩缺此属性导致 pre-existing 失败）。

### 验证方法

- headless 冒烟：池添加/读取/去重、逐动作显示、综合分平均、清空复位均正确。
- `pytest tests/test_app_ui_lifecycle.py tests/test_app_ui_dual_camera.py tests/test_app_controls.py tests/test_error_handling.py -q` 全绿。

---

## 2026-07-14: [feat] 八段锦评分接入——同动作正侧加权合并 + 背后七颠 DTW 分压缩映射

### 问题描述

八段锦 8 式模板已建（`templates/baduanjin/`，与散打隔离），实测暴露两个现有考试评分口径不支持的需求：

1. **两手攀足 / 攒拳怒目各有正+侧两个模板**，但 `_run_auto_compare` 是池级——每个模板各出一个独立动作子分。结果：①成绩表出「XX_正面」「XX_侧面」两列而非一列；②综合分里这两式被计两次（权重变成别式的 2 倍）。实测攀足正面 0.708（前俯团身、深度丢在镜头轴向）明显差于侧面 0.855，需按视角加权合成一个分。
2. **背后七颠 DTW 定位崩溃、分数作废**：低位移提踵动作，`normalize_pose_xy_v3` 抹掉高度信息、`subsequence_dtw` 开放边界退化——实测匹配到视频开头 4 帧 / 结尾 11 帧的静止片段，虚高分 0.94 与动作质量无关。尝试多窗口长度约束、提踵高度几何（踝-髋垂直距离）均证伪：几何信号淹没在 MediaPipe 脚部关键点噪声里。该式考察不严「是人就过」，但要求可复现、有微弱区分度（排除随机数：不可复现、挡不住申诉、违反系统禁 `Math.random` 原则）。

### 修改内容

- **`apps/recording_postprocess.py`**：
  - 新增 `_VIEW_WEIGHTS`（opt-in 注册表，同 `_KICK_KINDS` 模式）+ `merge_view_weighted_actions()`：把登记动作的 `<名>_正面`/`<名>_侧面` 子分按归一权重合成单个动作分。攀足侧 0.75+正 0.25（→0.818）、攒拳怒目 50/50。仅正侧**都存在**才合并，缺一原样保留。`front_score`/`side_score` 仍为各池原始均值（成绩表正/侧列不变），只影响 combined 与逐动作列（去重双计数）。
  - 新增 `_SCORE_SQUEEZE` 注册表：背后七颠 DTW 分线性压到 `[0.75,0.90]`——人人及格偏上（最低 75）、保留同一视频恒定可复现的微弱区分度（认真 0.942→89.1 vs 敷衍 0.70→85.5，差 3.6 分）。插在 `_run_pool` 的 geom/dtw 之后覆盖 `final`。
  - **散打零回归**：模板 stem 无 `_正面/_侧面` 后缀、不在两张注册表内 → 两处均 no-op。
- **`tests/test_recording_postprocess.py`**：新增 4 个单测覆盖正侧加权（成对/散打 no-op/单视角不误合）与压缩映射（区间/单调/可复现）。

### 验证方法

- `pytest tests/test_recording_postprocess.py tests/test_exam_roster.py tests/test_exam_panel.py -q` → 80 passed（含新增 4 个；散打契约「1正+1侧→action_scores 长度 2、combined 86」不破）。
- 数值自检：攀足合并 0.8182、攒拳怒目 0.8095、七颠压缩全区间落 [0.75,0.90] 且单调可复现。
- 端到端：真实考试录像 `record_20260714_003231_299384` 走完整考试流程，攒拳怒目正确合并为单个 `view=combined` 项（front 0.742+side 0.748→0.745）。

---

## 2026-07-13: [fix] 持久化摄像头与考试名单，修复中止后重考无法启动

### 问题描述

1. Tkinter 主/次摄像头选择只存在内存中，重启应用后恢复默认；手动刷新时还会先把主摄下拉值覆盖为“正在检测摄像头…”，导致刷新结果必然回退第一项。
2. 考试名单导入后只写入 `ExamSession.rows`，关闭考试面板或重启应用会重新得到空会话。
3. 现场 `outputs/exam_20260713_045138/成绩汇总.xlsx` 已留下两条 `pending` 重考行，但主会话停止后阶段为 `aborted`；原逻辑仅处理 `completed -> ready`，所以点击重考虽已追加，随后仍无法开始。开考校验又把所有非 `ready` 状态误报为“请先导入名单”。
4. 考试表格每次更新都会删除并重建全部行，后台占用/评分更新可能在“选择考生”和“点击重考”之间清掉选择。

### 修改内容

- **`core/paths.py` / `apps/app_ui.py`**：在 `user_prefs.json` 增加 `camera_selection`，按摄像头索引保存主/次选择；启动和刷新枚举时优先恢复偏好。设备暂时缺失只临时回退，不覆盖偏好，设备重新出现后自动选回；只有用户显式选择才写盘。刷新不再用状态文本覆盖主摄下拉值。
- **`apps/exam_panel.py`**：成功导入后保存 `exam_roster_path`，面板创建时自动重新读取有效名单；源文件缺失或损坏时保持 `idle` 并显示“请重新导入”，不崩溃。开考提示区分“名单为空”“考试已中止”“名单已完成”等真实状态。
- **名单替换安全**：考试处于叫号、等待、录制、收尾或暂停阶段时禁止更换名单；终态换名单时，仍有 `processing` 行的旧成绩簿先移交 App sink 接收晚到评分，无后台任务时才关闭，并统一解除旧场次 active/occupancy/manual-lock 状态。
- **`core/exam_session.py` / `apps/exam_panel.py`**：新增统一的重考入队状态恢复；`completed` 重考从队尾新行开始，`aborted` 重考恢复 `ready` 但保留原指针，让剩余未考名单继续按序执行，重考仍按既有契约排队尾，避免跳过中间考生。
- **`apps/exam_panel.py`**：Treeview 改用稳定 `row_id` 增量更新，不再删除重建全部行；后台刷新保留选择。重考成功后选中并滚动到新行，提供明确可见反馈。
- **测试**：新增 `tests/test_user_prefs.py`、`tests/test_exam_panel.py`，并扩展 `tests/test_app_ui_lifecycle.py`、`tests/test_exam_session.py`，覆盖偏好合并/损坏降级、摄像头缺失后恢复、名单路径恢复、表格选择保持、`aborted -> retest -> ready -> start` 及不跳过剩余名单。

### 验证方法

- `pytest tests/test_exam_panel.py tests/test_exam_session.py tests/test_user_prefs.py -q` -> **passed**。
- `pytest tests/test_app_controls.py tests/test_app_ui_dual_camera.py -q` -> **117 passed**（补齐录制 toggle / 双摄 worker 测试桩：`_sync_record_stop_enabled`、考试占用锁、runtime rotate、preview stage 推进）。
- `pytest tests/test_app_ui_lifecycle.py -q -k "camera_enumeration_restores_persisted_primary_and_secondary or missing_preferred_cameras_fallback_without_forgetting_then_restore or explicit_camera_selections_are_persisted"` -> **3 passed**。
- `python -m py_compile core/paths.py core/exam_session.py apps/exam_panel.py apps/app_ui.py` 通过。

---

## 2026-07-13: [fix] 综合分改直接平均 + 低鞭腿去几何 + 低鞭腿夹角评分（试验后弃用）

### 问题描述

1. 同一人打同一套模板考出 38~60 分且波动大。综合分原口径 `combined=0.4×正面池均值+0.6×侧面池均值` 多套了一层池权重，用户要求改为**所有动作子分直接平均**。
2. 低鞭腿几何分（原「站立腿大腿高度三角峰」）实测异常：真实考试录像里所有踢腿动作张开角都在 95~127°（踢到水平以上），低鞭腿 DTW 窗口峰值帧算出 104°，按「踢过头」判 0，把该动作子分从 0.77(纯DTW) 拖到 0.38。根因是这次录像低/高鞭高度没拉开 + 高度基准（站立腿膝/髋）定位不稳。

### 修改内容

- **`apps/recording_postprocess.py`**：
  - `_run_auto_compare` 综合分改为 `combined = mean(所有动作子分)`（正+侧，含踢腿几何混合分），不再套 `w_front/w_side`。删除 `w_front/w_side` 参数。`front_score`/`side_score` 仍算各池均值供成绩表正/侧列。
  - `_KICK_KINDS` 移除「低鞭腿」——低鞭腿现自动只走纯 DTW（`name not in _KICK_KINDS` 则跳过几何）。高鞭腿/侧踹腿/正蹬腿仍几何+DTW 各半。
- **`core/kick_quality.py`**：低鞭腿 `_score_low_whip` 从「高度三角峰」重写为「两腿张开夹角梯形」（保留在库，`_KICK_KINDS` 不再引用，供后续需要时启用）。实测三张标准低鞭腿摆拍夹角 = 64/73/84°（画骨架量出），定 64~84° 满分平台、>90° 归零、<40° 归零。夹角用真实像素反归一化（`width`/`height` 来自 raw meta）抵消横竖画幅畸变，顶点用髋中点、不依赖踢腿/站立腿判定（比高度法稳）。`score_kick_quality` 新增 `width`/`height` 参数。旋钮 `LOW_WHIP_FULL_MIN/MAX=64/84`、`LOW_WHIP_ZERO_HIGH/LOW=90/40`。

### 验证方法

- `core/kick_quality.py` 自检：低鞭腿夹角 64/73/84°→1.0、87°→~0.5、≥90°/40°→0；高鞭/侧踹/正蹬高度映射不变。self-check OK。
- 画骨架核对：三张手机摆拍低鞭腿 MediaPipe heavy 检测准确，夹角 64/73/84°；record_024815 侧面视频低鞭腿 DTW 窗口峰值帧（绝对帧#403）实为踢到水平的动作（104°），确认「这次踢太高」非算法误判。
- `pytest tests/test_recording_postprocess.py -q`：51 passed。

---

## 2026-07-13: [fix] 考试录像自动裁剪把 34.8s 砍成 1.9s，评分崩坏（38 分事故根因）

### 问题描述

同一人打同一套散打模板考 3 遍，8 个动作分数在 25~80 之间乱蹦（本该都在 0.8 上下），综合分 38~60 不可信。只读排查确诊：原始考试录像 side.avi = 1044 帧 / 34.8s（完整 8 动作），经 `core/exam_clip.py` 的 `estimate_action_range` 裁剪后 side_exam.mp4 只剩 56 帧 / 1.9s——95% 的帧被当「走位/静止」删掉。8 个模板（46~149 帧）被硬塞进 56 帧里跑 subsequence DTW，多数模板比视频还长，窗口互相重叠乱锁，分数随机。

根因是 `estimate_action_range` 优先走 `find_active_range`，后者只取**最长单个连续活跃段**，对「连续打 8 个动作」的录像只保留能量最高的一小段。隔离实验：用裁剪前完整 34.8s 视频跑同样 7 个侧面模板，DTW 全部落在 0.75~0.83、窗口按动作时序干净铺开不互抢——证明评分方法/归一化/DTW 都没问题，唯一问题就是裁剪。（与上一条「建模板」防线是**不同链路**：那条防 `create_template_from_video`，本条防考试录像 `exam_clip`。）

### 修改内容

- **`core/exam_clip.py`**：`estimate_action_range` 加 `smart_crop: bool = False`（考试生产默认）与 `tail_quiet_pct: float = 40.0`。
  - `smart_crop=False`（考试默认）：进场已由 `presence_gate` 挡干净，头部只固定掐 `trim_head_s`（0.5s）；**中段一律不裁**（整段评，隔离实验结论）；尾部**不按时间硬掐、按能量从末尾回扫到最后一次超阈动作**，末动作后留 0.3s 余量。正常离场留下的 ~2s 空场能量低会被切掉；`force_finish` 强停时尾部仍是动作则一帧不切。区间过短（<max(15,0.5fps)）一律回退整段，绝不 raise。
  - `smart_crop=True`：完整保留旧的 `find_active_range` 活跃段 + 时间裁剪逻辑，供 `estimate_action_range_from_feature_seq`（sim/单测）显式使用。`find_active_range` 本体一字不动（pose33_v3 golden 不漂）。
  - 新增 `_smooth_1d`（移动平均，防单帧能量毛刺）。
- **`tests/test_exam_clip.py`**：旧的 `estimate_action_range` 时间裁剪测试显式传 `smart_crop=True`；新增考试默认路径 3 例（无 energy 保留整尾、尾部空场被回扫切掉、force_finish 末动作不被吞）。

### 校准旋钮（默认值，现场可调）

- `trim_head_s=0.5`（掐头固定）
- `tail_quiet_pct=40.0`（尾部回扫安静阈分位；越低越保守越少切尾）
- 末动作后余量 `0.3s`（硬编码）
- `smart_crop=False`（生产默认；sim/test 显式开旧逻辑）

### 验证方法

- 真实 34.8s / 1044 帧 side.avi + front.avi 跑 `prepare_exam_pair`（生产默认）：裁后 front_frames=1004 / trim=[15,1019] / **保留 96.2%**（不再砍到 56）；裁后 side_exam 跑 7 个侧面模板 DTW 全部 0.759~0.833（对齐隔离实验 0.75~0.83），窗口按动作时序铺开不互抢。
- `pytest tests/test_exam_clip.py tests/test_recording_postprocess.py tests/test_exam_session.py tests/test_presence_gate.py tests/test_pose33_v3_golden.py -q`：88 passed（golden 不漂）。

---

## 2026-07-13: [fix] 长视频生成模板时拦截异常短自动裁剪

### 问题描述

`find_active_range` 的既有语义是从运动能量高于 P70 的区间中只保留最长连续段。单动作视频可借此提取一次代表动作，但 80.73s 的整套动作视频包含多个活动组，默认结果仅为 1759..1825（67 帧 / 2.23s / 2.8%），会静默生成语义错误的模板。当前评分链和多模板池按“每个动作一个模板”工作，不能把整套长视频直接塞入单个 DTW 模板。

### 修改内容

- **`core/action_compare.py`**：新增 `TemplateAutoCropAnalysis`、`analyze_template_auto_crop` 和 `TemplateAutoCropReviewRequired`。保持 `find_active_range` 与原自动主段完全不变，仅增加模板专用诊断：按 0.3s 平滑、0.15s 最短活动、2s 间隔合并、0.5s 候选 padding 形成活动组。仅当输入不少于 30s、存在多个活动组且自动主段不超过 `max(5s, 原片10%)` 时阻止无边界生成，并返回输入/保留时长、比例和候选帧区间。显式 `start/end` 继续优先并可通过人工确认。
- 新模板 metadata 增加 `crop_selection`、`selected_retained_ratio`、`template_scope=single_action`、`auto_crop_strategy`、`auto_activity_group_count/groups`、`auto_review_required`，保留既有 `auto_start_frame/auto_end_frame`。
- **`apps/make_template.py`**：CLI 复用同一诊断和保护；异常时以清晰消息退出，显式 `--start/--end` 可继续；metadata 与核心入口对齐。
- **`apps/app_ui.py`**：Tkinter 模板生成完成后展示实际帧区间、时长和保留比例，并在原始结果中带回 template meta；长视频异常直接显示人工确认说明。
- **`frontend/src/App.vue`**：Vue/Tauri 从 bridge 已返回的 `templateMeta` 展示实际帧区间、时长和保留比例；异步 `job.failed` 沿既有路径展示长视频人工确认错误。
- **`tests/test_template_auto_crop.py`**：覆盖 120s 多活动视频拦截、短单动作旧行为、20s 双活动组不误拦、无显式边界不落错误模板、显式边界覆盖与诊断 metadata。
- 未修改共享 `find_active_range`，因此考试裁剪、学员活跃段、body_core 和既有评分行为不变。

### 验证方法

- `pytest tests/test_template_auto_crop.py tests/test_template_metadata.py tests/test_pose33_v3_golden.py tests/test_ui_backend_analysis.py -q`：39 passed。
- `pytest tests/test_template_auto_crop.py tests/test_template_metadata.py tests/test_pose33_v3_golden.py tests/test_batch_backend_args.py -q`：47 passed。
- `pytest tests/test_recording_postprocess.py -q`：51 passed。
- `npm --prefix frontend run build` 与 `npm --prefix frontend run test`：通过。
- `py_compile core/action_compare.py apps/make_template.py apps/app_ui.py`：通过；`git diff --check` 无空白错误（仅 Windows 行尾提示）。
- 真实缓存 `outputs/_tpl_energy_cache.npz`：80.73s / 自动 2.23s / 2.8% / 9 个活动组，`review=True`，正确阻止静默生成。
- `E:\测试视频\20260711` 的 8 个 1280x720@30fps 连续视频以 heavy 实测：输入 17.53..23.33s、自动主段 1.20..2.40s、活动组 1..2，全部 `review=False`，未误拦单动作连续素材。
- 扩展 Tkinter 基线 `pytest tests/test_app_ui_lifecycle.py tests/test_app_controls.py tests/test_app_ui_dual_camera.py -q` 为 135 passed / 18 failed；失败均位于开始本任务前已有的多模板池、录制控件和双摄 pipeline 未提交改动（测试桩缺 `_current_template_lists` / `_sync_record_stop_enabled` 等），与本次模板生成路径无关，未越界修改。

## 2026-07-13: [feat] 四个腿法(低/高鞭腿、侧踹腿、正蹬腿)踢腿高度几何质量评分并入动作子分

### 问题描述

考试逐动作出分原只有 DTW 黑盒相似度,不评"踢得够不够高"。为四个踢腿动作加几何质量评分并入该动作子分:
- **低鞭腿**:踢到膝盖高=及格(0.6),大腿中点=满分(1.0),往上到髋关节递减回0.6,再高继续降(三角峰,端点 `Q_BOUND=0.6`)。
- **高鞭腿**:两腿夹角90°/踢到腰(髋)平=合格,踢到头(鼻)平=优秀。
- **侧踹腿 / 正蹬腿**:提到髋平=合格(0.6),越往上越高,到头(下巴)平=满分。同高鞭高度映射,但无分腿夹角门(伸腿动作,两踝张开角对其无意义)。

### 修改内容

- **新增 `core/kick_quality.py`**(纯几何,含 `__main__` assert 自检):`score_kick_quality(landmarks_window, valid_mask, kind)` → (0..1 或 None, detail)。步骤:定位踢腿(窗口内踝帧间位移大者)→ 峰值帧(踝 y 最小)→ 按 kind 评分。三种 kind:`low_whip` 三角峰(中点1.0/大腿端点 `Q_BOUND=0.85`);`high_whip` 高度映射(膝→0、髋=`Q_PASS=0.6`、鼻→1.0)+夹角90°提示门;`kick_up`(侧踹/正蹬)同高度映射但无夹角门。高度映射抽为共享 `_height_map_score`。用 raw Pose33(非 DTW 归一化特征——后者排除鼻子且旋转对齐扭曲高度);高度用 y 比值不受横拍长宽比影响。校准旋钮 `Q_BOUND/Q_PASS/PASS_ANGLE_DEG` 显式常量。
- **`apps/recording_postprocess.py`**:`_run_pool` 里 side 池含踢腿模板(动作名∈`_KICK_KINDS`={低鞭腿:low_whip,高鞭腿:high_whip,侧踹腿:kick_up,正蹬腿:kick_up})时,`extract_pose_raw_series` 对该视频提一次 raw 复用;每踢腿模板用其 DTW 窗口 `slice_pose_raw_series` 切片 → `score_kick_quality` → `blended = KICK_GEOM_WEIGHT*geom + (1-KICK_GEOM_WEIGHT)*dtw`(`KICK_GEOM_WEIGHT=0.5`);geom=None 退回纯 DTW。action_scores 每项增 `dtw_score`/`geom_score`/`geom_detail`,`score`=blended。非踢腿动作不变。
- 透传/成绩表**无需改**:`_result_payload` 整 dict `to_jsonable`、`PostprocessUpdate.action_scores` 已透传、exam_panel 读 `d["score"]`(=blended)、宽表用 blended。

### 验证方法

- `core/kick_quality.py` 自检通过(合成峰值帧:low中点→1.0、髋高→Q_BOUND、鼻高→1.0、膝下→0、帧不足→None;kick_up 髋→Q_PASS、鼻→1.0)。
- 真实侧面视频(record_20260713_013406_368237/side.mp4)跑 `_run_auto_compare`:高鞭腿几何1.00(头平)→0.879;侧踹腿几何1.00(头平)→0.889;正蹬腿几何0.916(过腰近头)→0.847;低鞭腿几何0.594(踝在大腿185%即踢过髋、偏离中点)→0.673;非踢腿动作几何列空、纯 DTW 不变。
- `pytest tests/test_recording_postprocess.py -q` 51 passed(action_scores 新增字段不破坏现有断言)。
- 待观察:低鞭"185%"提示这次低鞭踢偏高或侧面视角下踢腿/站立腿定位待复核;`KICK_GEOM_WEIGHT/Q_BOUND/Q_PASS` 为校准旋钮,待真实数据标定。

## 2026-07-12: [feat] 考试/一条龙支持正/侧多模板池 + 逐动作出分

### 问题描述

散打规定套路考核里学员连贯打一套多个动作（直拳/摆拳/勾拳/高低鞭腿/侧踹/正蹬=侧面机位，摇闪=正面机位）。原考试/一条龙链路每边只支持一个模板（`front_template`/`side_template` → `compare_dual_streams` 出单一正/侧/综合分），无法为"一条视频含多个动作"的套路各动作分别出分。需求：UI 给正/侧两个独立模板列表各可加多个模板；检测时正对正池、侧对侧池；每模板各出一个动作子分；各动作分入成绩表（宽表每动作一列）；考试+一条龙都改。

### 修改内容

核心原则：把"每边一个 Path"泛化为"每边一个 Path 列表"，列表长度 1 时行为等同旧单模板；旧 prefs 单值键、旧 job 单字段保留读取兼容。

- **`apps/recording_postprocess.py`**：
  - 新增复数 prefs 键 `auto_compare_front_templates`/`auto_compare_side_templates`；`load_configured_template_lists()`/`save_configured_template_lists()`/`validate_auto_compare_template_lists()`（每路径过 `_load_heavy_template` 校验，单个坏模板跳过不整体失败，两池都空才 `template_missing`）；`template_action_name()`（动作名=文件 stem）。
  - `DualRecordingJob` 增 `front_templates`/`side_templates` tuple，`__post_init__` 用单字段回填复数（旧构造兼容）。
  - **`_run_auto_compare` 重写**：正对正池、侧对侧池，每模板各跑一次 `self._compare`(=`compare_video_to_template`) 出一个子分；`front_score`/`side_score`=各池均值，`combined`=w_front0.4/w_side0.6 加权（仅一侧非空则退化为该侧）；结果新增 `action_scores` 逐动作明细。**弃用 `compare_dual_streams` 自动正侧拆分**——考试双机位视角固定，逐模板 subsequence DTW 更直接（行为变更，已验证单模板场景不劣化）。
  - 可注入比对 seam `compare` 默认从 `compare_dual_streams` 改为 `compare_video_to_template`（单模板形状）。
  - `_result_payload`/`PostprocessUpdate` 增 `action_scores` 字段并透传。
- **`core/exam_roster.py`**：`ExamResultRow` 增 `action_scores: dict`；`collect_action_names()` 按首次出现顺序去重；`write_scorebook_xlsx` 动作列追加表尾（表头 `<动作>分`，×100 一位小数，缺失留空），`row_to_export_values` 接动作名列表参数。
- **`apps/exam_panel.py`**：`apply_postprocess_update_to_scorebook` completed 分支把 `update.action_scores` 转 `{name:score}` 写入 row；更新说明文案。
- **`apps/app_ui.py`**：录制区两单行输入框改**正/侧两个 Listbox + 添加/移除按钮**（`_build_template_pool_ui`/`_add_pool_templates`/`_remove_pool_selected`/`_persist_template_pools`/`_current_template_lists`）；变更即落复数 prefs；提交 job 填 `front_templates`/`side_templates`；`_exam_preflight` 改用 `validate_auto_compare_template_lists`（至少一侧非空即可，不再强制正侧齐全）。
- **`scripts/match_segments_probe.py`**：新增诊断探针（提取一次特征复用给多模板，打印分辨率自检+逐动作窗口+重叠检测）。

测试同步（`tests/test_recording_postprocess.py`）：`compare` mock 与 `_result()` 改为单模板形状；`compare_calls==1`→`==2`（正/侧各一模板各比对一次）；FIFO order 断言含双模板；result key set 增 `action_scores`；模板覆盖重构点同步清空复数字段。

### 验证方法

- `pytest tests/test_recording_postprocess.py tests/test_exam_roster.py tests/test_exam_session.py -q` 全通过（51+26）。
- 成绩宽表导出：构造两行不同动作集 → 动作列追加表尾、按首现顺序、缺失留空、×100，验证通过。
- 多模板逐动作出分闭环：侧面池[直拳,侧踹腿] vs 直拳视频 → 直拳 0.879 / 侧踹腿 0.524，有区分度；正面池空则 front_score=None、综合退化侧面均值，验证通过。
- 全量 `py_compile` 通过。
- **性能优化（同批）**：新增 `core/action_compare.py:compare_video_to_templates`（一条视频 vs 多模板，**视频姿态只提取一次**复用给池内每个模板做 DTW）；`DualRecordingPostProcessor` 比对 seam 从单模板 `compare_video_to_template` 改为池级 `compare_video_to_templates`（`_run_auto_compare._run_pool` 每视角调一次）。实测 7 模板侧面池从"逐模板重复提取超 2 分钟" → **15.3s**。测试 mock 同步改池形状（`_pool_result`）。
- app_ui 5 个 UI 测试为改动前既有失败（`_sync_record_stop_enabled` 等，与本改动无关，已 stash 复核）。

## 2026-07-11: [feat] pose33_v3 评分 baseline 从 2.0 校准为 3.0

### 问题描述

baseline=2.0 是当初随手设定、未经真人数据校准的刻度常数。用同一人连打同一套两遍做实测，系统只给 84% 左右，而人工判断应在 90%。4 组"同人连打两遍"数据显示 avg_cost 中位数 ≈ 0.333，baseline=2.0 时对应 ~85.8%，需要上调。

### 修改内容

校准依据：4 组独立"同人两遍"样本（avg_cost：0.335 / 0.300 / 0.333 / 0.372），中位数 0.333；
要让该中位数映射到 0.90 分，需 baseline = 0.333 × 0.90 / 0.10 ≈ 3.0（取整、好解释）。

改动文件（pose33_v3 评分主路径，全部从 2.0 改为 3.0）：

- **`core/feature_layout.py`**：`POSE33_V3.default_baseline` 2.0 → 3.0（权威来源，补校准注释）
- **`core/online_matcher.py`**：`_POSE33_V3_BASELINE` 2.0 → 3.0
- **`core/action_compare.py`**：4 处（`_multi_subsequence_matches` 默认参数、单次 DTW 局部赋值、`compare_action_to_template` 默认参数、`compare_dual_streams` 默认参数）
- **`apps/match_template.py`**：局部 `baseline = 2.0` → 3.0
- **`apps/recording_postprocess.py`**：`compare_dual_streams` 调用处显式参数 2.0 → 3.0
- **`analysis/offline_matching_profile.py`**：`POSE33_BASELINE` 2.0 → 3.0（同步离线分析口径）

测试同步更新：

- `tests/test_s3_calibration.py`：`assert POSE33_V3.default_baseline == 3.0`
- `tests/test_recording_postprocess.py`：比对调用 kwargs 断言 `baseline: 3.0`
- `tests/test_s5_offline_profile.py`：`assert offline_matching_profile.POSE33_BASELINE == 3.0`
- `tests/fixtures/pose33_v3/golden.json`：重新生成（single_template score 0.837→0.885，dual combined_percent 38→46）

不动的：`analysis/calibrate_body_core.py`（body_core 历史标定参照，独立口径）。

### 验证方法

- 核心套件 130 passed（`test_s3_calibration` / `test_recording_postprocess` / `test_body_core_layout` / `test_s5_offline_profile` / `test_pose33_v3_golden` / `test_rule_availability` / `test_tech_eval_contract`）全绿。
- 剩余失败均为改动前预存（`test_app_ui_lifecycle` / `test_yolo_backend_contract` / `test_ui_backend_sessions` 等，用 git stash 前后对比确认）。
- 新刻度下同人两遍中位数得分：`3.0/(3.0+0.333) ≈ 90%`，符合预期。

## 2026-07-11: [chore] 生成整套动作 heavy 标准模板（record_20260711_191628_631417）

### 问题描述

需要把 `E:\测试视频\20260711\record_20260711_191628_631417.mp4` 处理为 heavy + pose33_v3 标准模板，且要求纳入「整套动作」的有效帧（从整套开始到结束），仅裁掉前后准备/静止时间——而非某个单一动作段。

### 修改内容

- 无代码改动，仅用既有 `apps/make_template.py` 生成模板产物（gitignored，不入库）。
- 首跑默认自动分段（`find_active_range`）只取到「能量最高的最长连续段」1759..1825（67 帧 ≈2.2s），属单个动作而非整套，不满足需求。
- 逐帧计算 `motion_energy` 曲线（heavy + `normalize_pose_xy_v3`）定位整套边界：视频共 2422 帧/80.7s；帧 0–79 站定静止（能量 ≈0.001），帧 80 起运动爬升；结尾动作持续到 ≈2360，帧 2370 起归零。
- 以 `--pose heavy --start 80 --end 2370` 手动覆盖范围生成模板：
  `templates/record_20260711_191628_631417_heavy.npz`（features `(2291, 22, 2)`，`start_frame=80`/`end_frame=2370`/`fps=30`/`feature_layout=pose33_v3`/`backend=mediapipe`/`normalizer_version=v3`），并出 `*.preview.mp4`（codec avc1）供人工核对。

### 验证方法

- 载入 npz 校验 `features.shape=(2291,22,2)` 与 meta 字段（pose_variant=heavy、feature_layout=pose33_v3、start/end=80/2370）均正确。
- 导出带骨架标注预览视频，用于人工确认整套动作首尾裁剪合理。
- 未改动任何源码，`pose33_v3` 默认路径行为不变。

## 2026-07-11: [feat] 一条龙正/侧模板填入 UI；仅双缺报错

### 问题描述

自动比对（检测一条龙）硬编码 `standard_*_heavy.npz`，主界面无法填入/查看模板；缺一侧即失败，侧摄线不够时无法单侧试跑。

### 修改内容

- **UI**（`apps/app_ui.py` 录制区）：「正面模板 / 侧面模板」两条路径 +「填入…」；状态行显示 ✓/✗；路径持久化 `user_prefs`（`auto_compare_front_template` / `auto_compare_side_template`）。
- **一条龙契约**（`recording_postprocess`）：`validate_auto_compare_templates` — **仅正/侧都不可用** 才 `template_missing`；仅一侧可用则单流 DTW（`compare_video_to_template`），另一侧分显示 `-`。
- 考试 preflight 仍要求两侧齐全（`validate_template_pair`）。
- 提交 job 改用 UI 当前路径；比分展示支持单侧 `None`。

### 验证方法

- `pytest tests/test_recording_postprocess.py -q` → **51 passed**
- 定向：`test_auto_compare_runs_with_only_side_template`、双缺 `template_missing`

---

## 2026-07-11: [templates] 从实测视频生成 standard_front_heavy

### 问题描述

考试黑盒需 heavy+pose33_v3 标准正模板；用户提供 `E:\测试视频\20260711\record_20260711_191628_631417.mp4`。

### 修改内容

- `make_template.py --pose heavy --preview` 生成 `templates/standard_front_heavy.npz`
- 自动动作段：帧 1759..1825（len=67，fps=30）；预览 `templates/standard_front_heavy.preview.mp4`
- 侧模板 `standard_side_heavy.npz` 仍缺，双摄比对/考试需另补侧面标准视频

### 验证方法

- 命令 exit 0；npz 落盘；meta 含 pose heavy / pose33_v3

---

## 2026-07-11: [docs] 导入直拳+摆拳+踢腿场地实测数据

### 问题描述

场地本子测好的机位 / 考台尺寸此前只在手绘图 `直拳+摆拳+踢腿场地.jpg`，系统内无结构化记录，装机与 ROI 标定缺少可查数字。

### 修改内容

- 新增 `docs/venue_直拳摆拳踢腿.json`：考台上/下边界 102、左/右边界 89；**正摄正对上边界**（高 100 / 距台 216 / 投影距右边界 35.6）；**侧摄正对侧边界**（高 83 / 距台 249 / 投影距上边界 21.5）。
- `user_prefs.json` 写入同套 `venue_layout` 摘要 + 文档路径（本地偏好，gitignore）。
- 语义按用户确认：草图正/侧均旋转画法；21.5 / 35.6 为投影落点到框边距离；**不**从厘米推导 `exam_roi_norm`。
- 纠错：曾误把上边当 89、右边当 102，且正/侧机位对边反了；侧摄再纠为**左侧**边界（非右侧）。
- 再纠：机位距离=到对应边界的**直线垂直距离**；正摄→上边界 **216**；侧摄→左边界 **241**（草图曾误读为 249）。
- 「高」语义固定为**摄像机镜头对地垂直高度**（正 100 / 侧 83），非到考台斜距。
- 记忆与 JSON 强调：**源图正/侧位置均旋转画法，禁止按图面方向装机**，只认 `docs/venue_直拳摆拳踢腿.json`。

### 验证方法

- 通读 JSON 字段与源图标注一致；`python -c "json.load(...)"` 可解析。

---

## 2026-07-11: [ui] 删除主界面两路「旋转」下拉

### 问题描述

旋转已改为点击预览 +90°，摄像头行旁的两个旋转下拉多余且占位。

### 修改内容

- 移除 `rotate_combo` / `rotate_combo_2` 及「旋转：」标签与 combobox 事件。
- 保留内部 `rotate_var*` 与 `_runtime_rotate*`（点击预览仍同步角度，供 collect_state / 考试 ROI）。
- `_set_running_controls` 不再联动旋转控件；控件测试去掉对应断言。

### 验证方法

- `pytest tests/test_app_controls.py tests/test_app_ui_dual_camera.py -q`（相关子集）

---

## 2026-07-11: [feat] 预览点击旋转 / 自动比对滑块 / 黑盒固定 heavy

### 问题描述

1. 旋转依赖下拉且会话运行中锁定，必须停止才能改角度。
2. 「录制后自动比对」仅 Checkbutton，运行中锁定，切换不便。
3. 预览模型档与后端黑盒评分档语义混用。

### 修改内容

- **点击预览旋转**：主/侧预览 `Button-1` 各自顺时针 +90°（互不干扰）；`_runtime_rotate*` 线程安全，worker 每帧读取；下拉同步且运行中可改；录制 writer 未建时 `update_session_size`，已建则写盘 `resize` 适配。
- **自动比对滑块**：`仅录制 ⟷ 自动比对` Scale；运行中可快速切换；主线程写入 `_dual_auto_compare` 缓存，片段开录/提交读取；考试仍强制开并禁用滑块。
- **模型分轨**：主界面文案改为「预览姿态模型」；`CompareWindow` 模板生成/比对/tech_eval 强制 `heavy`；双摄后处理本已 heavy，保持契约。
- `RecordingController.session_size` 属性供写盘适配读取。

### 验证方法

- `py_compile apps/app_ui.py core/recording_controller.py` 通过。
- 定向回归（排除 5 条既有基线失败）：`test_app_ui_dual_camera` + `test_app_controls` + `test_app_ui_lifecycle` → **148 passed, 5 deselected**。
- 新增：`_next_rotate_cw` / `_fit_frame_to_size` / runtime 两路隔离单测。

---

## 2026-07-11: [feat] 全局骨架开关：默认关，预览与录制均不带骨架

### 问题描述

录制区「双摄录像写入骨架」仅在双摄路径生效；单摄始终叠骨架推理，且文案暗示只影响双摄写盘。需要统一为**全局骨架开关**：默认关闭，关闭时预览与录制都不带骨架。

### 修改内容

- 勾选文案改为「开启骨架（预览与录制；双摄开启后不自动比对）」。
- `_collect_state`：`record_skeleton` 单摄/双摄/文件源一律透传 UI 勾选（默认 `False`）；`auto_compare` 仍仅双摄有效。
- 单摄串行 worker：摄像头且未开骨架时不建 MediaPipe、不在线匹配，预览与录像写裸帧（仅 FPS 字）；离线文件仍始终 annotate。
- 单摄多 worker：仅「开骨架」时才走 `ParallelPoseEngine`；关骨架落入串行裸帧路径。
- 双摄既有门控保持不变（关骨架双路裸帧 + 不自动比对语义仍在「开骨架」时优先 `annotated_recording`）。
- 考试 preflight 文案同步为「开启骨架」。

### 验证方法

- `py_compile apps/app_ui.py` 通过。
- 骨架相关定向：`test_collect_state_captures_*_skeleton*`、`test_app_initializes_dual_record_skeleton_toggle_disabled`、`test_exam_preflight_rejects_skeleton_session` + 控件回归（排除既有 toggle 基线失败）→ **15 passed**。
- 双摄/生命周期全量（排除 2 条既有基线失败：`creates_two_mediapipe_pipelines` 计 occupancy 第三处构造、`pipeline_loading_keeps_raw_pair` write 断言）→ 预期通过。
- 既有基线失败与本改无关：`test_app_controls` 三条 `_sync_record_stop_enabled` stub 缺失、上述双摄两条。

---

## 2026-07-11: [docs] CLAUDE.md 同步考试系统与近期 commit

### 问题描述

考试系统（PR #72，`exam_*` 五模块 + `exam_panel` + `presence_gate` + `recording_postprocess` 透传）及双流评分等近期改动落地后，CLAUDE.md 未记录，新会话无法感知考务编排入口与硬约束。

### 修改内容

1. Core 模块职责补 `exam_session` / `exam_roster` / `exam_clip` / `exam_announcer` / `presence_gate` 五项。
2. Apps 入口补 `exam_panel.py`、`recording_postprocess.py`，并在 `app_ui.py` 说明「考试模式…」入口与 occupancy/begin-end 录制原语。
3. Architecture 新增「Exam System」小节：编排流程 + v1 成绩含义 / 全局模板 / 0..1 分数口径 / 增量落盘 / 不破坏手动录制快路径等契约（依据 `docs/exam_system_design.md` v0.2 S1–S7）。
4. Common Commands 增 `sim_exam_flow.py`；测试区增考试系统回归命令。

依据：`docs/exam_system_design.md` v0.2、commit `bf8c908` 及其后审查修复链（`be3a7fc`→`a4fab22`）、`requirements.txt` 新增 `openpyxl`。

### 验证方法

- `py_compile` 全部考试模块 + `sim_exam_flow.py` → OK。
- `pytest tests/test_exam_session.py tests/test_exam_roster.py -q` → **26 passed**。

## 2026-07-11: [fix] 考试系统审查第五轮（关窗停播报 / 纯#零值）

### 问题描述

主窗 `prepare_for_app_close` 未 `_announcer.close()`，语音线程可能残留；纯 `#` 格式下数值 0 被导入为 `"0"`，与 Excel 空显示不符。

### 修改内容

1. `prepare_for_app_close` 与普通关面板一致调用 `_announcer.close()`。
2. 无必选 `0` 位时数值 0 显示为空串，导入按 `empty_student_id` 拒绝（`#0` 仍为 `"0"`）。

### 验证方法

- `pytest tests/test_exam_roster.py tests/test_exam_session.py tests/test_app_ui_lifecycle.py::test_app_close_registers_open_panel_scorebook_for_flush_close -q` → **27 passed**。

---

## 2026-07-11: [fix] 考试系统审查第四轮（主窗关面板台账 / # 占位符）

### 问题描述

主窗 `root.destroy` 不触发考试面板 `_on_close`，当前 `panel.scorebook` 未进 sinks 导致不 flush/close；Excel `#` 被当成 `0` 强制补零，超宽时丢掉字面量前缀。

### 修改内容

1. **`ExamPanel.prepare_for_app_close`**：中止考试、flush 台账、登记 `_exam_scorebook_sinks`；主窗 `_on_close` 优先调用它，并把 sinks（含刚移交面板）计入关窗额外预算。
2. **学号格式**：`0` 必选补零、`#` 可选不补零（`###000`+123→`123`）；超出占位宽度时保留字面量（`"ID-"000`+12345→`ID-12345`）。

### 验证方法

- `pytest tests/test_exam_*.py tests/test_presence_gate.py tests/test_app_ui_lifecycle.py tests/test_recording_controller.py -q` → **104 passed**。

---

## 2026-07-11: [fix] 考试系统审查第三轮（begin 不 advance / discard 关窗 / multi-sink）

### 问题描述

PR 复核：开录失败仍自动跳过；关主窗 `_closing` 导致 discard 被跳过并正常评分；重开面板覆盖 sink 丢延迟成绩；3s 关窗预算截断裁剪；自定义学号格式静默裸数字；Scorebook close 后 writer 永久重试。

### 修改内容

1. **begin_failed**：不 advance，回 `wait_enter`+ARM，可重试/skip（符合设计 §6.6）。
2. **关窗顺序**：先 `on_session_stop`/discard，再 `_closing=True`；`_end_recording_segment(discard=True)` 在 closing 时仍执行。
3. **multi-sink**：`_exam_scorebook_sinks` 列表登记，新面板未匹配时继续投递旧 sink。
4. **关窗预算**：有考试裁剪/sink 时 deadline 额外 +30s/+8s；prepare 内 join clip 28s。
5. **学号格式**：支持 `000\-000`、`"ID-"000000`；无法解析的自定义格式整表拒绝。
6. **Scorebook.close**：有限次终刷后清 dirty 并退出 writer，文件锁定不永久重试。

### 验证方法

- `pytest tests/test_exam_*.py tests/test_presence_gate.py tests/test_app_ui_lifecycle.py tests/test_recording_controller.py -q` → **102 passed**。

---

## 2026-07-11: [fix] 考试系统审查第二轮 P1/P2（begin_failed / 暂停中止 / sink / 主停）

### 问题描述

PR #72 复核 head 后仍有：开录失败被 UPDATE recording 盖回、暂停后中止不停 writer、关面板丢后台成绩、主窗口停止不同步考试、裁剪线程未纳入关窗、裁剪帧假元数据、失败补考双计分、学号 `000-000` 格式。

### 修改内容

1. **begin_failed**：`enter_stable` 不再附带 `UPDATE_ROW(recording)`；成功开录后由面板写 recording；`update_row` 拒绝终态被弱状态覆盖。
2. **abort**：`paused_from==recording` / finishing 也发 `DISCARD_SEGMENT` 并标记 skipped；abort 后不再走 `record_stopped`。
3. **关面板 sink**：有 `processing` 行时把 Scorebook 交给 `App._exam_scorebook_sink`，后台比对继续回填；应用关闭时 flush/close。
4. **主窗口停止**：`_stop` / `_on_close` 调用 `ExamPanel.on_session_stop()` 中止考试。
5. **裁剪线程**：登记 `_exam_clip_threads`，关窗先 join 再 cancel postprocessor；关窗期考试更新仍入队。
6. **clip 帧一致**：写出帧数不等时 re-trim 文件，禁止假元数据。
7. **计分行**：有成功分时补考默认不计分；失败后重算仅成功行计分。
8. **学号格式**：支持 `000-000` 等 0/# + 分隔符掩码。

### 验证方法

- `pytest tests/test_exam_*.py tests/test_presence_gate.py tests/test_app_ui_lifecycle.py tests/test_recording_controller.py -q` → **99 passed**。

---

## 2026-07-11: [fix] 考试系统审查 P1/P2 缺陷修复（成绩不丢 / 关窗 / 裁剪 / 开考闸）

### 问题描述

PR 审查发现 7 个 P1 + 2 个 P2：旧片段成绩被丢、Scorebook 旧快照覆盖、重考清空整场、主线程整段重编码 OOM、骨架会话能开考却检测不到人、关面板不停录、双路状态分叉、学号丢补零、ROI 校验失败仍开考。

### 修改内容

1. **旧片段成绩回填**：`_drain_recording_compare_updates` 主界面比对条仍只跟最新片段，但考试面板按 `segment_id` 始终接收所有更新。
2. **Scorebook 写盘竞态**：`_write_lock` + `generation`/`written_generation`，旧快照不得覆盖更新代；flush/close 走 `_persist_latest(force=True)`。
3. **重考不重开整场**：已有 Scorebook 时 `_start_exam` 不再 `load_candidates`；`start_exam` 从首个 `pending` 叫号，不强制 pointer=0。
4. **裁剪内存/线程**：`prepare_exam_pair` 改为流式扫帧/写出（峰值 O(1 帧)）；考试派发在后台线程裁剪后再 submit，不堵 Tk。
5. **骨架会话 preflight 拒绝**；关考试面板先 `abort_exam`（录制中 discard）。
6. **双路 begin/toggle**：分叉/错误先 finalize 旧片段；双路须同态 recording，失败回滚 idle。
7. **学号按 number_format 显示文本**（如 123+`000000`→`000123`）；ROI 保存失败则中止开考。

### 验证方法

- `pytest tests/test_exam_roster.py tests/test_exam_session.py tests/test_exam_clip.py tests/test_presence_gate.py tests/test_app_ui_lifecycle.py tests/test_recording_controller.py -q` → **94 passed**。

---

## 2026-07-11: [test] 考试流程无摄像头模拟 + 成绩 flush 加固

### 问题描述

需要验证考试状态机/占用闸门/名单/裁剪链路是否流畅；模拟中发现 `ExamScorebook.flush` 仅靠后台线程时可能漏落盘。

### 修改内容

- 新增 `scripts/sim_exam_flow.py`：3 人闭环、skip/force/重考、10 人压力、裁剪、Excel、闸门防误触共 6 场景。
- `ExamScorebook.flush` 改为调用线程同步写盘兜底；临时 xlsx 用 ASCII 名。

### 验证方法

- `.\.venv\Scripts\python.exe scripts\sim_exam_flow.py` → **57 passed, SIMULATION_OK**（模拟钟约 34s/3 人、31s/10 人；墙钟 <2ms 量级，无摄像头/无 DTW）。

---

## 2026-07-11: [feat] 考试系统 v1 初版实现（按设计 v0.2）

### 问题描述

现场需要「导入名单 → 叫号 → ROI 到场 → 自动双路录制 → 空场 2s 结束 → 后台比对 → 写 Excel 成绩」闭环；设计见 `docs/exam_system_design.md` v0.2。

### 修改内容

- **核心**：`core/exam_roster.py`（openpyxl 导入/导出/台账/补考/异步写盘）、`core/presence_gate.py`、`core/exam_session.py`（状态机）、`core/exam_clip.py`（派发前帧截齐+动作裁剪）、`core/exam_announcer.py`（异步 TTS）。
- **UI**：`apps/exam_panel.py`；`apps/app_ui.py` 增加「考试模式…」、`_begin/_end_recording_segment` 原语、occupancy 第三条路径（lite、阶段性、裸帧录制）、考试派发元数据与裁剪。
- **后处理**：`DualRecordingJob` / `result.json` 可选 exam 字段；非考试帧数硬校验不变。
- **依赖**：`requirements.txt` 增加 `openpyxl`。
- 参数采用设计默认值（0.8s 进场 / 2s 空场 / 防串场等），实测后再调。

### 验证方法

- `pytest tests/test_exam_roster.py tests/test_presence_gate.py tests/test_exam_session.py tests/test_exam_clip.py -q` → **22 passed**。
- `py_compile` 相关模块通过；exam 元数据 smoke 通过。
- 现场待验：双摄 + heavy 正/侧模板 + 考试面板跑 2～10 人；`templates/` 须事先备好模板。

---

## 2026-07-11: [docs] 考试系统设计 v0.2（吸收审查 A/B/C/D）

### 问题描述

v0.1 设计在占用路径、帧数硬失败、走位污染 DTW、toggle 三态、S2 不可测、补考/防串场/合规等方面会被返工；审查给出 A（阻塞）/ B（产品）/ C（工程）/ D（v2）清单。

### 修改内容

- 升级 `docs/exam_system_design.md` → **v0.2**，主要落入：
  - **A1** occupancy 第三条路径（lite、阶段性、裸帧录制、非考试零推理不变）
  - **A2** 考试派发前 `min` 帧截齐，不改现网 postprocess 硬校验
  - **A3** 派发前 motion/首尾裁剪防走位污染
  - **A4** ROI = primary + 旋转后坐标
  - **A5** `_begin/_end_recording_segment` 原语
  - **A6** S2 量化门闩
  - **B1–B6** 分数 0..1、补考最后成功、防串场参数、force/skip、转换表、合规 §14
  - **C1–C5** 写盘占用/线程、Excel 学号与合并格、错归风险、模板须预置
  - **D** → §18 已知局限
- 仍无业务代码与依赖变更。

### 验证方法

- 通读 `docs/exam_system_design.md` 目录与 §6.6 / §8.6 / §10.2–10.3 是否覆盖清单条目。
- 无需 pytest。

---

## 2026-07-11: [docs] 考试系统设计文档（供审查）

### 问题描述

需要现场「叫号 → ROI 到场 → 自动录制 → 空场 2s 结束 → 后台比对 → 写 Excel 成绩」的考试流程。实现前先沉淀可独立审查的设计，避免直接改代码方向跑偏。

### 修改内容

- 新增 `docs/exam_system_design.md`（v0.1）：产品决策、非目标、复用边界、状态机、Excel 导入/导出契约、Presence 闸门参数、播报、与双摄后处理集成、模块清单、风险、验收与实现分期。
- 已确认决策写入文档：Tkinter、双摄、全局 heavy 模板、固定 ROI + 姿态、空场 2s。
- **本轮无业务代码、无依赖变更、无 UI 改动**；实现以审查结论为准。

### 验证方法

- 确认文件存在：`docs/exam_system_design.md`。
- 文档可独立阅读（含 mermaid 流程、表头契约、成功标准）；无需 pytest。

---

## 2026-07-10: [feat] 双摄录制/录制+检测一条龙开关

### 问题描述

Tkinter 双摄结束录制后默认总是进入「录制 + 黑盒检测」一条龙（后台转码 + heavy DTW 比对）。部分场景只想落盘录像、不触发检测，此前没有独立开关。

### 修改内容

- 录制分组新增勾选框「录制后自动比对（检测一条龙）」，默认开启，兼容现有一条龙行为；会话运行中与骨架开关一并锁定。
- `UiState.auto_compare` / `_dual_auto_compare` 在双摄会话启动时捕获；片段终结后经 `_RecordingPairFinalization` 写入 `DualRecordingJob.auto_compare`。
- `DualRecordingPostProcessor`：`auto_compare=False` 时仍顺序转码并校验录像，随后以 `skipped` / `auto_compare_disabled` 终态落 `result.json`，不读模板、不装 heavy 模型、不跑比对。骨架开启时仍优先 `annotated_recording` 跳过。
- 开录状态文案：开启比对显示「自动比对：录制中」，关闭显示「仅录制：录制中」。
- 回归：控件启停、collect_state 单/双摄、submit 透传、postprocess 仅录制跳过。

### 验证方法

- `.\.venv\Scripts\python.exe -m pytest tests/test_recording_postprocess.py tests/test_app_controls.py tests/test_app_ui_dual_camera.py tests/test_app_ui_lifecycle.py -q`：`186 passed`。
- 实机待验：双摄勾选/取消「录制后自动比对」各录一段，确认开启出分、关闭仅转码跳过比对。

---

## 2026-07-10: [perf/test] Tkinter 双摄快速首屏、裸帧过渡与 MediaPipe 门控复测

### 问题描述

Tkinter 双摄点击“开始”后，第二摄像头仍需现场打开且两路首帧串行等待；开启骨架录像时，两条 MediaPipe VIDEO pipeline 还会在任何画面出现前串行初始化，造成明显黑屏。旧异常测试同时存在“关闭骨架却期待创建两条 pipeline”的自相矛盾。MediaPipe `0.10.35` 和 GPU 是否值得进入 UI 也缺少同样本、同参数的门控证据。

### 修改内容

- 新增 `apps/camera_warmup.py`：按 `primary` / `secondary` 角色并发预热摄像头、持续保存 latest-only 帧，以 generation 隔离选择变化和旧会话，并支持两路原子 claim、超时/停止/关闭释放及异步 reaper。
- `apps/app_ui.py` 接入双摄选择预热和会话结束后重新预热；启动时复用 ready/in-flight capture，不再串行重复打开。预览改用单槽 `DualPreviewPacket`，Tk 主线程同一次 `_tick` 更新两路 Label，任一路失败时不残留单路画面。
- 开启骨架时，pipeline 仍由正式 worker 原线程串行创建，预热线程在加载期持续发布同步裸帧；两条 pipeline 全部 ready 后才整体切换到骨架帧。加载期录制按钮禁用，裸帧绝不进入 `_write_recording_pair`。
- 增加会话 stage/generation/render barrier、启动阶段六个计时点和 exactly-once 收敛，覆盖启动期停止、关窗、模型失败和快速重启。
- 修正双摄异常测试语义：骨架推理异常使用 `record_skeleton=True` 并验证两条 pipeline；无骨架写盘异常使用 `record_skeleton=False` 并验证零 pipeline。
- `analysis/bench_annotate_fps.py` 新增 `--mediapipe-only`，并输出实际 `active_delegate`、fallback 原因、峰值 RSS 和 RSS 增量。
- 隔离 venv 使用 6 段真实视频对比 MediaPipe `0.10.31` / `0.10.35`。`0.10.35` 未达到 10% 性能门槛，CPU FPS 中位数约低 1.6%-1.7%，峰值 RSS 高约 15%-19%；12 次 GPU 请求全部因 Windows build 禁用 GPU 而回退 CPU。因此保持 `mediapipe==0.10.31`、CPU 默认且不增加 GPU UI，结果置顶写入 `docs/mediapipe_gpu_delegate_report.md`。
- Spec acceptance 第一轮修复资源边界：禁止在 native `read()` 活跃时由其他线程强制 `release()`，仅使用 capture 显式 `interrupt()` 协作退出；无法取消的旧 open/read 进入退役隔离，同 index 在其收敛前禁止重复 open；release 异常保留资源并重试，不再静默标记完成。
- Legacy 单摄预开对相同 index 的 ready/in-flight 请求改为幂等；pool 的同 index 互斥锁获取支持 stop_event 取消，避免停止后留下排队 open。
- 启动计时日志只在 `running/failed/stopped` 终态输出，claim 失败不再被早先的 `starting` 覆盖；会话完成后清理对应 metrics，避免长期增长。
- 第二轮验收继续收口异常路径：positional-only factory 不再被误判为支持 `stop_event`；重复 `close()` 会继续重试 retained capture；legacy 单摄同步 open 和预开对设备锁使用 5 秒有界等待，永久 release 失败时明确拒绝而不是无限挂起。
- 报告补充当前锁定 `0.10.31` 的六视频 active-GPU 门控：pose-only `1.0020x`、pose+hands `1.0126x`，均远低于 `1.30x` / `1.20x`。

### 验证方法

- 双摄预热、UI、生命周期、控件和录制定向回归：`174 passed`。
- delegate、benchmark 和 pose33_v3 golden 门控：`35 passed`。
- 最终 Tkinter / MediaPipe / valid-mask 定向回归：`198 passed`。
- `py_compile apps/app_ui.py apps/camera_warmup.py core/vision_pipeline.py analysis/bench_annotate_fps.py` 通过。
- `git diff --check` 通过，仅有 Windows CRLF 提示。
- 真实双摄骨架关/开各 20 次、点击到两路实际 Tk 渲染 P95 `<=0.5s` 仍需连接物理双摄人工验收；程序已输出结构化启动计时点用于采集。

---

## 2026-07-10: [feat] Tkinter 录制保存目录持久化

### 问题描述

Tkinter「录制」分组的保存目录每次启动都回到默认 `outputs_dir()`，用户上次选择的目录不会被记住。

### 修改内容

- `core/paths.py`：新增 `load_record_dir()` / `save_record_dir()`，持久化到仓库根 `user_prefs.json`（单一 key `record_dir`）。读取时校验目录仍存在，缺失或坏 JSON 均回退 `outputs_dir()`；写入尽力而为，失败静默。仅用标准库 `json`，无新依赖、无配置框架。
- `apps/app_ui.py`：`record_dir_var` / `_record_base_dir` 初始值改为 `load_record_dir()`；`_choose_record_dir` 选目录后 `save_record_dir(d)`；`_on_record_toggle` 在 idle→recording 开录生效那一刻 `save_record_dir(next_base_dir)`（覆盖用户手打路径的情况）。

### 验证方法

- `py_compile apps/app_ui.py core/paths.py` 通过。
- paths 读写往返自测（临时 repo_root）：保存后读回一致、无效目录回退、坏 JSON 回退，全通过。
- `pytest tests/test_app_ui_lifecycle.py -q`：37 passed。

---

## 2026-07-10: [feat] 黑盒检测统一 heavy+pose33_v3 + 未勾选骨架时预览跳过推理

### 问题描述

「录制＋检测一条龙」两处需求：

1. 录制后自动比对（黑盒检测）此前锁定在 MediaPipe **full + v2 布局**（`standard_front/side_full.npz`），精度非最高，且与在线识别的 heavy+pose33_v3 模板格式分裂。用户要求黑盒检测全部改用精度最高的 heavy 链路（模板后续重录）。
2. 双摄 worker 每帧无条件跑双路 `annotate()` 实时推理，即便未勾选「双摄录像写入骨架」也照跑，heavy 双路实时推理开销大且对录后黑盒检测无用（黑盒检测会在录像上重新提特征）。

### 修改内容

- `apps/recording_postprocess.py`：`_EXPECTED_TEMPLATE_LAYOUT` 从 `pose_indices_11_32_xy_rot_scale_norm_v2` 改为 `pose33_v3`；新增 `_EXPECTED_TEMPLATE_POSE_VARIANT="heavy"`。默认模板路径改为 `standard_front_heavy.npz` / `standard_side_heavy.npz`。模型可用性检查 `pose_full`→`pose_heavy`，`model_missing` 文案同步。模板校验由「必须 full」改为「必须 heavy」。`compare_dual_streams` 调用 `pose_variant=None`→`"heavy"`（不再回退推断）。
- `apps/app_ui.py` 双摄 worker（`_worker_loop_dual`）：新增 `draw_skeleton = bool(state.record_skeleton)`。仅当勾选骨架时才创建 `MediaPipePipeline` 并跑 `annotate()`；未勾选时 `pipe/pipe2=None`，每帧用原始帧副本作预览（FPS 文字画在副本上），录像仍写原始帧，完全跳过双路 heavy 实时推理。finally 已有 `if pipeline is not None` 守护，安全。
- `templates/`：用现有标准 `正面.mp4`/`侧面.mp4` 生成占位默认模板 `standard_front_heavy.npz`/`standard_side_heavy.npz`（heavy+pose33_v3），满足 `test_fixed_templates_are_delivered_and_compatible` 契约；**用户后续用直拳标准视频重录并覆盖同名文件即可**。
- `tests/test_recording_postprocess.py`：`_write_template` 默认 `pose_variant="heavy"`/`layout="pose33_v3"`；invalid-pose_variant 变异改用 `full`（现为被拒项）；compare mock 断言 `pose_variant` `None`→`"heavy"`。

### 验证方法

- `py_compile apps/app_ui.py apps/recording_postprocess.py` 通过。
- `pytest tests/test_recording_postprocess.py -q`：`49 passed`。
- 双摄/控件/生命周期回归（分文件跑，规避多文件后台通道挂起）：`test_app_ui_dual_camera.py 45 passed`、`test_app_controls.py 10 passed`、`test_app_ui_lifecycle.py 37 passed`。
- 实机待验：未勾选骨架时双摄预览为无骨架原始画面且帧率提升、录制后黑盒检测走 heavy 模板出分；需接真实双摄执行。用户重录直拳标准模板后再确认相似度合理。

## 2026-07-10: [fix/test] Tkinter 双摄自动比对目录身份最终收口

### 问题描述

第六轮验收发现：文件系统返回零 inode 时，普通路径与 Windows `\\?\` / `\\?\UNC` namespace 未经过同一规范化回退。修复后的对抗复核又确认，同一目录若在相邻提交间从 `stat` 失败或零 inode 恢复为非零 inode，所有权键会在文本键与 `(st_dev, st_ino)` 键之间切换，仍可能让不同片段 ID 绕过去重并覆盖首份 `result.json`。

### 修改内容

- 新增统一、大小写不敏感的 Windows namespace 文本规范化，`stat` 失败和零 inode 均复用同一回退，兼容普通盘符路径、`\\?\` 与 `\\?\UNC`。
- 每次提交始终计算并登记规范文本键；文件系统提供非零 inode 时额外登记 stat 键。锁外完成路径解析和文件系统查询，锁内对最多两个键原子求交并登记，任一身份已被占用即拒绝。
- 新增零 inode namespace 回归，以及 `stat_error -> inode`、`zero_inode -> inode`、`inode -> stat_error`、`inode -> zero_inode` 四种切换回归；均覆盖任务执行中和完成后的重复提交，验证只比对一次且首份结果字节不变。

### 验证方法

- 后处理完整套件：`49 passed`；自动比对完整定向回归：`213 passed`。
- 全量 `pytest tests -q`：`564 passed, 1 skipped, 6 failed`；失败集合与接手基线一致，仍为 1 条 YOLO preview routing 和 5 条 `ui_backend` session，本任务未修改 Vue/Tauri/bridge 且未新增失败。
- 两路独立 post-fix 审查均为 PASS，未发现剩余 P0-P4；`py_compile`、`git diff --check` 和规范 `sync-check` 通过。
- Requirements-First 最终 `acceptance-finish` 返回 `accepted`：22 个验收问题全部关闭，无未决问题、修复或审查 agent。
- 实机待验：物理双摄连续两段、录制中主停止、骨架开启后跳过评分三条现场流程仍需接入真实摄像头执行。

## 2026-07-10: [fix/test] Tkinter 双摄自动比对第五轮 Windows 路径身份修复

### 问题描述

第五轮审查在 Windows 实机确认：普通 `C:\...`、扩展 `\\?\C:\...` 及可能的共享别名可指向同一物理目录，但 `resolve() + normcase()` 仍会生成不同字符串键，导致不同片段 ID 绕过目录所有权并覆盖同一 `result.json`。

### 修改内容

- 已存在的片段目录改用 `(st_dev, st_ino)` 文件系统身份作为所有权键，使普通路径、Win32 长路径 namespace、符号链接和共享别名在指向同一实体时统一归并。
- 文件系统身份计算保持在协调器锁外；锁内仍只做 O(1) 所有权检查和登记。目录不存在或文件系统不提供 inode 时，回退到移除 `\\?\` / `\\?\UNC\` 前缀后的平台规范化文本键。
- 新增 Windows 普通路径与扩展路径的集成回归，覆盖首次任务进行中和完成后两种抢占时机，验证仅执行一次比对且首份结果字节不变。

### 验证方法

- 自动比对完整定向回归：`206 passed`。
- 后处理子集：`42 passed`，包含 Windows namespace 别名用例。
- `py_compile` 与 `git diff --check` 通过，仅有 Windows 行尾提示。

## 2026-07-10: [fix/test] Tkinter 双摄自动比对第四轮隔离与异常恢复修复

### 问题描述

第四轮对抗审查发现两项 P2：后处理只按 `segment_id` 去重，不同 ID 若复用同一规范化目录会依次完成并覆盖同一个 `result.json`；双摄 worker 的推理或 writer 异常虽能释放资源和终结片段，却会跳过 `_post_done()`，导致主界面保持运行态、Start 持续禁用。

### 修改内容

- `DualRecordingPostProcessor.submit()` 在锁外解析并按平台大小写规则规范化 `segment_dir`，锁内为目录原子登记唯一 `segment_id` 所有权并保留至处理器销毁；同目录、`..` 别名或大小写别名的其他 ID 在入队前直接拒绝，不产生回调、比对或结果覆盖。
- 双摄 worker 将 `_post_done()` 收敛到最外层 `finally`，在两套 pipeline、两路 capture、OpenCV 窗口和录制片段全部收尾后恰好投递一次；覆盖正常结束、第二设备失败、模型初始化失败、推理异常和双 writer 异常。
- 新增不同 ID 目录别名抢占与双摄运行时异常完成通知回归，验证首份 JSON 字节不变、无串段通知，且异常后主界面可恢复非运行态。

### 验证方法

- 自动比对完整定向回归：`205 passed`。
- 后处理子集：`41 passed`；双摄子集：`54 passed`。
- `py_compile` 与 `git diff --check` 通过，仅有 Windows 行尾提示。

## 2026-07-10: [fix/test] Tkinter 双摄自动比对第三轮回调验收修复

### 问题描述

第三轮对抗审查发现三个 P3 边界：协调器回调线程内调用公开 `close()` 会尝试自连接并抛错；片段已发布 `cancelled` 后，最终路径补写若失败会再发送一个 `failed/result_write_failed` 终态；第二轮关窗防护把普通主停止也视为不可刷新，导致 worker 已复位为 idle 后录制状态文案仍残留。

### 修改内容

- `DualRecordingPostProcessor.close()` 在协调器 worker 自身调用时仍完成取消和退出哨兵登记，但跳过当前线程 join，使回调余下逻辑正常执行，worker 随后按队列顺序退出。
- 已成功发布 `cancelled` 的路径 enrichment 写盘失败时保留原 `result.json`，不再抛到通用 `result_write_failed` 通知；首次结果写盘失败的既有错误语义保持不变。
- `_refresh_recording_status()` 只在真正 `_closing` 时提前返回；普通主停止虽已设置 stop event，仍可在 worker 收尾后读取 idle 快照、清空旧录制文案并禁用“结束录制”。`_on_record_stop()` 的 closing/stop 防阻塞守卫不变。
- 新增回调自关闭、取消补写失败和主停止 worker 收尾三条确定性回归。

### 验证方法

- 自动比对完整定向回归：`204 passed`。
- 后处理子集：`40 passed`；Tkinter 生命周期、控件与双摄子集：`101 passed`。
- `py_compile` 与 `git diff --check` 通过，仅有 Windows 行尾提示。

## 2026-07-10: [fix/test] Tkinter 双摄自动比对第二轮并发验收修复

### 问题描述

第二轮首波与对抗审查发现两处剩余并发边界：后处理在共享协调器锁内执行原子 JSON 写盘，慢磁盘会阻塞新的 `submit()` 和关窗 `cancel_all()`；关窗后已调度的录制状态刷新仍可能等待录制锁、按旧快照重新启用控件，而“结束录制”入口也可能等待终结锁并突破 3 秒关闭预算。

### 修改内容

- `DualRecordingPostProcessor` 新增独立持久化锁，协调器状态锁仅保护写盘前后的取消、终态和去重判断，不再覆盖文件 I/O 或回调；写盘后再次判定取消，取消与 queued/completed 写入竞态时抑制旧状态通知并最终原子覆盖为 `cancelled`。
- `_refresh_recording_status()` 和 `_on_record_stop()` 在 `_closing` 或 `_stop_evt` 已置位时于读取 Tk 状态、获取录制锁或终结锁前直接返回，保证控件保持禁用且关窗轮询不被旧回调拖住。
- 新增阻塞 JSON 写入、阻塞录制锁和阻塞终结锁的确定性并发回归，覆盖 `submit()` / `cancel_all()` 及时返回、取消终态胜出、无旧 completed 通知及关窗控件不复活。

### 验证方法

- 自动比对完整定向回归：`201 passed`。
- 后处理并发子集：`38 passed`；Tkinter 生命周期、控件与双摄子集：`100 passed`。
- `py_compile` 与 `git diff --check` 通过，仅有 Windows 行尾提示。

## 2026-07-10: [fix/test/docs] Tkinter 双摄自动比对第一轮验收修复

### 问题描述

首轮多 agent 验收确认了 12 项取消、资源、并发和交付证据问题：DTW 评分阶段未贯穿取消，双摄 worker 异常路径未释放两路 native 资源，关窗轮询期间仍可开始新片段，后处理缺少片段目录隔离，`cancel_all()` 回调线程和终态通知不符合契约，转码异常错误码不准确；原实现提交还混入了规范开工前的双摄旋转/布局改动，任务完成日志指向不含实现的基线提交。

### 修改内容

- `subsequence_dtw()` / `subsequence_dtw_with_path()` 在代价矩阵、动态规划行、回溯和返回前响应取消；取消信号贯穿规则、正面评分、侧面评分与最终结果，正面评分取消后不再计算侧面或返回部分分数。
- 双摄 worker 在统一 `finally` 中关闭两套 MediaPipe pipeline、释放两路 capture 并清理 OpenCV 窗口；关窗立即禁用录制控件，录制 toggle 在 closing/stop 状态下于读取 Tk 状态前返回。
- 后处理在转码前校验正侧源路径均属于 `segment_dir`；`cancel_all()` 只置位并快速返回，所有回调由协调器线程触发，每片段只通知一次取消终态，同时允许内部补写最终转码路径；普通转码异常稳定归类为 `transcode_failed`。
- 用开工前 checkpoint 文件树重建独立提交 `8f0c9f8`，只保留既有双摄录制、旋转和自适应布局的 7 个路径；自动比对实现及首轮修复独立提交为 `09aa101`，旧混合提交 `b931222` 由安全分支保留。
- 仅修正 `tasks.md` / `progress.md` 的生成型完成日志和 `Last Known Commit` 为 `09aa101ea55421208e81739a6e88ccd6572f8656`，未改冻结任务正文、依赖、勾选状态或 task-plan hash。

### 验证方法

- 自动比对定向回归（后处理、控件、双摄、生命周期、录制控制器、转码、双流、模板 metadata、`pose33_v3` golden）：`196 passed`。
- 全量 `pytest tests -q`：`548 passed, 6 failed`；失败集合与接手基线一致，仍为 1 条 YOLO preview routing 和 5 条 `ui_backend` session，在本任务明确不修改的 Vue/Tauri/bridge 范围内。
- `py_compile`、`git diff --check` 和规范 `sync-check` 通过；提交父子关系、固定模板和后处理模块均通过 Git 对象校验。
- 实机待验：物理双摄连续两段、录制中主停止、骨架开启后跳过评分三条现场流程仍需接入真实摄像头执行。

## 2026-07-10: [feat/fix/test] Tkinter 双摄录制后自动后台比对

### 问题描述

Tkinter 双摄已能同步预览和录制，但片段结束后没有自动接入既有正面/侧面 DTW 比对；连续录制、主停止、writer 失败和关窗之间还缺少统一的片段终结、后台串行调度、结果留档与取消边界。带骨架录像若再次做姿态提取会产生失真分数，也需要明确跳过。

### 修改内容

- 新增无 Tk 依赖的 `DualRecordingPostProcessor`：按 `segment_id` 去重，单消费者 FIFO 顺序执行双路 H.264 转码、录像/固定模板/full 模型校验和 `compare_dual_streams()`；固定 `workers=1`、正侧权重 `0.4/0.6`、baseline `2.0`，规则与误差分析关闭。
- 每个片段目录通过临时文件原子替换生成 schema v1 `result.json`，完整记录成功、失败、跳过、取消、最终录像路径、转码 warning、三项分数和匹配区间；一次性写盘失败稳定归类为 `result_write_failed`，单任务失败不终止后续队列。
- Tkinter 双摄新增默认关闭且会话期间锁定的“录像写入骨架”开关；预览始终显示标注帧，关闭时 writer 保存旋转后的原始帧并自动比对，开启时保存标注帧、完成转码后写 `skipped/annotated_recording`。
- 显式“结束录制”、主“停止”、writer 错误、worker `finally` 和关窗统一经过串行终结器；锁内快照/释放、锁外非阻塞提交，零帧也提交为 `recording_empty`，每段最多提交一次，下一段可立即开始。
- 主窗口新增最近已提交片段的排队/转码/校验/比对状态、正面分、侧面分、综合百分比和稳定错误码；旧任务只更新自己的 JSON。后处理、双摄布局和摄像头枚举均用线程安全队列回到 Tk 主线程。
- 关窗先登记当前有效片段，再停止新提交并取消活动/排队任务；即使 3 秒等待预算耗尽也投递消费者退出哨兵。ffmpeg、VideoCapture 和 MediaPipe 在成功、失败、取消时统一释放。
- 交付并校验固定模板 `templates/standard_front_full.npz` 与 `templates/standard_side_full.npz`，保持现有 `full + v2` 兼容路径；Vue/Tauri、单摄、离线视频和评分算法不变。

### 验证方法

- T-003 Tk 生命周期与后处理定向套件：`122 passed`。
- 跨模块回归（含转码、双流比对、模板 metadata、录制控制器和 `pose33_v3` golden）：`177 passed`。
- 全量 `pytest tests -q`：`529 passed, 6 failed`；失败集合与接手基线一致，仍为 1 条 YOLO preview routing 和 5 条 `ui_backend` session 在制品，本任务未修改 Vue/Tauri/bridge 且未新增失败。
- 真实 ffmpeg/ffprobe 烟测：短 MJPG AVI 经实际 `transcode_to_h264()` 输出 `codec_name=h264` 的 1683-byte MP4，源 AVI 已删除且临时文件残留为 0。
- `py_compile` 与 `git diff --check` 通过（仅 Windows 行尾提示）。
- 实机待验：骨架关闭时连续录制两段并确认各自产生 `front/side` 视频及完成结果；录制中直接主“停止”并确认结果落盘；骨架开启时确认视频保留且结果明确为“带骨架录像未自动比对”。

## 2026-07-09: [feat/fix/test] 摄像头画面转正与双摄预览自适应

### 问题描述

USB 摄像头竖置后通常仍输出横向分辨率且不提供方向传感器，旧 Tkinter 入口没有逐路转正能力，姿态推理、预览和录制都会保留错误方向。双摄预览同时固定为上下堆叠，不能根据转正后的画面比例利用横向空间；90°/270° 旋转若不更新 writer 尺寸还会导致录制尺寸与帧不一致。

### 修改内容

- `apps/app_ui.py` 为两路摄像头分别增加 `0°/90°/180°/270°` 旋转下拉，严格解析四个合法角度，并在运行期间锁定控件；旋转只作用于实时摄像头，离线视频单/多 worker 路径保持原行为。
- 串行单摄、双摄两路和实时多 worker reader 均在 MediaPipe 推理前应用旋转，使 landmarks、在线匹配、预览与录制共用转正后的坐标系；90°/270° 先交换驱动上报尺寸，再由旋转后的首帧真实 shape 通过 `RecordingController.update_session_size()` 校正 writer，避免错误 `CAP_PROP` 造成空文件或坏文件。
- 双摄根据旋转后宽高比自动排布：两路都为竖画面时左右并排，横向/方形/混合方向时上下堆叠；先按设备尺寸切换，再用首帧真实尺寸校正，所有 Tk grid 更新通过主线程执行，单摄隐藏时同步清空第二行/列权重。
- 补齐双路录制一致性：共享录制锁覆盖片段级保存根目录/时间戳发布和两路 toggle/begin/write/stop/close，保证 front/side 的片段边界与目录一致；目录选择只更新 Tk 变量，真正的普通 `Path` 在下一次 `idle -> recording` 时锁内发布，worker 不再持锁读取 `StringVar`。UI 同时显示正面/侧面输出路径，任一路 writer 失败时结束两路片段并明确标注失败路。
- `core/recording_controller.py` 在片段停止/会话关闭时清理 `last_error`，且单摄模式忽略第二路错误，避免侧路失败污染同会话重试或下一次单摄/离线会话；worker 在 close 前把 Tk 尚未消费的错误转存到待提示队列，最后一帧失败也不会静默丢失。
- 新增 headless 回归，覆盖角度像素方向、非法值、writer 首帧尺寸校正、布局坐标/权重、状态采集、三条实时采集路径、离线视频隔离、运行态控件、第二路 writer 失败收敛，以及真实双线程下 toggle/时间戳/双写/stop 不交错。

### 验证方法

- `.\.venv\Scripts\python.exe -m pytest tests/test_app_ui_dual_camera.py tests/test_app_controls.py tests/test_app_ui_lifecycle.py tests/test_recording_controller.py -q`：`80 passed`。
- `.\.venv\Scripts\python.exe -m pytest tests/test_pose33_v3_golden.py tests/test_s5_hands_toggle.py tests/test_error_handling.py tests/test_input_source_state.py -q`：`34 passed`。
- 全量 `.\.venv\Scripts\python.exe -m pytest tests -q`：`468 passed, 6 failed, 5 warnings`；失败集合与接手前记录一致，仍为 1 条 YOLO preview routing 和 5 条 `ui_backend` session 在制品，本次未新增失败。
- `.\.venv\Scripts\python.exe -m py_compile apps\app_ui.py core\recording_controller.py tests\test_app_ui_dual_camera.py tests\test_app_controls.py tests\test_app_ui_lifecycle.py tests\test_recording_controller.py` 通过。
- `git diff --check` 通过，无 whitespace error（仅 Windows 行尾提示）。
- 自动化验证不依赖真实摄像头；两台物理摄像头的竖置方向与驱动尺寸仍需现场快速目视确认。

## 2026-07-09: [feat] 双摄双面视图两路同步落盘 + 按日期/录制段归档

### 问题描述

双摄双面视图（issue #58）录制时只有第一路（正面）落盘，第二路（侧面）仅预览不存储（`apps/app_ui.py` 原 `ponytail:` 注释标注的推迟点）。同时单一录制路径无法区分双摄两路来源。

### 修改内容

- `apps/app_ui.py` 新增第二路录制控制器 `self._rec2`（`path_provider=_record_path_cam2`），在 `_worker_loop_dual_camera` 中与第一路各自 `begin_session` / `write_frame` / `close_session`；`_on_record_toggle` / `_on_record_stop` 同步切换两路。`_rec2` 在单摄/文件模式恒 idle no-op。
- 录制路径按日期和片段归档：单摄/文件写入 `<保存目录>/<YYYYMMDD>/record_<时间戳>.mp4`；双摄两路共用同一时间戳与片段目录，分别写入 `<保存目录>/<YYYYMMDD>/record_<时间戳>/front.mp4` 和 `side.mp4`。`_dual_active` 在双摄循环进入时置 True、finally 复位。

### 验证方法

- `py_compile apps/app_ui.py` 通过。
- `pytest tests/test_app_ui_lifecycle.py tests/test_recording_controller.py -q`：`20 passed`。

## 2026-07-09: [fix/test/docs] 修复 PR #71 五项审查问题并对齐最新 main

### 问题描述

PR #71 存在五项场地前阻塞：在线直拳模板未交付、`mp4v` 回退可以产出非 H.264 MP4、摄像头预打开存在重叠任务竞态、新会话保留旧识别文案，以及未跟踪的 daemon 转码线程可在关窗时被截断。同时 PR 分支落后 `origin/main`，且夹带了 main 已有的 batch paired 重复提交。

### 修改内容

- 从最新 `origin/main` 重建 PR 分支，只重放 Tkinter、change 归档和 bridge 在制品三个有效提交，移除重复 batch commit；冲突以 main 的同目录 paired 行为和最新文档为准。
- `.gitignore` 只白名单放行 `templates/online/直拳_左手.npz` 与 `直拳_右手.npz`，其他本地模板仍忽略；新增默认模板库加载契约测试。
- `core/video_writer.py` 移除 `mp4v -> .mp4` 路径：H.264 不可用时只回退 `MJPG/XVID -> .avi`，再由 ffmpeg 转 H.264；转码先写同目录唯一临时 MP4，校验非空后原子替换，失败保留 AVI 和已有 MP4。
- `apps/app_ui.py` 为预打开请求增加 generation token，只接受最新且 `isOpened()` 的 cap，原子替换时释放旧 handle；新会话重置 `识别：待机`。
- H.264 转码改为受跟踪的 non-daemon worker；关窗按 50 ms 分片等待采集/转码任务，最多保留 GUI 3 秒，之后转码可在后台继续，ffmpeg 最长 30 分钟硬超时并在失败时保留 AVI。
- 按最新仓库事实源同步 `AGENTS.md` / `CLAUDE.md` 归档和 analysis 清单，并更新仍指向已删 `_browse_video` 或旧 YOLO 文案的回归测试。

### 验证方法

- Tkinter matcher / 平滑 / 并行 / 双摄 / 录制 / MediaPipe golden 定向安全网：`97 passed`。
- 最新文档与错误处理契约：`9 passed`。
- 全量 `pytest tests -q`：`421 passed, 6 failed, 1 skipped, 5 warnings`；剩余 5 条为 PR 第 4 个 commit 已标注的 `ui_backend` session 在制品，1 条为本轮明确暂不处理的 YOLO preview routing。
- 真实 ffmpeg + ffprobe 探测：MJPG AVI 转码后 `codec_name=h264`、MP4 `2769` bytes、源 AVI 已删除。
- `py_compile apps/app_ui.py core/video_writer.py tests/test_app_ui_lifecycle.py` 通过；`git diff --check` 无 whitespace error（仅 Windows 行尾提示）。

## 2026-07-09: [chore] change.md 归档与重建

### 问题描述

`change.md` 累积至 586 行（覆盖 2026-06-14 ~ 2026-07-09），按任务完成规范需归档并重建空文件。

### 修改内容

- `git mv change.md "change（2026.6~2026.7）.md"`，重建空 `change.md`，本条为首条记录。
- 同步更新 `CLAUDE.md` 归档清单：新增 `change（2026.6~2026.7）.md`。
- 历史归档：`change（start~2026.5）.md`、`change（2026.5~2026.6）.md`、`change（2026.6~2026.7）.md`。

### 验证方法

- `ls change*.md` → 三份归档 + 空 `change.md`。
- `git mv` 保留文件历史（非删除重建）。

## 2026-07-09: [apps/core] 接通 Tkinter 实时直拳识别、预览平滑与录制 H.264 转码

### 问题描述

`core/online_matcher.py`、`templates/online/` 左右手直拳模板与 `core/preview_smoother.py` 均已就绪且单测全绿，但 `apps/app_ui.py` 未接线：Tkinter 实时预览无法显示在线识别结果，预览骨架未平滑，录制回退的 MJPG AVI 也未转 H.264。

### 修改内容

- `apps/app_ui.py`：
  - 新增默认开启的「实时动作识别」开关 + `match_var` 命中/分数显示，以及 `_build_online_matcher`/`_feed_online_matcher`/`_post_match` 三个接入边界；模板缺失/不兼容安全降级为无 matcher，关窗/停止竞态不刷新已销毁 UI。
  - 开关值经 Tk 主线程收进 `UiState.online_match_enabled`，worker 不跨线程读 `BooleanVar`；matcher 仅接单摄实时流，双摄/离线不接。
  - 单线程摄像头按 `infer(raw) → matcher(raw) → smoother → draw` 执行；文件 VIDEO 路径保留原 `annotate()`，帧序号/时间戳语义不变。
  - 并行摄像头启用 `ParallelPoseEngine(defer_draw=True)`，在有序出口喂 raw matcher、平滑并绘制，用采集 monotonic 时间戳保留满队列丢帧的真实间隔（顺带消除原「时序平滑关闭」抖动）。
  - 绘制均以 raw landmarks 做动作分类、smoothed landmarks 画骨架，smoother 不回流识别/评分序列。
  - 新增 `_transcode_async`，接入手动结束录制与单线程/双摄/并行视频/并行摄像头四个会话收敛点；录制文件名加微秒防同秒重录与后台转码争用。
- `core/vision_pipeline.py`：`draw_pose_frame` / `MediaPipePipeline.draw` 新增可选 `action_pose_landmarks`（默认沿用绘制 landmarks，旧行为不变），平滑预览显式传 raw 做分类。
- `core/video_writer.py`：新增 `transcode_to_h264`，仅把 AVI 用 `libx264/medium/CRF20/yuv420p` 转 MP4；成功且目标非空才删 AVI，ffmpeg 缺失/失败保留原录像。
- `core/parallel_pose_engine.py`：`ParallelPoseEngine` 新增 `defer_draw`，`InferResult` 携带 raw `pose_landmarks`/`hands`/`frame` 供重排出口单线程平滑+绘制。
- 新增 `tests/test_online_matcher.py`、`tests/test_app_ui_online_matcher.py`、`tests/test_preview_smoother.py`、`tests/test_preview_draw_contract.py`、`tests/test_video_writer_transcode.py`。

### 验证方法

- 定向 + 安全网 **68 passed**：在线 matcher、Tkinter 接入、预览平滑/绘制边界、转码、`pose33_v3` golden、`valid_mask`、双摄、手部开关全绿。
- 首轮全量 `pytest tests -q` 为 **403 passed / 10 failed**。后续 PR 审查确认：其中 5 条 `ui_backend` session 与 1 条 YOLO preview routing 由 PR 第 4 个在制品提交引入，其余为落后最新文档/接口的旧测试；最终验证结果见本文件置顶记录。
- `py_compile apps/app_ui.py core/vision_pipeline.py core/video_writer.py core/parallel_pose_engine.py` → OK。
- 待场地人员按实际机位人工确认左右手命中、平滑观感、录制全链路，并现场标定 `MatcherConfig` 阈值。
