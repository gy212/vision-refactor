# 整体架构重构：分阶段任务清单

本目录把 `整体架构重构设计.md`（下称"设计文档"）§12 的迁移计划按阶段拆成独立文件，每个文件四部分：任务前检查、待决策事项、具体任务、验收标准。设计细节以设计文档为准，这里是执行清单；两边不一致时先改设计文档，再改这里。

文中的计数和代码位置是 2026-10-03 在 `main`（`5a6d99c`）上统计的。代码会变，开工前按各文件的"任务前检查"重新核对。现有文件的路径指旧项目（工作区的 `old/`，只读）中的同名文件，P0 导入后以本仓库为准；命令在本仓库根运行（见设计文档文首的仓库与目录说明）。

## 阶段一览

| 阶段 | 文件 | 前置 | 规模 | 状态 |
|---|---|---|---|---|
| P0a 基线导入（阻断 P1） | [P0-基线.md](P0-基线.md) 第 3a 节 | — | 小 | 已完成（2026-10-03，阶段 PR 待 P0b 完成后一起提） |
| P0b 打包与交付（阻断交付，最晚 P5 验收前） | [P0-基线.md](P0-基线.md) 第 3b 节 | P0a | 中 | 已完成（2026-10-03，构建 `2219a8f`）；§13.4 实机清单移到重构完成后统一测试（见“待办”） |
| P1 基础层 | [P1-基础层.md](P1-基础层.md) | P0a | 小 | 已完成（2026-10-04），allowlist 基线 148 条 |
| P2 解码与姿态 | [P2-解码与姿态.md](P2-解码与姿态.md) | P1 | 中 | 已完成（2026-10-05），allowlist 降到 114 条 |
| P3 评估 | [P3-评估.md](P3-评估.md) | P2 | 大 | 已完成（2026-10-10），allowlist 降到 75 条；D-4、D-9 已决，实施见 P3 第 5 节；练习 profile 接入 `evaluate_rules` 按 D-16 移交 P9 |
| P4 录制 | [P4-录制.md](P4-录制.md) | P1 | 中 | 已完成（2026-10-11），allowlist 降到 66 条 |
| P5 摄像头 | [P5-摄像头.md](P5-摄像头.md) | P1，计划排在 P4 之后（两者都改旧 worker 循环，见该文件）；验收要出 EXE 自检，需 P0b | 中 | 已完成（2026-10-12），allowlist 降到 60 条；D-6、V-1、V-3、V-6 已决并实施 |
| P6 实时会话 | [P6-实时会话.md](P6-实时会话.md) | P2、P5 | 大 | 已完成（2026-10-12），allowlist 降到 51 条；D-7、D-8/V-2 已决并实施 |
| P7 工作流 | [P7-工作流.md](P7-工作流.md) | P4、P6 | 中 | 已完成（2026-10-13） |
| P8 Tk 拆分 | [P8-Tk拆分.md](P8-Tk拆分.md) | P7 | 大 | 已完成（2026-10-14） |
| P9 收尾 | [P9-收尾.md](P9-收尾.md) | P3、P8 | 中 | 已完成（2026-10-14），allowlist 清零；D-12 已决并实施，D-15、D-16 已实施；重构完成 |

P0a、P1 之后分两条线：算法线 P2→P3，运行时线 P4→P5→P6→P7→P8。两条线有两个汇合点：P4 的评分步骤先接旧评分函数，P3 完成后替换；P6 要用 P2 的 `create_engine`、`VideoSource` 和 `extract_pose_sequence`，所以 P2 要先完成。两条线都会改 `apps/recording_postprocess.py` 和 `apps/app_ui.py`，并行只在开发期成立，合并是串行的：先合入者负责搬迁重叠文件，后合入者 rebase 后在新位置改（设计文档 §12）。原约定 P4 先于 P3 合入，实际 P3（2026-10-10）先于 P4（2026-10-11）合入，该约定已作废。

## 待办

- [ ] 1. 重构完成后统一实机测试（2026-10-03 用户决定）：重构期间不向甲方交付 EXE，P0b、P5、P9 原有的实机验收都不在阶段内做，P9 验收后用合并后的 `main` 构建 EXE，在甲方的目标双摄上一次走完设计文档 §13.4 的实机清单。清单：两台目标摄像头的枚举结果与名称；逐路遮挡确认正/侧身份；1080p 下的实际格式（D-17 后三种模式统一），以及普通、考试双摄 1080p 的 CPU 负载与首帧时间；模式切换连续 20 次；运行中拔掉一路；关窗后任务管理器中没有残留的 `vision_ui` 子进程；`desktop.log` 诊断字段齐全（字段以届时 V-7 的结论为准）。证据单独记录，不用自检代替。PR #75 的摄像头改动至今未在目标硬件上验证过。

以下 2–5 项是各阶段原本要用真实素材或真实设备做的验证（2026-10-06 用户决定：凡需要真实测试的，一律记为待办）。阶段内只用 fixture、合成视频、公开视频和模拟驱动验收，与第 1 项一起在 P9 验收后统一做；做法是在同一台机器上分别运行"旧 commit"和合并后的 `main`，结果逐项比对。证据记进 change.md，不用自检或模拟测试代替。

- [ ] 2. 真实视频的提取差分（P2）：P2 阶段用的是 Wikimedia 上的公开踢拳视频（本机没有真实录像，路径见 change.md 2026-10-05 的 P2 记录）。待办：用甲方的真实录像（单摄、双摄正侧各一对、学生练习 1080p，各至少 1 段）跑 `tests/real_video_diff/record.py`，旧 commit 用 `5721f1d`（P2 前），比较 landmarks、有效掩码、特征、视角分数、时间戳，要求 `compare.py` 0 差异。后续阶段若移动了脚本用到的入口，更新脚本里的导入即可。
- [ ] 3. 真实视频的评分差分（P3）：分数、匹配区间、规则状态、证据帧。阶段内用 fixture 回放和第 2 项的公开视频录基准；待办是用真实的标准动作与学员录像，在 P3 前的 commit 与 `main` 上各跑一遍比对入口、`score_rules`、`evaluate_series` 和考试池评分。只比较没有被已批准决策（§11.2）改变的项。
- [ ] 4. 真实录制的片段产物（P4）：用真实摄像头各录一段单摄、普通双摄（自动比对开/关）、考试、学生练习片段，在 P4 前的 commit 与 `main` 上比对目录结构、文件名和结果 JSON 字段。阶段内只用现有测试断言作对照。
- [ ] 5. 实时会话的性能（P6）：同一台机器、同一组设备，单摄与双摄的启动耗时（`_DualStartupMetrics` 与新的诊断输出）和预览帧率。在 P6 前的 commit 与 `main` 上各测一次，`main` 不得变差。阶段内只用模拟驱动验证功能，不做性能结论。

## 待决策事项总表

"未决时的做法"遵循设计文档 §1.3：默认保持现有行为。P0-1～P0-4 是本目录新增的交付事项，其余编号与设计文档 §11 一致。

| 编号 | 事项 | 最晚在 | 未决时的做法 |
|---|---|---|---|
| ~~P0-1~~ | ~~构建环境与 PyInstaller、pytest 的固定版本~~ | ~~P0~~ | ~~已决（2026-10-03）：用 `.venv`，固定版本见设计文档 §11.2~~ |
| ~~P0-2~~ | ~~合并 spec 后保留单文件版还是目录版~~ | ~~P0~~ | ~~已决（2026-10-03）：只保留单文件版~~ |
| ~~P0-3~~ | ~~交付包附带哪些文件~~ | ~~P0~~ | ~~已决（2026-10-03）：使用说明、配置示例、第三方许可三项都纳入~~ |
| ~~P0-4~~ | ~~从旧仓库导入哪些文件作为本仓库的基线~~ | ~~P0~~ | ~~已决（2026-10-03）：按建议导入（本仓库 `5574d95`）~~ |
| ~~D-13~~ | ~~测试改造范围~~ | ~~P4 开工前~~ | ~~已决（2026-10-11）：接受，每个 PR 列出被替换的测试和新测试~~ |
| ~~V-7~~ | ~~`desktop.log` 诊断行统一格式~~ | ~~P1~~ | ~~已决（2026-10-04）：接受，`[camera]` 行 P5 并入 `diag`~~ |
| ~~D-4~~ | ~~考试与 `batch --paired` 计分不一致~~ | ~~P3~~ | ~~已决（2026-10-10）：采纳建议，batch 新增 `--scheme`（缺省不变），单独 PR~~ |
| ~~D-9 / V-8~~ | ~~模板池混用 v2/v3 归一化~~ | ~~P3~~ | ~~已决（2026-10-10）：采纳建议，逐模板归一化并提示混用，单独 PR~~ |
| ~~D-16~~ | ~~练习 profile 的技术执行框架统一（与 D-3 分开）~~ | ~~P3~~ | ~~已决（2026-10-11，issue #8）：延期到 P9 补齐，范围与验收见设计文档 §11.2、P9 第 3 节第 8 项~~ |
| ~~V-4~~ | ~~单摄转码改由 `SegmentFinalizer` 执行~~ | ~~P4~~ | ~~已决（2026-10-11）：接受，P4 删除 `_transcode_async`~~ |
| ~~D-6~~ | ~~1080p 不可用时 `prefer` 还是 `require`~~ | ~~P5~~ | ~~已决（2026-10-12）：学生练习用 `require`~~ |
| ~~V-1~~ | ~~会话结束后租约归还池~~ | ~~P5~~ | ~~已决（2026-10-12）：接受~~ |
| ~~V-3~~ | ~~打开设备时校验名称~~ | ~~P5~~ | ~~已决（2026-10-12）：接受~~ |
| ~~V-6~~ | ~~刷新设备时跳过占用中的设备~~ | ~~P5~~ | ~~已决（2026-10-12）：接受~~ |
| ~~D-7~~ | ~~双摄预览平滑与在线识别~~ | ~~P6~~ | ~~已决（2026-10-12）：保持关闭，作为配置项保留~~ |
| ~~D-8 / V-2~~ | ~~单摄模型加载期显示裸帧~~ | ~~P6~~ | ~~已决（2026-10-12）：启用，可配置关闭~~ |
| ~~V-5~~ | ~~学生练习显示锁定，不再改写用户勾选~~ | ~~P7~~ | ~~已决（2026-10-13）：接受，删除 `_student_saved_*`~~ |
| ~~D-10~~ | ~~web 端代码处置~~ | ~~P0a~~ | ~~已了结（2026-10-03）：P0-4 未导入 web 端~~ |
| ~~D-12~~ | ~~删除 `core/qwen_coach.py`~~ | ~~P9~~ | ~~已决（2026-10-14）：删除，P9 实施~~ |
| ~~D-2~~ | ~~正式规则把补帧判为有效（issue #20）~~ | ~~重构完成后~~ | ~~已决（2026-10-14）：改用 `HOLD_LAST_INVALID`，删除 `HOLD_LAST_LEGACY`，单独 PR、重新生成 golden~~ |
| ~~D-3~~ | ~~两套规则对同一检查项结论不同（issue #21）~~ | ~~重构完成后~~ | ~~已决（2026-10-14）：各模式共用一套规则目录，按左右式取手，考试固定左式；细则见设计文档 §7.2 决策补充~~ |
| ~~D-5~~ | ~~正式测量改用宽高比修正后的坐标（issue #22）~~ | ~~与 D-3 一起~~ | ~~已决（2026-10-14）：各模式都用宽高比修正坐标，阈值不变，随 D-3 实施~~ |
| ~~D-11~~ | ~~body_core 标定用的 baseline 2.0 与 3.0 不一致（issue #23）~~ | ~~新样本到位后~~ | ~~已决（2026-10-14）：改为 3.0 后用新样本重跑标定（`docs/样本视频采集清单.md`）~~ |
| ~~D-17~~ | ~~考试采集格式与现行拍摄规格（正面竖屏 1080p、侧面横屏 1080p）不符~~ | ~~D-3 实施前~~ | ~~已决（2026-10-14）：三种模式统一按 1920×1080 `require` 采集，单独 PR 实施；已实施（2026-10-14），实机复核随待办第 1 项~~ |

已决：D-1（学生练习以 v4 为基线，`f7112f9`）、D-14（摄像头沿用 PR #75 的"池 + 租约"，`095aed2`）、D-15（`CameraPool` 体量：P5 原样迁入，P9 按职责拆分、只搬不改，2026-10-03）、V-7（诊断行统一格式，2026-10-04）。

## 附录 B 搬迁的阶段归属

设计文档 §12 表内没写阶段的文件搬迁，按目标层归入对应阶段。旧导入路径保留为只做转发的门面，P9 统一删除（2026-10-14 已全部删除）；门面里不得留逻辑。

| 阶段 | 搬迁 |
|---|---|
| P1 | `core/paths.py` → `foundation/paths.py`；三套偏好读写 → `foundation/prefs.py`；`motion_energy`、`find_active_range` → `foundation/signal.py`；`to_jsonable` → `foundation/jsonable.py` |
| P2 | `vision_pipeline` → `pose/engine.py`、`pose/render.py`、`pose/live_labels.py`；`pose_features` 的有效性、归一化、视角分数；`feature_layout`、`model_manager`、`parallel_pose_engine`、`preview_smoother`、`backend_router`、`yolo_adapter`；`body_core_compare` 的提取部分；`video_writer` → `media/writer.py` |
| P3 | `action_compare` 其余部分（模板、匹配、切分、模板创建、匹配预览导出）、`rule_scoring`、`online_matcher`、`kick_quality`、`action_feedback`、`feedback_geometry`、`feedback_phases`；`pose_features.subsequence_dtw`；`body_core_compare` 的匹配部分；`recording_postprocess` 的评分常量与模板校验；`analysis/tech_eval.py` → `evaluation/techeval/`；`make_template`、`match_template` 的实现 → `apps/cli/` |
| P4 | `recording_controller` → `recording/controller.py`；`recording_postprocess` 的后处理 → `recording/finalizer.py`；`exam_clip` → `workflows/exam/clip.py`（L5 包骨架由 P4 先建，只含 clip 与 scorebook）；`exam_roster` 的台账部分 → `workflows/exam/scorebook.py`；`feedback_history` → `services/practice_history.py`（报告渲染调用移出）；新建 `media/segment_layout.py`、`recording/service.py`、`recording/steps.py` |
| P5 | `camera_enum` → `media/camera/devices.py`；`camera_capture` → `media/camera/capture_process.py`；`camera_warmup` → `media/camera/pool.py` |
| P6 | `apps/main.py` 的实现 → `apps/cli/main.py` |
| P7 | `exam_session`、`exam_roster` 其余部分、`presence_gate`、`exam_announcer` → `workflows/` |
| P8 | `app_ui`、`exam_panel`、`feedback_panel`、`feedback_result_panel`、`desktop_smoke` → `apps/tk/`；`feedback_report` → `reporting/` |
| P9 | 删除全部门面；D-10 若已随 P0-4 了结则跳过，否则按 D-10 处置 `ui_backend` 与 `ui_backend_sidecar.spec`；按 D-12 处置 `qwen_coach` |

## 通用任务前检查

每个阶段开工前，除本阶段文件第 1 节外：

- [ ] 前置阶段已按其验收标准完成，"删除"清单没有遗留。
- [ ] 从最新 `main` 新建分支（建议 `refactor/p<N>-<主题>`），`git status` 干净。
- [ ] 全量测试与上一阶段结束时一致。旧仓库基线（`5a6d99c`）：`D:\DevTools\venvs\vision\Scripts\python.exe -m pytest -q tests` 为 1026 passed、6 failed（web 桥）、1 skipped。本仓库基线（P0a，2026-10-03）：`.\.venv\Scripts\python.exe -m pytest tests -q` 为 947 passed，无失败、无跳过；与旧基线相差的 86 项都是未导入的测试（web 端 7 个文件 80 项，两份门禁测试拆出的 6 项），见设计文档 §12。
- [ ] golden 与契约通过：`.\.venv\Scripts\python.exe -m pytest tests/test_pose33_v3_golden.py tests/test_valid_mask_migration.py tests/test_tech_eval_contract.py tests/test_rule_availability.py -q`
- [ ] 本阶段的待决策事项都有结论，并记入设计文档 §11（已决的划线）。
- [ ] 本阶段文件里的计数和代码位置已重新核对；有出入时先改设计文档。
- [ ] P1 之后：架构测试通过，记下 allowlist 条数。

## 通用验收标准

每个阶段结束时，除本阶段文件第 4 节外：

- [ ] "删除"清单逐项完成，用文件里给出的搜索确认没有残留。
- [ ] 门禁测试通过；全量测试全部通过（P0-4 不导入 web 端后，web 桥的 6 项失败不进入本仓库；若导入了，除这 6 项外全部通过）。
- [ ] golden 零漂移。只有已批准的决策可以改变数值，且单独 PR、单独重生成 golden。
- [ ] 架构测试通过，allowlist 条数没有增加，体量条目的行数上限没有调高（棘轮规则见 AGENTS.md"模块职责与体量"）。
- [ ] 本阶段新建或改动的模块：docstring 写明职责；超过 600 行的已在 PR 里说明拆分安排。
- [ ] 每个 PR：`git diff --check` 通过；change.md 写明问题描述、修改内容、同步依据、验证方法；Conventional Commits；列出被替换的测试和对应的新测试（D-13）。
- [ ] 推送阶段分支（`refactor/p<N>-<主题>`），用 `gh pr create --base main` 向本仓库开 PR。PR 合并后，依赖本阶段的后续阶段才能开工。
- [ ] 设计文档与本目录同步：§12 状态、受影响的章节、本阶段文件的状态。
- [ ] 涉及运行时的阶段（P0、P5、P6、P8、P9）：源码模式运行模式切换检查 `tests/test_camera_capture.py::test_tk_preview_student_switch_with_process_cameras`（真实 Tk + spawn），并构建 EXE 运行 `--offline-self-check`。整套自检只能在冻结包里跑，它要求模型已播种到临时 `LOCALAPPDATA`。
