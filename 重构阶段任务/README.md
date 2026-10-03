# 整体架构重构：分阶段任务清单

本目录把 `整体架构重构设计.md`（下称"设计文档"）§12 的迁移计划按阶段拆成独立文件，每个文件四部分：任务前检查、待决策事项、具体任务、验收标准。设计细节以设计文档为准，这里是执行清单；两边不一致时先改设计文档，再改这里。

文中的计数和代码位置是 2026-10-03 在 `main`（`5a6d99c`）上统计的。代码会变，开工前按各文件的"任务前检查"重新核对。现有文件的路径指旧项目（工作区的 `old/`，只读）中的同名文件，P0 导入后以本仓库为准；命令在本仓库根运行（见设计文档文首的仓库与目录说明）。

## 阶段一览

| 阶段 | 文件 | 前置 | 规模 | 状态 |
|---|---|---|---|---|
| P0 基线（阻断） | [P0-基线.md](P0-基线.md) | — | 小 | 进行中，剩余事项见该文件第 3 节 |
| P1 基础层 | [P1-基础层.md](P1-基础层.md) | P0 | 小 | 未开始 |
| P2 解码与姿态 | [P2-解码与姿态.md](P2-解码与姿态.md) | P1 | 中 | 未开始 |
| P3 评估 | [P3-评估.md](P3-评估.md) | P2 | 大 | 未开始 |
| P4 录制 | [P4-录制.md](P4-录制.md) | P1 | 中 | 未开始 |
| P5 摄像头 | [P5-摄像头.md](P5-摄像头.md) | P1，计划排在 P4 之后 | 中 | 未开始 |
| P6 实时会话 | [P6-实时会话.md](P6-实时会话.md) | P2、P5 | 大 | 未开始 |
| P7 工作流 | [P7-工作流.md](P7-工作流.md) | P4、P6 | 中 | 未开始 |
| P8 Tk 拆分 | [P8-Tk拆分.md](P8-Tk拆分.md) | P7 | 大 | 未开始 |
| P9 收尾 | [P9-收尾.md](P9-收尾.md) | P3、P8 | 小 | 未开始 |

P0、P1 之后分两条线：算法线 P2→P3，运行时线 P4→P5→P6→P7→P8。两条线有两个汇合点：P4 的评分步骤先接旧评分函数，P3 完成后替换；P6 要用 P2 的 `create_engine`、`VideoSource` 和 `extract_pose_sequence`，所以 P2 要先完成。

## 待决策事项总表

"未决时的做法"遵循设计文档 §1.3：默认保持现有行为。P0-1～P0-4 是本目录新增的交付事项，其余编号与设计文档 §11 一致。

| 编号 | 事项 | 最晚在 | 未决时的做法 |
|---|---|---|---|
| P0-1 | 构建环境与 PyInstaller、pytest 的固定版本 | P0 | 无法开始打包 |
| P0-2 | 合并 spec 后保留单文件版还是目录版 | P0 | 无法合并 spec |
| P0-3 | 交付包附带哪些文件 | P0 | 只交付 EXE（同 9-29 包） |
| P0-4 | 从旧仓库导入哪些文件作为本仓库的基线 | P0 | 无法导入，P0 不能开工 |
| D-13 | 测试改造范围 | P1 开工前 | 依赖半成品 `App` 的测试改不了，P4 起的删除清单无法完成 |
| V-7 | `desktop.log` 诊断行统一格式 | P1 | 现有日志行保持原样，`diag` 只用于新增事件 |
| D-4 | 考试与 `batch --paired` 计分不一致 | P3 | 三种方案并存，batch 不加 `--scheme` |
| D-9 / V-8 | 模板池混用 v2/v3 归一化 | P3 | 保持现状：按首个模板归一化 |
| V-4 | 单摄转码改由 `SegmentFinalizer` 执行 | P4 | 保留 `_transcode_async`，P4 删除清单相应缩减 |
| D-6 | 1080p 不可用时 `prefer` 还是 `require` | P5 | 保持现状：`prefer`，不提示 |
| V-1 | 会话结束后租约归还池 | P5 | 保持现状：关闭子进程后重新预热 |
| V-3 | 打开设备时校验名称 | P5 | 不校验 |
| V-6 | 刷新设备时跳过占用中的设备 | P5 | 保持现状：先等池内进程释放再探测 |
| D-7 | 双摄预览平滑与在线识别 | P6 | 关闭 |
| D-8 / V-2 | 单摄模型加载期显示裸帧 | P6 | 不显示 |
| V-5 | 学生练习显示锁定，不再改写用户勾选 | P7 | 保留改写后恢复，`_student_saved_*` 不删 |
| D-10 | web 端代码处置 | P9 | 保留不动 |
| D-12 | 删除 `core/qwen_coach.py` | P9 | 保留 |
| D-2 | 正式规则把补帧判为有效 | 重构完成后 | 维持 `HOLD_LAST_LEGACY` |
| D-3 | 两套规则对同一检查项结论不同 | 重构完成后，需教师/甲方确认 | 两个 profile 并存 |
| D-5 | 正式测量改用宽高比修正后的坐标 | 与 D-3 一起 | 不修正 |
| D-11 | body_core 标定用的 baseline 2.0 与 3.0 不一致 | 任意时间，只影响 batch/analysis | 保持 1.2826 |

已决：D-1（学生练习以 v4 为基线，`f7112f9`）、D-14（摄像头沿用 PR #75 的"池 + 租约"，`095aed2`）。

## 附录 B 搬迁的阶段归属

设计文档 §12 表内没写阶段的文件搬迁，按目标层归入对应阶段。旧导入路径保留为只做转发的门面，P9 统一删除；门面里不得留逻辑。

| 阶段 | 搬迁 |
|---|---|
| P1 | `core/paths.py` → `foundation/paths.py`；三套偏好读写 → `foundation/prefs.py`；`motion_energy`、`find_active_range` → `foundation/signal.py`；`to_jsonable` → `foundation/jsonable.py` |
| P2 | `vision_pipeline` → `pose/engine.py`、`pose/render.py`、`evaluation/live_labels.py`；`pose_features` 的有效性、归一化、视角分数；`feature_layout`、`model_manager`、`parallel_pose_engine`、`preview_smoother`、`backend_router`、`yolo_adapter`；`body_core_compare` 的提取部分；`video_writer` → `media/writer.py` |
| P3 | `action_compare` 其余部分（模板、匹配、切分、模板创建、匹配预览导出）、`rule_scoring`、`online_matcher`、`kick_quality`、`action_feedback`、`feedback_geometry`、`feedback_phases`；`pose_features.subsequence_dtw`；`body_core_compare` 的匹配部分；`recording_postprocess` 的评分常量与模板校验；`analysis/tech_eval.py` → `evaluation/techeval/`；`make_template`、`match_template` 的实现 → `apps/cli/` |
| P4 | `recording_controller` → `recording/controller.py`；`recording_postprocess` 的后处理 → `recording/finalizer.py`；`exam_clip` → `workflows/exam/clip.py`；`exam_roster` 的台账部分 → `workflows/exam/scorebook.py`；`feedback_history` → `services/practice_history.py` |
| P5 | `camera_enum` → `media/camera/devices.py`；`camera_capture` → `media/camera/capture_process.py`；`camera_warmup` → `media/camera/pool.py` |
| P6 | `apps/main.py` 的实现 → `apps/cli/main.py` |
| P7 | `exam_session`、`exam_roster` 其余部分、`presence_gate`、`exam_announcer` → `workflows/` |
| P8 | `app_ui`、`exam_panel`、`feedback_panel`、`feedback_result_panel`、`desktop_smoke` → `apps/tk/`；`feedback_report` → `reporting/` |
| P9 | 删除全部门面；按 D-10 处置 `ui_backend` 与 `ui_backend_sidecar.spec`，按 D-12 处置 `qwen_coach` |

## 通用任务前检查

每个阶段开工前，除本阶段文件第 1 节外：

- [ ] 前置阶段已按其验收标准完成，"删除"清单没有遗留。
- [ ] 从最新 `main` 新建分支（建议 `refactor/p<N>-<主题>`），`git status` 干净。
- [ ] 全量测试与上一阶段结束时一致。旧仓库基线（`5a6d99c`）：`D:\DevTools\venvs\vision\Scripts\python.exe -m pytest -q tests` 为 1026 passed、6 failed（web 桥）、1 skipped。P0 导入代码、新建 `.venv` 后，改用本仓库 `.venv` 测定的结果。
- [ ] golden 与契约通过：`.\.venv\Scripts\python.exe -m pytest tests/test_pose33_v3_golden.py tests/test_valid_mask_migration.py tests/test_tech_eval_contract.py tests/test_rule_availability.py -q`
- [ ] 本阶段的待决策事项都有结论，并记入设计文档 §11（已决的划线）。
- [ ] 本阶段文件里的计数和代码位置已重新核对；有出入时先改设计文档。
- [ ] P1 之后：架构测试通过，记下 allowlist 条数。

## 通用验收标准

每个阶段结束时，除本阶段文件第 4 节外：

- [ ] "删除"清单逐项完成，用文件里给出的搜索确认没有残留。
- [ ] 门禁测试通过；全量测试除 web 桥 6 项外全部通过（D-10 执行后全部通过）。
- [ ] golden 零漂移。只有已批准的决策可以改变数值，且单独 PR、单独重生成 golden。
- [ ] 架构测试通过，allowlist 条数没有增加。
- [ ] 每个 PR：`git diff --check` 通过；change.md 写明问题描述、修改内容、同步依据、验证方法；Conventional Commits；列出被替换的测试和对应的新测试（D-13）。
- [ ] 推送阶段分支（`refactor/p<N>-<主题>`），用 `gh pr create --base main` 向本仓库开 PR。PR 合并后，依赖本阶段的后续阶段才能开工。
- [ ] 设计文档与本目录同步：§12 状态、受影响的章节、本阶段文件的状态。
- [ ] 涉及运行时的阶段（P0、P5、P6、P8、P9）：源码模式运行模式切换检查 `tests/test_camera_capture.py::test_tk_preview_student_switch_with_process_cameras`（真实 Tk + spawn），并构建 EXE 运行 `--offline-self-check`。整套自检只能在冻结包里跑，它要求模型已播种到临时 `LOCALAPPDATA`。
