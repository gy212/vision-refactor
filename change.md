## 2026-11-26: [feat] 重构从视频制作模板链路、新增双摄同步预览选段弹窗与质量校验闸门

- 问题：原「动作分析（模板比对）」窗口内置的「从视频生成模板」折叠区存在按钮中文宽度截断（`▸ 从视频生...`）、固定盲裁（70 分位单峰误裁、无视觉预览确认、手填起止帧倒挂/越界静默退化为 1 帧）、非原子写盘与同名覆盖（强行写入 `<stem>_heavy.npz` 导致双摄 `front`/`side` 覆盖且破坏考试规则/几何查表 `stem`）、`workers > 1` 分段提取却标 `video`、空拍/低质量全 0 模板无拦截、主窗关窗漏管比对弹窗与运行期控件未锁定、以及 `apps/cli/make_template.py` 绕过公共服务且默认 `full` 与 Tk `heavy` 不一致等系统性隐患。
- 修改：
  1. **模板建库公共服务、质量体检闸门与原子写盘**：新增 `core/services/template_quality.py`，在 `core/services/templates.py` 与 `core/services/__init__.py` 补全 `discover_recording_videos`、`sanitize_action_name`、`planned_template_paths`、`TemplateQualityReport`/`TemplateQualityError`、`TemplateCreateResult` 与 `create_template_pair`；`create_template_from_video` 强制单线程顺序提取并单次调用 `extract_pose_sequence` 同时供 `derive_valid_mask` 做体检（帧数不足/全空/低检出率/核心躯干有效率不达标即拒止），严格校验显式起止帧并接入 `atomic_target` 原子写盘与 `overwrite=False` 防覆盖。
  2. **下游评估校验与 CLI 收口**：`core/evaluation/scoring/exam_pool.py` 为 `SCORE_SQUEEZE`/`KICK_KINDS` 统一剥离 `_正面`/`_侧面` 后缀；`core/evaluation/matching/engine.py` 在 `match_templates` 逐个校验模板池内各模板的 `layout`/`normalizer_version`/`pose_variant` 一致性；`apps/cli/make_template.py` 收口调用 `create_template_from_video`、默认 `--pose` 对齐为 `heavy` 并新增 `--no-auto-crop`。
  3. **独立复用「制作模板」弹窗与主界面/比对窗口联动**：新增 `apps/tk/controllers/template_builder.py` 与 `apps/tk/views/template_builder_dialog.py`，支持双摄录制目录与单视频（正面/侧面/不加视角后缀）加载、同步双路滑块与 `±1/±5` 帧微调实时画面预览、12 标准动作下拉+自定义命名、覆盖确认及质量体检反馈；`apps/tk/views/recording_panel.py` 与 `apps/tk/controllers/template_pools.py` 增加 `制作模板…` 入口与 `add_paths`/`add_created` 自动回填；`apps/tk/views/compare_dialog.py`、`apps/tk/controllers/compare.py` 与 `apps/tk/app.py` 改由 `从视频制作模板…` 唤起共享弹窗，修复运行期池按钮锁定、线程数输入容错、`InterruptedError` 显示 `已停止`、主窗关闭协同停任务以及 `CollapsibleSection` 全角中文宽度计算，抽离 `ProgressPanel`/`FramePreviewSlot` 到 `apps/tk/views/widgets.py`，并同步清理 `.harness-baseline.json` 中已消除的 `compare_dialog.py::silent_except` 债务。
  4. **单测补充**：新增 `tests/test_template_creation.py` 与 `tests/test_template_builder_ui.py`，同步更新 `tests/test_tk_compare_settings.py`。
  5. **PR #70 首轮审查修复收口**：新增 `core/foundation/atomic_io.py::atomic_replace_many` 事务回滚保旧模板（`_commit_staged_pair` 写盘前/预览前查 `stop_evt`）；`core/services/preview.py` 默认 `keep_partial=False`、取消清理残片并抛 `InterruptedError`，且 `.avi` 回退自动调 `transcode_to_h264` 转码 MP4；`apps/cli/make_template.py` 改用 `is not None` 处理 `0` 帧并打印真实预览路径；将 `VIEW_FRONT_SUFFIX`/`VIEW_SIDE_SUFFIX`/`VIEW_SUFFIXES`/`split_view_suffix`/`base_action_name` 上移至 `core/evaluation/templates.py`，`exam_pool.py` 复用并压回 600 行；`template_quality.py` 委托 `SegmentLayout.resolve_videos`、校验单文件 `VIDEO_SUFFIXES`、拦截 Windows 保留名与自带视角后缀，并抽取公共 `validate_frame_bounds` 供 `body_core_templates.py` 与 `templates.py` 共用且接入 `atomic_target`；`engine.py` 在提特征前前置校验模板池兼容（缺省 `pose_variant` 归一 `"full"`）并汇总全池 `running_mode`；UI 侧 `TemplateBuilderController.start_generate` 改传 `start_s`/`end_s` 兼容容器虚高帧数，弹窗按 `pool_mode="dual"|"single"|"none"` 隔离运行期快照/关闭未停妥重入/`CompareDialog` 双摄单侧入池/主界面 `single` 禁选并统一 `VIDEO_SUFFIXES` 对话框后缀，同时修复 `test_tk_compare_settings.py` 无 Tk 收集失败与 `test_template_builder_ui.py` 5 处私有直调警告并补全双方边界单测。
  6. **PR #70 二轮审查修复**：在 `core/foundation/atomic_io.py` 抽取 `_replace_with_retry` 供 `replace_file`/`atomic_commit_staged`/`_rollback_replaced` 统一执行提交与回滚 `PermissionError` 有界重试，并按 `os.path.normcase(str(t.resolve(strict=False)))` 拦截 Windows 大小写等价重复目标；`core/services/preview.py` 与 `core/services/templates.py` 改为预览先写同目录临时文件、再由 `atomic_commit_staged` 把 `.npz` 与预览一起事务提交，取消/异常绝不破坏既有文件，同时将 `analyze_template_auto_crop` 等下沉 `template_quality.py` 使 `templates.py` 压回 496 行；`tests/tk_app_doubles.py` 改为 `pytest.importorskip("tkinter")` 兼容无 Tk 采集，`TemplateBuilderDialog` 新增 `self._running` 消除 `set_running(False)` 与 `controller.busy` 竞态，并在 `test_foundation_atomic_io.py`/`test_video_writer_transcode.py`/`test_template_creation.py`/`test_template_auto_crop.py`/`test_template_builder_ui.py` 补全全部边界单测。
- 验证：WSL `./.venv/wsl/bin/python` 运行 `ruff format --check`（全部新增/修改 `.py` 文件通过）、`square check .`（退出码 0，无新增违规，`exam_pool.py` 600 行、`templates.py` 496 行、`app.py` 299 行、`template_builder_dialog.py` 533 行、`test_template_builder_ui.py` 0 条 `test_private_access`）与受影响全套 pytest；并在 `/mnt/d/CodeProject/vision/new` Windows `.venv` 复跑真实 Tk 与 CI 门禁全套 pytest、Ruff 校验与 diff 检查全部通过。

## 2026-11-22: [fix] 修复 PR #69 审查指出的步法窗重叠、按次标量门控、掩码连坐与滞留微抖问题

- 问题：PR #69 第一轮审查指出 3 项 P1 缺陷：1）`step_footwork_windows` 相邻两次步法窗直接截断到 `prev_end`/`next_start`，导致连贯节奏（间隔 `0.3s～0.6s`）下跨次重叠串味，且 `combo_transition_windows` 用 `len(pairs) + 1` 编号在首轮缺前手时错标次数；2）7 条按次标量规则缺少逐次实例有效性门控，多次重复中某一次遮挡被其余正常次数稀释伪造为通过，同时 `measure_step_landing_lag` 双脚强制同帧与门连坐、`measure_combo_punch_pause` 侧面强制四点同见连坐；3）`measure_punch_post_peak_hold` 与 `_refine_hook_peak` 在全窗取极值，滞留尾端 `+1e-4`/`+0.002` 微抖即可将滞留时长归零或将摆拳峰值拖至收拳前夕。
- 修改：
  1. **步法窗中点切分与衔接次数单源一致**：`core/evaluation/step_reps.py::step_footwork_windows` 对齐 `exam_pool.rep_windows` 以 `0.5 * (prev_rep["end"] + next_rep["start"])` 中点截断并保证相邻步法帧区间严格不相交；`core/evaluation/measures/feet.py::foot_settle_index` 仅在显式传入 `peak_time`（`step_late_landing`）时经 `_first_planted_plateau` 识别跨过出拳峰值的真静止落稳平台（`ptp <= 0.015`、`max(|vx|) < 0.08`、`t_end >= peak_time`），既防止连贯下一次预迈步尾段拖偏落点，又不误截两段式慢跟步、恒速慢拖步与两拍垫步；`core/evaluation/event_phases.py::combo_transition_windows` 改为 `for rep_idx, rear in enumerate(rear_motions, 1)`，与 `_combination_records` 次数身份保持单源一致。
  2. **引擎级按次标量门控与双脚/双腕掩码解耦**：在 `core/evaluation/rules/spec.py`（`PhaseVerdict.per_instance`）、`rule_definitions.py`（7 条按次标量策略 `foot_forward_displacement`、`rear_foot_overtake`、`step_landing_sync`、`punch_post_peak_recovery`、`combo_punch_pause_gap`、`pad_step_settle_pause`、`retract_ankle_reach`）与 `engine.py`（`_resolve_per_instance`）统一接入逐次独立门控——任一次达有效门槛且违规则保留 `violation`，无违规但任一次有效帧不足则退回 `unable` 并明示原因；`core/evaluation/measures/feet.py` 拆出 `_single_foot_lag` 与 `measure_step_landing_frames`（`step_rules.py` 传入 `min_valid_ratio=ctx.params.valid_ratio`），对 `front_cut` 自身校验有效占比并隔离 `~cut` 无关尾窗遮挡（违规证据仅标 `front_ankle` 真实可见的 `cut` 帧），同时保留 `measure_step_landing_lag` 兼容返回；`core/evaluation/rules/punch_rules.py::measure_combo_punch_pause` 放宽侧面掩码为有效躯干加至少一条完整肩腕链可见。
  3. **出拳滞留与摆拳峰值抗平台微抖**：`core/evaluation/measures/motion.py::measure_punch_post_peak_hold` 改为以首次进入峰值平台带 `max(return_reach, 0.85 * max_reach)` 作为滞留起点，向后统计至首次回落（或 `held_ending` 至窗尾）；`core/evaluation/hook_events.py::_refine_hook_peak` 通过 `_leading_arc_mask` 将精修弧段封顶于首次到达 `0.85` 平台后的 `_HOOK_PEAK_LEADING_SECONDS`（`0.25s`）前沿区间，且正面以有符号内摆进度 `signed_cross` 定位平台起点避免外引帧截断环弧 C66。
  4. **对抗回归单测**：新建 `tests/test_per_instance_rule_boundaries.py`（复用 `test_hook_and_step_rules.py` 与 `test_low_kick_rules.py` 公开构造器），全覆盖连贯双次 `step_hook_combo`/`step_straight_combo`、首轮缺前手次轮编号、两段式慢跟步/匀速慢拖步/两拍垫步边界、全部 7 条按次标量规则单次遮挡 `unable` 与违规次保留、`step_late_landing` 双脚解耦与 `front_cut` 双向遮挡边界、`hook_combo` 远侧手腕遮挡停顿检出，以及 `+1e-4`/`+0.002` 平台微抖滞留、长滞留摆拳前沿峰值与正面环弧 C66 / 侧面 `0.25s` 横摆弧稳定性。
- 验证：WSL `./.venv/wsl/bin/python` 亲跑 PR #69 Review 3 段复现反例与复审 7 组对抗用例全部通过；通过 `square check .`、`ruff format --check --line-length 100`、`ruff check`、`git -c core.whitespace=cr-at-eol diff --check` 以及门禁、黄金基线（`pose33_v3/golden.json` 与 14 份 HTML golden 零漂移）和受影响规则/低鞭腿/考试/报告 pytest。

## 2026-11-22: [docs] 同步规范文档索引表述并将按需契约升级为操作前必读

- 问题：`AGENTS.md` 规范文档索引中仍将 Issue / PR / Review 契约写为“按需读取契约”，未明确“执行对应操作前必须先完整读取，未读严禁执行”，且分组与条目命名尚未与最新 Square 规范索引对齐。
- 修改：参照用户提供截图同步更新 `AGENTS.md` 的“规范文档索引”段落：调整“强制必读契约”与“可插拔技术栈样例包”表述并去掉冗余后缀，将“按需读取契约”升级为“按需必读契约”，明确执行 Issue、PR、Review 等操作前**必须先完整读取对应契约，未读严禁执行对应操作**，并同步三条子项名称（保留本仓 `square review` 挂载与 `square contract-check` 校验命令提示；同步依据：用户 2026-11-22 截图指令）。
- 验证：确认索引内 6 份 `specs/` 链接文件均存在，`git -c core.whitespace=cr-at-eol diff --check` 零空白错误，`square check .` 静态门禁通过。

## 2026-11-22: [docs] 改写全量 46 条检查标准为大白话并简化双摄肩宽比显示（v5.22）

- 问题：12 类动作全部 149 条检查项的 `standard` 文案与双摄肩宽比测量项显示名此前仍含“倍躯干长”“正交机位近似”“垂向位置”“归一肩宽比”“锁定角”等程序术语，不利于教练与学生直接阅读。
- 修改：
  1. **全量 46 条大白话标准改写（纯文案层，零公式/阈值/视角/点位/check id 变动）**：按已确认对照表逐条更新 `core/evaluation/rules/rule_descriptions.py` 与 `core/evaluation/rules/segment_blueprints.py` 的全部 46 条唯一共享标准（保留 `PARAMS`/`KICK_PARAMS`/`HOOK_SHOULDER_PARAMS` 单一事实来源插值，统一去掉拳法/低鞭腿不必要的重复覆盖），覆盖全部 12 动作 149 条目录检查项。
  2. **简化双摄肩宽比显示名与版本升级**：`core/evaluation/paired_torso.py` 将 `"双视角归一肩宽比（正/侧）"`（单位 `"倍（正交机位近似）"`）简化为 `"正面与侧面肩宽比"`（单位 `"倍"`）；升级 `core/evaluation/rules/sanda_v5.py` `RULE_VERSION = "sanda-rules-mediapipe-2026-11-22-v5.22"`。
  3. **结构化护栏与单测同步**：在 `tests/test_rule_reuse.py` 新增 `test_catalog_standards_are_plain_language_and_paired_torso_label_simplified` 护栏（断言全量 149 项零黑话、精确 46 条唯一共享标准及 `paired_torso` 显示名与单位），同步更新 `tests/test_rule_evidence_entries.py` 与 `tests/test_stance_and_kick_rules.py` 的新标准子串断言，并修补 `tests/test_timing_reference_selection.py` 中 `capture` 桩未透传 `**kwargs` 致 `step_combo_windows(..., action=...)` 报错的回归问题；确认 `pose33_v3/golden.json` 与 14 份 HTML golden 零漂移。
- 验证：WSL `./.venv/wsl/bin/python` 运行 `square check .`、`ruff format --check --line-length 100`、`ruff check`、`git -c core.whitespace=cr-at-eol diff --check` 与 `test_timing_reference_selection.py` 及 26 套门禁/规则/黄金基线/报告/考试/开关 pytest 全部通过。

## 2026-11-22: [feat] 实施实战式与低鞭腿 6 项既有规则优化（v5.21）

- 问题：客户与用户复核旧规则后锁定 6 项优化：实战式/拳法膝角 175° 略严、低鞭腿护手开肘 30° 过严、低鞭腿支撑脚碾脚实为内旋（脚跟转至领先于脚尖）而非外旋、低鞭腿 `finish` 锚点帧未对准侧面腿蹬出最远帧且 `arm_swing` 不应在 `motion` 误报、低鞭腿踢高上限应以支撑腿髋（异侧髋）为准而非同侧髋、实战式 `stance_toes` 应改为侧面只查后脚尖是否落后于后脚跟（不查前脚）。
- 修改：
  1. **实战式与公共护手阈值优化**：`rule_descriptions.py` 将 `PARAMS["stance_knee_max"]` 调至 `179.0`（统一作用于 `stance:start`/`stance:end` 及全部拳法复用段）；`rule_descriptions.py`、`feedback_config.py` 与 `measures/guard.py` 将 `kick_guard_elbow_flare` 调至 `45.0`（前/后低鞭腿统一共用）。
  2. **足部前后投影公共原语共用（优化 ③ + ⑥）**：在 `core/evaluation/measures/feet.py` 抽离共享原语 `foot_heel_toe_forward_projection` 并清理废弃的 `measure_toe_spread`/`foot_vectors`/`vector_lengths`/`vector_angle_deg`；`measure_pivot_heel` 改为内旋判定 `proj = (heel_x - toe_x) * dir_x / safe_t <= 0.0`（指标名 `"支撑脚跟相对脚尖前向投影"`，文案改为 `"支撑脚未碾脚内旋"`）；`stance_rules.py` 与 `rule_definitions.py` 将 `stance_toes` 改为侧面 `rear_foot_toe_projection`（`PRIMARY_VIEW["stance_toes"] = "side"`、`required = (rear_heel, rear_toe)`、`resolve_window_facing` 缺失走 `unable`，`toe_proj < 0.0` 判错，保留 `id` 不变兼容旧开关）。
  3. **低鞭腿 `finish` 锚点收紧、甩手收口与支撑腿髋上限（优化 ④ + ⑤）**：`kick_events.py` 结合 `kicking_leg_side_safe` 与抬腿区间按侧面脚尖相对踢腿髋前向伸展（缺脚尖退踝、无朝向退欧氏距离）定位腿蹬出最远帧 `ext_idx`/`t_ext`；`kick_phases.py` 将 `finish` 收紧至 `t_ext ± 0.06s` 并按 `min_frames` 在事件内扩足；`segment_blueprints.py` 将 `arm_swing` 收口为仅在 `finish` 检查（目录总数 `151 → 149`），同步清理 `issue_groups.py`；`kick_rules.py` 与 `measures/kicks.py` 将 `kick_height` 上限改为支撑腿髋 `sup_hip`（指标名 `"脚尖相对支撑腿髋高度"`）；升级 `sanda_v5.py` `RULE_VERSION = "sanda-rules-mediapipe-2026-11-22-v5.21"`。
  4. **Golden 与单测更新**：经逐行核对 `tests/fixtures/pose33_v3/golden.json` 仅 `stance:start/end::stance_toes` 从 `front_rule_checks` 迁至 `side_rule_checks`（其余分数/切分/检查项零漂移）后按规程刷新 `golden.json`；新建 `tests/test_stance_and_kick_rules.py` 覆盖 6 项优化正反等价类、左右式对称与缺帧/缺朝向/互换/低帧率边界，同步更新受影响既有单测。
- 验证：WSL `./.venv/wsl/bin/python` 运行 `square check .`、`ruff format --check --line-length 100`、`ruff check`、`git -c core.whitespace=cr-at-eol diff --check` 与全量受影响/门禁 pytest 全部通过。

## 2026-11-22: [feat] 落地摆拳独有规则、上步前后手摆拳组合与公共步法/姿态/衔接全量规则

- 问题：系统尚缺 `step_hook_combo`（上步前后手摆拳组合）动作编排与逐次切分；摆拳事件仍走旧 `action_events` 脆弱兜底（基线缺帧即失效、不支持 `held` 保持且未精修打到瞬间峰值）并缺少 C61–C67 独有规则；同时公共拳法（C40 峰值后未收回、C47 组合拳衔接停顿、C48 后直拳前手护颚）、公共姿态发力（C44/C45 躯干前后倾、拳法低头/屈膝/分视角转胯/提踵蹬转）与公共步法（C22 双摄侧身并脚、C23 未跟步、C24 后脚超前脚、C25 落脚滞后）尚未跨直拳/摆拳/垫步统一接通。经本地实拍核验，C38 贴下颚在正面 2D X 轴投影 100% 落入 `[肩线中点, 前侧肩]`，按约定本轮不做 C38。
- 修改：
  1. **Layer 1 底座测量与摆拳/步法/衔接时间轴**：新增 `core/evaluation/measures/hooks.py`（`measure_hook_arm_bend`、`measure_hook_forearm_level`、`measure_hook_over_center`）；在 `measures/feet.py` 抽离公共 `FootProgression`、`foot_forward_progression`、`foot_settle_index`、`measure_foot_forward_displacement`、`measure_rear_foot_overtake`（`tail_median`/`peak_median`）、`measure_step_feet_together`、`measure_step_landing_lag` 与 `measure_punch_heel_pivot`，精简 `measures/kicks.py` 删除重复垫步公式；扩展 `measures/posture.py`（`measure_trunk_lean_sagittal` 及视角可选标签的 `measure_hip_turn_width`）与 `measures/motion.py`（`measure_punch_post_peak_hold`）；新建 `core/evaluation/hook_events.py`（缺帧鲁棒基线、`allow_held=True` 与正/侧打到瞬间峰值精修）并在 `action_events.py` / `action_timing.py` 全面接通；扩展 `action_catalog.py`（新增 12 类动作 `step_hook_combo`、派生段 `combo_transition`/`step_footwork` 与 `motion_segments`）、`punch_events.py`（`baseline_facing` 与支持边收边出的 `preceding_punch(..., rear_peak=...)`）、`event_phases.py`（共享 `combo_transition_windows`）、`step_reps.py`（共享直拳/摆拳逐次切分与 `step_footwork_windows`）、`phases.py`、`scoring/exam_pool.py`（`side_combo_repetitions` 接入 `step_hook_combo`，保持 `side_repetitions` 2 参兼容），并在 `rules/spec.py`、`rules/window_context.py`、`rule_checks.py`、`practice.py` 透传 `event_details`/`combo_pairs`/`repetitions`。
  2. **Layer 2 公共/私有规则、全量蓝图与报告适配**：新建公共步法模块 `core/evaluation/rules/step_rules.py`（4 项策略，`pad_step` 与 `step_footwork` 共享推进与越前底层）；扩展公共 `punch_rules.py`（迁入 `wrist_shoulder_height_gap`，新增 `punch_post_peak_recovery` 与 `combo_punch_pause_gap`）与 `body_posture_rules.py`（新增 `trunk_sagittal_tilt_angle` 与 `punch_heel_lift_pivot`，`frontal_hip_width_ratio` 支持前手正面/后手侧面零复制复用）；重写 `hook_rules.py` 仅保留 3 项摆拳独有策略并删除 `hook_arm_shoulder_level`，精简 `straight_rules.py`（3 项）与 `kick_rules.py`（5 项）；在 `feedback_config.py` 新增 12 个有限正数阈值（旧 `feedback_config.json` 与 4 族权重向前兼容）；清理 `rule_descriptions.py` 未使用的 `FORMAL` 常量并补齐公共/私有元数据；在 `rule_definitions.py` 泛化双向 `paired` 证据点位；在 `segment_blueprints.py` 抽离 `_punch_posture_blueprints` 与 `_hook_private_blueprints`，为拳法复用项提供拳法专属文案而不改坏低鞭腿与 `stance` 8 项（`golden.json` 零漂移），为 `rear_straight` 补齐 `guard_jaw` 并统一通用文案；升级 `sanda_v5.py` `RULE_VERSION = "sanda-rules-mediapipe-2026-11-22-v5.20"`（12 动作 / 151 唯一检查项 / 11 UI 分组）；扩展 `core/reporting/issue_groups.py` 合并全部攻击段过程姿态提醒，`practice_model.py` 改用 `SEGMENT_LABELS` 展示派生段中文名；同步更新 `AGENTS.md` 代码地图。
  3. **Layer 3 单测闭环**：新建 `tests/test_hook_and_step_rules.py` 覆盖摆拳独有规则、事件/峰值精修、物理方向纠偏（前/后手转胯与出拳提踵）、C40/C47/C48、C22–C25（含非侧身帧过滤与三态 `unable` 边界）及 `step_hook_combo` 练习/报告/考试池端到端闭环；同步更新 `test_rule_reuse.py`、`test_practice_check_toggles.py`、`test_tk_practice_checks.py`、`test_step_straight_combo.py`、`test_independent_event_phases.py`、`test_low_kick_rules.py` 的目录计数与共享测量导入断言。
- 验证：WSL `./.venv/wsl/bin/python` 运行 `square check .`、`ruff format --check --line-length 100`、`ruff check`、`git -c core.whitespace=cr-at-eol diff --check`，以及 `test_pose33_v3_golden.py`（零漂移）、`test_valid_mask_migration.py`、`test_tech_eval_contract.py`、`test_rule_availability.py`、`test_architecture.py`、`test_rule_engine_entry.py`、`test_rule_reuse.py`、`test_hook_and_step_rules.py`、`test_step_straight_combo.py`、`test_low_kick_rules.py`、`test_straight_refinements.py`、`test_practice_check_toggles.py`、`test_independent_event_phases.py`、`test_tk_practice_checks.py`、`test_action_feedback.py -k "not real_tk"` 与受影响报告/考试单测全部通过。

## 2026-11-21: [refactor] 统一收拢散打评估公共检查项并严格隔离动作独有私有项

- 问题：原 `core/evaluation/measures/` 与 `core/evaluation/rules/` 存在三套割裂实现（`stance + *_straight`、`KICK_SEGMENTS`、`*_hook` 旧分支），`measures/kicks.py` 复制了护手掉肩、护手开肘、护手护颚及低头/侧倾/翻胯/碾脚等非踢腿独有测量，`rule_definitions.py` 与 `sanda_v5.py` 通过动作家族 `if/elif` 与 `standard_config` 硬分叉；摆拳停留在旧分支导致护手误落入 `FeedbackConfig` 默认值、漏挂前护手后拉且在后摆拳出拳手上误挂 4 条护手检查项。
- 修改：
  1. **收拢公共测量层**（`core/evaluation/measures/`）：新建 `measures/guard.py` 作为公共护手几何 SSOT（掉肩、护颚、抬肘、正面开肘、正/侧视肘退后）；将通用姿态与脚部测量（`measure_head_down`、`measure_body_lean`、`measure_hip_turn_width`、`measure_toe_spread`、`measure_pivot_heel`）归位至 `measures/posture.py` 与 `measures/feet.py`；物理删除 `measures/kicks.py` 中 7 个非踢腿独有/重复函数，保持 `rules/` 零 `.norm`/`.arccos`。
  2. **重构公共规则层与极简动作私有层**（`core/evaluation/rules/`）：新建 `window_context.py` 与 4 个公共规则域模块（`stance_rules.py`、`guard_rules.py` 含唯一视角规律 `guard_view`、`punch_rules.py`、`body_posture_rules.py` 支持可选 `mask_filter` 注入），以及 3 个仅含动作独有项的极简私有模块（`straight_rules.py`、`hook_rules.py`、`kick_rules.py`）；建立声明式 `STRATEGY_REGISTRY`（`rule_definitions.py`）与 `SEGMENT_CHECK_BLUEPRINTS`（`segment_blueprints.py`），按公共域/私有域重组纯数据 `rule_descriptions.py`，由 `sanda_v5.py` 全量蓝图驱动并升级 `RULE_VERSION` 为 `sanda-rules-mediapipe-2026-11-21-v5.19`；物理删除 7 个旧杂糅/错名文件（`straight_feet.py`、`straight_posture.py`、`straight_motion.py`、`straight_measures.py`、`catalog_measurements.py`、`kick_measures.py`、`kick_step.py`）；保留 `feedback_config.py` 全部字段并在 `check_front_weights` 白名单保留旧键兼容（`guard_low`/`guard_recover`/`stance_head`/`vertical_fist`）；同步收紧 `.harness-baseline.json` 消减 `rules/` 5 处存量 `nested_ternary`（`entry_ceiling` 63→58）。
  3. **摆拳接入公共项与零漂移红线**：`stance`、`front_straight`、`rear_straight`、`pad_step`、`front_low_kick`、`rear_low_kick` 保持全部已确认/标定阈值与数值零漂移（`golden.json` 零改动直接通过）；`front_hook` 接入公共护手（`guard_shoulder` 0.08、`guard_back` 0.18、`guard_elbow` -0.18，走 `guard_view`）并保留独有 `hook_arm_shoulder_level`（0.20 + `any`）；`rear_hook` 接入公共预摆（`punch_predrop` 0.15、`punch_prepull` 0.22）、公共护手（含补齐前手 `guard_back` 0.18 与 `guard_jaw` 0.34 `front`）、独有 `shoulder_level`（0.20 + `any`）及公共收回（`punch_recover` 0.34 `front`），并删除出拳右手上误挂的 4 条旧项（目录总数保持 99，`front_hook`=24、`rear_hook`=28、`hook_combo`=36）。
  4. **下游与契约测试同步**：更新 `core/evaluation/punch_phases.py` docstring 与 `core/reporting/issue_groups.py`（摆拳公共护手 `motion`/`finish` 合并出卡）；更新 `tests/test_low_kick_rules.py`、`tests/test_straight_refinements.py`、`tests/test_step_straight_combo.py`、`tests/test_rule_reuse.py` 导入与摆拳新公共项断言，并在 `tests/test_rule_reuse.py` 新增公共/私有边界与跨动作复用结构契约测试。
- 验证：WSL `./.venv/wsl/bin/python` 运行 CI 硬门禁（`test_pose33_v3_golden.py`、`test_valid_mask_migration.py`、`test_tech_eval_contract.py`、`test_rule_availability.py`、`test_architecture.py`）63 passed、`square check .` 通过（基线 88→83）；12 个规则/测量直接回归套件 145 passed；开关/报告/反馈回归套件零新增失败；23 个改动/新增 `.py` 文件通过 `ruff format --check --line-length 100` 与 `ruff check`；`git -c core.whitespace=cr-at-eol diff --check` 零空白错误。

## 2026-11-21: [docs] 将工程与沟通两份核心契约设为强制必读

- 问题：`AGENTS.md` 规范索引中仅将 `specs/engineering_contract.md` 与 `specs/communication_contract.md` 标为“常驻遵循契约”，未明确开工前强制必读要求。
- 修改：更新 `AGENTS.md` 规范文档索引，将两份常驻契约改为“强制必读契约”，要求开始任何任务、需求分析、沟通、编码与写测前先完整阅读并严格遵循（同步依据：用户 2026-11-21 明确指令）。
- 验证：`git -c core.whitespace=cr-at-eol diff --check` 零空白错误，`square check .` 静态门禁通过。

## 2026-11-20: [feat] 实施学生练习检查项全局统一开关（核心、报告与 Tk 桌面端）

- 问题：学生练习检查项此前默认全开且无全局开关，无法关闭不想检查的项目；同时 `PrefKey` 哈希依赖 `default`/`parse`/`dump`/`fallback` 字段，偏好写入失败静默掩盖保存错误，练习视频、序列、桌面入口缺少共用的前置筛选、全关校验及锁定态/退出时的弹窗收口保护。
- 修改：
  1. **L0 偏好存储**（`core/foundation/prefs.py`、`core/foundation/__init__.py`）：`PrefKey` 的 `default`、`parse`、`dump`、`fallback` 设为 `field(..., compare=False)` 使哈希与相等性仅依赖 `name`；`PreferenceStore.update/set` 新增 `strict: bool = False`（默认维持尽力而为语义不变）；新增 `PRACTICE_DISABLED_CHECKS`（`practice_disabled_checks`）与 `_parse_disabled_checks`，导出 `load_practice_disabled_checks` 与 `save_practice_disabled_checks`（`strict=True`，写盘失败抛出 `OSError`）全局保序去重存储（不按动作区分、不设 `scope_ids`）。
  2. **L3 全量目录与评估源头过滤**（`core/evaluation/rules/sanda_v5.py`、`core/evaluation/__init__.py`、`core/evaluation/practice.py`）：新增并导出 `catalog_checks(stance=DEFAULT_STANCE)` 收集并排序全系统 99 个唯一检查项；新增并导出共用 `select_practice_checks()`，在 `evaluate_series()` 最开头及 `analyze_practice()` 视频校验与推理前拦截当前动作全关（抛 `ValueError("当前动作至少需要保留 1 个开启的检查项")`），自动忽略其他动作段及过期旧 ID；保留完整 `synchronized_windows()` / `shared_action_record()` / `_repetitions()`，过滤 `RuleProfile` 并按需跳过 `paired_torso_measurements()`，仅存在跳过项时写入 `disabledCheckIds` 与 `totalCatalogChecks`（默认全开不写新键）。
  3. **L4 报告只读说明**（`core/reporting/saved_results.py`、`core/reporting/practice_model.py`、`core/reporting/text.py`）：抽取共用 `skipped_checks_note()`，严格只读记录自带的 `disabledCheckIds` 与 `totalCatalogChecks`，在 `build_report_model()` 副标题与 `format_report()` 摘要下方追加跳过与启用比例说明（默认全开零新增文字）。
  4. **Tk 桌面控制器、设置弹窗与入口护栏**（新建 `apps/tk/controllers/practice_checks.py`、`apps/tk/views/practice_checks_dialog.py`；修改 `apps/tk/views/__init__.py`、`apps/tk/controllers/practice_analysis.py`、`apps/tk/views/practice_controls.py`、`apps/tk/controllers/practice.py`、`apps/tk/dispatcher.py`、`apps/tk/shutdown.py`）：提供 9 组 99 项全局统一设置弹窗（含 `_set_vars` 批量切换、保存时动作 0 开启项非阻断提示，以及写盘 `OSError` 弹错保窗保留勾选）；`PracticeControls` 实现按钮计数联动、`_close_checks_dialog()`/`close_dialogs()`、`set_recording()` 锁定态自动关窗、`open_checks_dialog()` 禁点守卫、`_analyzer()` 同步 `set_status` 与 `show_result`；主线程 `PracticeAnalysis.analyzer()` 快照透传并支持 `action` 校验；`PracticeController.start()`、`start_practice_analysis()`（含 `ready_to_judge` 失败回退与 `showwarning`）、`exit()`（收口关闭双弹窗）及 `PracticeControls.import_video()`/`reanalyze()` 前置复用 `select_practice_checks` 拦截动作全关；`dispatcher.py`/`shutdown.py` 补全无 `tkinter` 环境 `TclError` 导入回退。
  5. **单测**（`tests/test_foundation_prefs.py`、新建 `tests/test_practice_check_toggles.py`、新建 `tests/test_tk_practice_checks.py`）：覆盖 `PrefKey` 哈希、偏好往返与写盘 `OSError` 抛出、99 项目录、跨动作复用、`select_practice_checks` 与双入口拦截、`phaseWindows` 保留、历史报告锁定，以及 Tk 控制器、弹窗、锁定关窗、`ready_to_judge` 拦截和 `exit()` 收口测试。
- 验证：
  - WSL `./.venv/wsl/bin/python`：`ruff format --check`（19 个修改/新增 `.py` 文件：`19 files already formatted`）、`ruff check`（上述 19 个 `.py` 文件：`All checks passed!`）、`git -c core.whitespace=cr-at-eol diff --check origin/main`（0 报错）；`pytest tests/test_architecture.py tests/test_pose33_v3_golden.py tests/test_valid_mask_migration.py tests/test_tech_eval_contract.py tests/test_rule_availability.py tests/test_foundation_prefs.py tests/test_practice_check_toggles.py tests/test_feedback_report.py tests/test_feedback_html_export.py tests/test_report_html_golden.py tests/test_evidence_budget.py -q`（`144 passed`）。
  - Windows `.venv/Scripts/python.exe`：`pytest tests/test_tk_practice_checks.py tests/test_action_feedback.py tests/test_feedback_result_panel.py -v`（`82 passed`，`0 failed, 0 skipped`；含 `test_tk_practice_checks.py` 11 项、`test_action_feedback.py` 52 项、`test_feedback_result_panel.py` 19 项真实 Tk 用例全部通过）。

## 2026-11-18: [fix] 修复 PR #67 审查指出的摆拳预摆阶段、肘点校验与旧记录横幅等问题

- 问题：PR #67 审查（`gy212` 3 条 P1 与 CodeRabbit 意见）指出：1) `rear_hook` 及 `hook_combo` 的 `guard_back`（`role="rear"`）错放在 `motion`/`finish`，且 `event_phases` 未给摆拳生成 `prepare`/`recovered` 窗口；2) 摆拳 `shoulder_level` 仅检查肩腕两点，缺失肘关键点仍判通过；3) 旧历史 `pending_rule` 记录（`needsRerecord=False`）全转 `unable` 后横幅硬编码 `needs_rerecord=True` 并提示重录与遮挡；4) `kick_measures._measure_baseline_relative_rule` 缺少 `stance:start` 守卫、`engine.py` 多段 `absence` 存在帧不足次时可能误判 `pass`、`text.py`/`timeline.py`/`practice_history.py`/`self_check.py` 存在缺省字段与 0 项时间轴绿勾等边界问题。
- 修改：
  1. `core/evaluation/event_phases.py` 与 `core/evaluation/rules/{sanda_v5,catalog_measurements,rule_definitions}.py`：`_hook` 动作同步生成 `prepare`/`recovered` 相位窗；将 `rear_hook` 中后侧 `guard_back` 移至 `prepare` 并补齐名称/标准文案；新增 `hook_arm_shoulder_level` 策略，摆拳 `shoulder_level` 要求 `(shoulder, elbow, wrist)` 三点齐全且同查拳肘平肩；
  2. `core/evaluation/rules/{kick_measures,engine}.py`：低鞭腿基线相对测量缺失 `stance:start` 时记录问题并返回空掩码；多段 `absence` 若未确认违规且存在有效帧不足次则返回 `unable`（抽出 `_preflight_unable` 与 `_insufficient_outcome` 保持函数精简）；
  3. `core/reporting/{practice_model,text,html/timeline}.py`、`apps/tk/{self_check,views/practice_history}.py`、`AGENTS.md` 与 `docs/`：旧版 `unable`/`pending_rule` 在 `needsRerecord=False` 时保留 `needs_rerecord=False` 与客观文案；补全 `text._check_labels`、0 项阶段中性标记 `—`、`check_detail_text`/`EvidenceWindow` 容错及 `self_check` 非空断言；补充专项回归测试并重生成 `report_html` 与 `result_page` 基准。
- 验证：WSL 通过 `ruff format --check`、`ruff check` 与 19 个核心/回归套件；Windows `.venv` 通过含 Tk 的 4 个集成与快照套件；`git -c core.whitespace=cr-at-eol diff --check origin/main` 零报错。

## 2026-11-18: [refactor] 废除未自动检查与样本标定门禁并全量启用检查项

- 问题：此前系统在 `AGENTS.md` 硬约束与 `qualification.py` / `straight_policy.py` / `kick_policy.py` / `availability.py` 中设立了“未自动检查（`pending_rule`）”“试用（`trial`）”及基于未确认 100% 标准的视频样本误报率/检出率上线门禁，导致用户已给出完整判定规则并完成底层实现的检查项（如 `12_后低鞭腿_练习分析报告` 中的 17 项检查）被门禁层短路拦截为“未自动检查”，且报告与 Tk 桌面端存在第四类未自动检查展示与“（试用）”前缀。
- 修改：
  1. **规则与评估层**：物理删除 `core/evaluation/rules/{qualification,straight_policy,kick_policy,availability}.py`；精简 `spec.py`、`engine.py`、`sanda_v5.py`（规则版本升级为 `sanda-rules-mediapipe-2026-11-18-v5.18`）、`adapters.py`、`practice.py` 与 `feedback_config.py`，清理未入表的死定义 `stance_head` / `vertical_fist` 及摆拳硬编码 `blocked` 拦截，保留全部 11 个动作共 100 个既有检查项并在 `lite` / `full` / `heavy` 全模型下直接正式启用，系统检查结论严格收敛为三态（练习：`candidate` / `not_observed` / `unable`；考试与 batch：`violation` / `pass` / `unable`）；
  2. **报告导出与 Tk 桌面层**：修改 `core/reporting/{practice_model,text}.py`、`core/reporting/html/{renderer,icons,timeline}.py`、`core/reporting/html/templates/{report,incomplete_section,incomplete_row}.html` 与 `apps/tk/{self_check,views/practice_history,views/practice_result/*}.py`、`scripts/generate_feedback_icons.py`，彻底移除“未自动检查”胶囊、横幅分级、筛选按钮与 6 张 `pending_rule` 图标资源，保留正交的教师人工复核字段 `review`（未发生撤销时界面与报告 100% 仅显示三类）；在 `practice_model._normalize_check`、`text._effective_status` 与 `practice_history.check_state_text` 兜底将存量历史记录中的旧 `pending_rule` 自动归入 `unable` 并清洗残留 `（试用）` 后缀，确保历史数据无损兼容且计数守恒；
  3. **规范与基准同步**：重写 `AGENTS.md` 硬约束为“动作规则直接生效与三态结果红线”（严禁私自拿视频素材当标准真值算合格率关停规则或改阈值），在 4 份历史标定 `docs/` 顶部加注废止提示，更新评估/报告/UI 测试套件并同步重生成 `tests/fixtures/pose33_v3/golden.json`（仅 11 条 `detail` 去掉 `试用检查。` 前缀，评分与指标零漂移）、14 份 `tests/fixtures/report_html/*.html` 及 `tests/fixtures/result_page/snapshots.json`。
- 验证：WSL 原生环境通过 `test_architecture.py`、`test_pose33_v3_golden.py`、`test_valid_mask_migration.py`、`test_tech_eval_contract.py`、`test_rule_availability.py` 及相关规则/报告单测；Windows `.venv` 通过含 Tk 的 4 个集成与结果页快照套件；改动文件 `ruff format --check`、`core`/`apps`/`scripts` `ruff check` 与 `git -c core.whitespace=cr-at-eol diff --check origin/main` 全部通过。

## 2026-11-15: [fix] 修复 PR #66 审查指出的低鞭腿切分、垫步尺度、逐次缺省判定与参数一致性问题

- 问题：PR #66 CodeRabbit 审查指出 6 项问题：1) 持续抬腿中途发生连续 6 帧及以上丢点再恢复高位时，滞回状态机未等回落即重开第二次区间；2) 相邻两次已各自闭合且满足峰值与时长的独立踢腿，因间隔 `<= 0.35s` 被误合并；3) 垫步位移先逐帧计算 `x / torso`，在后踝静止、躯干投影缩小时会伪造垫步推进；4) `absence=True`（如 `arm_swing`）在多次重复合并窗内对全帧求 `.all()`，导致其中一次未摆臂被另一次掩盖；5) `kick_step.py` 将标量布尔有效值广播到整段 `valid[sub]`，绕过引擎 `valid_ratio` 门禁；6) `standard_config` 未对踢腿动作段统一应用 `KICK_PARAMS`，且 `kick_policy.py` 未核对 `parameterOverrides` 是否被篡改。
- 修改：
  1. `core/evaluation/kick_events.py`：长缺帧结束旧区间后进入待回落等待（短缺口 `<= 0.35s` 续接同次，完整抬腿后的长丢点须见到回落才重新触发，且 `landed_after` 校验首个有限样本低落），新增 `_merge_raw_intervals` 保留各自闭合的独立完整踢腿；
  2. `core/evaluation/measures/kicks.py` / `core/evaluation/rules/kick_step.py` / `kick_measures.py`：`_rear_ankle_progression` 改为先平滑原始投影再按整窗躯干中位数统一归一化；`measure_pad_step_*` 与 `measure_kick_retract_segment` 全部返回逐帧有效掩码并直接写入 `valid[sub]`，不再标量广播，并为双踝同见易交叉的 `pad_step_ankle_overtake` 对齐侧视 `0.5` 有效比例覆盖；
  3. `core/evaluation/rules/engine.py` / `spec.py`：提取 `_instance_groups` 与 `_resolve_absence`，对存在 `phase_instances` 的 `absence` 规则改为按每次重复切片独立判全错，仅在违规次内标注问题帧；
  4. `core/evaluation/rules/rule_definitions.py` / `kick_policy.py`：`standard_config` 对低鞭腿与垫步动作段统一应用 `KICK_PARAMS`，并在 `_registered_phase_ok` 校验 `parameterOverrides == KICK_PARAMS`；
  5. `tests/test_low_kick_rules.py` 与 `docs/低鞭腿规则标定报告_20261114.md`：更新逐帧掩码断言、补充 6 项回归用例并同步 `hip_turn` 实测统计。
- 验证：重跑 256 组 `full`/`heavy` 缓存生产链路，47 上步前低鞭腿与 17 原地后低鞭腿逐次切分及 `exam_pool` 全对零错配，全部 42 实例模型定级零变化；CI 门禁、`test_low_kick_rules.py` 及相关规则/计时/反馈测试全通过；`ruff format --check`、`ruff check` 与 `git diff --check` 通过。

## 2026-11-14: [feat] 接入低鞭腿检查项、逐次切分与双模型标定

- 问题：原系统仅直拳组合接入完整规则检查与逐次切分，上步前低鞭腿、前低鞭腿与后低鞭腿缺少侧视滞回切分、垫步与踢腿几何规则（踢高、收腿、后/前侧护手、摆臂、低头、侧倾等），部分既有规则未纳入分模型资格门禁，且易受片头缺帧与左右腿互换影响。
- 修改：统一三类低鞭腿动作接入侧视 `side_kick` 动作阶段管线，`step_front_low_kick` 与 `rear_low_kick` 开启逐次切分、单独 `front_low_kick` 共用同款侧视相位不做拆段；下沉几何计算至 `core/evaluation/measures/kicks.py`（补齐 3 点摆臂可见性与双踝同见防互换掩码），实战式起止窗回归完整 `motion_in.stable` + `settled_stance_windows`；按 `full`/`heavy` 含 `unable` 总分母独立定级并阻断 `lite` 与未达标项（`full` 启用 14 项、`heavy` 启用 16 项，0 未授权 `trial`）；规则版本升级为 v5.17（`sanda-rules-mediapipe-2026-11-14-v5.17`）。
- 验证：256 组 `full`/`heavy` 实拍缓存（47 上步前低鞭腿、17 原地后低鞭腿）逐次切分与 `exam_pool` 全对零错配，全部 42 实例/模型状态与标定红线一致；CI 核心架构/契约门禁及低鞭腿、规则复用、动作反馈测试全通过（无 Tk 环境的 GUI 用例跳过），`ruff check`、`ruff format --check` 与 `git diff --check` 通过。
- 限制：踢高 (`kick_height`) 与后低鞭腿侧倾缺错例、后低鞭腿前护手掉肩 (`guard_shoulder`) 仅 1 人错例，暂保持 `validation_pending`；脚跟轴转 (`pivot_heel`)、绷脚 (`foot_flexed < 165°`)、未翻髋 (`hip_turn`)、护肘角度 (`guard_elbow > 30°`) 及后鞭腿护颌 (`guard_jaw`) 因单目二维投影混叠或高误报维持 `calibration_failed` 阻断；`lite` 模型未标定统一阻断。详见 `docs/低鞭腿规则标定报告_20261114.md`。

## 2026-11-08: [fix] 按减少误报优先调整公共实战式站距范围

- 问题：客户报告站距比值 1.21～1.25，被原 0.8～1.2 范围提示；客户先提 1.0～1.35，随后用户授权按样本选择并明确优先减少误报。
- 修改：最终试用范围为 0.70～1.35，所有动作的公共实战式检查统一生效；默认配置、报告标准和试用说明共用参数。补齐站距测量的公共参数应用，旧配置不再覆盖新标准。规则版本 v5.16。
- 验证：7 段 full 缓存比较候选范围，三段正确对照由两段提示变为零段提示；4 段标注错误素材中 3 段仍提示，但标签与实际取帧存在方向矛盾，不计作准确率达标。定向及核心门禁 102 项通过，golden 仅同步两条站距说明，格式与空白检查通过。
- 限制：一段标注错误素材与正确示范测值重叠，仍不能区分；继续试用。客户原片缺失，未改写原 HTML。详见 `docs/实战式站距调整_20261108.md`。

## 2026-11-08: [fix] 增加公共直拳伸肘容错

- 问题：客户确认正确的后手直拳，报告最大肘角为 168.10°，被原 177° 下限提示弯肘。
- 修改：公共直拳下限调整为 168°，保留 12° 余量；前后手、单拳和组合共用。默认参数、报告标准及试用说明同步，规则版本 v5.15，继续标注试用。
- 验证：定向与核心门禁 108 项通过；教师示范及 C51 两名执行者的 full 缓存验证通过。165° 候选会漏掉最大 167.01° 的已有弯肘，故最终采用 168°。格式与空白检查通过。
- 边界：客户只有 HTML，按其测量值推算新阈值不再提示，未重跑原片或改写原报告。详情见 `docs/直拳伸肘容错_20261108.md`。

## 2026-11-08: [docs] 统一强制默认使用简体中文（含 Commit 标题、Issue、PR 与审查）

- 问题描述：此前未明确禁止全英文撰写 Git Commit 标题，易出现中英文 Commit 标题混用。
- 修改内容：在 `AGENTS.md` §12 及 `spec/issue_spec.md`、`spec/pr_spec.md`、`spec/review_spec.md` 中明确规定：全仓库交互、注释、Commit 标题与正文、Issue、PR 及审查报告默认统一使用简体中文（保留必要的英文专业术语、代码符号与 Conventional Commits 类型前缀），严禁全英文撰写 Commit 标题。
- 验证方法：`git -c core.whitespace=cr-at-eol diff --cached --check` 无空白错误，全分支与工作树同步更新。

## 2026-11-08: [chore] 合并依赖清单、归拢打包 spec 至 packaging/ 并清理根目录遗留文件

- 问题描述：根目录遗留早期测试产生的 `user_prefs.json`、已废弃的 `requirements-spike.txt`、与 `requirements.txt` 重复的 `requirements-dev.txt`，以及散落在根目录的 `app_ui_onefile.spec`。
- 修改内容：
  1. 删除本地遗留 `user_prefs.json` 与废弃的 `requirements-spike.txt`；将 `requirements-dev.txt` 中的开发/打包依赖合并至 `requirements.txt` 并移除 `requirements-dev.txt`。
  2. `git mv app_ui_onefile.spec packaging/pyinstaller/app_ui_onefile.spec`，以 `ROOT` 锚定打包路径，并同步更新 `packaging/pyinstaller/build_info.py`、`tests/test_packaging_build_info.py` 与 `AGENTS.md`。
- 验证方法：`pytest tests/test_packaging_build_info.py tests/test_architecture.py -q` 通过，`scripts/check_format.py origin/main` 与 `git -c core.whitespace=cr-at-eol diff --cached --check` 通过。

## 2026-11-08: [docs] 建立 spec/ 专项规范体系并归档根目录历史设计文档

- 问题描述：根目录散落历史重构设计文档与归档文件；Issue、PR 与 Review 缺乏集中统一的专项规范目录及 `AGENTS.md` 自动触发路由。
- 修改内容：
  1. 新建 `spec/` 目录，集中维护 `spec/issue_spec.md`、`spec/pr_spec.md` 与 `spec/review_spec.md`（由 `DeepReview.md` 重命名移入），统一强化标题格式、“语言简洁易懂、内容完整不丢小节”及反锚定/破立交底模板。
  2. 将根目录散落的历史设计文档（`camera_session_architecture.md`、`摄像头架构设计交接说明.md`、`整体架构重构设计.md`、`重构阶段任务/`）及 `change（2026.7~2026.11）.md` 统一 `git mv` 归档至 `docs/archive/`；清理本地临时录制残留目录并在 `.gitignore` 忽略本地素材目录 `img/`、`vido/`。
  3. 更新 `AGENTS.md`：新增“专项任务规范（按需自动读取 `spec/`）”强制路由表与 §12 必读约束，并全量同步至所有本地分支与工作树。
- 验证方法：`git -c core.whitespace=cr-at-eol diff --cached --check` 无空白错误，`pytest tests/test_architecture.py -q` 通过。

## 2026-11-08: [docs] 优化 DeepReview.md 审查输出结构（内思外判分离与倒金字塔模板）

- 问题描述：首版 `DeepReview.md` 将内部反锚定推演过程直接作为外部报告结构，导致开头抽象理论过多、缺乏直观方案对比表、测试缺口与阻断项分散，人类阅读不够清晰直观。
- 修改内容：优化 `DeepReview.md` §4 为“内思外判分离（倒金字塔法则）”——顶部 3 行直接亮明最终裁决与一句话总结，第 1 节用表格直观对比“当前做法 vs 更优解法”，第 2 节按优先级将“代码位置、问题本质、复现对比、应该怎么改、对应测试缺口”五要素内聚展示，第 3 节给出极简复审验收标准，并严禁对无问题维度做填空式废话汇报。
- 验证方法：`git -c core.whitespace=cr-at-eol diff --cached --check` 通过，全分支与工作树同步更新。

## 2026-11-08: [docs] 补齐 PR #64 审查修复与实测说明

- 记录未配对阶段丢失、计时选源条件失配两个审查问题，以及原片核对发现的收势误识别、最终补修和验证边界。
- 验证：核心门禁及入口保存契约 77 项通过，逐次覆盖 9 项通过；两份真实缓存最终结果与报告已保存，错误中间结果保留作定位证据。变更记录统一补回 `new/change.md`，并随 PR 提交同名文件。

## 2026-11-08: [fix] 长时间丢点后不补造出拳

- 问题：真实 C35 素材连续丢点后，手已前伸才重新可见，公共识别把后续下放误计为新拳；此前这次误识别被组合配对掩盖。
- 修改：正侧面直拳均使用已有的归位后重启条件，超过丢点容差后先看到回到已知起始位置，再识别下一次出拳；短暂丢点容差、姿势阈值不变。
- 验证：5 个新增入口场景与既有相关测试共 57 项通过，误计排除且随后真正出拳保留。重放同两份真实缓存：教师仍 4 个事件、37 项未发现；C35 收势误事件消失，保留 2 个事件，24 项未发现、13 项疑似、0 项无法判断；两份判定及全部测量值均与补修前一致。

## 2026-11-08: [fix] 计时选源核对真实识别条件

- 问题：PR #64 新增的侧面方向判定需要起始鼻髋参照，但机位选择只看肩腕可见性；因此会选中无法执行识别的侧面，错过完整可用的正面。
- 修改：选源和直拳观测共用必要点及起始参照判定，先选可执行的机位，再识别一次；共享同一份原始护位窗口。缺失条件逐段保存，上步组合仍遵守侧面计时限制。
- 验证：5 项反例通过，覆盖鼻点全程或仅起始失效、单拳与普通组合换源、只识别一次以及上步机位限制。另两个审查场景贯通练习、公共规则、考试、batch，事件、阶段、结论和保存记录一致。

## 2026-11-08: [fix] 组合缺成员时保留独立动作检查

- 问题：PR #64 审查发现，上步组合仍由后手配对结果决定前手阶段；已识别的前手会因缺后手、后手参照失效或未配对而丢失检查。
- 修改：公共层先为全部基础事件生成身份和阶段，组合只引用真实事件并说明缺失成员；取消伪造前手阶段和评分起点。结束实战式以最后一个实际观测事件为边界，短结束窗口也保留已有证据。
- 验证：独立事件、组合计时及共享时序 56 项通过，覆盖仅前手、后手参照失效、组合前后额外前手、缺成员评分区间与短窗口。旧组合序号断言改为独立事件身份和组合引用契约。

## 2026-11-08: [docs] 记录公共检查接入的实测结果

- 新增 `docs/公共检查接入修复_20261108.md`，说明客户报告上下文、基础段复用范围、三个修复步骤、五组真实缓存前后结果及验证边界。
- 同步依据：本轮规则、计时、证据测试与 `outputs/shared-rule-consumers/validation/` 实际报告。客户没有提供原片，明确不声称核实客户的峰值；剩余无法判断不计为通过。
- 验证：核对已保存的前后结果、两份单拳四入口记录、实际报告原帧，以及格式和空白检查。

## 2026-11-08: [fix] 无法判断时展示实际取帧和失败原因

- 问题：实际已经取帧，报告却只显示“有效帧不足”，已测数值和画面丢失，无法判断是遮挡、参照不足还是阶段缺失。
- 修改：规则引擎保留无法判断时的真实窗口代表帧和已有数值，注明具体关键点、可用数量或参照失败。练习、考试、batch 共用证据；HTML 可展开查看，Tk 显示测量摘要。图片预算优先覆盖各阶段，结束诊断图不再被重复站姿图挤掉。
- 验证：证据与呈现 51 项通过、1 项因 Linux Tk 环境跳过；报告导出 27 项通过，Windows 结果页 18 项通过。核心门禁及公共入口、保存契约合计 75 项通过；golden 只更新本次有意改变的两路实战式规则结果，模板、评分、关节误差和技术指标保持原基准。

## 2026-11-08: [fix] 单拳与组合共用直拳事件和阶段

- 问题：单拳没有消费已有的出拳前、收回后和稳定站姿能力；把侧面前伸算法直接用于正面会漏掉已有动作。
- 修改：按实际机位选择可观测位移，直拳事件、预摆和收回阶段共用同一入口；组合仅负责配对。普通组合顺序不齐仍保留独立动作段，所有动作共用稳定实战式选帧。
- 验证：计时和逐次覆盖 60 项通过；相关练习计时 4 项通过。真实缓存中单后拳 1 次、普通组合 5 个事件均保留，教师正确示范仍 37 项未发现问题。客户原片缺失，不声称已核实客户的具体峰值。

## 2026-11-08: [fix] 所有适用动作接入公共检查

- 问题：单拳和普通直拳组合仍走旧标准，部分项目要求两面都有效，已有公共检查没有实际接通。
- 修改：按实战式、基础直拳段绑定唯一测量与机位；验证记录按段、阶段、方法和模型复用。接入出拳前、收回后检查，淘汰六个出拳侧占位项及无消费方的旧公式。参照保留有效帧，单帧失效不再连坐整段；同段过程与结束提醒共用合并逻辑。
- 验证：规则定向 74 项通过；公共入口行为用例 6 项通过。替换了“新标准只允许上步组合使用”的旧断言，改验相同基础段跨动作、跨入口保持一致。

## 2026-11-08: [chore] change.md 归档与重建

- 问题描述：`change.md` 累积至 2936 行（覆盖 2026-07-09 ~ 2026-11-08），按变更记录规范需归档并重建新文件。
- 修改内容：
  1. `git mv change.md "change（2026.7~2026.11）.md"`，新建 `change.md` 并以本条作为首条记录。
  2. 同步更新 `AGENTS.md` 归档清单：新增本仓库 `change（2026.7~2026.11）.md`。
- 验证方法：`git status` 与 `git -c core.whitespace=cr-at-eol diff --cached --check` 检查通过，`pytest tests/test_architecture.py -q` 通过。
