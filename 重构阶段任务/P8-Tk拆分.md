# P8 Tk 拆分

> 依据：设计文档 §9.1、§9.3、§3 第 7、39 项。前置：P7。规模：大，阶段内分多次提交。状态：进行中（2026-10-13 开工，分支 `refactor/p8-tk-split`）。

## 1. 任务前检查

- [ ] 通用检查（见 README）。
- [ ] D-13 已决，前几个阶段的测试替换关系都有记录。
- [ ] 重新核对（2026-10-03 统计，P4–P7 之后会变小）：
  - `App`（`apps/app_ui.py:1106`）4855 行、135 个方法、162 个实例属性；`_build_ui` 398 行；`CompareWindow`（:263，511 行）、`SettingsWindow`（:853，251 行）。
  - 轮询循环 3 个：App 的 `_tick`、`ExamPanel._panel_tick`（`apps/exam_panel.py:803`）、`FeedbackControls`（`apps/feedback_panel.py`）的轮询。
  - `.after(` 共 29 处、4 个文件：app_ui 22、exam_panel 3、feedback_panel 2、desktop_smoke 2。逐个标出其中从后台线程调用的 `after(0, …)`。
  - 关窗（`apps/app_ui.py`）：App 的 `_on_close`（:4390，115 行）与 `_poll_close_workers`（:4506，90 行）。
  - 结果页与报告：`FeedbackResultPanel`（`apps/feedback_result_panel.py:70`，1257 行）；`core/feedback_report.py` 的 `build_report_model`（280 行）、`render_html_report`（517 行）。
  - 练习分析的迟到结果过滤：`_active_feedback_token`。
  - 半成品 `App` 构造（`App.__new__`、`SimpleNamespace(`）：`tests/test_app_ui_lifecycle.py` 32 处、`tests/test_app_ui_dual_camera.py` 16 处。
- [ ] 按设计文档附录 A 的 Tk 能力清单（14 项）在当前版本上走查一遍并记录，作为拆分后的对照。

## 2. 待决策事项

| 编号 | 问题 | 建议 | 未决时 |
|---|---|---|---|
| D-13 | 测试改造范围（P1 已决则跳过） | 接受 | 48 处半成品构造删不掉，`App` 类也就删不掉 |

## 3. 具体任务

1. `apps/tk/dispatcher.py` 的 `UiDispatcher`：后台线程只调用 `dispatcher.post(callback, token=None)`；Tk 主线程每 16 ms 排空一次；token 过期或窗口正在关闭时丢弃回调（关闭协调器自己的除外）。取代 3 个轮询循环和所有跨线程的 `after(0, …)`。
2. `apps/tk/shutdown.py` 的 `ShutdownCoordinator`：按 §9.1.3 的 7 步关窗（工作流准备关闭 → 停止会话 → 录制器关闭会话 → finalizer `cancel_all` 并发布取消终态 → `CameraService.close()` → `ScorebookRegistry` 落盘 → 轮询 `TaskGroup.pending()` 到 `close_deadline()` 后 `root.destroy()`）。窗口销毁后拒绝一切界面回调。
3. 控制器 `apps/tk/controllers/`：session、recording、exam、practice、compare、settings，用假服务单测。练习控制器持有分析任务的 `Epoch`，取代 `_active_feedback_token`。
4. 视图 `apps/tk/views/`：main_window、capture_panel、preview_surface、recording_panel、exam_window、practice_controls、practice_result/、compare_dialog、settings_dialog、widgets。`_build_ui` 按界面区域拆分。
5. 结果页 `views/practice_result/`：`FeedbackResultPanel` 拆成头部、概览卡片、相位条、检查列表、证据区、诊断区、底部操作栏，只消费报告模型。
6. 报告：`reporting/practice_model.py`（原 `build_report_model`、标题摘要、原因分组、相位时间线数据）；`reporting/html/`（HTML 模板文件 + 小型渲染器，取代 `render_html_report`；证据帧经 `media.video_source`；路径脱敏函数保留）。
7. 组合根 `apps/tk/app.py`：创建 `PreferenceStore`、`TaskGroup`、`UiDispatcher`、`CameraService`、`RecordingService`、`SegmentFinalizer`、`LiveSessionManager`、`ModeManager`、`WorkflowHost`、`ScorebookRegistry` 等对象并连线，不含业务逻辑。`apps/app_ui.py` 留入口壳；`apps/desktop_launcher.py` 改为启动 `apps/tk/app.py`；`apps/desktop_smoke.py` → `apps/tk/self_check.py`。
8. 测试：控制器与服务用假服务、假设备单测；真实 Tk 主循环只保留少量冒烟（沿用"真实 spawn 子进程 + 模拟驱动"的模式切换检查）；改写 48 处半成品构造，保留行为断言，删除对私有属性的断言。

删除（本阶段完成条件）：

- `App` 类；
- `_tick`、`_panel_tick` 和 `FeedbackControls` 的轮询；
- 所有跨线程的 `after(0, …)`；
- `render_html_report` 中的内联 HTML。

## 4. 验收标准

- [ ] 通用验收（见 README）。
- [ ] 体量：`apps/tk/app.py` 不超过 300 行，每个控制器不超过 400 行，每个视图不超过 600 行，函数不超过 100 行（allowlist 中的旧项除外）。这几项预算写进架构测试长期生效，不只在本阶段验收时检查一次。
- [ ] 架构测试：`core/` 不导入 tkinter 或 `PIL.ImageTk`；`.after(` 只出现在 `apps/tk/`；`apps/tk` 中 `getattr(self, "_…")` 只减不增。
- [ ] 控制器单测、真实 Tk 冒烟、源码模式的模式切换检查、EXE 自检全部通过。
- [ ] 附录 A 的 Tk 能力清单逐项走查，与第 1 节的记录一致。
- [ ] 同一份分析结果，新旧 HTML 报告和结果页的内容一致（文字、证据帧、相位条）。

## 5. 分批计划与进度（断点续做用）

用户决定（2026-10-13）：一个分支一个 PR；每批提交时全量测试通过，提交后立即推送分支。接手时先读本节，再看 `git log main..HEAD`。

开工基线：全量 1197 passed、1 skipped；allowlist 49 条。

| 批 | 内容 | 状态 |
|---|---|---|
| B1 | 报告：`reporting/practice_model.py`、`reporting/evidence.py`、`reporting/html/`（模板 + 渲染器 + 时间线 SVG + 图标 + 脱敏）；`core/feedback_report.py` 改门面；HTML 差分基准 `tests/fixtures/report_html/`（旧实现录制，14 例逐字节一致）；spec 收入模板目录 | 已完成 |
| B2 | 结果页拆成 `apps/tk/views/practice_result/`（头部、概览卡片、相位条、检查列表、证据区、诊断区、底部操作栏），`apps/feedback_result_panel.py` 改门面 | 已完成 |
| B3 | `apps/tk/dispatcher.py`（`UiDispatcher`）与 `apps/tk/shutdown.py`（`ShutdownCoordinator`）及单测；App、ExamPanel、FeedbackControls、CompareWindow、SettingsWindow 的轮询与跨线程 `after(0)` 改走 dispatcher | 已完成 |
| B4 | `CompareWindow` → `views/compare_dialog.py` + `controllers/compare.py`；`SettingsWindow` → `views/settings_dialog.py` + `controllers/settings.py`；`CollapsibleSection` 等 → `views/widgets.py` | 已完成 |
| B5 | `ExamPanel` → `views/exam_window.py` + `controllers/exam.py`；`FeedbackControls` → `views/practice_controls.py` + `controllers/practice.py`（持有分析任务 `Epoch`） | 已完成 |
| B6 | `App` 拆分：`views/main_window.py`、`capture_panel.py`、`preview_surface.py`、`recording_panel.py`；`controllers/session.py`、`recording.py`；组合根 `apps/tk/app.py`；`apps/app_ui.py` 留入口壳；删除 `App` 类；改写 48 处半成品构造 | 已完成 |
| B7 | `apps/desktop_smoke.py` → `apps/tk/self_check.py`（拆 `run`，并加 HTML 图文报告导出检查，验证模板随包分发）；`desktop_launcher` 改启动 `apps/tk/app.py` | 未开始 |
| B8 | 架构测试加 Tk 体量预算与 `.after(` 只在 `apps/tk/`；附录 A 走查；设计文档 §9/§12、README、change.md；EXE 自检；开 PR | 未开始 |

各批要点（写 change.md 与 PR 说明时汇总）：

- B1：HTML 基准 `tests/report_html_cases.py` + `tests/fixtures/report_html/`，测试 `tests/test_report_html_golden.py`；模板用 `string.Template`，读取时统一 LF（Windows 检出可能是 CRLF）。`core.reporting` 的 `__init__` 导出模型、证据帧与 `render_html_report`。报告测试改从 `core.reporting` 导入。allowlist 49 → 44。
- B2：结果页拆成 `style`、`header`（头部 + 结论横幅）、`overview`、`phase_bar`、`check_list`（含纯函数 `filter_checks`/`initial_check_id`）、`evidence_view`、`review_box`、`diagnostics`、`action_bar`、`panel`（容器）。构造参数 `history_store` 改为 `history_root`（App 传 `feedback_controls.history().root`）。删掉面板里 App 从不走到的兜底分支：没有 `on_review_toggle` 时自己调 `history_store.review`、没有 `on_export_report` 时自己弹保存框并渲染 HTML（App 始终注入这两个回调，导出与复核留在练习控制器）。`_active_photos` 无界列表删除，正在显示的画面由各视角框持有引用。测试改用公共方法 `select_phase/select_status/select_check/review/practice_again/view_history/export`、`record`、`photo_count`。差分：`tests/result_page_snapshot.py` + `tests/fixtures/result_page/snapshots.json`（拆分前旧面板 3 份记录 50 步控件树快照摘要），测试 `tests/test_result_page_snapshot.py`。allowlist 44 → 41。
- B3：`UiDispatcher`（`post(callback, token=None, *, during_close=False)`、`every_tick`、`start`/`pump`/`close`/`destroy`；token 是"仍有效"判定，`epoch_token(epoch)` 由 `Epoch` 生成）与 `ShutdownCoordinator`（步骤由 `ShutdownSteps` 注入：工作流准备 → dispatcher 进入关闭态并停会话 → 后台任务释放（收尾片段、`cancel_all`、摄像头）→ Tk 分片轮询会话线程与 finalizer → 到截止时刻或全部空闲后 flush、销毁）。App 的逐帧节奏改由 dispatcher 调度（`_tick` 不再自己登记 `after`，B6 删除）；`_on_close` 拆成 `_close_workflows`、`_enter_closing`、`_close_extra_budget_s`、`_release_for_close`、`_flush_for_close` 交给协调器，`_poll_close_workers` 与 `_CLOSE_*` 常量删除（移到 `apps/tk/shutdown.py`）。CompareWindow、SettingsWindow 构造时注入 `post`，后台线程不再直接 `after(0)`；ExamPanel 播报文字经 `app.ui.post`；FeedbackControls 删掉队列与 100 ms 轮询，分析线程经注入的 `post` 交回 `_finish`。新测试 `tests/test_tk_dispatcher.py`、`tests/test_tk_shutdown.py`；`test_app_ui_lifecycle.py` 的关窗测试改断言协调器（`_shutdown.releasing/join_release/deadline`），两条直接调 `_poll_close_workers` 的测试改为经 `_on_close` + 受控时钟；两条断言 `_tick` 自己登记 `after` 的测试改断言不再登记。allowlist 41 → 38，全量 1225 passed。
- B4：`CompareWindow` → `apps/tk/views/compare_dialog.py`（`CompareDialog`，布局拆成模板池 / 目标视频 / 执行 / 结果四段）+ `apps/tk/controllers/compare.py`（`CompareController` 用 `TaskGroup` 跑生成与比对，纯函数 `parse_frame`、`usable_template`、`template_summary`、`compare_report`）；`SettingsWindow` → `views/settings_dialog.py`（`SettingsDialog`）+ `controllers/settings.py`（`SettingsController` 下载任务，纯函数 `active_model_keys`、`model_state`、`current_summary`）；`CollapsibleSection` → `views/widgets.py`。窗口已关时投递的回调由视图包一层 `winfo_exists` 丢弃。线程数改在 Tk 线程读取一次（旧实现在工作线程里读 `IntVar`）。`core.pose` 的 `__init__` 补导出模型清单与下载（`MEDIAPIPE_MODELS`、`ModelSpec`、`is_installed`、`installed_size_mb`、`download_model`），视图不再经 `core.model_manager` 门面。新测试 `tests/test_tk_compare_settings.py`（控制器假服务 + 两窗口真实 Tk 冒烟）；`test_error_handling.py` 改 patch `CompareDialog`。allowlist 38 → 36，全量 1232 passed。
- B5：`ExamPanel` → `apps/tk/views/exam_window.py`（`ExamWindow`，只管布局、文件对话框与弹框）+ `apps/tk/controllers/exam.py`（`ExamController`：名单、ROI、开考、重考、导出、关窗与主窗关闭收尾；纯函数 `start_refusal`、`retest_refusal`、`roi_problem`、`session_config`；经 `ExamHost`/`ExamView` 协议协作，自己实现 `WorkflowUi`）。原 108 行的 `_retest_current` 拆成 `retest_target`、`retest_refusal`、`retest`。App 作为 `ExamHost` 新增 `post`、`exam_record_dir`、`exam_closed`，`exam_panel` 字段改存控制器。`FeedbackControls` → `apps/tk/views/practice_controls.py`（`PracticeControls`）+ `apps/tk/views/practice_history.py`（`HistoryWindow`、`EvidenceWindow`）+ `apps/tk/controllers/practice_analysis.py`（`PracticeAnalysis`：历史存储、`TaskGroup` 后台分析、`Epoch` 代次——改学号与取消推进代次，带工作流 token 的过期结果只恢复控件）。分析配置只在 Tk 线程读一次。`apps/exam_panel.py`、`apps/feedback_panel.py` 删除（无外部使用者）。`clamp_workers`/`default_workers` 移到 `apps/tk/controllers/session.py`（抵消 App 新增的宿主方法行数）。结果页的导出与复核仍在 App，B6 随练习模式一起移到 `controllers/practice.py`。测试：`tests/test_exam_panel.py` 改为对 `ExamController` 的假视图 + 假宿主测试（13 条行为断言保留，删去对面板私有字段的拼装），新增拒绝文字与真实 Tk 冒烟 2 条；`test_app_ui_lifecycle.py` 的关窗移交台账测试改用 `ExamController`；`test_action_feedback.py` 3 条真实 Tk 测试与 `test_m2_adversarial_stress.py` 1 条改用 `PracticeControls`/`analysis` 公共接口（去掉 `_start`、`active_token`）；`test_clamp_workers.py` 改导入路径。allowlist 36 → 34，全量 1234 passed。
- B6：`App` 类删除。视图 `apps/tk/views/`：`main_window.py`（`MainWindow`：滚动左栏、教师块、练习块、状态区、预览与结果页切换）、`capture_panel.py`、`preview_surface.py`（`choose_dual_preview_layout`、按角色出图）、`recording_panel.py`、`status_panel.py`、`practice_block.py`；视图只暴露控件与变量，命令经 `connect(...)` 注入。控制器 `apps/tk/controllers/`：`session.py`（`SessionController`：`UiState` 快照、启动/停止、会话事件经 `post` 回 Tk 线程并计数 `finished()`、旋转只存 `LiveControls.rotation`）、`recording.py`（`RecordingController`：FrameSink/SegmentControl、手动录制、自动比对、模板池在 Tk 线程缓存为元组供 `plan()` 读取、后处理更新以 `during_close=True` 投递）、`cameras.py`（`CameraController`：枚举结果经 `post` 回 Tk 线程，删去队列；偏好设备恢复与持久化；预热规格）、`practice.py`（练习模式、结果页导出与复核）、`exam_host.py`（`ExamHost` 适配器）、`flags.py`（`LiveFlags` 取代 App 私有标志）。组合根 `apps/tk/app.py`（`DesktopApp`，181 行）装配控制器与视图，逐帧工作经 `ui.every_tick`，关窗步骤交 `ShutdownCoordinator`，flush 先 `ui.pump()`；`apps/app_ui.py` 改为入口壳。`UiDispatcher.destroy` 取消已登记的 `after`。`desktop_smoke.py` 改用新接口（`run` 行数不变）。删掉无用的 `source_var`、`save_var/out_var/out_path`、`rotate_var`、`progress_var` 与模板路径缓存。测试（D-13）：删除 `test_app_ui_lifecycle.py`、`test_app_ui_dual_camera.py`、`test_app_controls.py`，行为断言迁入新文件 `tests/test_tk_session.py`（启停、旋转、预览布局、会话事件、在线匹配、双摄）、`test_tk_recording.py`（手动/自动录制、片段计划、后处理状态）、`test_tk_cameras.py`（预热规格、枚举回 Tk 线程、偏好设备）、`test_tk_close.py`（关窗协调、考试成绩表移交、真实窗口关闭）与替身 `tests/tk_app_doubles.py`（真实隐藏 Tk 根 + 假摄像头池 + `FakeLive`）；无 Tk 的旋转与缩放用例移到 `test_live_session.py`。`test_student_presence.py`、`test_feedback_result_panel.py`、`test_m2_adversarial_stress.py`、`test_ui_controls.py`、`test_layout_structure.py`、`test_error_handling.py`、`test_s5_hands_toggle.py`、`test_camera_capture.py::test_tk_preview_student_switch_with_process_cameras` 改用 `DesktopApp` 公共接口；`test_app_ui_online_matcher.py` 的 App 匹配用例并入 `test_tk_session.py`；静态扫描（`test_s5_realtime_latest_frame.py`、`test_s5_ui_no_go.py`、`test_artifact_roots.py`）扩到 `apps/tk/**`。去掉而无对应新测试的：只断言私有字段或已删死代码的用例（`_parse_rotate`/`_degrees_to_rotate_label`、`App.__init__`/`_build_ui`/`_stop` 的 AST 检查、`out_path`/`save_var`、`root.after_calls` 计数）。allowlist 去掉 `app_ui` 5 条（`ENTRY_CEILING` 34 → 29）。全量 1222 passed（用例合并后总数下降，无新增失败）；`sim_exam_flow` 37 passed、SIMULATION_OK；真实 spawn 摄像头切换测试通过。遗留：`controllers/exam.py` 454 行超过控制器 400 行预算，B8 加 Tk 预算前拆分。
- 修复（单独提交）：结果页「导出图文报告」回调 `App._on_student_export_report` 使用 `ACTIONS` 但从未导入，点击即 NameError（旧项目基线 `5a6d99c` 已存在）。补导入，新增 `test_feedback_result_panel.py::test_export_report_from_result_page_writes_html`。
