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
| B2 | 结果页拆成 `apps/tk/views/practice_result/`（头部、概览卡片、相位条、检查列表、证据区、诊断区、底部操作栏），`apps/feedback_result_panel.py` 改门面 | 未开始 |
| B3 | `apps/tk/dispatcher.py`（`UiDispatcher`）与 `apps/tk/shutdown.py`（`ShutdownCoordinator`）及单测；App、ExamPanel、FeedbackControls、CompareWindow、SettingsWindow 的轮询与跨线程 `after(0)` 改走 dispatcher | 未开始 |
| B4 | `CompareWindow` → `views/compare_dialog.py` + `controllers/compare.py`；`SettingsWindow` → `views/settings_dialog.py` + `controllers/settings.py`；`CollapsibleSection` 等 → `views/widgets.py` | 未开始 |
| B5 | `ExamPanel` → `views/exam_window.py` + `controllers/exam.py`；`FeedbackControls` → `views/practice_controls.py` + `controllers/practice.py`（持有分析任务 `Epoch`） | 未开始 |
| B6 | `App` 拆分：`views/main_window.py`、`capture_panel.py`、`preview_surface.py`、`recording_panel.py`；`controllers/session.py`、`recording.py`；组合根 `apps/tk/app.py`；`apps/app_ui.py` 留入口壳；删除 `App` 类；改写 48 处半成品构造 | 未开始 |
| B7 | `apps/desktop_smoke.py` → `apps/tk/self_check.py`（拆 `run`，并加 HTML 图文报告导出检查，验证模板随包分发）；`desktop_launcher` 改启动 `apps/tk/app.py` | 未开始 |
| B8 | 架构测试加 Tk 体量预算与 `.after(` 只在 `apps/tk/`；附录 A 走查；设计文档 §9/§12、README、change.md；EXE 自检；开 PR | 未开始 |
