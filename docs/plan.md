# 学生练习检查项全局统一开关实施方案（plan.md）

> **设计原则（全局共用复用）**：系统中检查项标识 `check["id"]`（`segment:phase:role:code`，全库共 99 个唯一项）本身即按基础动作段跨动作与左/右式共用。开关直接**按检查项 `id` 全局统一存储与复用**，不按单独动作区分——在一处关闭某个公共检查项（如开始实战式护肘、前手直拳伸肘），所有包含该动作段的单动作、组合动作及左/右式全部自动生效，严禁按动作拆成独立配置或维护分动作作用域列表。

## 1. 用户体验与生效边界

- **入口位置**：在左侧学生练习面板（[practice_controls.py](file:///home/CodeProject/vision/new/apps/tk/views/practice_controls.py)）表单下方新增按钮 **`检查项设置（已开 X/99 项）`**；录制或后台分析进行中自动置灰锁定，关闭面板时同步销毁已打开的弹窗。
- **弹窗操作**：点击弹出独立的「检查项设置」窗口（[practice_checks_dialog.py](file:///home/CodeProject/vision/new/apps/tk/views/practice_checks_dialog.py)，逻辑收敛至无 Tk 依赖的 [practice_checks.py](file:///home/CodeProject/vision/new/apps/tk/controllers/practice_checks.py)），调用 `catalog_checks(stance)` 按基础动作段将全库全部 99 个唯一检查项统一归为 9 组（「开始实战式」、7 个基础动作段、「结束实战式」），不按 11 个组合动作重复拆分。支持**按组全选/取消**、**全部开启（恢复默认）**、**保存**与**取消**；保存时校验全库至少保留 1 个开启项。
- **持久化与跨动作复用**：全局共用一个有序关闭列表存入 `user_prefs.json`（键 `"practice_disabled_checks"`，默认空列表即全部 99 项开启），不设任何按动作区分的字典或 `scope_ids`。
- **生效范围**：
  - **生效**：学生练习全部入口（实时录制后点「动作评判」、点「导入旧视频」、个人历史里点「补充分析所选记录」），由 `PracticeAnalysis.analyzer()` 在主线程读取一次关闭集合快照透传给 `analyze_practice(..., disabled_checks=...)`。
  - **不生效（严格隔离）**：考试模式与离线 `batch` 批处理完全不受影响，始终全量检查；绝不改动 `feedback_config.json`。

## 2. 分层代码改动清单

### 2.1 本地偏好存储（L0 基础层）
- 修改 [prefs.py](file:///home/CodeProject/vision/new/core/foundation/prefs.py) 与 [\_\_init\_\_.py](file:///home/CodeProject/vision/new/core/foundation/__init__.py)：
  - **加固 `PrefKey` 哈希契约**：将 `PrefKey` 的 `default`、`parse`、`dump`、`fallback` 四个字段均设为 `field(..., compare=False)`，使 `PrefKey` 的相等性与哈希仅依赖稳定的顶层键名 `name: str`。
  - **全局统一关闭集合**：新增 `PRACTICE_DISABLED_CHECKS: PrefKey[tuple[str, ...]]`（JSON 顶层键 `"practice_disabled_checks"`，默认值 `()`，磁盘存 JSON 字符串数组 `["stance:start:front:stance_elbow", ...]`；非 `list` 静默回退 `()`，列表内仅保留非空字符串 `strip` 后保序去重）。
  - **公开读写接口**：导出 `load_practice_disabled_checks(store=None) -> tuple[str, ...]` 与 `save_practice_disabled_checks(disabled_ids, *, store=None) -> tuple[str, ...]`（仅保留非空字符串 `strip` 后保序去重写入 `PRACTICE_DISABLED_CHECKS` 并返回该元组，不按动作拆分也不设 `scope_ids`）。

### 2.2 全量检查项目录与练习评估源头过滤（L3 评估层）
- 修改 [sanda_v5.py](file:///home/CodeProject/vision/new/core/evaluation/rules/sanda_v5.py) 与 [\_\_init\_\_.py](file:///home/CodeProject/vision/new/core/evaluation/__init__.py)：
  - 新增并导出 `catalog_checks(stance: str = DEFAULT_STANCE) -> list[dict[str, Any]]`：校验 `stance in STANCES`（否则抛 `ValueError("请选择支持的左式/右式")`），遍历 `ACTIONS` 调用 `build_checks(act, stance)` 按 `check["id"]` 保序去重收集全系统全部 99 个唯一检查项，并按「`stance:start` → 非 `stance` 动作段首次出现顺序 → `stance:end`」排序，不复制规则定义。
- 修改 [practice.py](file:///home/CodeProject/vision/new/core/evaluation/practice.py) 与 [\_\_init\_\_.py](file:///home/CodeProject/vision/new/core/evaluation/__init__.py)：
  - 在公开入口 `analyze_practice()` 与 `evaluate_series()` 同步增加可选关键字参数 `disabled_checks: Collection[str] = ()`（默认空，既有调用与测试零改动）。
  - **公开并共用同一筛选与全关拦截函数** `select_practice_checks(action, stance, *, pose_variant="full", disabled_checks=())`（经 `core.evaluation` 导出供桌面层与核心双入口统一复用）：
    1. 生成当前动作完整清单 `all_checks = build_checks(action, stance, pose_variant=pose_variant)`；
    2. 对照 `all_checks` 筛出属于当前动作的已关闭项 `skipped_ids`（**自动忽略不属于当前动作的其他动作段 ID 与过期旧 ID**；仅含无关或过期 ID 时当前动作按全开执行）；
    3. 计算剩余开启清单 `enabled_checks`，若为空则统一抛出 `ValueError("当前动作至少需要保留 1 个开启的检查项")`；
    4. **触发时机**：`analyze_practice()` 在检查视频文件与启动 `extract_series` 推理**之前**先读配置并调用拦截；`evaluate_series()` 在函数最开头（`synchronized_windows()` 之前）调用拦截。
  - **源头过滤但保留完整动作计时**：
    - `synchronized_windows()`、`shared_action_record()`、`_repetitions()` **照常基于完整动作全量运行，绝不跳过**；
    - 用 `enabled_ids = {c["id"] for c in checks}` 过滤 `rule_profile(action, stance, cfg).rules` 构建过滤后的 `RuleProfile`，`paired_torso_measurements()` 仅在过滤后仍存在多视角规则时计算；开启项严格保持 `candidate` / `not_observed` / `unable` 三态；
    - 当 `skipped_ids` 非空时，在落盘 `result` 顶层写入 `"disabledCheckIds": skipped_ids` 与 `"totalCatalogChecks": len(all_checks)`（为空时不写入这两个字段，保持默认全开输出逐字节不变）。

### 2.3 报告与历史记录只读展示（L4 报告层）
- 修改 [saved_results.py](file:///home/CodeProject/vision/new/core/reporting/saved_results.py)、[practice_model.py](file:///home/CodeProject/vision/new/core/reporting/practice_model.py) 与 [text.py](file:///home/CodeProject/vision/new/core/reporting/text.py)：
  - 严格只读保存记录自身的 `result.get("disabledCheckIds")` 与 `result.get("totalCatalogChecks")`（绝不读实时偏好），由共享辅助函数 `skipped_checks_note()` 生成统一文案；
  - 当 `disabledCheckIds` 为非空列表时（`skipped_n = len(disabled_ids)`，`enabled_n` 为当前记录去重后检查项数，`total_n` 取合法 `int` 型 `totalCatalogChecks` 否则回退 `enabled_n + skipped_n`）：
    - 在 `build_report_model()` 的 `headline.subtitle` 末尾追加 `f" 本次已按设置跳过 {skipped_n} 个检查项（启用 {enabled_n}/{total_n} 项）。"`；
    - 在 `format_report()` 摘要行下方加入 `f"检查项范围：本次已按设置跳过 {skipped_n} 个检查项（启用 {enabled_n}/{total_n} 项）。"`。
  - 默认全开时两处均不增加任何文字，既有 HTML 黄金快照零漂移。

### 2.4 桌面端控制器、全局设置弹窗与接线（Apps 入口层）
- 新建 [practice_checks.py](file:///home/CodeProject/vision/new/apps/tk/controllers/practice_checks.py)（无 Tk 依赖）：提供 `build_check_groups(stance)`（9 组共 99 项）、`checks_button_text(disabled_ids=None)`、`checks_summary_text`、`check_item_label`、`validate_and_save_checks(all_ids, enabled_ids)`（全关抛 `ValueError("至少需要保留 1 个开启的检查项")`），以及直接复用 `core.evaluation.select_practice_checks` 的 `zero_enabled_actions(disabled_ids)` 与 `validate_action_checks(action, stance, disabled_ids=None, *, pose_variant="full")`。
- 新建 [practice_checks_dialog.py](file:///home/CodeProject/vision/new/apps/tk/views/practice_checks_dialog.py) 并在 [\_\_init\_\_.py](file:///home/CodeProject/vision/new/apps/tk/views/__init__.py) 导出 `PracticeChecksDialog`：提供 `is_open()`/`focus()`、顶部已启用计数与「全部开启（恢复默认）」、中部 9 组可滚动勾选项及组内全选/取消（复用内部 `_set_vars(cids, enabled)`）、底部保存与取消；保存成功后调用 `zero_enabled_actions(saved)`，若有动作 0 开启项则弹出非阻断提示列明动作名后再关窗。
- 修改 [practice_analysis.py](file:///home/CodeProject/vision/new/apps/tk/controllers/practice_analysis.py)、[practice_controls.py](file:///home/CodeProject/vision/new/apps/tk/views/practice_controls.py) 与 [practice.py](file:///home/CodeProject/vision/new/apps/tk/controllers/practice.py)：
  - `PracticeAnalysis.analyzer(model_label, *, action=None, stance=DEFAULT_STANCE)` 在主线程读取 `load_practice_disabled_checks()` 快照，传入 `action` 时立即调用 `select_practice_checks` 前置校验并透传 `disabled_checks`；
  - `PracticeControls` 挂载设置按钮、联动 `set_recording` 与 `close`，`analyze_recording()` 透传 `action`/`stance`，`import_video()` 与 `reanalyze()` 在弹选文件或启动后台线程前调 `validate_action_checks` 前置拦截；`PracticeController.start()` 在开双摄 `preflight` 前校验并将 `ValueError` 并入 `identity_error` 拦截；
  - 为 [dispatcher.py](file:///home/CodeProject/vision/new/apps/tk/dispatcher.py) 与 [shutdown.py](file:///home/CodeProject/vision/new/apps/tk/shutdown.py) 补充 `TclError` 导入回退，确保无 `tkinter` 环境可运行非 GUI 控制器测试而不影响真实 Tk 行为。

## 3. 专项验收条件与测试覆盖

1. **偏好存储与哈希修复**（[test_foundation_prefs.py](file:///home/CodeProject/vision/new/tests/test_foundation_prefs.py)）：`PrefKey(default={})` 可正常 `set`/`get` 且哈希不报错；`save_practice_disabled_checks` / `load_practice_disabled_checks` 支持持久化、保序去重、非法数据回退 `()` 与恢复全开 `()`。
2. **全量目录、跨动作复用、双入口全关拦截与报告锁定**（[test_practice_check_toggles.py](file:///home/CodeProject/vision/new/tests/test_practice_check_toggles.py)）：
   - `catalog_checks("left")` / `catalog_checks("right")` 返回全库 99 个唯一检查项且首尾为开始/结束实战式；
   - 关闭公共检查项在 `front_straight`、`step_straight_combo` 与左右式下自动复用生效；
   - `select_practice_checks()`、`analyze_practice()` 与 `evaluate_series()` 全关均抛 `ValueError("当前动作至少需要保留 1 个开启的检查项")` 且 `analyze_practice()` 不启动 `extract_series`；仅含其他动作段或过期旧 ID 时按全开完成且不写 `disabledCheckIds`；
   - 关闭起止实战式后仍完整保留 `phaseWindows` 阶段时间（且无多视角规则时跳过 `paired_torso_measurements`）；后续修改全局偏好后再读旧记录，桌面模型、纯文本与 HTML 报告均锁定保存时的启用数与跳过数。
3. **桌面端控制器、弹窗与入口护栏测试**（[test_tk_practice_checks.py](file:///home/CodeProject/vision/new/tests/test_tk_practice_checks.py)）：6 项无 GUI 测试覆盖 9 组 99 项分组、按钮文案、全关拦截、持久化、`zero_enabled_actions`/`validate_action_checks` 及 `PracticeAnalysis.analyzer()` 快照与动作前置校验；3 项受 `pytest.importorskip("tkinter")` 保护的 GUI 用例覆盖设置弹窗、控制面板按钮锁与 `import_video`/`reanalyze`/`PracticeController.start` 拦截。
