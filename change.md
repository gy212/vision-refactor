## 2026-11-20: [feat] 实施学生练习检查项全局统一开关（核心、报告与 Tk 桌面端）

- 问题：学生练习检查项此前默认全开且无全局开关，无法关闭不想检查的项目；同时 `PrefKey` 哈希依赖 `default`/`parse`/`dump`/`fallback` 字段，且练习视频、序列、桌面入口缺少共用的前置筛选与全关校验。
- 修改：
  1. **L0 偏好存储**（`core/foundation/prefs.py`、`core/foundation/__init__.py`）：`PrefKey` 的 `default`、`parse`、`dump`、`fallback` 设为 `field(..., compare=False)` 使哈希与相等性仅依赖 `name`；新增 `PRACTICE_DISABLED_CHECKS`（`practice_disabled_checks`）与 `_parse_disabled_checks`，导出 `load_practice_disabled_checks` 与 `save_practice_disabled_checks` 全局保序去重存储（不按动作区分、不设 `scope_ids`）。
  2. **L3 全量目录与评估源头过滤**（`core/evaluation/rules/sanda_v5.py`、`core/evaluation/__init__.py`、`core/evaluation/practice.py`）：新增并导出 `catalog_checks(stance=DEFAULT_STANCE)` 收集并排序全系统 99 个唯一检查项；新增并导出共用 `select_practice_checks()`，在 `evaluate_series()` 最开头及 `analyze_practice()` 视频校验与推理前拦截当前动作全关（抛 `ValueError("当前动作至少需要保留 1 个开启的检查项")`），自动忽略其他动作段及过期旧 ID；保留完整 `synchronized_windows()` / `shared_action_record()` / `_repetitions()`，过滤 `RuleProfile` 并按需跳过 `paired_torso_measurements()`，仅存在跳过项时写入 `disabledCheckIds` 与 `totalCatalogChecks`（默认全开不写新键）。
  3. **L4 报告只读说明**（`core/reporting/saved_results.py`、`core/reporting/practice_model.py`、`core/reporting/text.py`）：抽取共用 `skipped_checks_note()`，严格只读记录自带的 `disabledCheckIds` 与 `totalCatalogChecks`，在 `build_report_model()` 副标题与 `format_report()` 摘要下方追加跳过与启用比例说明（默认全开零新增文字）。
  4. **Tk 桌面控制器、设置弹窗与入口护栏**（新建 `apps/tk/controllers/practice_checks.py`、`apps/tk/views/practice_checks_dialog.py`；修改 `apps/tk/views/__init__.py`、`apps/tk/controllers/practice_analysis.py`、`apps/tk/views/practice_controls.py`、`apps/tk/controllers/practice.py`、`apps/tk/dispatcher.py`、`apps/tk/shutdown.py`）：提供 9 组 99 项全局统一设置弹窗（含 `_set_vars` 批量切换与保存时动作 0 开启项非阻断提示）、按钮计数联动与录制锁定；主线程 `PracticeAnalysis.analyzer()` 快照透传并支持 `action` 校验；`PracticeController.start()`、`PracticeControls.import_video()` 与 `reanalyze()` 前置复用 `select_practice_checks` 拦截动作全关；`dispatcher.py`/`shutdown.py` 补全无 `tkinter` 环境 `TclError` 导入回退。
  5. **单测与文档**（`tests/test_foundation_prefs.py`、新建 `tests/test_practice_check_toggles.py`、新建 `tests/test_tk_practice_checks.py`、`docs/plan.md`）：覆盖 `PrefKey` 哈希与偏好往返、99 项目录、跨动作复用、`select_practice_checks` 与双入口拦截、`phaseWindows` 保留、历史报告锁定以及 Tk 控制器/弹窗/入口护栏，并同步更新 `docs/plan.md`。
- 验证（WSL `./.venv/wsl/bin/python`）：
  - `ruff format --check`（19 个修改/新增 `.py` 文件）：`19 files already formatted`；`ruff check`（上述 19 个 `.py` 文件）：`All checks passed!`；`git -c core.whitespace=cr-at-eol diff --check`：0 报错；
  - `pytest tests/test_architecture.py -q`：`18 passed`；
  - `pytest tests/test_pose33_v3_golden.py tests/test_valid_mask_migration.py tests/test_tech_eval_contract.py tests/test_rule_availability.py -q`：`45 passed`；
  - `pytest tests/test_foundation_prefs.py tests/test_practice_check_toggles.py tests/test_tk_practice_checks.py tests/test_feedback_report.py tests/test_feedback_html_export.py tests/test_report_html_golden.py tests/test_evidence_budget.py -q`：`86 passed, 3 skipped`（其中 `test_tk_practice_checks.py` 为 `6 passed, 3 skipped`，3 个真实 Tk GUI 用例在 WSL 按 `pytest.importorskip("tkinter")` 跳过）；
  - `pytest tests/test_action_feedback.py -k "not real_tk" -q`：`49 passed, 3 deselected`（直接运行 `test_action_feedback.py` 时，3 个 `test_real_tk_*` 用例因 WSL 缺 `tkinter` 报 `ModuleNotFoundError: No module named 'tkinter'`）。

## 2026-11-20 学生练习检查项开关方案审阅

- 问题：核对 `docs/plan.md` 中按动作保存设置和练习入口至少开启一项的约定。
- 审阅结果：建议补齐偏好键的可哈希表示及视频、序列两个入口共用的全关校验，再开始实施。详见 `outputs/review-plan/20261120/review.md`。
- 验证：用临时偏好文件复现字典默认值导致保存失败。用合成关键点模拟检查项全部滤空，复现序列入口返回“已检查项目未发现明显问题”。

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
