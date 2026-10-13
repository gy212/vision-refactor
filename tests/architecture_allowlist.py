# -*- coding: utf-8 -*-
"""架构测试的现有违规登记（棘轮，只减不增）；规则见 AGENTS.md"模块职责与体量"与设计文档 §13.2。

每条 ``键: (登记值, 计划消除阶段)``。实际值必须与登记值完全相等：变大判新增违规，
变小时在同一提交里把登记值改成实际值；消除后删掉该条，并把 ``ENTRY_CEILING`` 改成新的条数。
搬迁文件时条目随文件改到新路径，登记值沿用原值，不新增条目、不调高登记值。P9 清零。
"""

# 旧的平铺模块按设计文档附录 B 的去向归层（配置，不计入违规条数）。迁走后删掉对应行；
# P2 迁走的模块留下的转发门面仍按目标层归层，P9 删门面时一并删掉。
LEGACY_TARGETS = {
    "core": "core.foundation",
    "core.paths": "core.foundation",
    "core.video_writer": "core.media",
    "core.vision_pipeline": "core.pose",
    "core.pose_features": "core.evaluation",  # P3 起也转发 matching.dtw，按更高的一层归层
    "core.feature_layout": "core.pose",
    "core.model_manager": "core.pose",
    "core.parallel_pose_engine": "core.pose",
    "core.preview_smoother": "core.pose",
    "core.backend_router": "core.pose",
    "core.yolo_adapter": "core.pose",
    "core.action_compare": "core.services",  # P3 起转发 evaluation 与 services，按更高的一层归层
    "core.rule_scoring": "core.evaluation",
    "core.body_core_compare": "core.evaluation",
    "core.online_matcher": "core.evaluation",
    "core.kick_quality": "core.evaluation",
    "core.action_feedback": "core.reporting",  # P3 起也转发 reporting.text，按更高的一层归层
    "core.feedback_geometry": "core.evaluation",
    "core.feedback_phases": "core.evaluation",
    "core.recording_controller": "core.recording",
    "core.feedback_history": "core.services",
    "core.feedback_report": "core.reporting",
    "core.qwen_coach": "core.services",
    "core.exam_session": "core.workflows",
    "core.exam_roster": "core.workflows",
    "core.exam_clip": "core.workflows",
    "core.presence_gate": "core.workflows",
    "core.exam_announcer": "core.workflows",
}

# 规则模块不得直接写 arccos / linalg.norm（测量走 evaluation.measures）；这些旧模块按规则模块检查。
LEGACY_RULE_MODULES = {"core.rule_scoring", "core.feedback_geometry", "core.action_feedback"}

LAYERING: dict[str, tuple[int, str]] = {}

CYCLES: dict[str, tuple[int, str]] = {}

PRIVATE: dict[str, tuple[int, str]] = {}

PRIMITIVES: dict[str, tuple[int, str]] = {
    'np_load: analysis/calibrate_body_core.py': (1, 'P9'),
    'np_load: analysis/offline_matching_profile.py': (6, 'P9'),
    'np_load: batch/batch_export_skeleton.py': (1, 'P9'),
    'subsequence_dtw: analysis/calibrate_body_core.py': (1, 'P9'),
    'subsequence_dtw: analysis/offline_matching_profile.py': (2, 'P9'),
    'subsequence_dtw: scripts/match_segments_probe.py': (1, 'P9'),
    'thread: apps/app_ui.py': (4, 'P8'),
    'thread: apps/feedback_panel.py': (1, 'P8'),
    'tk_after: apps/app_ui.py': (15, 'P8'),
    'tk_after: apps/desktop_smoke.py': (2, 'P8'),
    'tk_after: apps/exam_panel.py': (1, 'P8'),
    'tk_after: apps/feedback_panel.py': (2, 'P8'),
    'video_capture: analysis/bench_annotate_fps.py': (4, 'P9'),
    'video_capture: analysis/spike_yolo_baseline.py': (2, 'P9'),
    'video_capture: core/qwen_coach.py': (1, 'P9'),  # D-12 未决：保留到 P9 按 D-12 处置
}

CLONES: dict[str, tuple[int, str]] = {}

MODULE_SIZE: dict[str, tuple[int, str]] = {
    'analysis/bench_annotate_fps.py': (1018, 'P9'),
    'apps/app_ui.py': (3528, 'P8'),
    'core/media/camera/pool.py': (1115, 'P9'),  # 原 apps/camera_warmup.py（1125），D-15：P9 按职责拆分
    'core/qwen_coach.py': (916, 'P9'),
}

FUNCTION_SIZE: dict[str, tuple[int, str]] = {
    'analysis/bench_annotate_fps.py::_default_cases': (103, 'P9'),
    'analysis/bench_annotate_fps.py::bench_sample': (144, 'P9'),
    'analysis/bench_annotate_fps.py::main': (131, 'P9'),
    'analysis/calibrate_body_core.py::run': (166, 'P9'),
    'analysis/offline_matching_profile.py::run_fixture_profile': (112, 'P9'),
    'apps/app_ui.py::App.__init__': (113, 'P8'),
    'apps/app_ui.py::App._build_ui': (391, 'P8'),
    'apps/app_ui.py::App._set_running_controls': (124, 'P8'),
    'apps/app_ui.py::CompareWindow._build': (128, 'P8'),
    'apps/desktop_smoke.py::run': (142, 'P8'),
    'apps/exam_panel.py::ExamPanel._retest_current': (108, 'P8'),
    'batch/batch_dual_compare.py::_run_body_core_batch': (115, 'P9'),
    'batch/batch_dual_compare.py::main': (237, 'P9'),
    'batch/batch_export_skeleton.py::main': (120, 'P9'),
    'batch/batch_tech_eval.py::main': (190, 'P9'),
    'core/evaluation/techeval/cog.py::eval_cog_front': (101, 'P9'),  # 原 analysis/tech_eval.py，P3 迁入
    'core/evaluation/techeval/cog.py::eval_cog_side': (139, 'P9'),
    'core/evaluation/techeval/com.py::_compute_body_com_single': (176, 'P9'),
    'core/evaluation/techeval/com.py::eval_cog_com': (128, 'P9'),
    'core/evaluation/techeval/events.py::detect_retract_events_side': (103, 'P9'),
    'scripts/sim_exam_flow.py::scenario_5': (107, 'P9'),
}

DEFENSIVE_GETATTR: dict[str, tuple[int, str]] = {
    'apps/app_ui.py': (44, 'P8'),
}

ENTRY_CEILING = 41
