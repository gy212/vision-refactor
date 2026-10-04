# -*- coding: utf-8 -*-
"""S6 default switch decision regression tests (YOLO migration Issue #28)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DECISION = _REPO_ROOT / "docs" / "yolo_default_switch_decision.md"
OPTIMIZED_PLAN = _REPO_ROOT / "docs" / "yolo_migration_plan_optimized.md"
GPU_REPORT = _REPO_ROOT / "docs" / "yolo_gpu_recheck_report.md"


def test_s6_decision_records_all_default_paths_stay_mediapipe():
    text = DECISION.read_text(encoding="utf-8")
    for keyword in (
        "YOLO 迁移 S6 默认切换决策（Issue #28）",
        "全部不切默认",
        "apps/main.py",
        "apps/app_ui.py",
        "继续 MediaPipe",
        "J1 corr=0.280",
        "J4 一致率=0.50",
        "CUDA 已修复",
        "6/6 样本有效",
        "Hands 关 FPS 比仅 1/6",
        "Hands 开 0/6",
        "score_authorized=False",
        "yolo_body_mp_pose_supplement",
    ):
        assert keyword in text, f"S6 默认切换决策缺少关键依据或结论：{keyword}"


def test_s6_decision_has_per_chain_outcome_and_reopen_gates():
    text = DECISION.read_text(encoding="utf-8")
    for chain in (
        "实时预览（CLI / Tkinter UI）",
        "Vue/Tauri 实时预览",
        "模板匹配",
        "规则 / 技术评估",
        "Hybrid",
    ):
        assert chain in text

    for gate in (
        "Enterprise license",
        "6/6 样本有效",
        "1.30 / 1.20",
        "pass/fail 一致率",
        "一键回滚配置",
        "calibration_status",
        "score_authorized",
    ):
        assert gate in text, f"S6 默认切换前置条件缺失：{gate}"


def test_plan_and_gpu_report_point_to_s6_no_default_switch():
    plan = OPTIMIZED_PLAN.read_text(encoding="utf-8")
    gpu = GPU_REPORT.read_text(encoding="utf-8")

    for text, name in ((plan, OPTIMIZED_PLAN.name), (gpu, GPU_REPORT.name)):
        for keyword in (
            "#28",
            "全部不切默认",
            "离线 / 实验入口",
        ):
            assert keyword in text, f"{name} 缺少 #28 默认切换决策同步说明：{keyword}"

    legacy_plan_promises = (
        "实时预览默认 YOLO body，手部仍用 MediaPipe Hands。",
        "YOLO 作为可选加速。",
        "UI、CLI、batch 都能显示结果来源。",
    )
    for legacy in legacy_plan_promises:
        assert legacy not in plan, f"S6 计划仍保留会误导默认切换的旧表述：{legacy}"
