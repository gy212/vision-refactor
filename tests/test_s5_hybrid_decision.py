# -*- coding: utf-8 -*-
"""S5 Hybrid decision regression tests (YOLO migration Issue #27)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

REPORT = _REPO_ROOT / "docs" / "yolo_gpu_recheck_report.md"
PLAN = _REPO_ROOT / "docs" / "yolo_migration_plan.md"
OPTIMIZED_PLAN = _REPO_ROOT / "docs" / "yolo_migration_plan_optimized.md"


def test_hybrid_decision_report_records_no_implementation():
    assert REPORT.exists(), f"GPU / Hybrid 决议报告缺失：{REPORT}"
    text = REPORT.read_text(encoding="utf-8")

    for keyword in (
        "Hybrid 决议（Issue #27）",
        "不实现 `yolo_body_mp_pose_supplement`",
        "6/6 个样本跑完有效 GPU benchmark",
        "Hands 关 1/6 达标",
        "Hands 开 0/6 达标",
        "0.41",
        "torch.cuda.is_available() == True",
        "score_authorized=False",
        "另开实现子任务",
    ):
        assert keyword in text, f"Hybrid 决议缺少关键证据或边界：{keyword}"


def test_no_hybrid_runtime_code_path_exists():
    runtime_roots = ("apps", "batch", "core", "analysis")
    hits: list[Path] = []
    for root_name in runtime_roots:
        root = _REPO_ROOT / root_name
        for path in root.rglob("*.py"):
            if "yolo_body_mp_pose_supplement" in path.read_text(encoding="utf-8"):
                hits.append(path.relative_to(_REPO_ROOT))
    assert hits == []


def test_plan_docs_do_not_recommend_hybrid_implementation():
    for path in (PLAN, OPTIMIZED_PLAN):
        assert path.exists(), f"迁移计划文档缺失：{path}"
        text = path.read_text(encoding="utf-8")
        for keyword in (
            "Issue #27",
            "docs/yolo_gpu_recheck_report.md",
            "当前不实现",
            "另开实现子任务",
        ):
            assert keyword in text, f"{path.name} 缺少 Hybrid 决议同步说明：{keyword}"

    legacy_promises = (
        "实现 `yolo_body_mp_pose_supplement`，只在离线规则/技术评估明确开启时跑 MediaPipe Pose。",
        "Hybrid 只作为离线显式模式：`yolo_body_mp_pose_supplement`。",
        "| P4 实现 Hybrid 补点 | 后置到 S5，且仅离线显式开启 |",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in (PLAN, OPTIMIZED_PLAN))
    for legacy in legacy_promises:
        assert legacy not in combined, f"计划文档仍保留会误导实现 Hybrid 的旧表述：{legacy}"
