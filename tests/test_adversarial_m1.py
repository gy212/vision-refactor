"""Adversarial stress-testing suite for Milestone 1 (core/feedback_report.py).

Challenges:
1. Item conservation invariant across 1,200+ randomized, malformed, duplicate, and edge-case configurations.
2. 7-level headline hierarchy exhaustiveness grid search (161,051 cases in 0..10 space) and bug reproduction.
3. Prohibited grading / score / penalty points / progress bar scanner.
4. Height-normalized landmark coordinates robustness on non-square frames and extreme landmark inputs.
"""
from dataclasses import asdict
import itertools
import math
from pathlib import Path
import random
import re
import tempfile
from typing import Any

import cv2
import numpy as np
import pytest

from core.feedback_report import (
    StatusCounts,
    HeadlineSummary,
    NormalizedReportModel,
    build_report_model,
    get_headline_summary,
    render_html_report,
    extract_evidence_frame,
)
from tests.test_feedback_report import make_synthetic_check, make_synthetic_record, make_dummy_video


# ===========================================================================
# 1. ITEM CONSERVATION INVARIANT STRESS TEST
# ===========================================================================

def generate_random_check(idx: int, edge_mode: str = "normal") -> dict:
    statuses = ["candidate", "not_observed", "unable", "pending_rule"]
    reviews = ["pending", "confirmed", "revoked"]

    if edge_mode == "missing_id":
        cid = None if random.random() < 0.5 else ""
    elif edge_mode == "duplicate_id":
        cid = f"check_{idx % 5}"  # Collide across 5 IDs
    else:
        cid = f"check_{idx}"

    status = random.choice(statuses) if edge_mode != "invalid_status" else "unknown_state_xyz"
    review = random.choice(reviews) if edge_mode != "invalid_review" else "invalid_review_abc"

    chk: dict[str, Any] = {
        "code": f"code_{idx}",
        "name": f"Check {idx}",
        "status": status,
        "review": review,
    }
    if cid is not None:
        chk["id"] = cid

    if edge_mode == "missing_fields":
        keys_to_pop = [k for k in ["code", "name", "standard", "source", "reason"] if random.random() < 0.5]
        for k in keys_to_pop:
            chk.pop(k, None)
    elif edge_mode == "malformed_fields":
        chk["measurements"] = "not_a_list"
        chk["fusion"] = ["not_a_dict"]
        chk["evidence"] = 12345
    return chk


def test_item_conservation_randomized_stress():
    """Generate 1,200+ randomized edge-case check configurations and assert conservation in 100% of cases."""
    random.seed(42)
    edge_modes = [
        "normal",
        "missing_id",
        "duplicate_id",
        "invalid_status",
        "invalid_review",
        "missing_fields",
        "malformed_fields",
    ]

    total_runs = 0
    conservation_successes = 0

    for i in range(1200):
        mode = random.choice(edge_modes)
        num_checks = random.choice([0, 1, 2, 5, 20, 50, 100, 250])
        raw_checks = [generate_random_check(j, edge_mode=mode) for j in range(num_checks)]

        rec = make_synthetic_record(checks=raw_checks)
        if random.random() < 0.1:
            rec["result"]["checks"] = None
        elif random.random() < 0.1:
            rec["result"]["checks"] = "not_a_list"

        model = build_report_model(rec)
        counts = model.counts

        assert counts.verify_conservation(), f"Conservation failed for mode={mode}, n={num_checks}: {counts}"
        assert len(model.all_checks) == counts.total_checks
        assert len(model.candidate_checks) == counts.candidate_count
        assert len(model.revoked_checks) == counts.revoked_count
        assert len(model.not_observed_checks) == counts.not_observed_count
        assert len(model.unable_checks) == counts.unable_count
        assert len(model.pending_rule_checks) == counts.pending_rule_count

        total_runs += 1
        conservation_successes += 1

    assert total_runs == conservation_successes == 1200


# ===========================================================================
# 2. 7-LEVEL HEADLINE HIERARCHY & PEDAGOGICAL NEUTRALITY GRID SEARCH
# ===========================================================================

def evaluate_headline_neutrality_violations():
    """Grid search all combinations of (cand, rev, not_obs, unab, pend) in 0..10 (161,051 cases)."""
    violations = []

    for cand, rev, not_obs, unab, pend in itertools.product(range(11), repeat=5):
        total = cand + rev + not_obs + unab + pend
        counts = StatusCounts(
            total_checks=total,
            candidate_count=cand,
            confirmed_count=0,
            unconfirmed_count=cand,
            revoked_count=rev,
            not_observed_count=not_obs,
            unable_count=unab,
            pending_rule_count=pend
        )

        hl = get_headline_summary(counts, [], needs_rerecord=False)

        # Invariant 1: If total_checks == 0, never report success/green
        if total == 0:
            if hl.level == "success" or hl.status_tone == "success" or "未发现问题" in hl.title:
                violations.append({
                    "type": "EMPTY_CHECKS_FALSE_SUCCESS",
                    "counts": (cand, rev, not_obs, unab, pend),
                    "headline": hl.title,
                    "level": hl.level,
                    "tone": hl.status_tone
                })
                continue

        # Invariant 2: If all non-revoked checks are unable (cand=0, not_obs=0, unab>0),
        # MUST NEVER report success/green or "未发现问题"
        if cand == 0 and not_obs == 0 and unab > 0:
            if hl.level == "success" or hl.status_tone == "success" or "未发现问题" in hl.title:
                violations.append({
                    "type": "ALL_UNABLE_FALSE_SUCCESS",
                    "counts": (cand, rev, not_obs, unab, pend),
                    "headline": hl.title,
                    "level": hl.level,
                    "tone": hl.status_tone
                })
                continue

        # Invariant 3: If not_observed == 0 (no check was ever verified compliant),
        # headline should NEVER claim "已检查项目未发现问题" with success tone!
        if not_obs == 0 and ("已检查项目未发现问题" in hl.title and hl.status_tone == "success"):
            violations.append({
                "type": "ZERO_NOT_OBSERVED_FALSE_SUCCESS",
                "counts": (cand, rev, not_obs, unab, pend),
                "headline": hl.title,
                "level": hl.level,
                "tone": hl.status_tone
            })
            continue

        # Invariant 4: If pending_rule > 0 and cand == 0, never report success/green
        if cand == 0 and pend > 0:
            if hl.level == "success" or hl.status_tone == "success":
                violations.append({
                    "type": "PENDING_RULE_FALSE_SUCCESS",
                    "counts": (cand, rev, not_obs, unab, pend),
                    "headline": hl.title,
                    "level": hl.level,
                    "tone": hl.status_tone
                })
                continue

        # Invariant 5: If cand > 0, never report success
        if cand > 0:
            if hl.level == "success" or hl.status_tone == "success":
                violations.append({
                    "type": "CANDIDATE_FALSE_SUCCESS",
                    "counts": (cand, rev, not_obs, unab, pend),
                    "headline": hl.title,
                    "level": hl.level,
                    "tone": hl.status_tone
                })
                continue

    return violations


def test_headline_grid_search_metrics():
    """Verify grid search execution and document exact violation counts."""
    violations = evaluate_headline_neutrality_violations()
    assert len(violations) == 0, f"Expected 0 violations, got {len(violations)}"


def test_reproduce_headline_neutrality_bug_all_unable_with_revoked():
    """When student has unable checks and a teacher-revoked candidate,
    headline reports Level 1 '本次未能完成有效判断', level='error', tone='unable'.
    """
    counts = StatusCounts(
        total_checks=3,
        candidate_count=0,
        confirmed_count=0,
        unconfirmed_count=0,
        revoked_count=1,
        not_observed_count=0,
        unable_count=2,
        pending_rule_count=0
    )
    assert counts.verify_conservation() is True
    hl = get_headline_summary(counts, [], needs_rerecord=False)

    assert hl.title == "本次未能完成有效判断"
    assert hl.level == "error"
    assert hl.status_tone == "unable"


def test_reproduce_headline_neutrality_bug_empty_checks():
    """When total_checks == 0, headline reports '暂无检查记录' with info level."""
    counts = StatusCounts(
        total_checks=0,
        candidate_count=0,
        confirmed_count=0,
        unconfirmed_count=0,
        revoked_count=0,
        not_observed_count=0,
        unable_count=0,
        pending_rule_count=0
    )
    assert counts.verify_conservation() is True
    hl = get_headline_summary(counts, [], needs_rerecord=False)
    assert hl.title == "暂无检查记录"
    assert hl.level == "info"


# ===========================================================================
# 3. PROHIBIT GRADING ARTIFACTS ADVERSARIAL SCANNER
# ===========================================================================

def test_prohibit_grading_exhaustive_scan():
    """Verify absolutely no scores, penalty points, or progress bars in models or HTML reports."""
    prohibited_patterns = [
        re.compile(r"总分[：:\s]*\d+"),
        re.compile(r"得分[：:\s]*\d+"),
        re.compile(r"扣分[：:\s]*\d+"),
        re.compile(r"扣\s*\d+\s*分"),
        re.compile(r"合格率[：:\s]*\d+%"),
        re.compile(r"准确率[：:\s]*\d+%"),
        re.compile(r"优秀率[：:\s]*\d+%"),
        re.compile(r"<progress"),
        re.compile(r"role=[\"\']progressbar[\"\']"),
        re.compile(r"progress-bar"),
        re.compile(r"\b100\s*分\b"),
        re.compile(r"评分进度条"),
        re.compile(r"动作得分"),
    ]

    scenarios = [
        make_synthetic_record(checks=[]),
        make_synthetic_record(checks=[make_synthetic_check(code=f"c_{i}", status="candidate") for i in range(15)]),
        make_synthetic_record(checks=[make_synthetic_check(code=f"c_{i}", status="not_observed") for i in range(15)]),
        make_synthetic_record(checks=[make_synthetic_check(code=f"c_{i}", status="unable") for i in range(15)]),
    ]

    for rec in scenarios:
        model = build_report_model(rec)
        html_out = render_html_report(rec)

        texts_to_check = [
            model.headline.title,
            model.headline.subtitle,
            model.headline.reason,
            html_out
        ]

        for text in texts_to_check:
            for pat in prohibited_patterns:
                match = pat.search(text)
                assert match is None, f"Prohibited grading artifact matched '{match.group(0)}' in: {text[:200]}"


# ===========================================================================
# 4. HEIGHT-NORMALIZED LANDMARK COORDINATES & ASPECT RATIO SAFETY
# ===========================================================================

def test_landmark_height_normalization_aspect_ratios(tmp_path):
    """Test non-square aspect ratios (1920x1080, 640x480, 1080x1920, 200x50, 50x200)."""
    resolutions = [
        (1920, 1080),  # 16:9
        (640, 480),    # 4:3
        (1080, 1920),  # 9:16 vertical
        (200, 50),     # 4:1 wide banner
        (50, 200),     # 1:4 tall banner
    ]

    for w, h in resolutions:
        vpath = make_dummy_video(tmp_path / f"test_{w}x{h}.avi", width=w, height=h, frames=5)

        lms = [[0.0, 0.0, 0.0, 1.0] for _ in range(33)]
        lms[11] = [0.4, 0.3, 0.0, 1.0]
        expected_x = int(round(0.4 * h))
        expected_y = int(round(0.3 * h))
        w_scaled_x = int(round(0.4 * w))

        frame = extract_evidence_frame(vpath, 0, landmarks=lms, highlight_joints={11})
        assert frame is not None
        assert frame.shape == (h, w, 3)

        if expected_x < w and expected_y < h:
            assert frame[expected_y, expected_x].tolist() != [0, 0, 0]
            if w_scaled_x != expected_x and w_scaled_x < w:
                if abs(w_scaled_x - expected_x) > 10:
                    bg_color = [0, 100, 150]
                    assert frame[expected_y, w_scaled_x].tolist() == bg_color, (
                        f"Joint was incorrectly drawn at width-scaled x ({w_scaled_x}) instead of height-scaled ({expected_x})!"
                    )

        # Extreme non-crashing cases
        safe_extreme_cases = [
            ("NaN", [[float("nan"), float("nan"), 0, 1]] * 33),
            ("Inf", [[float("inf"), float("-inf"), 0, 1]] * 33),
            ("Negative", [[-2.5, -3.0, 0, 1]] * 33),
            ("Huge", [[1e7, 1e7, 0, 1]] * 33),
            ("EmptyList", []),
            ("NoneLandmarks", None),
            ("ShortLandmarks", [[0.5, 0.5, 0.0, 1.0]]),
        ]

        for name, extreme_lms in safe_extreme_cases:
            res_frame = extract_evidence_frame(vpath, 0, landmarks=extreme_lms, highlight_joints={11, 13, 15})
            if extreme_lms is not None:
                assert res_frame is not None


def test_reproduce_extract_evidence_frame_crash_on_malformed_landmarks(tmp_path):
    """BUG REPRODUCTION 3: IndexError crash when landmark elements are empty or 1D."""
    vpath = make_dummy_video(tmp_path / "crash_test.avi", width=640, height=480, frames=5)

    # 1. Empty sublists: [[]] * 33 (should not raise IndexError)
    res1 = extract_evidence_frame(vpath, 0, landmarks=[[] for _ in range(33)], highlight_joints={11, 13, 15})
    assert res1 is not None

    # 2. 1D point: [[0.5]] * 33 (should not raise IndexError)
    res2 = extract_evidence_frame(vpath, 0, landmarks=[[0.5] for _ in range(33)], highlight_joints={11, 13, 15})
    assert res2 is not None

    # 3. render_html_report unhandled crash prevented
    hist_root = tmp_path / "history"
    rec_dir = hist_root / ("e" * 32)
    rec_dir.mkdir(parents=True)
    video_path = rec_dir / "front.mp4"
    make_dummy_video(video_path, width=640, height=480, frames=5)

    rec = make_synthetic_record(
        evidence=[{
            "id": "front:0",
            "view": "front",
            "frame": 0,
            "timeSeconds": 0.1,
            "landmarks": [[] for _ in range(33)],
            "validMask": [True] * 33
        }],
        videos={"front": "front.mp4", "side": "side.mp4"}
    )
    rec["id"] = "e" * 32

    out = render_html_report(rec, history_root=hist_root)
    assert out is not None
