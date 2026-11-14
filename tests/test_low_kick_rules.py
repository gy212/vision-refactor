# -*- coding: utf-8 -*-
"""低鞭腿（上步前低鞭腿、前低鞭腿、原地后低鞭腿）编排、逐次切分、规则测量与标定门禁回归测试。"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.evaluation import (  # noqa: E402
    FeedbackConfig,
    evaluate_series,
    judge_views,
    series_from_raw,
)
from core.evaluation.action_catalog import (  # noqa: E402
    ACTION_SEGMENTS,
    ACTIONS,
    PHASES,
    SEGMENT_LABELS,
    action_definition,
)
from core.evaluation.action_timing import NEEDS_KICK_SIDE, NEEDS_SIDE, shared_timeline  # noqa: E402
from core.evaluation.body_sides import sides_for_stance  # noqa: E402
from core.evaluation.kick_events import (  # noqa: E402
    kick_repetitions,
    segment_kick_events,
)
from core.evaluation.measures.kicks import (  # noqa: E402
    kicking_leg_side_safe,
    measure_body_lean,
    measure_foot_flexed,
    measure_kick_guard_elbow,
    measure_kick_height,
    measure_kick_retract_segment,
    measure_pad_step_missing,
    measure_pad_step_overtake,
    measure_pad_step_pause,
)
from core.evaluation.rule_checks import ACTION_BY_NAME  # noqa: E402
from core.evaluation.rules.kick_measures import kick_landmarks  # noqa: E402
from core.evaluation.rules.kick_policy import apply_kick_availability  # noqa: E402
from core.evaluation.rules.sanda_v5 import build_checks  # noqa: E402
from core.evaluation.scoring import exam_pool  # noqa: E402
from core.pose.landmarks import (  # noqa: E402
    L_HIP,
    L_SHOULDER,
    MOUTH_L,
    MOUTH_R,
    NOSE,
    R_HIP,
    R_SHOULDER,
)
from core.pose.raw_series import Pose33RawSeries  # noqa: E402
from core.reporting.issue_groups import group_action_issues  # noqa: E402
from core.reporting.practice_model import build_report_model  # noqa: E402


def _make_kick_sequence(
    *,
    stance: str = "left",
    view: str = "side",
    n_kicks: int = 2,
    fps: float = 30.0,
    pad_disp: float = 0.22,
    pad_overtake: float = -0.02,
    pad_pause_s: float = 0.0,
    retract_offset: float = 0.04,
    drop_guard_shoulder: bool = False,
    omit_arm_swing: bool = False,
    in_place_landing: bool = False,
    missing_head_frames: int = 0,
    mask_ankle_on_first_peak: bool = False,
) -> tuple[np.ndarray, dict]:
    """构造可运行完整练习/考试链路的合成正侧双视角 Pose33 序列（含可注入动作缺陷）。"""
    dir_x = 1.0 if stance == "left" else -1.0
    if view == "side":
        total = 35 + n_kicks * 60 + 25
    else:
        total = 35 + n_kicks * 60 + 25
    pause_frames = int(round(pad_pause_s * fps))
    if pause_frames > 0:
        total += n_kicks * pause_frames

    lm = np.zeros((total, 33, 4), dtype=np.float32)
    lm[:, :, 3] = 0.95
    f_sh, f_el, f_wr, f_hp, f_kn, f_ak, f_hl, f_to = sides_for_stance("front", stance)
    r_sh, r_el, r_wr, r_hp, r_kn, r_ak, r_hl, r_to = sides_for_stance("rear", stance)

    if view == "side":
        lm[:, NOSE, :2] = (0.50 + 0.04 * dir_x, 0.12)
        lm[:, MOUTH_L, :2] = (0.50 + 0.03 * dir_x, 0.14)
        lm[:, MOUTH_R, :2] = (0.50 + 0.03 * dir_x, 0.14)
        lm[:, f_sh, :2] = (0.50, 0.25)
        lm[:, r_sh, :2] = (0.50, 0.25)
        lm[:, f_hp, :2] = (0.50, 0.55)
        lm[:, r_hp, :2] = (0.50, 0.55)
        # 默认高架护手与预备姿势
        lm[:, f_el, :2] = (0.50 + 0.04 * dir_x, 0.30)
        lm[:, f_wr, :2] = (0.50 + 0.08 * dir_x, 0.18)
        lm[:, r_el, :2] = (0.50 + 0.03 * dir_x, 0.30)
        lm[:, r_wr, :2] = (0.50 + 0.06 * dir_x, 0.18)
        lm[:, f_kn, :2] = (0.50 + 0.10 * dir_x, 0.73)
        lm[:, f_ak, :2] = (0.50 + 0.14 * dir_x, 0.90)
        lm[:, f_hl, :2] = (0.50 + 0.13 * dir_x, 0.92)
        lm[:, f_to, :2] = (0.50 + 0.17 * dir_x, 0.92)
        lm[:, r_kn, :2] = (0.50 - 0.08 * dir_x, 0.73)
        lm[:, r_ak, :2] = (0.50 - 0.10 * dir_x, 0.90)
        lm[:, r_hl, :2] = (0.50 - 0.11 * dir_x, 0.92)
        lm[:, r_to, :2] = (0.50 - 0.07 * dir_x, 0.92)
    else:
        lm[:, NOSE, :2] = (0.50, 0.12)
        lm[:, MOUTH_L, :2] = (0.49, 0.14)
        lm[:, MOUTH_R, :2] = (0.51, 0.14)
        lm[:, L_SHOULDER, :2] = (0.40, 0.25)
        lm[:, R_SHOULDER, :2] = (0.60, 0.25)
        lm[:, L_HIP, :2] = (0.44, 0.55)
        lm[:, R_HIP, :2] = (0.56, 0.55)
        lm[:, f_el, :2] = (0.41, 0.30)
        lm[:, f_wr, :2] = (0.44, 0.18)
        lm[:, r_el, :2] = (0.59, 0.30)
        lm[:, r_wr, :2] = (0.56, 0.18)
        lm[:, [f_kn, r_kn], 1] = 0.73
        lm[:, [f_ak, r_ak], 1] = 0.90
        lm[:, [f_hl, r_hl], 1] = 0.92
        lm[:, [f_to, r_to], 1] = 0.92
        lm[:, f_kn, 0] = 0.43 if stance == "left" else 0.57
        lm[:, f_ak, 0] = 0.43 if stance == "left" else 0.57
        lm[:, f_hl, 0] = 0.42 if stance == "left" else 0.58
        lm[:, f_to, 0] = 0.44 if stance == "left" else 0.56
        lm[:, r_kn, 0] = 0.57 if stance == "left" else 0.43
        lm[:, r_ak, 0] = 0.57 if stance == "left" else 0.43
        lm[:, r_hl, 0] = 0.58 if stance == "left" else 0.42
        lm[:, r_to, 0] = 0.56 if stance == "left" else 0.44

    cursor = 25
    for rep in range(n_kicks):
        pad_start = cursor
        pad_end = pad_start + 14
        kick_start = pad_end + pause_frames
        kick_peak = kick_start + 10
        kick_end = kick_start + 22

        if view == "side":
            rear_start_x = 0.50 - 0.10 * dir_x
            rear_target_x = (
                max(
                    rear_start_x * dir_x + pad_disp,
                    (0.50 + 0.14 * dir_x) * dir_x + pad_overtake,
                )
                * dir_x
            )
            if pad_disp < 0.05:
                rear_target_x = rear_start_x + pad_disp * dir_x
            for i in range(pad_start, pad_end + 1):
                u = (i - pad_start) / max(1, pad_end - pad_start)
                lm[i, r_ak, 0] = rear_start_x + u * (rear_target_x - rear_start_x)
                lm[i, r_kn, 0] = lm[i, r_ak, 0] + 0.02 * dir_x
                lm[i, r_hl, 0] = lm[i, r_ak, 0] - 0.01 * dir_x
                lm[i, r_to, 0] = lm[i, r_ak, 0] + 0.03 * dir_x
            if pause_frames > 0:
                lm[pad_end + 1 : kick_start, r_ak, 0] = rear_target_x
                lm[pad_end + 1 : kick_start, r_kn, 0] = rear_target_x + 0.02 * dir_x
                lm[pad_end + 1 : kick_start, r_hl, 0] = rear_target_x - 0.01 * dir_x
                lm[pad_end + 1 : kick_start, r_to, 0] = rear_target_x + 0.03 * dir_x

            # 支撑脚外旋（toe 与 heel 重合略后）与踢腿抬膝伸腿
            lm[kick_start : kick_end + 1, r_to, 0] = (
                lm[kick_start, r_hl, 0] - 0.01 * dir_x
            )
            land_x = 0.50 + (0.24 if in_place_landing else 0.14) * dir_x
            for i in range(kick_start, kick_end + 1):
                if i <= kick_peak:
                    u = (i - kick_start) / max(1, kick_peak - kick_start)
                    ank_y = 0.90 - 0.25 * u
                    ank_x = 0.50 + (0.14 + 0.22 * u) * dir_x
                else:
                    u = (i - kick_peak) / max(1, kick_end - kick_peak)
                    ank_y = 0.65 + 0.25 * u
                    end_x = 0.50 + retract_offset * dir_x
                    if u < 0.7:
                        ank_x = (
                            0.50 + (0.36 - (u / 0.7) * (0.36 - retract_offset)) * dir_x
                        )
                    else:
                        ank_x = end_x if not in_place_landing else land_x
                lm[i, f_ak, :2] = (ank_x, ank_y)
                lm[i, f_kn, :2] = (ank_x - 0.08 * dir_x, 0.55 + (ank_y - 0.55) * 0.55)
                lm[i, f_to, :2] = (ank_x + 0.04 * dir_x, ank_y + 0.01)
                lm[i, f_hl, :2] = (ank_x - 0.02 * dir_x, ank_y + 0.02)

            if in_place_landing:
                lm[kick_end + 1 :, f_ak, :2] = (land_x, 0.90)
                lm[kick_end + 1 :, f_kn, :2] = (land_x - 0.02 * dir_x, 0.73)
                lm[kick_end + 1 :, f_to, :2] = (land_x + 0.03 * dir_x, 0.92)
                lm[kick_end + 1 :, f_hl, :2] = (land_x - 0.01 * dir_x, 0.92)

            if mask_ankle_on_first_peak and rep == 0:
                lm[kick_peak - 3 : kick_peak + 4, f_ak, 3] = 0.0

            if drop_guard_shoulder:
                lm[kick_start : kick_end + 1, r_wr, 1] = 0.44
                lm[kick_start : kick_end + 1, r_el, 1] = 0.36

        if not omit_arm_swing:
            swing_sh_x = lm[kick_start : kick_end + 1, f_sh, 0]
            lm[kick_start : kick_end + 1, f_el, 0] = swing_sh_x
            lm[kick_start : kick_end + 1, f_el, 1] = 0.37
            lm[kick_start : kick_end + 1, f_wr, 0] = swing_sh_x
            lm[kick_start : kick_end + 1, f_wr, 1] = 0.49

        cursor = kick_end + 25

    if missing_head_frames > 0:
        lm[:missing_head_frames, :, 3] = 0.0

    meta = {"width": 100, "height": 100, "fps": float(fps), "start_frame": 0}
    return lm, meta


def test_kick_catalog_and_aliases():
    assert action_definition("step_front_low_kick").segments == (
        "pad_step",
        "front_low_kick",
    )
    assert ACTION_SEGMENTS["step_front_low_kick"] == ("pad_step", "front_low_kick")
    assert action_definition("front_low_kick").segments == ("front_low_kick",)
    assert action_definition("rear_low_kick").segments == ("rear_low_kick",)
    for name in ("step_front_low_kick", "front_low_kick", "rear_low_kick"):
        assert action_definition(name).timing == "side_kick"
        assert not hasattr(action_definition(name), "stances")
    assert action_definition("step_front_low_kick").scoring == "per_repetition"
    assert action_definition("front_low_kick").scoring == "whole_clip"
    assert action_definition("rear_low_kick").scoring == "per_repetition"
    assert ACTION_BY_NAME["原地后低鞭腿"] == "rear_low_kick"
    assert ACTIONS["rear_low_kick"] == "后低鞭腿"
    assert SEGMENT_LABELS["pad_step"] == "后脚垫步"
    assert PHASES["retract"] == "收腿过程"
    assert kick_landmarks("arm_downward_extension", "front", "left", "front") == (
        11,
        13,
        15,
    )


def test_kick_segment_reuse_and_strategy_isolation():
    step_checks = [
        c
        for c in build_checks("step_front_low_kick", "left", pose_variant="heavy")
        if c["segment"] == "front_low_kick"
    ]
    single_checks = [
        c
        for c in build_checks("front_low_kick", "left", pose_variant="heavy")
        if c["segment"] == "front_low_kick"
    ]

    def _sig(items):
        return [
            (
                c["code"],
                c["phase"],
                c["bodyPart"],
                c["strategy"],
                c["availability"]["state"],
                c["blockedReason"],
                c.get("primaryView"),
            )
            for c in items
        ]

    assert step_checks
    assert _sig(step_checks) == _sig(single_checks)

    straight_elbow = [
        c
        for c in build_checks("front_straight", "left")
        if c["segment"] == "front_straight" and c["code"] == "guard_elbow"
    ]
    kick_elbow = [
        c
        for c in build_checks("front_low_kick", "left")
        if c["segment"] == "front_low_kick" and c["code"] == "guard_elbow"
    ]
    assert straight_elbow and all(
        c["strategy"] == "elbow_shoulder_height" for c in straight_elbow
    )
    assert kick_elbow and all(
        c["strategy"] == "upper_arm_torso_flare" for c in kick_elbow
    )


def test_kick_qualification_gating_and_tamper_resistance():
    lite_checks = [
        c
        for c in build_checks("step_front_low_kick", "left", pose_variant="lite")
        if c["segment"] in {"pad_step", "front_low_kick"}
    ]
    assert lite_checks
    assert all(
        c["availability"]["state"] == "validation_pending" and c["blockedReason"]
        for c in lite_checks
    )

    for variant in ("full", "heavy"):
        step_by = {
            (c["segment"], c["phase"], c["code"]): c
            for c in build_checks("step_front_low_kick", "left", pose_variant=variant)
            if c["segment"] in {"pad_step", "front_low_kick"}
        }
        rear_by = {
            (c["segment"], c["phase"], c["code"]): c
            for c in build_checks("rear_low_kick", "left", pose_variant=variant)
            if c["segment"] == "rear_low_kick"
        }
        # 绝不允许擅自以 trial 状态绕过 blockedReason
        assert all(
            c["availability"]["state"] != "trial"
            for c in (*step_by.values(), *rear_by.values())
        )
        for k in (
            ("pad_step", "prepare", "pad_step_missing"),
            ("pad_step", "prepare", "pad_step_overtake"),
            ("pad_step", "prepare", "pad_step_pause"),
            ("front_low_kick", "retract", "kick_retract"),
            ("front_low_kick", "motion", "arm_swing"),
            ("front_low_kick", "motion", "guard_shoulder"),
            ("front_low_kick", "motion", "guard_back"),
            ("front_low_kick", "motion", "head_down"),
        ):
            assert step_by[k]["availability"]["state"] == "enabled"
            assert step_by[k]["blockedReason"] == ""
        for k in (
            ("rear_low_kick", "motion", "arm_swing"),
            ("rear_low_kick", "finish", "arm_swing"),
            ("rear_low_kick", "motion", "guard_back"),
        ):
            assert rear_by[k]["availability"]["state"] == "enabled"
            assert rear_by[k]["blockedReason"] == ""

        # body_lean 按模型独立登记：heavy 启用，full 阻断
        expected_lean = "enabled" if variant == "heavy" else "calibration_failed"
        for phase in ("motion", "finish"):
            lean = step_by[("front_low_kick", phase, "body_lean")]
            assert lean["availability"]["state"] == expected_lean
            assert bool(lean["blockedReason"]) == (expected_lean != "enabled")

        # 缺正例或少于 2 人：validation_pending 并阻断
        for k in (("front_low_kick", "finish", "kick_height"),):
            assert step_by[k]["availability"]["state"] == "validation_pending"
            assert step_by[k]["blockedReason"]
        for k in (
            ("rear_low_kick", "finish", "kick_height"),
            ("rear_low_kick", "motion", "guard_shoulder"),
            ("rear_low_kick", "finish", "guard_shoulder"),
            ("rear_low_kick", "motion", "body_lean"),
            ("rear_low_kick", "finish", "body_lean"),
        ):
            assert rear_by[k]["availability"]["state"] == "validation_pending"
            assert rear_by[k]["blockedReason"]

        # 实证未达标（高误报 / 总检出不足 70%）：calibration_failed 并阻断
        for phase, code in (
            ("finish", "head_down"),
            ("finish", "pivot_heel"),
            ("finish", "foot_flexed"),
            ("finish", "hip_turn"),
            ("finish", "guard_elbow"),
        ):
            assert (
                step_by[("front_low_kick", phase, code)]["availability"]["state"]
                == "calibration_failed"
            )
            assert step_by[("front_low_kick", phase, code)]["blockedReason"]
        for phase, code in (
            ("finish", "guard_back"),
            ("motion", "head_down"),
            ("finish", "head_down"),
            ("finish", "pivot_heel"),
            ("finish", "foot_flexed"),
            ("finish", "hip_turn"),
            ("finish", "guard_elbow"),
            ("finish", "guard_jaw"),
        ):
            assert (
                rear_by[("rear_low_kick", phase, code)]["availability"]["state"]
                == "calibration_failed"
            )
            assert rear_by[("rear_low_kick", phase, code)]["blockedReason"]

    tampered = copy.deepcopy(step_by[("front_low_kick", "retract", "kick_retract")])
    tampered["strategy"] = "forged_strategy"
    apply_kick_availability([tampered], "heavy")
    assert tampered["availability"]["state"] == "validation_pending"
    assert tampered["blockedReason"]

    wrong_phase = copy.deepcopy(step_by[("pad_step", "prepare", "pad_step_missing")])
    wrong_phase["phase"] = "finish"
    apply_kick_availability([wrong_phase], "heavy")
    assert wrong_phase["availability"]["state"] == "validation_pending"


def test_kick_side_repetition_splitting_and_exam_pool_compat():
    for stance in ("left", "right"):
        lm_side, meta = _make_kick_sequence(stance=stance, view="side", n_kicks=2)
        reps = kick_repetitions(
            series_from_raw(lm_side, meta), "step_front_low_kick", stance
        )
        assert len(reps) == 2

    lm_single, meta = _make_kick_sequence(stance="left", view="side", n_kicks=1)
    assert (
        len(
            kick_repetitions(
                series_from_raw(lm_single, meta), "step_front_low_kick", "left"
            )
        )
        == 1
    )

    lm_head, meta = _make_kick_sequence(
        stance="left", view="side", n_kicks=2, missing_head_frames=12
    )
    assert (
        len(
            kick_repetitions(
                series_from_raw(lm_head, meta), "step_front_low_kick", "left"
            )
        )
        == 2
    )

    lm_masked, meta = _make_kick_sequence(
        stance="left", view="side", n_kicks=2, mask_ankle_on_first_peak=True
    )
    assert (
        len(
            kick_repetitions(
                series_from_raw(lm_masked, meta), "step_front_low_kick", "left"
            )
        )
        == 2
    )

    lm_inplace, meta = _make_kick_sequence(
        stance="left", view="side", n_kicks=2, in_place_landing=True
    )
    events = segment_kick_events(
        series_from_raw(lm_inplace, meta), "step_front_low_kick", "left"
    )
    assert len(events) == 2
    assert all(ev["closed"] for ev in events)

    lm_exam, meta_exam = _make_kick_sequence(stance="left", view="side", n_kicks=2)
    raw_series = Pose33RawSeries(lm_exam, meta_exam)
    cfg = FeedbackConfig(pose_variant="heavy")
    # 兼容原有双参数 side_repetitions 与统一入口 action_side_repetitions
    assert exam_pool.side_repetitions(raw_series, cfg) == []
    assert (
        len(exam_pool.action_side_repetitions(raw_series, "step_front_low_kick", cfg))
        == 2
    )
    assert exam_pool.action_side_repetitions(raw_series, "front_low_kick", cfg) == []


def test_kick_geometry_thresholds_and_anti_swap_masking():
    n = 15
    lm = np.zeros((n, 33, 4), dtype=np.float32)
    lm[:, :, 3] = 0.95
    lm[:, NOSE, :2] = (0.54, 0.12)
    lm[:, L_SHOULDER, :2] = (0.50, 0.25)
    lm[:, R_SHOULDER, :2] = (0.50, 0.25)
    lm[:, L_HIP, :2] = (0.50, 0.55)
    lm[:, R_HIP, :2] = (0.50, 0.55)  # torso = 0.30
    lm[:, 25, :2] = (0.55, 0.75)  # front knee
    lm[:, 26, :2] = (0.45, 0.75)  # rear knee
    lm[:, 27, :2] = (0.60, 0.90)  # front ankle
    lm[:, 28, :2] = (0.40, 0.90)  # rear ankle
    mask = np.ones((n, 33), dtype=bool)
    times = np.arange(n, dtype=float) / 30.0

    # pad_step_missing：0.05/0.30=0.17 < 0.45 违规；0.18/0.30=0.60 通过
    lm_small = lm.copy()
    lm_small[:, 28, 0] = np.linspace(0.40, 0.45, n)
    bad, valid, _ = measure_pad_step_missing(lm_small, mask, times, "left", facing=1.0)
    assert valid and bad
    lm_good = lm.copy()
    lm_good[:, 28, 0] = np.linspace(0.40, 0.58, n)
    bad, valid, _ = measure_pad_step_missing(lm_good, mask, times, "left", facing=1.0)
    assert valid and not bad

    # pad_step_overtake：允许两脚并齐(=0.0)，>0.20 判超越
    lm_flush = lm.copy()
    lm_flush[:, 28, 0] = 0.60
    bad, valid, _ = measure_pad_step_overtake(lm_flush, mask, times, "left", facing=1.0)
    assert valid and not bad
    lm_over = lm.copy()
    lm_over[:, 28, 0] = 0.70  # (0.70 - 0.60) / 0.30 = 0.33 > 0.20
    bad, valid, _ = measure_pad_step_overtake(lm_over, mask, times, "left", facing=1.0)
    assert valid and bad

    # pad_step_pause：落稳停顿 > 0.60s 违规
    t_long = np.linspace(0.0, 1.2, 36)
    lm_pause = np.repeat(lm[:1], 36, axis=0)
    lm_pause[:8, 28, 0] = np.linspace(0.40, 0.58, 8)
    lm_pause[8:, 28, 0] = 0.58
    bad, valid, _ = measure_pad_step_pause(
        lm_pause, np.ones((36, 33), dtype=bool), t_long, "left", facing=1.0
    )
    assert valid and bad

    # kick_height：高过同侧髋 (< -0.06) 或明显低于支撑膝 (> +0.08) 均违规
    torso = np.full(n, 0.30)
    lm_h = lm.copy()
    lm_h[:, 31, 1] = 0.66  # 处于髋 0.55 与支撑膝 0.75 之间
    bad_arr, val_arr, _ = measure_kick_height(lm_h, mask, torso, 31, 23, 26)
    assert val_arr.all() and not bad_arr.any()
    lm_h[:, 31, 1] = 0.50  # (0.50 - 0.55)/0.30 = -0.167 < -0.06
    assert measure_kick_height(lm_h, mask, torso, 31, 23, 26)[0].all()
    lm_h[:, 31, 1] = 0.80  # (0.80 - 0.75)/0.30 = +0.167 > +0.08
    assert measure_kick_height(lm_h, mask, torso, 31, 23, 26)[0].all()

    # kick_retract：落地前未收脚 > 0.40 违规
    lm_ret = lm.copy()
    lm_ret[:, 27, 0] = 0.70  # (0.70 - 0.50)/0.30 = 0.67 > 0.40
    bad_r, val_r, _ = measure_kick_retract_segment(lm_ret, mask, torso, 27, 23, 1.0)
    assert val_r and bad_r

    # 锁定角度：foot_flexed < 165°、body_lean > 35°、guard_elbow > 30°
    lm_ang = lm.copy()
    lm_ang[:, 25, :2] = (0.0, 0.0)
    lm_ang[:, 27, :2] = (0.3, 0.0)
    lm_ang[:, 31, :2] = (0.59, 0.05)  # 约 170°
    assert not measure_foot_flexed(lm_ang, mask, torso, 25, 27, 31)[0].any()
    lm_ang[:, 31, :2] = (0.50, 0.20)  # 135° < 165°
    assert measure_foot_flexed(lm_ang, mask, torso, 25, 27, 31)[0].all()

    lm_lean = lm.copy()
    lm_lean[:, L_SHOULDER, :2] = (0.40, 0.25)
    lm_lean[:, R_SHOULDER, :2] = (0.60, 0.43)  # atan2(0.18, 0.20) ~ 42° > 35°
    assert measure_body_lean(lm_lean, mask)[0].all()
    lm_lean[:, R_SHOULDER, :2] = (0.60, 0.30)  # ~ 14° <= 35°
    assert not measure_body_lean(lm_lean, mask)[0].any()

    lm_el = lm.copy()
    lm_el[:, 12, :2] = (0.60, 0.25)  # R_SHOULDER
    lm_el[:, 24, :2] = (0.60, 0.55)  # R_HIP
    lm_el[:, 14, :2] = (0.78, 0.45)  # R_ELBOW (~42° > 30°)
    assert measure_kick_guard_elbow(lm_el, mask, torso, 14, 12, 24)[0].all()
    lm_el[:, 14, :2] = (0.65, 0.45)  # ~14° <= 30°
    assert not measure_kick_guard_elbow(lm_el, mask, torso, 14, 12, 24)[0].any()

    # 抗侧视互换掩码：支撑脚踝显著高于踢腿脚踝 (>0.15*torso) 时清除 23..32 关键点
    lm_sw = lm.copy()
    lm_sw[:, 28, 1] = 0.60
    lm_sw[:, 27, 1] = 0.85
    safe = kicking_leg_side_safe(lm_sw, mask, 27, 28, torso)
    assert not safe[:, 23:33].any()

    # 半数帧缺有效点时，按双侧均有效帧统计互换比例（4/4=100% > 0.5）
    lm_sw2 = np.zeros((8, 33, 4), dtype=np.float32)
    lm_sw2[:, 28, 1] = 0.60
    lm_sw2[:, 27, 1] = 0.85
    mask_sw2 = np.ones((8, 33), dtype=bool)
    mask_sw2[4:, 28] = False
    safe2 = kicking_leg_side_safe(
        lm_sw2, mask_sw2, 27, 28, np.full(8, 0.30), majority_threshold=0.5
    )
    assert not safe2[:, 23:33].any()


def test_end_to_end_kick_evaluation_defects_unable_and_issue_grouping():
    cfg = FeedbackConfig(pose_variant="heavy")
    lm_f_ok, mf = _make_kick_sequence(stance="left", view="front", n_kicks=2)
    lm_s_ok, ms = _make_kick_sequence(stance="left", view="side", n_kicks=2)
    ok_res = evaluate_series(
        "step_front_low_kick",
        "left",
        {"front": series_from_raw(lm_f_ok, mf), "side": series_from_raw(lm_s_ok, ms)},
        cfg,
    )
    assert ok_res["repetitions"]["detected"] == 2
    assert not ok_res.get("incompleteRecovery")
    assert not [
        c
        for c in ok_res["checks"]
        if c["segment"] in {"pad_step", "front_low_kick"} and c["status"] == "candidate"
    ]
    failed_checks = [
        c
        for c in ok_res["checks"]
        if c["code"] in {"pivot_heel", "foot_flexed", "hip_turn", "guard_elbow"}
        and c["segment"] == "front_low_kick"
    ]
    assert failed_checks and all(c["status"] == "pending_rule" for c in failed_checks)

    # 注入四类实测缺陷：无垫步、收腿不足、摆臂缺位、掉后护手
    lm_f_bad, mf_b = _make_kick_sequence(
        stance="left", view="front", n_kicks=2, omit_arm_swing=True
    )
    lm_s_bad, ms_b = _make_kick_sequence(
        stance="left",
        view="side",
        n_kicks=2,
        pad_disp=0.02,
        retract_offset=0.22,
        drop_guard_shoulder=True,
        omit_arm_swing=True,
    )
    views_bad = {
        "front": series_from_raw(lm_f_bad, mf_b),
        "side": series_from_raw(lm_s_bad, ms_b),
    }
    bad_res = evaluate_series("step_front_low_kick", "left", views_bad, cfg)
    cands = {c["code"] for c in bad_res["checks"] if c["status"] == "candidate"}
    assert {"pad_step_missing", "kick_retract", "arm_swing", "guard_shoulder"} <= cands

    # 考试/批量链路同样检出且不可用帧输出 unable（非 not_observed）
    jv = judge_views(views_bad, "step_front_low_kick", "left", cfg)
    side_states = {c.code: c.state for c in jv["side"][0]}
    assert side_states["pad_step_missing"] == "violation"
    assert side_states["kick_retract"] == "violation"

    # 分组逻辑应将 front_low_kick 的 motion/finish guard_shoulder 合并为 1 条
    rec = {
        "schemaVersion": 1,
        "id": "b" * 32,
        "studentId": "001",
        "studentName": "测试",
        "action": "step_front_low_kick",
        "actionLabel": ACTIONS["step_front_low_kick"],
        "stance": "left",
        "stanceLabel": "左式",
        "createdAtUtc": "2026-11-14T00:00:00Z",
        "createdAtLocal": "2026-11-14 08:00:00",
        "revision": 1,
        "result": bad_res,
    }
    model = build_report_model(rec)
    grouped = group_action_issues(model.candidate_checks, "step_front_low_kick")
    gs_groups = [c for c in grouped if c.code == "guard_shoulder"]
    assert len(gs_groups) == 1 and len(gs_groups[0].members) == 2

    # 遮挡护手关键点时应返回 unable 而非静默通过
    lm_f_occ, _ = _make_kick_sequence(stance="left", view="front", n_kicks=2)
    lm_s_occ, _ = _make_kick_sequence(stance="left", view="side", n_kicks=2)
    lm_s_occ[:, 16, 3] = 0.0  # 遮挡后手腕 (R_WRIST)
    occ_res = evaluate_series(
        "step_front_low_kick",
        "left",
        {
            "front": series_from_raw(lm_f_occ, mf),
            "side": series_from_raw(lm_s_occ, ms),
        },
        cfg,
    )
    gs_occ = [
        c
        for c in occ_res["checks"]
        if c["segment"] == "front_low_kick" and c["code"] == "guard_shoulder"
    ]
    assert gs_occ and all(c["status"] == "unable" for c in gs_occ)


def test_kick_remediation_regression_cases():
    """复核整改回归：3点摆臂遮挡、完整稳定实战式基线、手放下零误触发、统一前低鞭腿与缺侧视提示。"""
    cfg = FeedbackConfig(pose_variant="heavy")

    # 1. arm_swing 需肩-肘-腕 3 点齐全：遮挡肘点（即使手腕下垂）必须判为 unable 而非静默通过
    lm_f_arm, mf = _make_kick_sequence(
        stance="left", view="front", n_kicks=2, omit_arm_swing=True
    )
    lm_s_arm, ms = _make_kick_sequence(
        stance="left", view="side", n_kicks=2, omit_arm_swing=True
    )
    lm_f_arm[:, 15, 1] = 0.40
    lm_f_arm[:, 13, 3] = 0.0
    lm_s_arm[:, 13, 3] = 0.0
    arm_res = evaluate_series(
        "step_front_low_kick",
        "left",
        {
            "front": series_from_raw(lm_f_arm, mf),
            "side": series_from_raw(lm_s_arm, ms),
        },
        cfg,
    )
    arm_occ = [
        c
        for c in arm_res["checks"]
        if c["segment"] == "front_low_kick" and c["code"] == "arm_swing"
    ]
    assert arm_occ and all(c["status"] == "unable" for c in arm_occ)

    # 2. 实战式起止窗取完整 settled_stance_windows 边界（不截断碎窗），且首尾手放下时不兜底误判为 candidate
    lm_f_ok, mf = _make_kick_sequence(stance="left", view="front", n_kicks=2)
    lm_s_ok, ms = _make_kick_sequence(stance="left", view="side", n_kicks=2)
    views_ok = {
        "front": series_from_raw(lm_f_ok, mf),
        "side": series_from_raw(lm_s_ok, ms),
    }
    wins, _, _, _ = shared_timeline(views_ok, "step_front_low_kick", "left", cfg)
    edge = math.ceil(cfg.stance_window_seconds * 30)
    assert len(wins["side"][("stance", "start")]) >= edge
    assert len(wins["front"][("stance", "start")]) >= edge

    lm_f_hd = lm_f_ok.copy()
    lm_s_hd = lm_s_ok.copy()
    for arr in (lm_f_hd, lm_s_hd):
        arr[:41, [15, 16], 1] = 0.58
        arr[122:, [15, 16], 1] = 0.58
    hd_res = evaluate_series(
        "step_front_low_kick",
        "left",
        {
            "front": series_from_raw(lm_f_hd, mf),
            "side": series_from_raw(lm_s_hd, ms),
        },
        cfg,
    )
    assert not [
        c
        for c in hd_res["checks"]
        if c["segment"] == "stance" and c["status"] == "candidate"
    ]

    # 3. 单独 front_low_kick 统一走 side_kick 标定阶段（无 repetitions 分段且结论与 step_front_low_kick 一致）
    lm_f1, mf1 = _make_kick_sequence(
        stance="left", view="front", n_kicks=1, pad_disp=0.0, in_place_landing=True
    )
    lm_s1, ms1 = _make_kick_sequence(
        stance="left", view="side", n_kicks=1, pad_disp=0.0, in_place_landing=True
    )
    views_1 = {
        "front": series_from_raw(lm_f1, mf1),
        "side": series_from_raw(lm_s1, ms1),
    }
    single_res = evaluate_series("front_low_kick", "left", views_1, cfg)
    step_res = evaluate_series("step_front_low_kick", "left", views_1, cfg)
    assert "repetitions" not in single_res
    assert "repetitions" in step_res
    single_map = {
        c["id"]: (c["status"], c.get("blockedReason", ""), c["availability"]["state"])
        for c in single_res["checks"]
        if c["segment"] == "front_low_kick"
    }
    step_map = {
        c["id"]: (c["status"], c.get("blockedReason", ""), c["availability"]["state"])
        for c in step_res["checks"]
        if c["segment"] == "front_low_kick"
    }
    assert single_map and single_map == step_map

    # 缺侧面视频提示按是否含上步区分 NEEDS_KICK_SIDE 与 NEEDS_SIDE
    front_only = {"front": series_from_raw(lm_f_ok, mf)}
    assert (
        evaluate_series("rear_low_kick", "left", front_only, cfg)["diagnostics"][
            "front"
        ]["reason"]
        == NEEDS_KICK_SIDE
    )
    assert (
        evaluate_series("front_low_kick", "left", front_only, cfg)["diagnostics"][
            "front"
        ]["reason"]
        == NEEDS_KICK_SIDE
    )
    assert (
        evaluate_series("step_front_low_kick", "left", front_only, cfg)["diagnostics"][
            "front"
        ]["reason"]
        == NEEDS_SIDE
    )
