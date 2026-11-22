# -*- coding: utf-8 -*-
"""摆拳规则（C61–C68）、摆拳事件切分与公共步法/姿态/衔接规则（C22–C25、C40–C48）契约与边界测试。"""

from __future__ import annotations

import math
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from core.evaluation import (
    ACTIONS,
    FeedbackConfig,
    SEGMENT_LABELS,
    build_checks,
    evaluate_series,
    judge_views,
    rule_profile,
    rule_series,
)
from core.evaluation.action_catalog import DERIVED_SEGMENTS, action_definition, motion_segments
from core.evaluation.body_sides import sides_for_stance
from core.evaluation.hook_events import hook_reference, observe_hook
from core.evaluation.measures.feet import (
    measure_foot_forward_displacement,
    measure_punch_heel_pivot,
    measure_rear_foot_overtake,
    measure_step_feet_together,
    measure_step_landing_lag,
)
from core.evaluation.measures.hooks import (
    measure_hook_arm_bend,
    measure_hook_forearm_level,
    measure_hook_over_center,
)
from core.evaluation.measures.motion import measure_punch_post_peak_hold
from core.evaluation.measures.posture import measure_hip_turn_width, measure_trunk_lean_sagittal
from core.evaluation.measures.torso import chain_torso
from core.evaluation.motion_events import MotionEvent
from core.evaluation.paired_torso import FRONTAL_PROJECTION_RATIO, paired_torso_measurements
from core.evaluation.punch_events import preceding_punch
from core.evaluation.rules.engine import evaluate_rules
from core.evaluation.rules.spec import RuleContext
from core.evaluation.scoring import exam_pool
from core.evaluation.step_reps import step_combo_windows
from core.pose.landmarks import (
    L_HEEL,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    MOUTH_L,
    MOUTH_R,
    NOSE,
    R_HEEL,
    R_HIP,
    R_SHOULDER,
)
from core.pose.raw_series import Pose33RawSeries
from core.reporting import build_report_model, group_action_issues

FPS = 30.0


def _stance_pose(stance: str = "left", view: str = "side") -> np.ndarray:
    """构造可稳定识别起始护位与左右式对称的基准姿态（躯干长 = 0.30）。"""
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[:, :2], pose[:, 3] = (0.50, 0.50), 0.95
    dir_x = 1.0 if stance == "left" else -1.0
    f_sh, f_el, f_wr, f_hp, f_kn, f_ak, f_hl, f_to = sides_for_stance("front", stance)
    r_sh, r_el, r_wr, r_hp, r_kn, r_ak, r_hl, r_to = sides_for_stance("rear", stance)
    if view == "side":
        pose[[NOSE, MOUTH_L, MOUTH_R], :2] = [
            (0.50 + 0.05 * dir_x, 0.16),
            (0.50 + 0.04 * dir_x, 0.18),
            (0.50 + 0.04 * dir_x, 0.18),
        ]
        pose[[f_sh, r_sh, f_hp, r_hp], :2] = [
            (0.50 + 0.03 * dir_x, 0.25),
            (0.50 - 0.03 * dir_x, 0.25),
            (0.50 + 0.03 * dir_x, 0.55),
            (0.50 - 0.03 * dir_x, 0.55),
        ]
        pose[[f_el, f_wr, r_el, r_wr], :2] = [
            (0.50 + 0.06 * dir_x, 0.38),
            (0.50 + 0.08 * dir_x, 0.22),
            (0.50 - 0.01 * dir_x, 0.38),
            (0.50 + 0.02 * dir_x, 0.22),
        ]
        pose[[f_kn, r_kn, f_ak, r_ak], :2] = [
            (0.50 + 0.10 * dir_x, 0.72),
            (0.50 - 0.08 * dir_x, 0.72),
            (0.50 + 0.12 * dir_x, 0.88),
            (0.50 - 0.10 * dir_x, 0.88),
        ]
        pose[[f_hl, f_to, r_hl, r_to], :2] = [
            (0.50 + 0.10 * dir_x, 0.90),
            (0.50 + 0.15 * dir_x, 0.90),
            (0.50 - 0.12 * dir_x, 0.90),
            (0.50 - 0.07 * dir_x, 0.90),
        ]
    else:
        pose[[NOSE, MOUTH_L, MOUTH_R], :2] = [(0.50, 0.16), (0.49, 0.18), (0.51, 0.18)]
        pose[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP], :2] = [
            (0.55, 0.25),
            (0.45, 0.25),
            (0.54, 0.55),
            (0.46, 0.55),
        ]
        pose[[f_el, r_el], :2] = [(pose[f_sh, 0], 0.38), (pose[r_sh, 0], 0.38)]
        pose[f_wr, :2] = (0.52 if stance == "left" else 0.48, 0.20)
        pose[r_wr, :2] = (0.48 if stance == "left" else 0.52, 0.20)
        pose[[f_kn, f_ak, f_hl, f_to], 0] = pose[f_hp, 0]
        pose[[r_kn, r_ak, r_hl, r_to], 0] = pose[r_hp, 0]
        pose[[f_kn, r_kn], 1] = 0.72
        pose[[f_ak, r_ak], 1] = 0.88
        pose[[f_hl, r_hl], 1] = 0.90
        pose[[f_to, r_to], 1] = 0.92
    return pose


def _make_step_hook_views(
    *,
    stance: str = "left",
    reps: int = 2,
    seconds: float = 6.5,
    combo_gap_s: float = 0.08,
    front_overlap: bool = False,
    rear_follow_disp: float = 0.12,
    rear_overtake_offset: float = -0.06,
    rear_land_lag_s: float = 0.0,
    feet_together: bool = False,
    hold_last_rear: bool = False,
    delayed_drop_rear: bool = False,
) -> dict[str, dict]:
    """构造同步正/侧双视角上步前后手摆拳组合序列，支持按场景注入步法与衔接缺陷。"""
    n = int(round(seconds * FPS))
    times = np.arange(n, dtype=float) / FPS
    dir_x = 1.0 if stance == "left" else -1.0
    f_sh, f_el, f_wr, _, _, f_ak, f_hl, f_to = sides_for_stance("front", stance)
    r_sh, r_el, r_wr, _, _, r_ak, r_hl, r_to = sides_for_stance("rear", stance)

    side_lm = np.repeat(_stance_pose(stance, "side")[None], n, axis=0)
    front_lm = np.repeat(_stance_pose(stance, "front")[None], n, axis=0)
    base_side = side_lm[0].copy()

    for k in range(reps):
        t0 = 1.2 + 2.2 * k
        f_start, f_end = t0, t0 + 0.32
        r_start = (f_end - 0.08) if front_overlap else (f_end + combo_gap_s)
        r_end = r_start + (0.80 if (delayed_drop_rear and k == reps - 1) else 0.36)
        f_peak_t, r_peak_t = 0.5 * (f_start + f_end), r_start + 0.18

        for i in np.flatnonzero((times >= f_start) & (times <= f_end)):
            u = math.sin(math.pi * (times[i] - f_start) / max(f_end - f_start, 1e-4))
            side_lm[i, f_el, :2] = (base_side[f_el, 0] + 0.14 * dir_x * u, 0.26)
            side_lm[i, f_wr, :2] = (base_side[f_wr, 0] + 0.12 * dir_x * u, 0.25)
            side_lm[i, f_hl, 1] = base_side[f_hl, 1] - 0.05 * u
            front_lm[i, [L_HIP, R_HIP], 0] = (0.52, 0.48)
            front_lm[i, [f_el, f_wr], :2] = [(front_lm[0, f_sh, 0], 0.25), (0.50, 0.25)]

        is_held = hold_last_rear and k == reps - 1
        rear_mask = (times >= r_start) if is_held else ((times >= r_start) & (times <= r_end))
        for i in np.flatnonzero(rear_mask):
            if is_held:
                u = min(1.0, (times[i] - r_start) / 0.18)
            elif delayed_drop_rear and k == reps - 1:
                if times[i] <= r_peak_t:
                    u = (times[i] - r_start) / max(r_peak_t - r_start, 1e-4)
                elif times[i] <= r_end - 0.08:
                    u = 1.0
                else:
                    u = max(0.0, (r_end - times[i]) / 0.08)
            else:
                u = math.sin(math.pi * (times[i] - r_start) / max(r_end - r_start, 1e-4))
            side_lm[i, r_el, :2] = (base_side[r_el, 0] + 0.18 * dir_x * u, 0.26)
            side_lm[i, r_wr, :2] = (base_side[r_wr, 0] + 0.16 * dir_x * u, 0.25)
            side_lm[i, r_hl, 1] = base_side[r_hl, 1] - 0.06 * u
            side_lm[i, [L_HIP, R_HIP], 0] = (0.50 + 0.005 * dir_x, 0.50 - 0.005 * dir_x)
            front_lm[i, [L_HIP, R_HIP], 0] = (0.55, 0.45)
            front_lm[i, [r_el, r_wr], :2] = [(front_lm[0, r_sh, 0], 0.25), (0.50, 0.25)]

        step_lo, step_hi = t0 - 0.25, min(seconds - 0.1, r_peak_t + 0.55)
        r_step_start, r_step_end = r_start - 0.10, r_peak_t + rear_land_lag_s
        for i in np.flatnonzero((times >= step_lo) & (times <= step_hi)):
            t = times[i]
            fu = float(np.clip((t - step_lo) / max(f_peak_t - step_lo, 1e-4), 0.0, 1.0))
            ru = float(np.clip((t - r_step_start) / max(r_step_end - r_step_start, 1e-4), 0.0, 1.0))
            f_shift = 0.08 * dir_x * fu
            if rear_overtake_offset > 0.0:
                target_r = base_side[f_ak, 0] + f_shift + rear_overtake_offset * dir_x
                r_x = base_side[r_ak, 0] + (target_r - base_side[r_ak, 0]) * ru
            else:
                r_x = base_side[r_ak, 0] + rear_follow_disp * dir_x * ru
            side_lm[i, [f_ak, r_ak], 0] = (base_side[f_ak, 0] + f_shift, r_x)
            if feet_together:
                side_lm[i, [f_hl, r_hl], 0] = (0.51, 0.49)
            else:
                side_lm[i, f_hl, 0] = side_lm[i, f_ak, 0] - 0.02 * dir_x
                side_lm[i, r_hl, 0] = side_lm[i, r_ak, 0] - 0.02 * dir_x
            side_lm[i, [f_to, r_to], 0] = side_lm[i, [f_hl, r_hl], 0] + 0.05 * dir_x

    return {
        "front": rule_series(front_lm, range(n), FPS, 1.0),
        "side": rule_series(side_lm, range(n), FPS, 1.0),
    }


stance_pose = _stance_pose
make_step_hook_views = _make_step_hook_views


@pytest.mark.parametrize("stance,facing", [("left", 1.0), ("right", -1.0)])
def test_hook_arm_bend_distinguishes_tight_normal_and_straight(stance, facing):
    lm = np.repeat(_stance_pose(stance, "side")[None], 4, axis=0)
    sh, el, wr, *_ = sides_for_stance("front", stance)
    lm[:, sh, :2] = (0.50, 0.25)
    lm[:, el, :2] = (0.50 + 0.20 * facing, 0.25)  # 大臂长 = 0.20
    # 0: 正90°投影0(过); 1: 腕在肘后0.60>0.50(紧弯); 2: 腕在肘前0.60<=0.70(过); 3: 腕在肘前0.80>0.70(直臂)
    for idx, off in enumerate((0.20, 0.08, 0.32, 0.36)):
        lm[idx, wr, :2] = (0.50 + off * facing, 0.25)
    mask = np.ones((4, 33), dtype=bool)
    torso = chain_torso(lm, mask)

    bad, valid, _ = measure_hook_arm_bend(lm, mask, torso, sh, el, wr, facing)
    assert valid.all() and bad.tolist() == [False, True, False, True]

    mask[1, el] = False
    _, valid_miss, _ = measure_hook_arm_bend(lm, mask, torso, sh, el, wr, facing)
    assert valid_miss.tolist() == [True, False, True, True]


def test_hook_forearm_level_and_over_center_symmetries_and_bounds():
    lm = np.repeat(_stance_pose("left", "front")[None], 4, axis=0)
    mask = np.ones((4, 33), dtype=bool)
    lm[:, 13, :2] = (0.60, 0.25)
    for i, y in enumerate((0.25, 0.28, 0.30, 0.20)):  # 0°, 8.53°, 14.04°(下), 14.04°(上)
        lm[i, 15, :2] = (0.40, y)
    bad, valid, _ = measure_hook_forearm_level(lm, mask, 13, 15)
    assert valid.all() and bad.tolist() == [False, False, True, True]

    for att_x, opp_x in ((0.60, 0.40), (0.40, 0.60)):
        lm_c = np.repeat(_stance_pose("left", "front")[None], 4, axis=0)
        lm_c[:, [L_SHOULDER, R_SHOULDER], :2] = [(att_x, 0.25), (opp_x, 0.25)]
        inward = 1.0 if opp_x > att_x else -1.0
        lm_c[:, L_WRIST, 0] = 0.50 + np.array([-0.08, 0.0, 0.05, 0.07]) * inward
        bad_c, val_c, _ = measure_hook_over_center(lm_c, mask, L_WRIST, L_SHOULDER, R_SHOULDER)
        assert val_c.all() and bad_c.tolist() == [False, False, False, True]


def test_hook_events_robust_baseline_held_and_peak_refinement():
    cfg, n = FeedbackConfig(), 90
    times, baseline = np.arange(n, dtype=float) / FPS, np.arange(10)
    lm_f = np.repeat(_stance_pose("left", "front")[None], n, axis=0)
    lm_f[:, [L_SHOULDER, R_SHOULDER, L_WRIST], :2] = [(0.60, 0.25), (0.40, 0.25), (0.58, 0.22)]
    mask_f = np.ones((n, 33), dtype=bool)
    mask_f[2, L_WRIST] = False  # 缺 1 帧基线仍可识别
    for t in range(25, 50):
        lm_f[t, L_WRIST, 0] = 0.58 - 0.22 * math.sin(math.pi * (t - 25) / 24)
    lm_f[37, L_WRIST, 0] = 0.34  # C66 摆过头最远越线顶点
    torso_f = chain_torso(lm_f, mask_f)
    ref_f = hook_reference(lm_f, mask_f, torso_f, baseline, "front", "left", "front", cfg)
    assert ref_f.reason == "" and ref_f.motion_frames == 9
    obs_f = observe_hook(
        lm_f, mask_f, times, torso_f, baseline, [baseline], "front", "left", cfg, "front"
    )
    assert len(obs_f.events) == 1 and obs_f.events[0].peak == 37

    lm_s = np.repeat(_stance_pose("left", "side")[None], n, axis=0)
    mask_s = np.ones((n, 33), dtype=bool)
    lm_s[25:, [13, 15], :2] = [(0.66, 0.26), (0.74, 0.25)]
    lm_s[30, [13, 15], :2] = [(0.72, 0.26), (0.73, 0.25)]  # 弧段内小臂投影最短帧
    torso_s = chain_torso(lm_s, mask_s)
    obs_s = observe_hook(
        lm_s, mask_s, times, torso_s, baseline, [baseline], "front", "left", cfg, "side"
    )
    assert (
        len(obs_s.events) == 1 and obs_s.events[0].ending == "held" and obs_s.events[0].peak == 30
    )


def test_trunk_lean_sagittal_and_punch_heel_lift_physical_directions():
    lm = np.repeat(_stance_pose("left", "side")[None], 4, axis=0)
    mask = np.ones((4, 33), dtype=bool)
    for i, deg in enumerate((0.0, 8.0, 15.0, -15.0)):
        rad = math.radians(deg)
        lm[i, [L_HIP, R_HIP], :2] = (0.50, 0.60)
        lm[i, [L_SHOULDER, R_SHOULDER], :2] = (
            0.50 + 0.30 * math.sin(rad),
            0.60 - 0.30 * math.cos(rad),
        )
    bad, valid, _ = measure_trunk_lean_sagittal(lm, mask, max_angle=10.0)
    assert valid.all() and bad.tolist() == [False, False, True, True]

    base = np.repeat(_stance_pose("left", "side")[None], 5, axis=0)
    base_mask = np.ones((5, 33), dtype=bool)
    base_torso = chain_torso(base, base_mask)
    cur = base[:3].copy()
    cur[0, R_HEEL, 1] = cur[0, 32, 1]
    cur[1, R_HEEL, 1] = cur[1, 32, 1] - 0.012  # 0.04 < 0.05
    cur[2, R_HEEL, 1] = cur[2, 32, 1] - 0.036  # 0.12 >= 0.05
    cur_mask = np.ones((3, 33), dtype=bool)
    bad_h, val_h, _ = measure_punch_heel_pivot(
        cur, cur_mask, chain_torso(cur, cur_mask), base, base_mask, base_torso, R_HEEL, 32
    )
    assert val_h.all() and bad_h.tolist() == [True, True, False]

    cur[2, R_HEEL, :2] = cur[2, 32, :2]  # 零长脚部须判无效
    _, val_zero, _ = measure_punch_heel_pivot(
        cur, cur_mask, chain_torso(cur, cur_mask), base, base_mask, base_torso, R_HEEL, 32
    )
    assert val_zero.tolist() == [True, True, False]


def test_rear_punch_hip_turn_uses_side_view_without_front_false_positive():
    cfg = FeedbackConfig()
    profile = rule_profile("rear_straight", "left", cfg)
    rear_hip = next(r for r in profile.rules if r.code == "rear_straight:finish:rear:hip_turn")
    front_hip = next(
        r
        for r in rule_profile("front_straight", "left", cfg).rules
        if r.code == "front_straight:finish:front:hip_turn"
    )
    assert rear_hip.view == "side" and front_hip.view == "front"

    base_s = np.repeat(_stance_pose("left", "side")[None], 5, axis=0)
    cur_ok = base_s[:4].copy()
    cur_ok[:, [L_HIP, R_HIP], 0] = (0.505, 0.495)
    lm_seq = np.concatenate([base_s, cur_ok], axis=0)
    mask_seq = np.ones(lm_seq.shape[:2], dtype=bool)
    ctx = RuleContext(
        view="side",
        action="rear_straight",
        stance="left",
        phase_windows={"stance:start": np.arange(5), "rear_straight:finish": np.arange(5, 9)},
        baseline=np.arange(5),
        params=cfg,
    )
    out_ok = evaluate_rules(lm_seq, mask_seq, profile, context=ctx)
    hip_out = next(o for o in out_ok if o.spec.code == "rear_straight:finish:rear:hip_turn")
    assert hip_out.state == "pass" and hip_out.measurements[0]["name"] == "侧面两髋水平宽比值"

    lm_bad = np.concatenate([base_s, base_s[:4]], axis=0)
    out_bad = evaluate_rules(lm_bad, mask_seq, profile, context=ctx)
    assert (
        next(o for o in out_bad if o.spec.code == "rear_straight:finish:rear:hip_turn").state
        == "violation"
    )
    m3, m5 = np.ones((3, 33), bool), np.ones((5, 33), bool)
    _, _, front_details = measure_hip_turn_width(
        base_s[:3], m3, chain_torso(base_s[:3], m3), base_s, m5, chain_torso(base_s, m5)
    )
    assert "正面两髋水平宽比值" in front_details


def test_c40_post_peak_recovery_handles_normal_held_and_delayed_drop():
    base = np.repeat(_stance_pose("left", "side")[None], 5, axis=0)
    bmask = np.ones((5, 33), dtype=bool)
    btorso, times = chain_torso(base, bmask), np.arange(30, dtype=float) / FPS
    m30, m12 = np.ones((30, 33), bool), np.ones((12, 33), bool)

    lm_ok = np.repeat(base[:1], 30, axis=0)
    lm_ok[5, 15, 0] += 0.18
    lm_ok[6:10, 15, 0] += 0.10
    bad_ok, val_ok, _ = measure_punch_post_peak_hold(
        lm_ok, m30, chain_torso(lm_ok, m30), times, base, bmask, btorso, 11, 15, 1.0
    )
    assert val_ok.all() and not bad_ok

    lm_delay = np.repeat(base[:1], 30, axis=0)
    lm_delay[5:22, 15, 0] += 0.18
    bad_d, val_d, det_d = measure_punch_post_peak_hold(
        lm_delay, m30, chain_torso(lm_delay, m30), times, base, bmask, btorso, 11, 15, 1.0
    )
    assert val_d.all() and bad_d and det_d["出拳峰值后滞留时长"][0] > 0.45

    lm_held = np.repeat(base[:1], 12, axis=0)
    lm_held[2:, 15, 0] += 0.18
    bad_h, val_h, _ = measure_punch_post_peak_hold(
        lm_held,
        m12,
        chain_torso(lm_held, m12),
        times[:12],
        base,
        bmask,
        btorso,
        11,
        15,
        1.0,
        held_ending=True,
    )
    assert val_h.all() and bad_h


def test_c47_overlap_pairing_and_pause_threshold():
    front_ev = MotionEvent(40, 45, 54, "returned")
    assert preceding_punch([front_ev], 20, 50) is None
    assert preceding_punch([front_ev], 20, 50, rear_peak=58) == front_ev

    res_ok = evaluate_series(
        "step_hook_combo", "left", _make_step_hook_views(reps=2, front_overlap=True)
    )
    assert (
        next(c for c in res_ok["checks"] if c["code"] == "combo_punch_pause")["status"]
        == "not_observed"
    )

    res_bad = evaluate_series(
        "step_hook_combo", "left", _make_step_hook_views(reps=2, combo_gap_s=0.45)
    )
    assert (
        next(c for c in res_bad["checks"] if c["code"] == "combo_punch_pause")["status"]
        == "candidate"
    )


def test_c22_paired_feet_together_ignores_non_side_frames_and_requires_side_frames():
    n = 20
    front = rule_series(
        np.repeat(_stance_pose("left", "front")[None], n, axis=0), range(n), FPS, 1.0
    )
    side = rule_series(np.repeat(_stance_pose("left", "side")[None], n, axis=0), range(n), FPS, 1.0)
    front["landmarks"][:10, [L_SHOULDER, R_SHOULDER], 0] = (0.60, 0.40)
    side["landmarks"][:10, [L_SHOULDER, R_SHOULDER, L_HEEL, R_HEEL], 0] = (0.52, 0.48, 0.50, 0.50)
    front["landmarks"][10:, [L_SHOULDER, R_SHOULDER], 0] = (0.54, 0.46)
    side["landmarks"][10:, [L_SHOULDER, R_SHOULDER, L_HEEL, R_HEEL], 0] = (0.54, 0.46, 0.56, 0.44)

    paired = paired_torso_measurements({"front": front, "side": side})
    ratios, obs = paired["side"]
    assert obs.all() and (ratios[:10] > FRONTAL_PROJECTION_RATIO).all()
    assert (ratios[10:] <= FRONTAL_PROJECTION_RATIO).all()

    bad, valid, _ = measure_step_feet_together(
        side["landmarks"], side["valid_mask"], paired["side"], np.arange(n), min_ratio=1.0
    )
    assert not valid[:10].any() and valid[10:].all() and not bad[valid].any()

    side["landmarks"][10:, [L_HEEL, R_HEEL], 0] = (0.52, 0.48)
    bad2, valid2, _ = measure_step_feet_together(
        side["landmarks"], side["valid_mask"], paired["side"], np.arange(n), min_ratio=1.0
    )
    assert valid2[10:].all() and bad2[valid2].all()

    c22 = next(
        r
        for r in rule_profile("step_hook_combo", "left", FeedbackConfig()).rules
        if r.code == "step_footwork:motion::step_feet_together"
    )
    assert c22.view == "side" and c22.required_views == ("front", "side")
    assert c22.evidence_joints_by_view["front"] == (L_SHOULDER, R_SHOULDER)
    assert set(c22.evidence_joints_by_view["side"]) == {L_HEEL, R_HEEL, L_SHOULDER, R_SHOULDER}


def test_c23_c24_c25_step_footwork_measures_and_noise_immunity():
    n = 30
    times = np.arange(n, dtype=float) / FPS
    lm = np.repeat(_stance_pose("left", "side")[None], n, axis=0)
    mask = np.ones((n, 33), dtype=bool)
    f_ak, r_ak = sides_for_stance("front", "left")[5], sides_for_stance("rear", "left")[5]

    lm_no = lm.copy()
    lm_no[:, r_ak, 0] = np.linspace(0.40, 0.44, n)
    bad_no, val_no, _ = measure_foot_forward_displacement(
        lm_no, mask, times, r_ak, min_disp=0.25, facing=1.0
    )
    assert val_no.all() and bad_no

    lm_yes = lm.copy()
    lm_yes[:, r_ak, 0] = np.linspace(0.40, 0.50, n)
    bad_yes, val_yes, _ = measure_foot_forward_displacement(
        lm_yes, mask, times, r_ak, min_disp=0.25, facing=1.0
    )
    assert val_yes.all() and not bad_yes

    lm_mid = lm.copy()
    lm_mid[:, [f_ak, r_ak], 0] = (0.60, 0.52)
    lm_mid[10:14, r_ak, 0] = 0.65
    bad_ov, val_ov, _ = measure_rear_foot_overtake(
        lm_mid, mask, r_ak, f_ak, overtake_max=0.02, facing=1.0, mode="peak_median"
    )
    assert val_ov.all() and bad_ov

    lm_spike = lm.copy()
    lm_spike[:, [f_ak, r_ak], 0] = (0.60, 0.52)
    lm_spike[12, r_ak, 0] = 0.80
    bad_sp, val_sp, _ = measure_rear_foot_overtake(
        lm_spike, mask, r_ak, f_ak, overtake_max=0.02, facing=1.0, mode="peak_median"
    )
    assert val_sp.all() and not bad_sp

    lm_late = lm.copy()
    lm_late[:24, r_ak, 0] = np.linspace(0.40, 0.52, 24)
    lm_late[24:, r_ak, 0] = 0.52
    bad_l, val_l, det_l = measure_step_landing_lag(
        lm_late, mask, times, f_ak, r_ak, front_peak=None, rear_peak=0.40, max_lag=0.20, facing=1.0
    )
    assert val_l.all() and bad_l and det_l["脚步落稳滞后出拳峰值"][0] > 0.20
    bad_sync, val_sync, _ = measure_step_landing_lag(
        lm_late, mask, times, f_ak, r_ak, front_peak=None, rear_peak=0.85, max_lag=0.20, facing=1.0
    )
    assert val_sync.all() and not bad_sync


def test_step_progression_tail_retreat_and_sparse_gaps_regressions():
    """回归 P1-1/P1-2：尾窗后撤复位下 C23 不误报且 C25 不漏检；长缺口稀疏有效帧零告警不误报。"""
    f_ak, r_ak = sides_for_stance("front", "left")[5], sides_for_stance("rear", "left")[5]

    views = _make_step_hook_views(stance="left", reps=2, rear_follow_disp=0.14)
    side_t = np.asarray(views["side"]["times"], dtype=float)
    views["side"]["landmarks"][(side_t >= 2.05) & (side_t <= 2.35), r_ak, 0] = 0.30
    res = evaluate_series("step_hook_combo", "left", views)
    c23 = next(
        c for c in res["checks"] if c["id"] == "step_footwork:motion:rear:step_follow_missing"
    )
    assert c23["status"] == "not_observed"

    n = 36
    times = np.arange(n, dtype=float) / FPS
    lm_ret = np.repeat(_stance_pose("left", "side")[None], n, axis=0)
    lm_ret[:24, r_ak, 0] = np.linspace(0.40, 0.54, 24)
    lm_ret[24:27, r_ak, 0] = 0.54
    lm_ret[27:, r_ak, 0] = np.linspace(0.54, 0.30, n - 27)
    bad_ret, val_ret, det_ret = measure_step_landing_lag(
        lm_ret,
        np.ones((n, 33), bool),
        times,
        f_ak,
        r_ak,
        front_peak=None,
        rear_peak=0.40,
        facing=1.0,
    )
    assert val_ret.all() and bad_ret and det_ret["脚步落稳滞后出拳峰值"][0] > 0.20

    n_sp = 30
    times_sp = np.arange(n_sp, dtype=float) / FPS
    lm_sp = np.repeat(_stance_pose("left", "side")[None], n_sp, axis=0)
    mask_sp = np.ones((n_sp, 33), dtype=bool)
    mask_sp[:, r_ak] = False
    for idx, x_val in ((0, 0.40), (6, 0.48), (12, 0.54), (18, 0.54), (24, 0.54)):
        mask_sp[idx, r_ak] = True
        lm_sp[idx, r_ak, 0] = x_val
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bad_sp, val_sp, det_sp = measure_step_landing_lag(
            lm_sp, mask_sp, times_sp, f_ak, r_ak, front_peak=None, rear_peak=0.45, facing=1.0
        )
        bad_sp_late, _, det_sp_late = measure_step_landing_lag(
            lm_sp, mask_sp, times_sp, f_ak, r_ak, front_peak=None, rear_peak=0.10, facing=1.0
        )
    assert caught == []
    assert int(val_sp.sum()) == 5 and not bad_sp and det_sp["脚步落稳滞后出拳峰值"][0] <= 0.20
    assert bad_sp_late and det_sp_late["脚步落稳滞后出拳峰值"][0] > 0.20

    # 混合缺口（起止各有 2 帧连续、中间迈步稀疏）不得跳过真实落稳帧落到窗尾。
    lm_hy = np.repeat(_stance_pose("left", "side")[None], n_sp, axis=0)
    mask_hy = np.ones((n_sp, 33), dtype=bool)
    mask_hy[:, r_ak] = False
    for idx, x_val in (
        (0, 0.40),
        (1, 0.40),
        (7, 0.48),
        (13, 0.54),
        (19, 0.54),
        (25, 0.54),
        (28, 0.54),
        (29, 0.54),
    ):
        mask_hy[idx, r_ak] = True
        lm_hy[idx, r_ak, 0] = x_val
    with warnings.catch_warnings(record=True) as caught_hy:
        warnings.simplefilter("always")
        bad_hy, _, det_hy = measure_step_landing_lag(
            lm_hy, mask_hy, times_sp, f_ak, r_ak, front_peak=None, rear_peak=0.45, facing=1.0
        )
    assert caught_hy == [] and not bad_hy and det_hy["脚步落稳滞后出拳峰值"][0] <= 0.05


@pytest.mark.parametrize("stance", ["left", "right"])
def test_step_hook_combo_end_to_end_practice_and_step_defects(stance):
    assert ACTIONS["step_hook_combo"] == "上步前后手摆拳组合"
    assert SEGMENT_LABELS["step_footwork"] == "上步步法"
    assert SEGMENT_LABELS["combo_transition"] == "组合拳衔接"
    assert motion_segments("step_hook_combo") == ("front_hook", "rear_hook")
    assert (
        set(action_definition("step_hook_combo").segments) - set(motion_segments("step_hook_combo"))
        == DERIVED_SEGMENTS
    )

    views = _make_step_hook_views(stance=stance, reps=2)
    res = evaluate_series("step_hook_combo", stance, views)
    assert len(res["checks"]) == 66 and res["repetitions"]["detected"] == 2
    by_id = {c["id"]: c for c in res["checks"]}
    for cid in (
        "step_footwork:motion:rear:step_follow_missing",
        "step_footwork:motion:rear:step_overtake",
        "step_footwork:motion::step_late_landing",
        "step_footwork:motion::step_feet_together",
        "combo_transition:motion::combo_punch_pause",
        "rear_hook:motion:rear:punch_unrecovered",
    ):
        assert by_id[cid]["status"] == "not_observed"

    stage_titles = [
        s.title
        for s in build_report_model(
            {"action": "step_hook_combo", "stance": stance, "result": res}
        ).phases
    ]
    assert "组合拳衔接" in stage_titles and "上步步法" in stage_titles

    bad_res = evaluate_series(
        "step_hook_combo",
        stance,
        _make_step_hook_views(
            stance=stance, reps=2, rear_follow_disp=0.01, feet_together=True, hold_last_rear=True
        ),
    )
    bad_by_id = {c["id"]: c for c in bad_res["checks"]}
    assert bad_by_id["step_footwork:motion::step_feet_together"]["status"] == "candidate"
    assert bad_by_id["step_footwork:motion:rear:step_follow_missing"]["status"] == "candidate"
    assert bad_by_id["rear_hook:motion:rear:punch_unrecovered"]["status"] == "candidate"

    ov_res = evaluate_series(
        "step_hook_combo",
        stance,
        _make_step_hook_views(
            stance=stance,
            reps=2,
            rear_overtake_offset=0.06,
            rear_land_lag_s=0.35,
            delayed_drop_rear=True,
        ),
    )
    ov_by_id = {c["id"]: c for c in ov_res["checks"]}
    assert ov_by_id["step_footwork:motion:rear:step_overtake"]["status"] == "candidate"
    assert ov_by_id["step_footwork:motion::step_late_landing"]["status"] == "candidate"
    assert ov_by_id["rear_hook:motion:rear:punch_unrecovered"]["status"] == "candidate"


def test_step_hook_combo_exam_pool_stepwise_scoring_and_issue_grouping(monkeypatch, tmp_path):
    views = _make_step_hook_views(stance="left", reps=2)
    diag: dict = {}
    windows = step_combo_windows(
        views["side"], "left", FeedbackConfig(), diag, action="step_hook_combo"
    )
    assert len(diag["repetitions"]) == 2
    assert ("step_footwork", "motion") in windows and ("combo_transition", "motion") in windows

    meta = {"width": 100, "height": 100, "fps": FPS, "start_frame": 0}
    raw_side = Pose33RawSeries(views["side"]["landmarks"], meta)
    assert (
        len(exam_pool.action_side_repetitions(raw_side, "step_hook_combo", FeedbackConfig())) == 2
    )

    monkeypatch.setattr(
        exam_pool, "load_feedback_config", lambda: (_ for _ in ()).throw(ValueError("skip rules"))
    )
    seen_windows: list = []

    def compare_windows(templates, video, win_list, **_kw):
        seen_windows.append((Path(video).name, list(win_list)))
        return [
            [
                SimpleNamespace(score=0.7 + 0.1 * i, start_frame=a + 1, end_frame=b - 1)
                for i, (a, b) in enumerate(win_list)
            ]
            for _ in templates
        ]

    tpl = [tmp_path / "上步前后手摆拳组合.npz"]
    res = exam_pool.score_exam_pools(
        tpl,
        tpl,
        Path("f.mp4"),
        Path("s.mp4"),
        compare=lambda *_a, **_k: [],
        raw_series=lambda *_a, **_k: raw_side,
        stop_evt=Event(),
        compare_windows=compare_windows,
    )
    assert len(seen_windows) == 2 and len(seen_windows[0][1]) == 2
    assert res.front_score == pytest.approx(0.75) and res.side_score == pytest.approx(0.75)

    judged = judge_views(views, "step_hook_combo", "left", FeedbackConfig())
    assert len(judged["side"][0]) > 0 and len(judged["front"][0]) > 0
    for code in ("head_down", "hip_turn", "trunk_lean_sagittal", "stance_knee"):
        sample = build_checks("front_hook", "left")
        pair = [
            c
            for c in sample
            if c["segment"] == "front_hook"
            and c["code"] == code
            and c["phase"] in {"motion", "finish"}
        ]
        for chk in pair:
            chk.update(status="candidate", review="pending", reason="触发", evidence=[])
        grouped = group_action_issues(
            build_report_model(
                {"action": "front_hook", "stance": "left", "result": {"checks": pair}}
            ).candidate_checks,
            "front_hook",
        )
        assert len(grouped) == 1 and len(grouped[0].members) == 2


def test_c48_guard_jaw_on_rear_punches_three_state_boundaries_and_punch_vs_kick_copy():
    punch_checks = {c["id"]: c for c in build_checks("rear_straight", "left")}
    kick_checks = {c["id"]: c for c in build_checks("front_low_kick", "left")}
    assert punch_checks["rear_straight:motion::head_down"]["name"] == "出拳时低头"
    assert kick_checks["front_low_kick:motion::head_down"]["name"] == "踢腿时低头"
    assert punch_checks["rear_straight:finish:rear:pivot_heel"]["name"] == "出拳发力脚未蹬转提踵"
    assert kick_checks["front_low_kick:finish:rear:pivot_heel"]["name"] == "支撑脚未碾脚内旋"
    jaw_std = punch_checks["rear_straight:finish:front:guard_jaw"]["standard"]
    assert "摆拳" not in jaw_std and "0.34" in jaw_std

    cfg = FeedbackConfig()
    lm_f = np.repeat(_stance_pose("left", "front")[None], 8, axis=0)
    lm_f[4:, L_WRIST, :2] = (0.70, 0.45)
    profile = rule_profile("rear_straight", "left", cfg)
    ctx_f = RuleContext(
        view="front",
        action="rear_straight",
        stance="left",
        phase_windows={"stance:start": np.arange(4), "rear_straight:finish": np.arange(4, 8)},
        baseline=np.arange(4),
        params=cfg,
    )
    out_c48 = evaluate_rules(lm_f, np.ones(lm_f.shape[:2], bool), profile, context=ctx_f)
    assert (
        next(o for o in out_c48 if o.spec.code == "rear_straight:finish:front:guard_jaw").state
        == "violation"
    )

    frontal_views = _make_step_hook_views(stance="left", reps=2)
    frontal_views["front"]["landmarks"][:, [L_SHOULDER, R_SHOULDER], 0] = (0.65, 0.35)
    frontal_views["side"]["landmarks"][:, [L_SHOULDER, R_SHOULDER], 0] = (0.51, 0.49)
    res_frontal = evaluate_series("step_hook_combo", "left", frontal_views)
    assert (
        next(c for c in res_frontal["checks"] if c["code"] == "step_feet_together")["status"]
        == "unable"
    )

    res_single = evaluate_series(
        "step_hook_combo", "left", {"side": _make_step_hook_views(stance="left", reps=1)["side"]}
    )
    assert (
        next(c for c in res_single["checks"] if c["code"] == "step_feet_together")["status"]
        == "unable"
    )
