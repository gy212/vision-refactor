# -*- coding: utf-8 -*-
"""步法窗中点切分、按次标量规则逐次门控、双脚/双腕掩码解耦与滞留微抖边界回归测试。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.evaluation import (
    FeedbackConfig,
    evaluate_series,
    judge_views,
    rule_series,
    series_from_raw,
)
from core.evaluation.action_timing import shared_timeline
from core.evaluation.body_sides import sides_for_stance
from core.evaluation.hook_events import observe_hook
from core.evaluation.measures.feet import measure_step_landing_lag
from core.evaluation.measures.motion import measure_punch_post_peak_hold
from core.evaluation.measures.torso import chain_torso
from core.evaluation.step_reps import step_combo_windows
from core.pose.landmarks import L_HIP, L_SHOULDER, L_WRIST, R_HIP, R_SHOULDER, R_WRIST
from tests.test_hook_and_step_rules import FPS, make_step_hook_views, stance_pose
from tests.test_low_kick_rules import make_kick_sequence


def _compact_step_combo_views(
    *,
    action: str = "step_hook_combo",
    stance: str = "left",
    gap_s: float = 0.40,
    first_rep_no_follow: bool = False,
    skip_first_front: bool = False,
    step_shift: float = 0.12,
) -> dict[str, dict]:
    """构造连贯双次（间隔 gap_s）“先落稳再出拳”上步直拳/摆拳组合序列。"""
    n = int(round(5.0 * FPS))
    times = np.arange(n, dtype=float) / FPS
    dir_x = 1.0 if stance == "left" else -1.0
    _, f_el, f_wr, _, _, f_ak, f_hl, f_to = sides_for_stance("front", stance)
    _, r_el, r_wr, _, _, r_ak, r_hl, r_to = sides_for_stance("rear", stance)
    side_lm = np.repeat(stance_pose(stance, "side")[None], n, axis=0)
    front_lm = np.repeat(stance_pose(stance, "front")[None], n, axis=0)
    base_side = side_lm[0].copy()
    is_straight = action == "step_straight_combo"
    f_reach = 0.20 if is_straight else 0.12
    r_reach = 0.24 if is_straight else 0.16
    starts = (1.00, 1.60 + gap_s)
    r_prev = 0.0
    for k, t0 in enumerate(starts):
        r_gain = 0.0 if (first_rep_no_follow and k == 0) else step_shift
        for i in np.flatnonzero(times >= t0 - 0.25):
            u = float(np.clip((times[i] - (t0 - 0.25)) / 0.30, 0.0, 1.0))
            side_lm[i, f_ak, 0] = base_side[f_ak, 0] + (step_shift * k + step_shift * u) * dir_x
            side_lm[i, r_ak, 0] = base_side[r_ak, 0] + (r_prev + r_gain * u) * dir_x
            side_lm[i, [f_hl, r_hl], 0] = side_lm[i, [f_ak, r_ak], 0] - 0.02 * dir_x
            side_lm[i, [f_to, r_to], 0] = side_lm[i, [f_hl, r_hl], 0] + 0.05 * dir_x
        r_prev += r_gain
        if not (skip_first_front and k == 0):
            for i in np.flatnonzero((times >= t0) & (times <= t0 + 0.28)):
                u = math.sin(math.pi * (times[i] - t0) / 0.28)
                side_lm[i, [f_el, f_wr], :2] = [
                    (base_side[f_el, 0] + 0.14 * dir_x * u, 0.26),
                    (base_side[f_wr, 0] + f_reach * dir_x * u, 0.25),
                ]
                side_lm[i, f_hl, 1] = base_side[f_hl, 1] - 0.05 * u
                front_lm[i, [L_HIP, R_HIP], 0] = (0.52, 0.48)
        for i in np.flatnonzero((times >= t0 + 0.24) & (times <= t0 + 0.60)):
            u = math.sin(math.pi * (times[i] - (t0 + 0.24)) / 0.36)
            side_lm[i, [r_el, r_wr], :2] = [
                (base_side[r_el, 0] + 0.18 * dir_x * u, 0.26),
                (base_side[r_wr, 0] + r_reach * dir_x * u, 0.25),
            ]
            side_lm[i, r_hl, 1] = base_side[r_hl, 1] - 0.06 * u
            side_lm[i, [L_HIP, R_HIP], 0] = (0.50 + 0.005 * dir_x, 0.50 - 0.005 * dir_x)
    return {
        "front": rule_series(front_lm, range(n), FPS, 1.0),
        "side": rule_series(side_lm, range(n), FPS, 1.0),
    }


def _make_multi_hook_combo_views(
    *,
    reps: int = 3,
    pause_s: float = 0.08,
    stance: str = "left",
) -> dict[str, dict]:
    """构造多组正面计时的前后手摆拳组合（hook_combo）双视角序列。"""
    seconds = 1.6 + reps * (1.2 + pause_s) + 1.0
    n = int(round(seconds * FPS))
    times = np.arange(n, dtype=float) / FPS
    dir_x = 1.0 if stance == "left" else -1.0
    f_sh, f_el, f_wr, *_ = sides_for_stance("front", stance)
    r_sh, r_el, r_wr, *_ = sides_for_stance("rear", stance)
    side_lm = np.repeat(stance_pose(stance, "side")[None], n, axis=0)
    front_lm = np.repeat(stance_pose(stance, "front")[None], n, axis=0)
    base_side = side_lm[0].copy()
    base_front = front_lm[0].copy()
    cursor = 1.0
    for _ in range(reps):
        f_start, f_end = cursor, cursor + 0.32
        r_start, r_end = f_end + pause_s, f_end + pause_s + 0.36
        for i in np.flatnonzero((times >= f_start) & (times <= f_end)):
            u = math.sin(math.pi * (times[i] - f_start) / (f_end - f_start))
            front_lm[i, f_wr, :2] = (base_front[f_wr, 0] - 0.16 * dir_x * u, 0.25)
            front_lm[i, f_el, :2] = (base_front[f_sh, 0], 0.25)
            side_lm[i, f_wr, :2] = (base_side[f_wr, 0] + 0.14 * dir_x * u, 0.25)
            side_lm[i, f_el, :2] = (base_side[f_el, 0] + 0.12 * dir_x * u, 0.26)
        for i in np.flatnonzero((times >= r_start) & (times <= r_end)):
            u = math.sin(math.pi * (times[i] - r_start) / (r_end - r_start))
            front_lm[i, r_wr, :2] = (base_front[r_wr, 0] + 0.18 * dir_x * u, 0.25)
            front_lm[i, r_el, :2] = (base_front[r_sh, 0], 0.25)
            side_lm[i, r_wr, :2] = (base_side[r_wr, 0] + 0.18 * dir_x * u, 0.25)
            side_lm[i, r_el, :2] = (base_side[r_el, 0] + 0.16 * dir_x * u, 0.26)
        cursor = r_end + 0.65
    return {
        "front": rule_series(front_lm, range(n), FPS, 1.0),
        "side": rule_series(side_lm, range(n), FPS, 1.0),
    }


@pytest.mark.parametrize("action", ["step_hook_combo", "step_straight_combo"])
def test_compact_step_combo_windows_disjoint_and_no_cross_rep_leak(action):
    """连贯双次（0.40s）步法窗严格不相交，标准动作不误报滞后，首轮不跟步稳定检出。"""
    views_ok = _compact_step_combo_views(action=action, gap_s=0.40)
    diag: dict = {}
    step_combo_windows(views_ok["side"], "left", FeedbackConfig(), diag, action=action)
    fw = [item for item in diag["phaseInstances"] if item["segment"] == "step_footwork"]
    assert len(fw) == 2
    assert fw[0]["end"] < fw[1]["start"]
    times = np.asarray(views_ok["side"]["times"])
    frames_1 = set(np.flatnonzero((times >= fw[0]["start"]) & (times <= fw[0]["end"])).tolist())
    frames_2 = set(np.flatnonzero((times >= fw[1]["start"]) & (times <= fw[1]["end"])).tolist())
    assert frames_1 and frames_2 and frames_1.isdisjoint(frames_2)

    res_ok = evaluate_series(action, "left", views_ok)
    by_code = {c["code"]: c for c in res_ok["checks"] if c["segment"] == "step_footwork"}
    assert by_code["step_late_landing"]["status"] == "not_observed"
    assert by_code["step_follow_missing"]["status"] == "not_observed"

    views_no_follow = _compact_step_combo_views(action=action, gap_s=0.40, first_rep_no_follow=True)
    res_bad = evaluate_series(action, "left", views_no_follow)
    c23 = next(c for c in res_bad["checks"] if c["code"] == "step_follow_missing")
    assert c23["status"] == "candidate"
    jv = judge_views(views_no_follow, action, "left", FeedbackConfig())
    c23_chk = next(c for c in jv["side"][0] if c.code == "step_follow_missing")
    assert c23_chk.state == "violation"
    assert {
        item["repetition"] for item in c23_chk.evidence_details if item["purpose"] == "violation"
    } == {1}


def test_combo_transition_preserves_repetition_index_when_first_front_missing():
    """首轮缺前手跳过衔接时，次轮 combo_transition 与 comboPairs 的 repetition 必须保持为 2。"""
    views = _compact_step_combo_views(action="step_hook_combo", skip_first_front=True)
    diag: dict = {}
    step_combo_windows(views["side"], "left", FeedbackConfig(), diag, action="step_hook_combo")
    assert len(diag["repetitions"]) == 2
    assert diag["repetitions"][0]["missingSegments"] == ["front_hook"]
    assert len(diag["comboPairs"]) == 1 and diag["comboPairs"][0]["repetition"] == 2
    trans = [item for item in diag["phaseInstances"] if item["segment"] == "combo_transition"]
    assert (
        len(trans) == 1
        and trans[0]["repetition"] == 2
        and trans[0]["eventId"] == "combo_transition:2"
    )

    # 同级验证：非上步组合（hook_combo via phases.py）首轮缺前手时同样保持 repetition=2
    vh = _make_multi_hook_combo_views(reps=2, pause_s=0.08)
    t_f = np.asarray(vh["front"]["times"])
    vh["front"]["landmarks"][(t_f >= 0.95) & (t_f <= 1.38)] = stance_pose("left", "front")
    _, diag_h, _, _ = shared_timeline(vh, "hook_combo", "left", FeedbackConfig())
    assert len(diag_h["front"]["comboPairs"]) == 1
    assert diag_h["front"]["comboPairs"][0]["repetition"] == 2
    trans_h = [
        item for item in diag_h["front"]["phaseInstances"] if item["segment"] == "combo_transition"
    ]
    assert len(trans_h) == 1 and trans_h[0]["repetition"] == 2


def test_per_instance_scalar_rules_return_unable_on_single_rep_occlusion():
    """全部 7 条按次标量规则在多次重复中某一次遮挡时严格返回 unable，绝不跨次稀释伪造通过。"""
    cfg = FeedbackConfig()
    # 1) step_overtake (2 reps, rep 2 rear_ankle occluded)
    v2 = make_step_hook_views(stance="left", reps=2)
    v2["side"]["valid_mask"][np.asarray(v2["side"]["times"]) >= 2.90, 28] = False
    res2 = evaluate_series("step_hook_combo", "left", v2, cfg)
    c24 = next(c for c in res2["checks"] if c["code"] == "step_overtake")
    assert c24["status"] == "unable"
    assert "存在有效帧不足的动作次" in c24["reason"]

    # 2) step_follow_missing / step_overtake / step_late_landing (3 reps, rep 3 rear_ankle occluded)
    v3 = make_step_hook_views(stance="left", reps=3, seconds=9.0)
    v3["side"]["valid_mask"][np.asarray(v3["side"]["times"]) >= 5.10, 28] = False
    res3 = evaluate_series("step_hook_combo", "left", v3, cfg)
    step_by_code = {c["code"]: c for c in res3["checks"] if c["segment"] == "step_footwork"}
    for code in ("step_follow_missing", "step_overtake", "step_late_landing"):
        assert step_by_code[code]["status"] == "unable", f"{code} unexpectedly passed"
        assert "存在有效帧不足的动作次" in step_by_code[code]["reason"]

    # 3) punch_unrecovered & combo_punch_pause (hook_combo 3 reps, rep 3 side arms occluded)
    vh = _make_multi_hook_combo_views(reps=3, pause_s=0.08)
    _, diag_h, _, _ = shared_timeline(vh, "hook_combo", "left", cfg)
    rep3_rear = next(
        p
        for p in diag_h["front"]["phaseInstances"]
        if p["segment"] == "rear_hook" and p["phase"] == "motion" and p["repetition"] == 3
    )
    rep3_trans = next(
        p
        for p in diag_h["front"]["phaseInstances"]
        if p["segment"] == "combo_transition" and p["repetition"] == 3
    )
    t_side = np.asarray(vh["side"]["times"])
    vh["side"]["valid_mask"][
        (t_side >= rep3_rear["start"]) & (t_side <= rep3_rear["end"]), R_WRIST
    ] = False
    trans_mask = (t_side >= rep3_trans["start"]) & (t_side <= rep3_trans["end"])
    vh["side"]["valid_mask"][trans_mask, L_WRIST] = False
    vh["side"]["valid_mask"][trans_mask, R_WRIST] = False
    res_h = evaluate_series("hook_combo", "left", vh, cfg)
    by_id_h = {c["id"]: c for c in res_h["checks"]}
    assert by_id_h["rear_hook:motion:rear:punch_unrecovered"]["status"] == "unable"
    assert "存在有效帧不足的动作次" in by_id_h["rear_hook:motion:rear:punch_unrecovered"]["reason"]
    assert by_id_h["combo_transition:motion::combo_punch_pause"]["status"] == "unable"
    assert (
        "存在有效帧不足的动作次" in by_id_h["combo_transition:motion::combo_punch_pause"]["reason"]
    )

    # 4) pad_step_pause / pad_step_missing / pad_step_overtake & kick_retract (3 kicks, rep 3 occluded)
    lm_f, mf = make_kick_sequence(stance="left", view="front", n_kicks=3)
    lm_s, ms = make_kick_sequence(stance="left", view="side", n_kicks=3)
    base_views_k = {"front": series_from_raw(lm_f, mf), "side": series_from_raw(lm_s, ms)}
    _, diag_k, _, _ = shared_timeline(base_views_k, "step_front_low_kick", "left", cfg)
    pad3 = next(
        p
        for p in diag_k["side"]["phaseInstances"]
        if p["segment"] == "pad_step" and p["repetition"] == 3
    )
    ret3 = next(
        p
        for p in diag_k["side"]["phaseInstances"]
        if p["segment"] == "front_low_kick" and p["phase"] == "retract" and p["repetition"] == 3
    )
    tk = np.asarray(base_views_k["side"]["times"])
    lm_s[(tk >= pad3["start"]) & (tk <= pad3["end"]), 28, 3] = 0.0
    lm_s[(tk >= ret3["start"]) & (tk <= ret3["end"]), L_HIP, 3] = 0.0
    occ_views_k = {"front": series_from_raw(lm_f, mf), "side": series_from_raw(lm_s, ms)}
    res_k = evaluate_series("step_front_low_kick", "left", occ_views_k, cfg)
    by_id_k = {c["id"]: c for c in res_k["checks"]}
    jv_k = judge_views(occ_views_k, "step_front_low_kick", "left", cfg)
    exam_k = {c.check_id: c for c in jv_k["side"][0]}
    for cid in (
        "pad_step:prepare:rear:pad_step_missing",
        "pad_step:prepare:rear:pad_step_overtake",
        "pad_step:prepare:rear:pad_step_pause",
        "front_low_kick:retract:front:kick_retract",
    ):
        assert by_id_k[cid]["status"] == "unable", f"{cid} unexpectedly passed"
        assert "存在有效帧不足的动作次" in by_id_k[cid]["reason"]
        assert exam_k[cid].state == "unable"


def test_per_instance_preserves_confirmed_violation_when_other_rep_occluded():
    """第 1 次已实锤违规时，即使第 2 次全遮挡导致整窗占比低于阈值，仍必须保留违规结论。"""
    v_mix = make_step_hook_views(stance="left", reps=2, rear_follow_disp=0.01)
    v_mix["side"]["valid_mask"][np.asarray(v_mix["side"]["times"]) >= 2.90, 28] = False
    res = evaluate_series("step_hook_combo", "left", v_mix)
    c23 = next(c for c in res["checks"] if c["code"] == "step_follow_missing")
    assert c23["status"] == "candidate"
    jv = judge_views(v_mix, "step_hook_combo", "left", FeedbackConfig())
    c23_chk = next(c for c in jv["side"][0] if c.code == "step_follow_missing")
    assert c23_chk.state == "violation"
    assert {
        item["repetition"] for item in c23_chk.evidence_details if item["purpose"] == "violation"
    } == {1}


def test_step_landing_lag_decouples_feet_masks_and_blocks_partial_pass():
    """前脚滞后不被后脚半程遮挡连坐吞成 unable；前脚在 front_cut 内缺帧但尾窗恢复须判 unable。"""
    f_ak = sides_for_stance("front", "left")[5]
    r_ak = sides_for_stance("rear", "left")[5]

    # 1) 前脚滞后 + 后脚 50% 遮挡 -> 保留 candidate
    v_lag = make_step_hook_views(stance="left", reps=1, seconds=4.0)
    t1 = np.asarray(v_lag["side"]["times"])
    for i, t in enumerate(t1):
        if 1.0 <= t <= 1.75:
            v_lag["side"]["landmarks"][i, f_ak, 0] = 0.62 + 0.12 * ((t - 1.0) / 0.75)
        elif t > 1.75:
            v_lag["side"]["landmarks"][i, f_ak, 0] = 0.74
    v_lag["side"]["valid_mask"][np.flatnonzero((t1 >= 0.7) & (t1 <= 2.4))[::2], r_ak] = False
    c25_occ = next(
        c
        for c in evaluate_series("step_hook_combo", "left", v_lag)["checks"]
        if c["code"] == "step_late_landing"
    )
    assert c25_occ["status"] == "candidate"

    # 2) 前脚在 front_cut 内全缺帧、尾窗恢复可见，后脚正常无滞后 -> 必须返回 unable
    v_cut_miss = make_step_hook_views(stance="left", reps=1, seconds=4.0)
    t2 = np.asarray(v_cut_miss["side"]["times"])
    v_cut_miss["side"]["valid_mask"][t2 <= 2.00, f_ak] = False
    c25_miss = next(
        c
        for c in evaluate_series("step_hook_combo", "left", v_cut_miss)["checks"]
        if c["code"] == "step_late_landing"
    )
    assert c25_miss["status"] == "unable"

    # 3) 底层核对：前脚在 front_cut 内缺帧而尾窗恢复，但后脚明确滞后 -> 返回 True + 后脚有效掩码
    n = 60
    times = np.arange(n, dtype=float) / FPS
    lm = np.repeat(stance_pose("left", "side")[None], n, axis=0)
    mask = np.ones((n, 33), dtype=bool)
    mask[times <= 1.15, f_ak] = False
    lm[:45, r_ak, 0] = np.linspace(0.40, 0.54, 45)
    lm[45:, r_ak, 0] = 0.54
    bad_r, valid_r, det_r = measure_step_landing_lag(
        lm, mask, times, f_ak, r_ak, front_peak=0.50, rear_peak=0.90, max_lag=0.20, facing=1.0
    )
    assert bad_r and valid_r.all() and det_r["脚步落稳滞后出拳峰值"][0] > 0.20


def test_hook_combo_pause_detects_with_side_far_wrist_occluded():
    """hook_combo 停顿 0.50s 时侧面遮挡远侧手腕仍应检出 candidate，双腕全遮挡则判 unable。"""
    views = _make_multi_hook_combo_views(reps=1, pause_s=0.50, stance="left")
    views["side"]["valid_mask"][:, R_WRIST] = False  # 侧面遮挡远侧右手腕
    res = evaluate_series("hook_combo", "left", views)
    c47 = next(c for c in res["checks"] if c["code"] == "combo_punch_pause")
    assert c47["status"] == "candidate"

    views_both = _make_multi_hook_combo_views(reps=1, pause_s=0.50, stance="left")
    views_both["side"]["valid_mask"][:, [L_WRIST, R_WRIST]] = False
    res_both = evaluate_series("hook_combo", "left", views_both)
    c47_both = next(c for c in res_both["checks"] if c["code"] == "combo_punch_pause")
    assert c47_both["status"] == "unable"


@pytest.mark.parametrize("jitter", [1e-4, 0.002])
def test_punch_post_peak_hold_resists_plateau_tail_jitter(jitter):
    """出拳滞留超 0.45s 再收回且滞留末端带正向微抖时，直拳与摆拳均稳定检出 punch_unrecovered。"""
    base = np.repeat(stance_pose("left", "side")[None], 5, axis=0)
    bmask, m30 = np.ones((5, 33), bool), np.ones((30, 33), bool)
    lm_delay = np.repeat(base[:1], 30, axis=0)
    lm_delay[5:22, L_WRIST, 0] += 0.18
    lm_delay[20, L_WRIST, 0] += jitter
    bad_d, val_d, det_d = measure_punch_post_peak_hold(
        lm_delay,
        m30,
        chain_torso(lm_delay, m30),
        np.arange(30, dtype=float) / FPS,
        base,
        bmask,
        chain_torso(base, bmask),
        L_SHOULDER,
        L_WRIST,
        1.0,
    )
    assert val_d.all() and bad_d and det_d["出拳峰值后滞留时长"][0] > 0.50

    # 端到端直拳（front_straight）滞留 0.55s + 尾端微抖 -> candidate
    n = int(round(3.5 * FPS))
    side_lm = np.repeat(stance_pose("left", "side")[None], n, axis=0)
    front_lm = np.repeat(stance_pose("left", "front")[None], n, axis=0)
    base_s = side_lm[0].copy()
    for i in range(30, 56):
        if i <= 35:
            u = (i - 30) / 5.0
        elif i <= 52:
            u = 1.0
        else:
            u = max(0.0, (56 - i) / 4.0)
        side_lm[i, L_WRIST, 0] = base_s[L_WRIST, 0] + 0.22 * u
    side_lm[51, L_WRIST, 0] += jitter
    res_st = evaluate_series(
        "front_straight",
        "left",
        {
            "front": rule_series(front_lm, range(n), FPS, 1.0),
            "side": rule_series(side_lm, range(n), FPS, 1.0),
        },
    )
    assert (
        next(c for c in res_st["checks"] if c["code"] == "punch_unrecovered")["status"]
        == "candidate"
    )

    # 端到端摆拳组合（step_hook_combo）长滞留 + 尾端微抖 -> candidate
    v_hook = make_step_hook_views(stance="left", reps=1, seconds=4.0, delayed_drop_rear=True)
    t_h = np.asarray(v_hook["side"]["times"])
    tail_idx = int(np.argmin(np.abs(t_h - 2.26)))
    v_hook["side"]["landmarks"][tail_idx, R_WRIST, 0] += jitter
    res_hk = evaluate_series("step_hook_combo", "left", v_hook)
    assert (
        next(c for c in res_hk["checks"] if c["id"] == "rear_hook:motion:rear:punch_unrecovered")[
            "status"
        ]
        == "candidate"
    )


def test_long_hold_hook_peak_stays_at_leading_edge_under_tail_jitter():
    """长滞留摆拳末端带微抖时，正/侧 observe_hook 的 event.peak 均锁定首次到达前沿不后漂。"""
    cfg, n = FeedbackConfig(), 110
    times, baseline = np.arange(n, dtype=float) / FPS, np.arange(12)

    # 1) 侧面长滞留摆拳（20..25 上升，25..50 滞留，50..54 回撤），第 47 帧加微抖和略短小臂投影
    lm_s = np.repeat(stance_pose("left", "side")[None], n, axis=0)
    mask_s = np.ones((n, 33), dtype=bool)
    base_s = lm_s[0].copy()
    for i in range(20, 55):
        if i <= 25:
            u = (i - 20) / 5.0
        elif i <= 50:
            u = 1.0
        else:
            u = max(0.0, (54 - i) / 4.0)
        lm_s[i, 13, :2] = (base_s[13, 0] + 0.14 * u, 0.26)
        lm_s[i, 15, :2] = (base_s[15, 0] + 0.16 * u, 0.25)
    lm_s[26, [13, 15], 0] = (0.71, 0.74)  # 首次到达前沿的小臂最短投影帧（长 0.03）
    lm_s[47, [13, 15], 0] = (0.735, 0.742)  # 滞留尾端带 +0.002 前伸微抖且小臂更短（长 0.007）
    obs_s = observe_hook(
        lm_s,
        mask_s,
        times,
        chain_torso(lm_s, mask_s),
        baseline,
        [baseline],
        "front",
        "left",
        cfg,
        "side",
    )
    assert len(obs_s.events) == 1
    assert obs_s.events[0].peak == 26

    # 2) 正面长滞留摆拳（20..25 上升，25..50 滞留，50..54 回撤），第 47 帧略增过中线微抖
    lm_f = np.repeat(stance_pose("left", "front")[None], n, axis=0)
    mask_f = np.ones((n, 33), dtype=bool)
    lm_f[:, [L_SHOULDER, R_SHOULDER, L_WRIST], :2] = [(0.60, 0.25), (0.40, 0.25), (0.58, 0.22)]
    for i in range(20, 55):
        if i <= 25:
            u = (i - 20) / 5.0
        elif i <= 50:
            u = 1.0
        else:
            u = max(0.0, (54 - i) / 4.0)
        lm_f[i, L_WRIST, 0] = 0.58 - 0.18 * u
    lm_f[26, L_WRIST, 0] = 0.39  # 首次到达前沿的最内摆帧
    lm_f[47, L_WRIST, 0] = 0.385  # 滞留尾端微抖更靠内
    obs_f = observe_hook(
        lm_f,
        mask_f,
        times,
        chain_torso(lm_f, mask_f),
        baseline,
        [baseline],
        "front",
        "left",
        cfg,
        "front",
    )
    assert len(obs_f.events) == 1
    assert obs_f.events[0].peak == 26


def test_foot_settle_plateau_guard_no_false_stop_on_slow_drag_or_two_beat_steps():
    """两段式慢跟步、全程匀速慢拖步不漏检 C25，且两拍正常垫步不误报 pad_step_pause。"""
    f_ak = sides_for_stance("front", "left")[5]
    r_ak = sides_for_stance("rear", "left")[5]

    # 1) 两段式慢跟步：rearPeak 前短停 0.25s 后继续拖步至 2.20s 落稳 -> C25=candidate
    v_two = make_step_hook_views(stance="left", reps=1, seconds=4.0)
    t_two = np.asarray(v_two["side"]["times"])
    lm_two = v_two["side"]["landmarks"]
    r0 = float(lm_two[0, r_ak, 0])
    for i, t in enumerate(t_two):
        if t < 1.00:
            x = r0
        elif t <= 1.20:
            x = r0 + 0.09 * ((t - 1.00) / 0.20)  # +0.30 torso
        elif t <= 1.45:
            x = r0 + 0.09  # rearPeak(~1.77s) 前短停 0.25s
        elif t <= 2.20:
            x = r0 + 0.09 + 0.075 * ((t - 1.45) / 0.75)  # 继续拖步 +0.25 torso
        else:
            x = r0 + 0.165
        lm_two[i, r_ak, 0] = x
    c25_two = next(
        c
        for c in evaluate_series("step_hook_combo", "left", v_two)["checks"]
        if c["code"] == "step_late_landing"
    )
    assert c25_two["status"] == "candidate"

    # 2) 全程匀速慢拖步（0.11 torso/s，无停顿）-> bad=True 且滞后 > 0.20s
    n_drag = int(round(2.8 * FPS))
    t_drag = np.arange(n_drag, dtype=float) / FPS
    lm_drag = np.repeat(stance_pose("left", "side")[None], n_drag, axis=0)
    mask_drag = np.ones((n_drag, 33), dtype=bool)
    for i, t in enumerate(t_drag):
        if t <= 0.50:
            lm_drag[i, r_ak, 0] = 0.40
        elif t <= 2.30:
            lm_drag[i, r_ak, 0] = 0.40 + 0.033 * (t - 0.50)  # 0.11 torso/s
        else:
            lm_drag[i, r_ak, 0] = 0.40 + 0.033 * 1.80
    bad_drag, val_drag, det_drag = measure_step_landing_lag(
        lm_drag,
        mask_drag,
        t_drag,
        f_ak,
        r_ak,
        front_peak=None,
        rear_peak=1.85,
        max_lag=0.20,
        facing=1.0,
    )
    assert val_drag.all() and bad_drag and det_drag["脚步落稳滞后出拳峰值"][0] > 0.20

    # 3) 两拍正常垫步（先小调整+0.30、短停0.25s，再主垫步+0.25于起脚前0.12s落稳）-> pad_step_pause=not_observed
    cfg = FeedbackConfig(pose_variant="heavy")
    lm_f, mf = make_kick_sequence(stance="left", view="front", n_kicks=1, pad_pause_s=0.35)
    lm_s, ms = make_kick_sequence(stance="left", view="side", n_kicks=1, pad_pause_s=0.35)
    base_k = {"front": series_from_raw(lm_f, mf), "side": series_from_raw(lm_s, ms)}
    wins_k, _, _, _ = shared_timeline(base_k, "step_front_low_kick", "left", cfg)
    pad_idx = wins_k["side"][("pad_step", "prepare")]
    tk = np.asarray(base_k["side"]["times"])
    t_end = float(tk[pad_idx[-1]])
    r_base = float(lm_s[0, r_ak, 0])
    for idx in pad_idx:
        dt_before = t_end - float(tk[idx])
        if dt_before >= 0.85:
            x = r_base
        elif dt_before >= 0.65:
            x = r_base + 0.09 * ((0.85 - dt_before) / 0.20)  # 第一拍 +0.30 torso
        elif dt_before >= 0.40:
            x = r_base + 0.09  # 中途短停 0.25s
        elif dt_before >= 0.12:
            x = r_base + 0.09 + 0.075 * ((0.40 - dt_before) / 0.28)  # 主垫步 +0.25 torso
        else:
            x = r_base + 0.165  # 起脚前 0.12s 落稳
        lm_s[idx, r_ak, 0] = x
    res_pad = evaluate_series(
        "step_front_low_kick",
        "left",
        {"front": series_from_raw(lm_f, mf), "side": series_from_raw(lm_s, ms)},
        cfg,
    )
    pad_pause = next(c for c in res_pad["checks"] if c["code"] == "pad_step_pause")
    assert pad_pause["status"] == "not_observed"


def test_step_landing_front_cut_ratio_and_tail_occlusion_boundaries():
    """front_cut 内低有效占比禁止借尾窗伪造通过；front_cut 满帧滞后不被无关尾窗遮挡吞掉且不选遮挡证据。"""
    f_ak = sides_for_stance("front", "left")[5]

    # 1) combo_gap_s=1.00：front_cut 内仅 27% 有效（遮掉真实滞后）、尾窗 100% 恢复 -> 必须 unable
    v_sparse = make_step_hook_views(stance="left", reps=1, seconds=5.0, combo_gap_s=1.00)
    t_sp = np.asarray(v_sparse["side"]["times"])
    for i, t in enumerate(t_sp):
        if 1.0 <= t <= 1.90:
            v_sparse["side"]["landmarks"][i, f_ak, 0] = 0.62 + 0.12 * ((t - 1.0) / 0.90)
        elif t > 1.90:
            v_sparse["side"]["landmarks"][i, f_ak, 0] = 0.74
    diag_sp: dict = {}
    step_combo_windows(
        v_sparse["side"], "left", FeedbackConfig(), diag_sp, action="step_hook_combo"
    )
    f_peak_sp = float(diag_sp["repetitions"][0]["frontPeak"])
    fw_sp = next(p for p in diag_sp["phaseInstances"] if p["segment"] == "step_footwork")
    cut_frames = np.flatnonzero((t_sp >= fw_sp["start"]) & (t_sp <= f_peak_sp + 0.60))
    v_sparse["side"]["valid_mask"][cut_frames[10:], f_ak] = False  # 仅保留 10 帧 (< 60%)
    c25_sp = next(
        c
        for c in evaluate_series("step_hook_combo", "left", v_sparse)["checks"]
        if c["code"] == "step_late_landing"
    )
    assert c25_sp["status"] == "unable"

    # 2) combo_gap_s=0.55：front_cut 内 100% 可见且明确滞后 0.45s、仅 ~cut 尾窗遮挡前脚 -> candidate 且证据全在可见 cut 帧
    v_tail = make_step_hook_views(stance="left", reps=1, seconds=4.5, combo_gap_s=0.55)
    t_tl = np.asarray(v_tail["side"]["times"])
    for i, t in enumerate(t_tl):
        if 1.0 <= t <= 1.82:
            v_tail["side"]["landmarks"][i, f_ak, 0] = 0.62 + 0.12 * ((t - 1.0) / 0.82)
        elif t > 1.82:
            v_tail["side"]["landmarks"][i, f_ak, 0] = 0.74
    diag_tl: dict = {}
    step_combo_windows(v_tail["side"], "left", FeedbackConfig(), diag_tl, action="step_hook_combo")
    f_peak_tl = float(diag_tl["repetitions"][0]["frontPeak"])
    v_tail["side"]["valid_mask"][t_tl > f_peak_tl + 0.60, f_ak] = False
    res_tl = evaluate_series("step_hook_combo", "left", v_tail)
    c25_tl = next(c for c in res_tl["checks"] if c["code"] == "step_late_landing")
    assert c25_tl["status"] == "candidate"
    viol_frames = [
        item["frame"] for item in c25_tl["evidenceDetails"] if item["purpose"] == "violation"
    ]
    assert viol_frames and all(bool(v_tail["side"]["valid_mask"][f, f_ak]) for f in viol_frames)

    # 3) 正常无滞后时仅 ~cut 尾窗遮挡前脚（后脚全程可见）-> 不误伤正常通过
    v_pass = make_step_hook_views(stance="left", reps=1, seconds=4.5, combo_gap_s=0.55)
    t_ps = np.asarray(v_pass["side"]["times"])
    diag_ps: dict = {}
    step_combo_windows(v_pass["side"], "left", FeedbackConfig(), diag_ps, action="step_hook_combo")
    f_peak_ps = float(diag_ps["repetitions"][0]["frontPeak"])
    v_pass["side"]["valid_mask"][t_ps > f_peak_ps + 0.60, f_ak] = False
    c25_ps = next(
        c
        for c in evaluate_series("step_hook_combo", "left", v_pass)["checks"]
        if c["code"] == "step_late_landing"
    )
    assert c25_ps["status"] == "not_observed"


def test_hook_peak_preserves_front_looping_c66_and_side_sweep_arc():
    """正面先外引再内摆的环弧 C66 与侧面 0.25s 连贯横摆弧均不被峰值前沿截断。"""
    cfg = FeedbackConfig()

    # 1) 正面环弧大幅摆拳：先外引至 (0.78, 0.14)，再在 0.30s 内内摆至第 44 帧 (0.34, 0.24)
    n_f = 90
    times_f = np.arange(n_f, dtype=float) / FPS
    baseline_f = np.arange(12)
    lm_f = np.repeat(stance_pose("left", "front")[None], n_f, axis=0)
    lm_f[:, [L_SHOULDER, R_SHOULDER, L_WRIST], :2] = [(0.60, 0.25), (0.40, 0.25), (0.58, 0.22)]
    for i in range(25, 35):
        u = (i - 25) / 9.0
        lm_f[i, L_WRIST, :2] = (0.58 + 0.20 * u, 0.22 - 0.08 * u)  # 外引：无符号距离已达高位
    for i in range(35, 45):
        u = (i - 34) / 10.0
        lm_f[i, L_WRIST, :2] = (0.78 - 0.44 * u, 0.14 + 0.10 * u)  # 0.30s 内摆过中线至 0.34
    for i in range(45, 55):
        u = (i - 44) / 10.0
        lm_f[i, L_WRIST, :2] = (0.34 + 0.24 * u, 0.24 - 0.02 * u)
    mask_f = np.ones((n_f, 33), dtype=bool)
    obs_f = observe_hook(
        lm_f,
        mask_f,
        times_f,
        chain_torso(lm_f, mask_f),
        baseline_f,
        [baseline_f],
        "front",
        "left",
        cfg,
        "front",
    )
    assert len(obs_f.events) == 1 and obs_f.events[0].peak == 44
    lm_s_stub = np.repeat(stance_pose("left", "side")[None], n_f, axis=0)
    res_c66 = evaluate_series(
        "front_hook",
        "left",
        {
            "front": rule_series(lm_f, range(n_f), FPS, 1.0),
            "side": rule_series(lm_s_stub, range(n_f), FPS, 1.0),
        },
        cfg,
    )
    c66 = next(c for c in res_c66["checks"] if c["code"] == "hook_over_center")
    assert c66["status"] == "candidate"

    # 2) 侧面连贯横摆弧：1.00..1.15s 前伸至 85%，随后 1.15..1.40s（0.25s）横摆至第 42 帧小臂最短投影
    n_s = 90
    times_s = np.arange(n_s, dtype=float) / FPS
    baseline_s = np.arange(12)
    lm_s = np.repeat(stance_pose("left", "side")[None], n_s, axis=0)
    base_s = lm_s[0].copy()
    for i in range(30, 50):
        t = times_s[i]
        if t <= 1.15:
            u = (t - 1.00) / 0.15
            wr_dx = 0.16 * 0.88 * u
            el_dx = 0.08 * u
        elif t <= 1.40:
            u = (t - 1.15) / 0.25
            wr_dx = 0.16 * (0.88 + 0.12 * u)
            el_dx = 0.08 + 0.095 * u  # 肘持续前追至第 42 帧（1.40s）投影最短
        else:
            u = max(0.0, (50 - i) / 8.0)
            wr_dx = 0.16 * u
            el_dx = 0.175 * u
        lm_s[i, 13, :2] = (base_s[13, 0] + el_dx, 0.26)
        lm_s[i, 15, :2] = (base_s[15, 0] + wr_dx, 0.25)
    mask_s = np.ones((n_s, 33), dtype=bool)
    obs_s = observe_hook(
        lm_s,
        mask_s,
        times_s,
        chain_torso(lm_s, mask_s),
        baseline_s,
        [baseline_s],
        "front",
        "left",
        cfg,
        "side",
    )
    assert len(obs_s.events) == 1 and obs_s.events[0].peak == 42
