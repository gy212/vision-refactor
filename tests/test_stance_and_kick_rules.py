# -*- coding: utf-8 -*-
"""Step 2 六项旧规则优化专项单测（实战式膝角/后脚外撇、护手开肘、低鞭腿碾脚内旋/最远伸展锚点/支撑腿髋上限）。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.evaluation import FeedbackConfig, evaluate_series, series_from_raw  # noqa: E402
from core.evaluation.body_sides import sides_for_stance  # noqa: E402
from core.evaluation.kick_events import segment_kick_events  # noqa: E402
from core.evaluation.kick_phases import kick_combo_windows  # noqa: E402
from core.evaluation.measures import feet as feet_mod  # noqa: E402
from core.evaluation.measures.feet import (  # noqa: E402
    foot_heel_toe_forward_projection,
    measure_pivot_heel,
    measure_rear_foot_splay,
)
from core.evaluation.measures.guard import measure_guard_elbow_flare  # noqa: E402
from core.evaluation.rules.engine import evaluate_rules  # noqa: E402
from core.evaluation.rules.kick_rules import kick_landmarks  # noqa: E402
from core.evaluation.rules.rule_descriptions import KICK_PARAMS, PARAMS, PRIMARY_VIEW  # noqa: E402
from core.evaluation.rules.sanda_v5 import (  # noqa: E402
    RULE_VERSION,
    build_checks,
    catalog_checks,
    rule_profile,
)
from core.evaluation.rules.spec import RuleContext  # noqa: E402
from core.pose.landmarks import (  # noqa: E402
    L_ELBOW,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    MOUTH_L,
    MOUTH_R,
    NOSE,
    R_ELBOW,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
)
from core.reporting.issue_groups import group_action_issues  # noqa: E402
from core.reporting.practice_model import build_report_model  # noqa: E402

FPS = 30.0


def _stance_frame(stance: str = "left", view: str = "side") -> np.ndarray:
    lm = np.zeros((33, 4), dtype=np.float32)
    lm[:, 3] = 0.95
    dir_x = 1.0 if stance == "left" else -1.0
    f_sh, f_el, f_wr, f_hp, f_kn, f_ak, f_hl, f_to = sides_for_stance("front", stance)
    r_sh, r_el, r_wr, r_hp, r_kn, r_ak, r_hl, r_to = sides_for_stance("rear", stance)
    if view == "side":
        lm[NOSE, :2] = (0.50 + 0.05 * dir_x, 0.12)
        lm[[MOUTH_L, MOUTH_R], :2] = (0.50 + 0.04 * dir_x, 0.14)
        lm[[f_sh, r_sh], :2] = (0.50, 0.25)
        lm[[f_hp, r_hp], :2] = (0.50, 0.55)
        lm[f_el, :2] = (0.50 + 0.05 * dir_x, 0.34)
        lm[f_wr, :2] = (0.50 + 0.10 * dir_x, 0.18)
        lm[r_el, :2] = (0.50 + 0.02 * dir_x, 0.34)
        lm[r_wr, :2] = (0.50 + 0.05 * dir_x, 0.18)
        lm[f_kn, :2] = (0.50 + 0.10 * dir_x, 0.73)
        lm[f_ak, :2] = (0.50 + 0.14 * dir_x, 0.90)
        lm[f_hl, :2] = (0.50 + 0.13 * dir_x, 0.92)
        lm[f_to, :2] = (0.50 + 0.17 * dir_x, 0.92)
        lm[r_kn, :2] = (0.50 - 0.08 * dir_x, 0.73)
        lm[r_ak, :2] = (0.50 - 0.10 * dir_x, 0.90)
        lm[r_hl, :2] = (0.50 - 0.11 * dir_x, 0.92)
        lm[r_to, :2] = (0.50 - 0.07 * dir_x, 0.92)
    else:
        lm[NOSE, :2] = (0.50, 0.12)
        lm[MOUTH_L, :2] = (0.49, 0.14)
        lm[MOUTH_R, :2] = (0.51, 0.14)
        lm[L_SHOULDER, :2] = (0.40, 0.25)
        lm[R_SHOULDER, :2] = (0.60, 0.25)
        lm[L_HIP, :2] = (0.44, 0.55)
        lm[R_HIP, :2] = (0.56, 0.55)
        lm[L_ELBOW, :2] = (0.41, 0.34)
        lm[L_WRIST, :2] = (0.44, 0.18)
        lm[R_ELBOW, :2] = (0.59, 0.34)
        lm[R_WRIST, :2] = (0.56, 0.18)
        lm[L_KNEE, :2] = (0.43, 0.73)
        lm[R_KNEE, :2] = (0.57, 0.73)
        lm[[27, 29, 31], 0] = (0.43, 0.42, 0.44)
        lm[[28, 30, 32], 0] = (0.57, 0.58, 0.56)
        lm[[27, 28], 1] = 0.90
        lm[[29, 30, 31, 32], 1] = 0.92
    return lm


def test_opt1_and_opt2_knee_179_and_guard_elbow_flare_45_shared_thresholds():
    assert RULE_VERSION == "sanda-rules-mediapipe-2026-11-22-v5.22"
    assert PARAMS["stance_knee_max"] == 179.0
    assert KICK_PARAMS["kick_guard_elbow_flare"] == 45.0
    assert FeedbackConfig().kick_guard_elbow_flare == 45.0
    assert len(catalog_checks("left")) == 149

    # 优化 ①：179° 阈值统一生效于实战式起止与全部拳法复用段
    cfg = FeedbackConfig()
    for action, seg, phase in (
        ("stance", "stance", "start"),
        ("stance", "stance", "end"),
        ("front_straight", "front_straight", "finish"),
        ("rear_hook", "rear_hook", "motion"),
    ):
        chk = next(
            c
            for c in build_checks(action, "left")
            if c["segment"] == seg and c["phase"] == phase and c["code"] == "stance_knee"
        )
        assert "179°" in chk["standard"]
        assert chk["parameterOverrides"]["stance_knee_max"] == 179.0

    lm = np.repeat(_stance_frame("left", "side")[None], 3, axis=0)
    angles = np.deg2rad([178.5, 179.0, 179.5])
    for hip, knee, ankle in ((23, 25, 27), (24, 26, 28)):
        lm[:, hip, :2], lm[:, knee, :2] = (0.5, 0.3), (0.5, 0.5)
        lm[:, ankle, :2] = np.column_stack((0.5 + 0.2 * np.sin(angles), 0.5 - 0.2 * np.cos(angles)))
    mask = np.ones(lm.shape[:2], bool)
    for action in ("stance", "front_straight", "front_hook"):
        rule = next(
            s for s in rule_profile(action, "left", cfg).rules if s.code.endswith(":stance_knee")
        )
        ctx = RuleContext(view="side", action=action, stance="left", params=cfg)
        bad, valid, _ = rule.check(lm, mask, lm, mask, ctx)
        assert valid.all() and bad.tolist() == [False, False, True]

    # 优化 ②：45° 护手开肘上限在前/后低鞭腿统一共用（44° 通过，46° 违规，缺点无效）
    for action in ("front_low_kick", "rear_low_kick"):
        chk = next(c for c in build_checks(action, "left") if c["code"] == "guard_elbow")
        assert "45°" in chk["standard"]
        assert chk["parameterOverrides"]["kick_guard_elbow_flare"] == 45.0

    lm_f = np.repeat(_stance_frame("left", "front")[None], 2, axis=0)
    torso = np.full(2, 0.30)
    for idx, deg in enumerate((44.0, 46.0)):
        rad = math.radians(deg)
        lm_f[idx, R_SHOULDER, :2] = (0.60, 0.25)
        lm_f[idx, R_HIP, :2] = (0.60, 0.55)
        lm_f[idx, R_ELBOW, :2] = (0.60 + 0.20 * math.sin(rad), 0.25 + 0.20 * math.cos(rad))
    bad_el, val_el, _ = measure_guard_elbow_flare(
        lm_f, np.ones(lm_f.shape[:2], bool), torso, R_ELBOW, R_SHOULDER, R_HIP
    )
    assert val_el.all() and bad_el.tolist() == [False, True]
    miss = np.ones(lm_f.shape[:2], bool)
    miss[:, R_ELBOW] = False
    assert not measure_guard_elbow_flare(lm_f, miss, torso, R_ELBOW, R_SHOULDER, R_HIP)[1].any()


@pytest.mark.parametrize("stance", ["left", "right"])
def test_opt3_and_opt6_shared_foot_projection_for_pivot_inward_and_stance_rear_toes(stance):
    assert not hasattr(feet_mod, "measure_toe_spread")
    assert PRIMARY_VIEW["stance_toes"] == "side"
    dir_x = 1.0 if stance == "left" else -1.0
    _, _, _, _, _, _, f_hl, f_to = sides_for_stance("front", stance)
    _, _, _, _, _, _, r_hl, r_to = sides_for_stance("rear", stance)

    # 共享原语核对：脚跟相对脚尖前向投影供优化 ③ 与优化 ⑥ 共同复用
    lm = np.repeat(_stance_frame(stance, "side")[None], 3, axis=0)
    mask = np.ones(lm.shape[:2], bool)
    torso = np.full(3, 0.30)
    # 帧 0: heel 领先 toe (+0.03); 帧 1: heel == toe (0.0); 帧 2: heel 落后 toe (-0.03)
    lm[0, r_hl, 0], lm[0, r_to, 0] = 0.50 + 0.03 * dir_x, 0.50
    lm[1, r_hl, 0], lm[1, r_to, 0] = 0.50, 0.50
    lm[2, r_hl, 0], lm[2, r_to, 0] = 0.50 - 0.03 * dir_x, 0.50
    proj, valid = foot_heel_toe_forward_projection(lm, mask, torso, r_hl, r_to, dir_x)
    assert valid.all() and proj[0] > 0 and proj[1] == pytest.approx(0.0) and proj[2] < 0

    # 优化 ③：低鞭腿碾脚内旋，脚跟领先于脚尖才算到位（proj <= 0 判错）
    bad_piv, val_piv, det_piv = measure_pivot_heel(lm, mask, torso, r_hl, r_to, dir_x)
    assert val_piv.all() and bad_piv.tolist() == [False, True, True]
    assert "支撑脚跟相对脚尖前向投影" in det_piv
    for action in ("front_low_kick", "rear_low_kick"):
        piv_chk = next(c for c in build_checks(action, stance) if c["code"] == "pivot_heel")
        assert piv_chk["name"] == "支撑脚未碾脚内旋"
        assert "内旋碾地" in piv_chk["standard"] and "脚跟要转到脚尖前面" in piv_chk["standard"]

    # 优化 ⑥：实战式后脚外撇仅看侧面后脚（toe_proj < 0 判错，重合 0.0 通过），前脚不查
    bad_spl, val_spl, det_spl = measure_rear_foot_splay(lm, mask, torso, r_hl, r_to, dir_x)
    assert val_spl.all() and bad_spl.tolist() == [True, False, False]
    assert "后脚尖相对后脚跟前向投影" in det_spl

    stance_checks = {c["id"]: c for c in build_checks("stance", stance)}
    for phase in ("start", "end"):
        item = stance_checks[f"stance:{phase}::stance_toes"]
        assert item["name"] == "后脚外撇过度（脚尖落后于脚跟）"
        assert (
            item["standard"]
            == "实战式后脚保持斜向前方，不能过度向外撇；侧面看后脚尖落后到后脚跟后面就算错（不查前脚）"
        )
        assert item["primaryView"] == "side"
        assert item["requiredLandmarks"] == [r_hl, r_to]

    # 端到端验证：前脚反向外撇但后脚正常 -> 通过；后脚尖落后于后脚跟 -> 违规；缺朝向 -> unable
    n = 24
    meta = {"width": 100, "height": 100, "fps": FPS, "start_frame": 0}
    lm_ok_s = np.repeat(_stance_frame(stance, "side")[None], n, axis=0)
    lm_ok_f = np.repeat(_stance_frame(stance, "front")[None], n, axis=0)
    lm_ok_s[:, f_to, 0] = lm_ok_s[:, f_hl, 0] - 0.08 * dir_x  # 前脚故意反向外撇，仍应忽略
    res_ok = evaluate_series(
        "stance",
        stance,
        {"front": series_from_raw(lm_ok_f, meta), "side": series_from_raw(lm_ok_s, meta)},
    )
    for phase in ("start", "end"):
        assert (
            next(c for c in res_ok["checks"] if c["id"] == f"stance:{phase}::stance_toes")["status"]
            == "not_observed"
        )

    lm_bad_s = lm_ok_s.copy()
    lm_bad_s[:, r_to, 0] = lm_bad_s[:, r_hl, 0] - 0.04 * dir_x  # 后脚尖落后于后脚跟
    res_bad = evaluate_series(
        "stance",
        stance,
        {"front": series_from_raw(lm_ok_f, meta), "side": series_from_raw(lm_bad_s, meta)},
    )
    for phase in ("start", "end"):
        assert (
            next(c for c in res_bad["checks"] if c["id"] == f"stance:{phase}::stance_toes")[
                "status"
            ]
            == "candidate"
        )

    # 缺失朝向参照（遮挡鼻/口/耳）必须记录测量问题并返回 unable
    lm_nof = lm_ok_s[:8].copy()
    mask_nof = np.ones(lm_nof.shape[:2], bool)
    mask_nof[:, [NOSE, MOUTH_L, MOUTH_R, 7, 8]] = False
    issues: list[str] = []
    cfg = FeedbackConfig()
    ctx = RuleContext(
        view="side",
        action="stance",
        stance=stance,
        phase_windows={"stance:start": np.arange(8)},
        baseline=np.arange(8),
        params=cfg,
        measurement_issues=issues,
    )
    out = evaluate_rules(lm_nof, mask_nof, rule_profile("stance", stance, cfg), context=ctx)
    toes_out = next(o for o in out if o.spec.code == "stance:start::stance_toes")
    assert toes_out.state == "unable" and "面朝方向" in toes_out.detail


@pytest.mark.parametrize("stance", ["left", "right"])
def test_opt4_kick_ext_idx_furthest_frame_narrow_finish_and_arm_swing_finish_only(stance):
    # 验证前/后低鞭腿均仅在 finish 检查 arm_swing，不在 motion 重复检查
    for action in ("front_low_kick", "rear_low_kick"):
        arms = [c for c in build_checks(action, stance) if c["code"] == "arm_swing"]
        assert len(arms) == 1 and arms[0]["phase"] == "finish"

    # 构造“提膝最高点(t1=38) 与 腿蹬出最远帧(t2=46)”明显分离的低鞭腿
    dir_x = 1.0 if stance == "left" else -1.0
    f_sh, f_el, f_wr, f_hp, f_kn, f_ak, f_hl, f_to = sides_for_stance("front", stance)
    r_sh, r_el, r_wr, r_hp, r_kn, r_ak, r_hl, r_to = sides_for_stance("rear", stance)

    n = 90
    meta = {"width": 100, "height": 100, "fps": FPS, "start_frame": 0}
    lm_s = np.repeat(_stance_frame(stance, "side")[None], n, axis=0)
    lm_f = np.repeat(_stance_frame(stance, "front")[None], n, axis=0)
    t_start, t_chamber, t_ext_true, t_end = 30, 38, 46, 56
    for i in range(t_start, t_end + 1):
        if i <= t_chamber:
            u = (i - t_start) / (t_chamber - t_start)
            ank_y = 0.90 - 0.30 * u  # 提膝最高点达 0.60
            ank_x = 0.50 + (0.14 + 0.04 * u) * dir_x  # 仍收在身下，前伸很小
        elif i <= t_ext_true:
            u = (i - t_chamber) / (t_ext_true - t_chamber)
            ank_y = 0.60 + 0.09 * u  # 鞭出击打高度略降至 0.69（仍达 70% 抬腿高）
            ank_x = 0.50 + (0.18 + 0.24 * u) * dir_x  # 腿蹬出最远至 +0.42
        else:
            u = (i - t_ext_true) / (t_end - t_ext_true)
            ank_y = 0.69 + 0.21 * u
            ank_x = 0.50 + (0.42 - 0.36 * u) * dir_x
        lm_s[i, f_ak, :2] = (ank_x, ank_y)
        lm_s[i, f_kn, :2] = (ank_x - 0.07 * dir_x, 0.55 + (ank_y - 0.55) * 0.5)
        lm_s[i, f_to, :2] = (ank_x + 0.05 * dir_x, ank_y)
        lm_s[i, f_hl, :2] = (ank_x - 0.02 * dir_x, ank_y + 0.02)
        lm_s[i, r_hl, 0] = 0.50 - 0.09 * dir_x
        lm_s[i, r_to, 0] = 0.50 - 0.12 * dir_x

    ser_s = series_from_raw(lm_s, meta)
    events = segment_kick_events(ser_s, "front_low_kick", stance)
    assert len(events) == 1
    assert events[0]["peak_idx"] == t_chamber
    assert events[0]["ext_idx"] == t_ext_true

    # finish 窗口严格收紧在 t_ext ± 0.06s（不含早期提膝帧 t_chamber）
    cfg = FeedbackConfig()
    wins = kick_combo_windows(ser_s, "front_low_kick", stance, cfg)
    fin_idx = wins[("front_low_kick", "finish")]
    assert t_ext_true in fin_idx and t_chamber not in fin_idx
    assert abs(float(ser_s["times"][fin_idx[-1]] - ser_s["times"][fin_idx[0]])) <= 0.14

    # 仅在最远伸展帧 finish 甩手、不在提膝期甩手 -> arm_swing 应通过；反之仅提膝期甩手 -> 违规
    lm_f_ok = lm_f.copy()
    lm_f_ok[t_ext_true - 2 : t_ext_true + 3, f_el, :2] = (lm_f_ok[0, f_sh, 0], 0.37)
    lm_f_ok[t_ext_true - 2 : t_ext_true + 3, f_wr, :2] = (lm_f_ok[0, f_sh, 0], 0.50)
    res_ok = evaluate_series(
        "front_low_kick", stance, {"front": series_from_raw(lm_f_ok, meta), "side": ser_s}, cfg
    )
    assert next(c for c in res_ok["checks"] if c["code"] == "arm_swing")["status"] == "not_observed"

    lm_f_bad = lm_f.copy()
    lm_f_bad[t_chamber - 2 : t_chamber + 3, f_el, :2] = (lm_f_bad[0, f_sh, 0], 0.37)
    lm_f_bad[t_chamber - 2 : t_chamber + 3, f_wr, :2] = (lm_f_bad[0, f_sh, 0], 0.50)
    res_bad = evaluate_series(
        "front_low_kick", stance, {"front": series_from_raw(lm_f_bad, meta), "side": ser_s}, cfg
    )
    assert next(c for c in res_bad["checks"] if c["code"] == "arm_swing")["status"] == "candidate"

    # 边界兜底：缺脚尖退回脚踝、缺面朝退欧氏距离、互换帧不劫持 ext_idx、低帧率 15fps 不因窗窄变 unable
    lm_no_toe = lm_s.copy()
    lm_no_toe[:, f_to, 3] = 0.0
    assert (
        segment_kick_events(series_from_raw(lm_no_toe, meta), "front_low_kick", stance)[0][
            "ext_idx"
        ]
        == t_ext_true
    )

    lm_no_face = lm_s.copy()
    lm_no_face[:, [NOSE, MOUTH_L, MOUTH_R, 7, 8], 3] = 0.0
    assert (
        segment_kick_events(series_from_raw(lm_no_face, meta), "front_low_kick", stance)[0][
            "ext_idx"
        ]
        == t_ext_true
    )

    lm_swap = lm_s.copy()
    lm_swap[t_chamber, r_ak, 1] = 0.50  # 支撑踝高于踢腿踝 -> 判定左右腿互换
    lm_swap[t_chamber, f_ak, 1] = 0.72
    lm_swap[t_chamber, f_to, 0] = 0.50 + 0.80 * dir_x  # 伪造极大前伸不应劫持 ext_idx
    assert (
        segment_kick_events(series_from_raw(lm_swap, meta), "front_low_kick", stance)[0]["ext_idx"]
        == t_ext_true
    )

    meta_15 = {"width": 100, "height": 100, "fps": 15.0, "start_frame": 0}
    ser_15 = series_from_raw(lm_s[::2], meta_15)
    wins_15 = kick_combo_windows(ser_15, "front_low_kick", stance, cfg)
    assert len(wins_15[("front_low_kick", "finish")]) >= cfg.min_valid

    # issue_groups 不再合并已收口为 finish-only 的 arm_swing
    single_arm = [c for c in res_bad["checks"] if c["code"] == "arm_swing"]
    grouped = group_action_issues(
        build_report_model(
            {"action": "front_low_kick", "stance": stance, "result": {"checks": single_arm}}
        ).candidate_checks,
        "front_low_kick",
    )
    assert len(grouped) == 1 and grouped[0].code == "arm_swing"


@pytest.mark.parametrize(
    ("action", "kick_role", "sup_role", "stance"),
    [
        ("front_low_kick", "front", "rear", "left"),
        ("front_low_kick", "front", "rear", "right"),
        ("rear_low_kick", "rear", "front", "left"),
        ("rear_low_kick", "rear", "front", "right"),
    ],
)
def test_opt5_kick_height_upper_bound_uses_support_hip_not_kick_hip(
    action, kick_role, sup_role, stance
):
    _, _, _, kick_hip, _, _, _, kick_toe = sides_for_stance(kick_role, stance)
    _, _, _, sup_hip, sup_kn, _, _, _ = sides_for_stance(sup_role, stance)
    assert kick_landmarks("kick_peak_height_band", kick_role, stance, "side") == (
        kick_toe,
        sup_hip,
        sup_kn,
    )

    chk = next(c for c in build_checks(action, stance) if c["code"] == "kick_height")
    assert "支撑腿胯部" in chk["standard"] and "同侧髋" not in chk["standard"]

    # 故意拉开两髋高度：kick_hip=0.50（更高），sup_hip=0.60（更低），sup_knee=0.78，torso=0.30
    # 令 kick_toe=0.55：低于 kick_hip（旧规则会误通过），但高于 sup_hip 达 0.167*torso > 0.06（新规则必判违规）
    cfg = FeedbackConfig()
    lm = np.repeat(_stance_frame(stance, "side")[None], 6, axis=0)
    lm[:, kick_hip, 1] = 0.50
    lm[:, sup_hip, 1] = 0.60
    lm[:, sup_kn, 1] = 0.78
    lm[:, kick_toe, 1] = 0.55
    mask = np.ones(lm.shape[:2], bool)
    scope = f"{action}:finish"
    ctx = RuleContext(
        view="side",
        action=action,
        stance=stance,
        phase_windows={scope: np.arange(6)},
        baseline=np.arange(6),
        params=cfg,
    )
    out_high = evaluate_rules(lm, mask, rule_profile(action, stance, cfg), context=ctx)
    kh_high = next(o for o in out_high if o.spec.code.endswith(":kick_height"))
    assert kh_high.state == "violation"
    assert any(m["name"] == "脚尖相对支撑腿髋高度" for m in kh_high.measurements)

    # 脚尖落在 sup_hip(0.60) 与 sup_knee(0.78) 之间 -> 通过
    lm_mid = lm.copy()
    lm_mid[:, kick_toe, 1] = 0.68
    out_mid = evaluate_rules(lm_mid, mask, rule_profile(action, stance, cfg), context=ctx)
    assert next(o for o in out_mid if o.spec.code.endswith(":kick_height")).state == "pass"

    # 遮挡支撑腿髋 -> 必为 unable（即使同侧髋可见）
    mask_occ = mask.copy()
    mask_occ[:, sup_hip] = False
    out_occ = evaluate_rules(lm_mid, mask_occ, rule_profile(action, stance, cfg), context=ctx)
    assert next(o for o in out_occ if o.spec.code.endswith(":kick_height")).state == "unable"


def test_opt4_kick_ext_idx_occlusion_swap_and_partial_toe_drop_regressions():
    """回归 P1-1/P1-2：峰值遮挡/互换不劫持至近地边界帧；最远帧局部丢脚尖对齐脚长后仍锁定最远帧。"""
    meta = {"width": 100, "height": 100, "fps": FPS, "start_frame": 0}
    _, _, _, f_hp, f_kn, f_ak, f_hl, f_to = sides_for_stance("front", "left")
    _, _, _, _, _, r_ak, r_hl, r_to = sides_for_stance("rear", "left")

    # P1-1：线性抬腿峰值在帧 42（30..56），膝盖同步抬高以支持 knee_lift 保活
    n = 90
    lm = np.repeat(_stance_frame("left", "side")[None], n, axis=0)
    for i in range(30, 57):
        u = (i - 30) / 12.0 if i <= 42 else (56 - i) / 14.0
        ank_y = 0.90 - 0.25 * u
        ank_x = 0.64 + 0.20 * u
        lm[i, f_ak, :2] = (ank_x, ank_y)
        lm[i, f_kn, :2] = (ank_x - 0.08, 0.73 - 0.24 * u)
        lm[i, f_to, :2] = (ank_x + 0.06, ank_y)
        lm[i, f_hl, :2] = (ank_x - 0.02, ank_y + 0.02)

    # 1a. 遮挡 34..52 帧踝+脚尖（由 knee_lift 保活），ext_idx 应回退至 peak_idx=42 而非贴地边界 53
    lm_occ = lm.copy()
    lm_occ[34:53, [f_ak, f_to], 3] = 0.0
    ev_occ = segment_kick_events(series_from_raw(lm_occ, meta), "front_low_kick", "left")
    assert len(ev_occ) == 1 and ev_occ[0]["ext_idx"] == ev_occ[0]["peak_idx"] == 42

    # 1b. 34..52 帧持续左右踝互换（真实坐标互换），ext_idx 应回退至 peak_idx=42 而非贴地起止边界帧
    lm_sw = lm.copy()
    lm_sw[34:53, f_ak, :2] = lm[34:53, r_ak, :2]
    lm_sw[34:53, r_ak, :2] = lm[34:53, f_ak, :2]
    ev_sw = segment_kick_events(series_from_raw(lm_sw, meta), "front_low_kick", "left")
    assert len(ev_sw) == 1 and ev_sw[0]["ext_idx"] == ev_sw[0]["peak_idx"] == 42

    # 1c. 34..52 帧遮挡踢腿髋，ext_idx 不得落到贴地边界帧
    lm_hp = lm.copy()
    lm_hp[34:53, f_hp, 3] = 0.0
    ev_hp = segment_kick_events(series_from_raw(lm_hp, meta), "front_low_kick", "left")
    assert len(ev_hp) == 1 and ev_hp[0]["ext_idx"] == ev_hp[0]["peak_idx"] == 42

    # P1-2：帧 38..46 脚踝由 0.80 增至最远 0.85，脚尖系统领先 +0.07；最远帧附近丢脚尖仍锁定 46
    lm_blur = np.repeat(_stance_frame("left", "side")[None], n, axis=0)
    for i in range(30, 57):
        if i <= 38:
            u = (i - 30) / 8.0
            ank_y = 0.90 - 0.30 * u
            ank_x = 0.64 + 0.16 * u  # 帧 38 达 0.80
        elif i <= 46:
            u = (i - 38) / 8.0
            ank_y = 0.60 + 0.08 * u
            ank_x = 0.80 + 0.05 * u  # 帧 43 为 0.83125，帧 46 真正最远 0.85
        else:
            u = (i - 46) / 10.0
            ank_y = 0.68 + 0.22 * u
            ank_x = 0.85 - 0.21 * u
        lm_blur[i, f_ak, :2] = (ank_x, ank_y)
        lm_blur[i, f_kn, :2] = (ank_x - 0.07, 0.55 + (ank_y - 0.55) * 0.5)
        lm_blur[i, f_to, :2] = (ank_x + 0.07, ank_y)
        lm_blur[i, f_hl, :2] = (ank_x - 0.02, ank_y + 0.02)
        lm_blur[i, r_hl, 0], lm_blur[i, r_to, 0] = 0.41, 0.38

    for drop_slice in (slice(44, 49), slice(39, 49)):
        lm_drop = lm_blur.copy()
        lm_drop[drop_slice, f_to, 3] = 0.0
        ev_drop = segment_kick_events(series_from_raw(lm_drop, meta), "front_low_kick", "left")
        assert len(ev_drop) == 1 and ev_drop[0]["ext_idx"] == 46


@pytest.mark.parametrize("stance", ["left", "right"])
def test_opt4_rear_low_kick_ext_idx_separation_and_finish_only_arm_swing(stance):
    """验证 rear_low_kick 在左右式下同样分离提膝最高点与最远伸展帧，且甩手仅在 finish 判定。"""
    dir_x = 1.0 if stance == "left" else -1.0
    _, _, _, _, _, _, f_hl, f_to = sides_for_stance("front", stance)
    r_sh, r_el, r_wr, _, r_kn, r_ak, r_hl, r_to = sides_for_stance("rear", stance)

    n = 90
    meta = {"width": 100, "height": 100, "fps": FPS, "start_frame": 0}
    lm_s = np.repeat(_stance_frame(stance, "side")[None], n, axis=0)
    lm_f = np.repeat(_stance_frame(stance, "front")[None], n, axis=0)
    t_start, t_chamber, t_ext_true, t_end = 30, 38, 46, 56
    for i in range(t_start, t_end + 1):
        if i <= t_chamber:
            u = (i - t_start) / (t_chamber - t_start)
            ank_y = 0.90 - 0.30 * u
            ank_x = 0.50 + (-0.10 + 0.24 * u) * dir_x
        elif i <= t_ext_true:
            u = (i - t_chamber) / (t_ext_true - t_chamber)
            ank_y = 0.60 + 0.09 * u
            ank_x = 0.50 + (0.14 + 0.28 * u) * dir_x
        else:
            u = (i - t_ext_true) / (t_end - t_ext_true)
            ank_y = 0.69 + 0.21 * u
            ank_x = 0.50 + (0.42 - 0.52 * u) * dir_x
        lm_s[i, r_ak, :2] = (ank_x, ank_y)
        lm_s[i, r_kn, :2] = (ank_x - 0.07 * dir_x, 0.55 + (ank_y - 0.55) * 0.5)
        lm_s[i, r_to, :2] = (ank_x + 0.06 * dir_x, ank_y)
        lm_s[i, r_hl, :2] = (ank_x - 0.02 * dir_x, ank_y + 0.02)
        lm_s[i, f_hl, 0] = 0.50 + 0.16 * dir_x
        lm_s[i, f_to, 0] = 0.50 + 0.12 * dir_x

    ser_s = series_from_raw(lm_s, meta)
    events = segment_kick_events(ser_s, "rear_low_kick", stance)
    assert len(events) == 1
    assert events[0]["peak_idx"] == t_chamber
    assert events[0]["ext_idx"] == t_ext_true

    cfg = FeedbackConfig()
    wins = kick_combo_windows(ser_s, "rear_low_kick", stance, cfg)
    fin_idx = wins[("rear_low_kick", "finish")]
    assert t_ext_true in fin_idx and t_chamber not in fin_idx

    lm_f_ok = lm_f.copy()
    lm_f_ok[t_ext_true - 2 : t_ext_true + 3, r_el, :2] = (lm_f_ok[0, r_sh, 0], 0.37)
    lm_f_ok[t_ext_true - 2 : t_ext_true + 3, r_wr, :2] = (lm_f_ok[0, r_sh, 0], 0.50)
    res_ok = evaluate_series(
        "rear_low_kick", stance, {"front": series_from_raw(lm_f_ok, meta), "side": ser_s}, cfg
    )
    assert next(c for c in res_ok["checks"] if c["code"] == "arm_swing")["status"] == "not_observed"

    lm_f_bad = lm_f.copy()
    lm_f_bad[t_chamber - 2 : t_chamber + 3, r_el, :2] = (lm_f_bad[0, r_sh, 0], 0.37)
    lm_f_bad[t_chamber - 2 : t_chamber + 3, r_wr, :2] = (lm_f_bad[0, r_sh, 0], 0.50)
    res_bad = evaluate_series(
        "rear_low_kick", stance, {"front": series_from_raw(lm_f_bad, meta), "side": ser_s}, cfg
    )
    assert next(c for c in res_bad["checks"] if c["code"] == "arm_swing")["status"] == "candidate"
