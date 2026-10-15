"""差分用例：DTW 原语、代表周期、单/多模板比对、双模板与正侧双流比对。"""
from __future__ import annotations

import numpy as np


def run(c) -> None:
    from core.evaluation import compare_dual_streams
    from core.evaluation import compare_dual_templates as compare_video_to_dual_templates
    from core.evaluation import estimate_period_frames as period
    from core.evaluation import match_video as compare_video_to_templates
    from core.evaluation import multi_subsequence_matches as multi
    from core.evaluation import select_representative_cycle as cycle
    from core.evaluation import subsequence_dtw, subsequence_dtw_with_path
    from core.pose import normalize_pose_xy_v3
    from core.pose.features import extract_features as _extract_pose_features
    from core.services import compare_video as compare_video_to_template

    rec, v, t = c.rec, c.v, c.t
    rng = np.random.default_rng(3)
    for i, (n, m) in enumerate(((5, 40), (12, 12), (1, 9), (20, 60))):
        q = rng.normal(size=(n, 22, 2)).astype(np.float32)
        s = rng.normal(size=(m, 22, 2)).astype(np.float64)
        rec(f"dtw/{i}", lambda: subsequence_dtw(q, s))
        rec(f"dtw_path/{i}", lambda: subsequence_dtw_with_path(q, s))

    feats = np.load(t["cycle"], allow_pickle=True)["features"]
    for fps in (30.0, 12.0, 0.0):
        rec(f"cycle/{fps}", lambda: cycle(feats, fps=fps))
        rec(f"period/{fps}", lambda: period(np.linalg.norm(np.diff(feats.reshape(len(feats), -1), axis=0), axis=1), fps))
    student, _fps, _ = _extract_pose_features(v["student"], pose_variant="full", normalizer=normalize_pose_xy_v3)
    rec("multi/default", lambda: multi(feats[:30], student))
    rec("multi/args", lambda: multi(feats[:30], student, baseline=2.0, max_matches=3, exclusion=8, offset=7))
    rec("extract/views", lambda: _extract_pose_features(
        v["gappy"], pose_variant="full", normalizer=normalize_pose_xy_v3, compute_view=True))

    for tname in ("front", "side", "cycle", "legacy_v1", "legacy_v2", "legacy_v3", "alias", "image_mode", "no_meta"):
        for vname in ("student", "gappy", "mirror"):
            rec(f"single/{tname}/{vname}", lambda: compare_video_to_template(t[tname], v[vname], workers=3))
    rec("single/pose_override", lambda: compare_video_to_template(t["front"], v["student"], pose_variant="lite"))
    rec("single/body_core_tpl", lambda: compare_video_to_template(t["body_core"], v["student"]))
    rec("single/short", lambda: compare_video_to_template(t["front"], v["short"]))
    rec("single/missing", lambda: compare_video_to_template(t["bad:missing"], v["student"]))

    pools = {"plain": ["front", "side", "cycle"], "mixed_v1_first": ["legacy_v1", "front", "legacy_v2"],
             "mixed_v3_first": ["front", "legacy_v1", "legacy_v2"], "image": ["image_mode", "front"],
             "exam": [k for k in t if k.startswith("pool:")], "empty": []}
    for pname, names in pools.items():
        for vname in ("student", "gappy"):
            rec(f"pool/{pname}/{vname}", lambda: compare_video_to_templates(
                [t[n] for n in names], v[vname], workers=2))
    rec("pool/baseline", lambda: compare_video_to_templates([t["front"]], v["student"], baseline=1.5))

    for vname in ("student", "gappy", "mirror", "long"):
        for rules in (False, True):
            for err in (False, True):
                rec(f"dual_tpl/{vname}/{rules}/{err}", lambda: compare_video_to_dual_templates(
                    t["front"], t["side"], v[vname], enable_rules=rules, enable_error_analysis=err))
    for action, stance in (("stance", "left"), ("front_straight", "left"), ("hook_combo", "right"),
                           ("rear_low_kick", "left"), ("STANCE", "left"), ("stance", "")):
        rec(f"dual_tpl/rule/{action}/{stance}", lambda: compare_video_to_dual_templates(
            t["cycle"], t["side"], v["student"], enable_rules=True, rule_action=action, rule_stance=stance))
    rec("dual_tpl/weights", lambda: compare_video_to_dual_templates(
        t["cycle"], t["front"], v["student"], w_front=0.9, w_side=0.5, baseline=2.0, enable_error_analysis=True))
    rec("dual_tpl/short", lambda: compare_video_to_dual_templates(t["front"], t["side"], v["short"], enable_rules=True))
    rec("dual_tpl/mixed_version", lambda: compare_video_to_dual_templates(t["front"], t["legacy_v2"], v["student"]))
    rec("dual_tpl/legacy_pair", lambda: compare_video_to_dual_templates(
        t["legacy_v1"], t["no_meta"], v["student"], enable_rules=True, enable_error_analysis=True))
    rec("dual_tpl/image", lambda: compare_video_to_dual_templates(t["image_mode"], t["side"], v["student"], workers=3))

    for fv, sv in (("student", "mirror"), ("front", "side"), ("gappy", "student"), ("long", "short")):
        for rules in (False, True):
            rec(f"dual_stream/{fv}/{sv}/{rules}", lambda: compare_dual_streams(
                t["front"], t["side"], v[fv], v[sv], enable_rules=rules, enable_error_analysis=True))
    rec("dual_stream/cycle", lambda: compare_dual_streams(
        t["cycle"], t["cycle"], v["student"], v["gappy"], enable_rules=True, rule_action="straight_combo",
        w_front=0.25, w_side=0.75, baseline=4.0))
    rec("dual_stream/mixed_version", lambda: compare_dual_streams(
        t["legacy_v1"], t["front"], v["student"], v["student"]))
    rec("dual_stream/image", lambda: compare_dual_streams(
        t["image_mode"], t["image_mode"], v["student"], v["gappy"], workers=2, enable_error_analysis=True))
