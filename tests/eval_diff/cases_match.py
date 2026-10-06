"""差分用例：DTW 原语、代表周期、单/多模板比对、双模板与正侧双流比对。"""
from __future__ import annotations

import importlib

import numpy as np


def _first(*paths):
    """按候选位置取函数：迁移前后它们的位置不同，差分只关心行为。"""
    for mod, name in paths:
        try:
            return getattr(importlib.import_module(mod), name)
        except (ImportError, AttributeError):
            continue
    raise ImportError(paths)


def run(c) -> None:
    from core import action_compare as ac
    from core.pose_features import subsequence_dtw, subsequence_dtw_with_path

    rec, v, t = c.rec, c.v, c.t
    rng = np.random.default_rng(3)
    for i, (n, m) in enumerate(((5, 40), (12, 12), (1, 9), (20, 60))):
        q = rng.normal(size=(n, 22, 2)).astype(np.float32)
        s = rng.normal(size=(m, 22, 2)).astype(np.float64)
        rec(f"dtw/{i}", lambda: subsequence_dtw(q, s))
        rec(f"dtw_path/{i}", lambda: subsequence_dtw_with_path(q, s))

    cycle = _first(("core.evaluation.segmentation", "select_representative_cycle"),
                   ("core.action_compare", "_select_representative_cycle"))
    period = _first(("core.evaluation.segmentation", "estimate_period_frames"),
                    ("core.action_compare", "_estimate_period_frames"))
    multi = _first(("core.evaluation.matching.repetition", "multi_subsequence_matches"),
                   ("core.action_compare", "_multi_subsequence_matches"))
    feats = np.load(t["cycle"], allow_pickle=True)["features"]
    for fps in (30.0, 12.0, 0.0):
        rec(f"cycle/{fps}", lambda: cycle(feats, fps=fps))
        rec(f"period/{fps}", lambda: period(np.linalg.norm(np.diff(feats.reshape(len(feats), -1), axis=0), axis=1), fps))
    student, _fps, _ = ac._extract_pose_features(v["student"], pose_variant="full", normalizer=ac.normalize_pose_xy_v3)
    rec("multi/default", lambda: multi(feats[:30], student))
    rec("multi/args", lambda: multi(feats[:30], student, baseline=2.0, max_matches=3, exclusion=8, offset=7))
    rec("extract/views", lambda: ac._extract_pose_features(
        v["gappy"], pose_variant="full", normalizer=ac.normalize_pose_xy_v3, compute_view=True))

    for tname in ("front", "side", "cycle", "legacy_v1", "legacy_v2", "legacy_v3", "alias", "image_mode", "no_meta"):
        for vname in ("student", "gappy", "mirror"):
            rec(f"single/{tname}/{vname}", lambda: ac.compare_video_to_template(t[tname], v[vname], workers=3))
    rec("single/pose_override", lambda: ac.compare_video_to_template(t["front"], v["student"], pose_variant="lite"))
    rec("single/body_core_tpl", lambda: ac.compare_video_to_template(t["body_core"], v["student"]))
    rec("single/short", lambda: ac.compare_video_to_template(t["front"], v["short"]))
    rec("single/missing", lambda: ac.compare_video_to_template(t["bad:missing"], v["student"]))

    pools = {"plain": ["front", "side", "cycle"], "mixed_v1_first": ["legacy_v1", "front", "legacy_v2"],
             "mixed_v3_first": ["front", "legacy_v1", "legacy_v2"], "image": ["image_mode", "front"],
             "exam": [k for k in t if k.startswith("pool:")], "empty": []}
    for pname, names in pools.items():
        for vname in ("student", "gappy"):
            rec(f"pool/{pname}/{vname}", lambda: ac.compare_video_to_templates(
                [t[n] for n in names], v[vname], workers=2))
    rec("pool/baseline", lambda: ac.compare_video_to_templates([t["front"]], v["student"], baseline=1.5))

    for vname in ("student", "gappy", "mirror", "long"):
        for rules in (False, True):
            for err in (False, True):
                rec(f"dual_tpl/{vname}/{rules}/{err}", lambda: ac.compare_video_to_dual_templates(
                    t["front"], t["side"], v[vname], enable_rules=rules, enable_error_analysis=err))
    for scope in ("stance", "punch", "STANCE", ""):
        rec(f"dual_tpl/scope/{scope}", lambda: ac.compare_video_to_dual_templates(
            t["cycle"], t["side"], v["student"], enable_rules=True, action_scope=scope))
    rec("dual_tpl/weights", lambda: ac.compare_video_to_dual_templates(
        t["cycle"], t["front"], v["student"], w_front=0.9, w_side=0.5, baseline=2.0, enable_error_analysis=True))
    rec("dual_tpl/short", lambda: ac.compare_video_to_dual_templates(t["front"], t["side"], v["short"], enable_rules=True))
    rec("dual_tpl/mixed_version", lambda: ac.compare_video_to_dual_templates(t["front"], t["legacy_v2"], v["student"]))
    rec("dual_tpl/legacy_pair", lambda: ac.compare_video_to_dual_templates(
        t["legacy_v1"], t["no_meta"], v["student"], enable_rules=True, enable_error_analysis=True))
    rec("dual_tpl/image", lambda: ac.compare_video_to_dual_templates(t["image_mode"], t["side"], v["student"], workers=3))

    for fv, sv in (("student", "mirror"), ("front", "side"), ("gappy", "student"), ("long", "short")):
        for rules in (False, True):
            rec(f"dual_stream/{fv}/{sv}/{rules}", lambda: ac.compare_dual_streams(
                t["front"], t["side"], v[fv], v[sv], enable_rules=rules, enable_error_analysis=True))
    rec("dual_stream/cycle", lambda: ac.compare_dual_streams(
        t["cycle"], t["cycle"], v["student"], v["gappy"], enable_rules=True, action_scope="punch",
        w_front=0.25, w_side=0.75, baseline=4.0))
    rec("dual_stream/mixed_version", lambda: ac.compare_dual_streams(
        t["legacy_v1"], t["front"], v["student"], v["student"]))
    rec("dual_stream/image", lambda: ac.compare_dual_streams(
        t["image_mode"], t["image_mode"], v["student"], v["gappy"], workers=2, enable_error_analysis=True))
