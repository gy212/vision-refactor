"""差分用例：学生练习（检查清单、逐帧序列、相位与正侧融合、文本报告、``feedback_config.json`` 覆盖）。"""
from __future__ import annotations

import json
from pathlib import Path

from tests.eval_diff import synth_practice as sp

SCENARIOS = (
    ("straight_combo", "left", ("front", "side"), "straight_combo", {}),
    ("straight_combo", "right", ("front", "side"), "straight_combo", {}),
    ("hook_combo", "left", ("front", "side"), "hook_combo", {}),
    ("front_low_kick", "left", ("front", "side"), "front_low_kick", {}),
    ("rear_low_kick", "right", ("front", "side"), "rear_low_kick", {}),
    ("stance", "left", ("front", "side"), "stance", {}),
    ("front_straight", "left", ("front",), "front_straight", {}),
    ("rear_hook", "left", ("side",), "rear_hook", {}),
    ("straight_combo", "left", ("front", "side"), "broken_combo", {}),
    ("front_straight", "left", ("front", "side"), "front_straight",
     {"fusion_support": 0.55, "trigger_ratio": 0.05, "guard_drop": 0.05}),
    ("rear_straight", "right", ("front", "side"), "rear_straight", {"sample_fps": 15, "max_edge": 160}),
    ("front_hook", "left", ("front", "side"), "rear_hook", {"min_valid": 2, "valid_ratio": 0.3}),
)


def _series(fg, work: Path, action_motion: str, stance: str, view: str, seed: int, cfg):
    video = sp.write_avi(work / "practice" / f"{action_motion}_{stance}_{view}.avi")
    lm, detected = sp.landmarks(action_motion, stance, view, seed)
    data = fg.extract_series(video, cfg, lambda: False, sp.scripted_pipeline(lm, detected))
    data["capture"]["analysisSeconds"] = 0.0  # 计时值不可复现，固定后再录
    return data


def _config_override(rec, fg, work: Path) -> None:
    root = work / "cfg_root"
    root.mkdir(parents=True, exist_ok=True)
    g = fg.load_feedback_config.__globals__
    original = g["repo_root"]
    g["repo_root"] = lambda: root
    try:
        rec("practice/config/missing", fg.load_feedback_config)
        for name, text in (("ok", json.dumps({"pose_variant": "lite", "trigger_ratio": 0.5,
                                               "front_weights": {"stance": {"default": 0.3}, "straight": {"default": 0.5},
                                                                 "hook": {"default": 0.5}, "kick": {"default": 0.5}}})),
                           ("bom", "﻿" + json.dumps({"min_valid": 4})), ("list", "[]"), ("bad_key", '{"nope": 1}'),
                           ("bad_value", '{"fusion_support": 0.4}'), ("bad_json", "{")):
            (root / "feedback_config.json").write_text(text, encoding="utf-8")
            rec(f"practice/config/{name}", fg.load_feedback_config)
    finally:
        g["repo_root"] = original


def real_practice(c) -> None:
    """真实/公开视频：MediaPipe 逐帧推理后跑练习规则（只在 real 模式）。"""
    from core import feedback_geometry as fg

    cfg = fg.FeedbackConfig(pose_variant="lite", max_edge=1280)
    views = {}
    for view in ("front", "side"):
        data = fg.extract_series(Path(c.v["practice_" + view]), cfg, lambda: False)
        data["capture"]["analysisSeconds"] = 0.0  # 计时值不可复现，固定后再录
        views[view] = data
    c.rec("real/practice/series", lambda: views)
    for action in ("straight_combo", "front_low_kick", "stance"):
        for stance in ("left", "right"):
            c.rec(f"real/practice/{action}/{stance}", lambda: fg.evaluate_series(action, stance, views, cfg))


def run(c) -> None:
    from core import action_feedback as af
    from core import feedback_geometry as fg

    rec, work = c.rec, c.work

    for action in af.ACTIONS:
        for stance in ("left", "right"):
            rec(f"checks/{action}/{stance}", lambda: af.build_checks(action, stance))
    rec("checks/bad", lambda: af.build_checks("nope", "left"))
    for cfg_kw in ({}, {"pose_variant": "x"}, {"return_distance": 0.3}, {"front_weights": {"stance": {}}}):
        rec(f"practice/cfg/{sorted(cfg_kw)}", lambda: fg.FeedbackConfig(**cfg_kw))
    for i, (action, stance, views, motion, cfg_kw) in enumerate(SCENARIOS):
        cfg = fg.FeedbackConfig(**cfg_kw)
        series = {view: _series(fg, work, motion, stance, view, 11 * i + (view == "side"), cfg) for view in views}
        rec(f"practice/series/{i}", lambda: series)
        result = fg.evaluate_series(action, stance, series, cfg)
        rec(f"practice/eval/{i}", lambda: result)
        record = {"studentId": "S001", "studentName": "测试", "createdAt": "2026-10-01T10:00:00",
                  "expiresAt": "2026-10-15T10:00:00", "result": result}
        rec(f"practice/report/{i}", lambda: af.format_report(record))
        for check in result["checks"][:6]:
            rec(f"practice/fusion/{i}/{check['id']}", lambda: af.format_fusion(check))
    legacy = {"studentId": "S2", "studentName": "旧", "createdAt": "x", "expiresAt": "y",
              "result": {"action": "stance", "stance": "left", "ruleVersion": "v3", "summary": "旧版", "checks": [],
                         "needsRerecord": True}}
    rec("practice/report/legacy", lambda: af.format_report(legacy))
    _config_override(rec, fg, work)
