"""差分用例：正式规则、踢腿质量、考试池评分（录后处理全流程）、模板校验、在线识别。"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from tests.eval_diff import synth


def _rules(rec, c) -> None:
    from core.pose_features import COCO17_SUPPORTED_CAPABILITIES, derive_valid_mask
    from core.rule_scoring import score_rules

    raws = {n: c.raw(n) for n in ("student", "front", "side")}
    noisy = raws["student"].copy()
    noisy[::3, 25:29, 3] = 0.1
    raws["noisy"] = noisy
    raws["tiny"] = raws["student"][:4]
    for name, lm in raws.items():
        for view in ("front", "side", "SIDE", ""):
            for scope in ("both", "stance", "punch", None):
                rec(f"rules/{name}/{view}/{scope}", lambda: score_rules(lm, view=view, action_scope=scope))
        rec(f"rules/{name}/mask", lambda: score_rules(
            lm, view="front", action_scope="both", valid_mask=derive_valid_mask(lm, 0.8), min_valid=12))
        rec(f"rules/{name}/coco", lambda: score_rules(
            lm, view="front", action_scope="both", supported_capabilities=COCO17_SUPPORTED_CAPABILITIES))
        rec(f"rules/{name}/min0", lambda: score_rules(lm, view="side", action_scope="both", min_valid=0))


def _kick(rec, c) -> None:
    from core.kick_quality import score_kick_quality

    student = c.raw("student")
    mask = student[..., 3] >= 0.5
    for kind in ("low_whip", "high_whip", "kick_up", "nope"):
        for s, e in ((0, len(student) - 1), (30, 60), (100, 140), (10, 11)):
            rec(f"kick/{kind}/{s}", lambda: score_kick_quality(student[s:e + 1], mask[s:e + 1], kind))
            rec(f"kick/{kind}/{s}/wh", lambda: score_kick_quality(
                student[s:e + 1], mask[s:e + 1], kind, width=1280, height=720))
    rec("kick/shape", lambda: score_kick_quality(student[:, :20], mask[:, :20], "kick_up"))


def _templates(rec, t) -> None:
    from apps import recording_postprocess as rp

    for key in sorted(k for k in t if k.startswith(("bad:", "pool:")) or k in ("front", "no_meta")):
        rec(f"tpl/try/{key}", lambda: rp.try_load_heavy_template(t[key]))
        rec(f"tpl/pair/{key}", lambda: rp.validate_template_pair(t["pool:直拳"], t[key]))
    rec("tpl/try/none", lambda: (rp.try_load_heavy_template(None), rp.try_load_heavy_template("  ")))
    bad = [t[k] for k in t if k.startswith("bad:")]
    rec("tpl/lists/mixed", lambda: rp.validate_auto_compare_template_lists(
        [t["pool:直拳"], *bad, t["pool:直拳"], str(t["pool:高鞭腿"])], bad))
    rec("tpl/lists/empty", lambda: rp.validate_auto_compare_template_lists(bad, None))
    rec("tpl/auto/one_side", lambda: rp.validate_auto_compare_templates(t["pool:直拳"], t["bad:shape"]))
    rec("tpl/auto/none", lambda: rp.validate_auto_compare_templates(None, t["bad:garbage"]))
    rec("tpl/auto/layout", lambda: rp.validate_auto_compare_templates(t["pool:直拳"], t["bad:layout"]))
    merge = getattr(rp, "merge_view_weighted_actions", None)
    if merge is None:  # P3 起迁入 evaluation.scoring.exam_pool
        from core.evaluation.scoring.exam_pool import merge_view_weighted_actions as merge
    rec("tpl/merge", lambda: merge([
        {"name": "两手攀足_侧面", "score": 0.4, "start": 1, "end": 9}, {"name": "直拳", "score": 0.7, "start": 2, "end": 5},
        {"name": "两手攀足_正面", "score": 0.9, "start": 3, "end": 8}, {"name": "攒拳怒目_正面", "score": 0.5, "start": 0, "end": 4}]))


def _exam_pool(rec, c) -> None:
    from apps import recording_postprocess as rp

    t, work = c.t, c.work
    pool = sorted(k for k in t if k.startswith("pool:"))
    cases = {"both": (pool, pool[:3]), "front_only": (pool, []), "side_only": ([], pool[2:]),
             "with_bad": ([t_ for t_ in pool] + ["bad:shape"], ["bad:nan", pool[0]])}
    for name, (front_keys, side_keys) in cases.items():
        seg = work / "exam" / name
        seg.mkdir(parents=True, exist_ok=True)
        front, side = seg / "front.mp4", seg / "side.mp4"
        c.place(front, "student")
        c.place(side, "side")
        updates = []
        done = threading.Event()

        def on_update(u, updates=updates, done=done):
            updates.append(u)
            if u.status in ("completed", "failed", "skipped", "cancelled"):
                done.set()

        proc = rp.DualRecordingPostProcessor(
            on_update=on_update, transcode=lambda p, _e: p, video_validator=lambda _p: True,
            model_available=lambda: True)
        job = rp.DualRecordingJob(
            segment_id=name, segment_dir=seg, front_source=front, side_source=side, front_frames=180,
            side_frames=180, front_template=t[pool[0]], side_template=t[pool[0]],
            front_templates=tuple(t[k] for k in front_keys), side_templates=tuple(t[k] for k in side_keys))
        proc.submit(job)
        assert done.wait(120), name
        proc.close(5.0)
        rec(f"exam/{name}/updates", lambda: [(u.status, u.message, u.front_score, u.side_score, u.combined_percent,
                                              u.error_code, u.action_scores) for u in updates])
        import json
        payload = json.loads((seg / "result.json").read_text(encoding="utf-8"))
        payload.pop("created_at"), payload.pop("completed_at")
        rec(f"exam/{name}/result", lambda: payload)


def _online(rec, c) -> None:
    from core import action_compare as ac
    from core import online_matcher as om

    v, t = c.v, c.t
    for key in ("good", "bad_layout", "bad_norm", "bad_shape", "bad_meta", "missing"):
        rec(f"online/load/{key}", lambda: om.load_template_library(t[f"online:{key}"]))
    lib = om.load_template_library(t["online:good"])
    feats, _fps, _ = ac._extract_pose_features(v["student"], pose_variant="full", normalizer=ac.normalize_pose_xy_v3)
    for s, e in ((0, 40), (25, 70), (100, 150), (0, 2)):
        for thr in (0.4, 0.95):
            rec(f"online/window/{s}-{e}/{thr}", lambda: om.match_window(feats[s:e], lib, om.MatcherConfig(reject_score_thr=thr)))
    rec("online/window/empty_lib", lambda: om.match_window(feats[:30], [], om.MatcherConfig()))
    results = []
    matcher = om.OnlineActionMatcher(lib, results.append, cfg=om.MatcherConfig(arm_thr=0.02, disarm_thr=0.01))
    for i, f in enumerate(feats):
        matcher.push(f, timestamp_ms=i * 33.4, frame_index=i)
    # 每个闭合窗口投递一个任务、产出一个结果；等结果数追上投递数再关（push 侧不暴露计数，只能轮询）。
    deadline = time.monotonic() + 30
    while len(results) < matcher._jobs.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.02)
    matcher.close()
    rec("online/stream", lambda: results)


def run(c) -> None:
    _rules(c.rec, c)
    _kick(c.rec, c)
    _templates(c.rec, c.t)
    _exam_pool(c.rec, c)
    _online(c.rec, c)
