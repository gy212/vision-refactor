"""差分用例：模板生成与自动裁剪、动作区间、正侧切分、技术指标、body_core、CLI、batch、离线剖析。"""
from __future__ import annotations

import contextlib
import csv
import io
import sys
from pathlib import Path

import numpy as np

from tests.eval_diff import synth


def _energy(n: int, bursts: tuple[tuple[int, int], ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = np.abs(rng.normal(0.0, 0.01, size=n)).astype(np.float32)
    for s, e_ in bursts:
        e[s:e_] += 0.3
    return e


def _templates_and_ranges(rec, v, work: Path) -> None:
    from core.services import analyze_template_auto_crop, create_template_from_video
    from core.evaluation import normalize_template_meta
    from core.evaluation import estimate_action_range
    from core.workflows.exam.clip import estimate_action_range_from_feature_seq

    out = work / "made"
    out.mkdir(parents=True, exist_ok=True)
    for i, kw in enumerate(({}, {"start": 5, "end": 60}, {"start": -5}, {"end": 999}, {"workers": 3})):
        rec(f"create/{i}", lambda: _template_dump(create_template_from_video(
            v["student"], pose_variant="lite", out_path=out / f"c{i}.npz", **kw)))
    rec("create/long_review", lambda: create_template_from_video(v["long"], out_path=out / "long.npz"))
    rec("create/long_manual", lambda: _template_dump(create_template_from_video(
        v["long"], start=0, end=400, out_path=out / "long_m.npz")))
    for i, (n, bursts, fps) in enumerate(((60, ((10, 30),), 30.0), (1500, ((100, 160), (900, 980)), 30.0),
                                          (1500, ((100, 1400),), 30.0), (1, (), 0.0), (2000, ((5, 40), (1900, 1990)), 25.0))):
        energy = _energy(n, bursts, i)
        rec(f"autocrop/{i}", lambda: _autocrop(analyze_template_auto_crop(energy, fps=fps)))
        for smart in (False, True):
            rec(f"range/{i}/{smart}", lambda: estimate_action_range(
                n + 1, fps=fps, energy=energy, smart_crop=smart))
    rec("range/none", lambda: estimate_action_range(40, fps=30.0))
    rec("range/empty", lambda: estimate_action_range(0))
    feats = np.load(synth.FIX_DIR / "front_template.npz", allow_pickle=True)["features"]
    rec("range/features", lambda: estimate_action_range_from_feature_seq(feats))
    for k, m in (("meta_none", None), ("meta_name", {"feature_layout_name": "x"}), ("meta_full", {"feature_layout": "y"})):
        rec(f"meta/{k}", lambda: normalize_template_meta(None if m is None else dict(m)))


def _template_dump(path: Path):
    d = np.load(path, allow_pickle=True)
    return path, d["features"], d["meta"].item()


def _autocrop(a):
    return a, a.selected_frames, a.retained_ratio, a.duration_s, a.selected_duration_s, a.requires_review, a.to_meta()


def _tech_eval(rec, v) -> None:
    from core.evaluation import evaluate_video_assets, evaluate_video_detail, evaluate_video_full, extract_pose_and_view_scores, split_front_side_segments

    for vname in ("student", "front", "side", "gappy", "mirror"):
        for stance in ("left", "right"):
            for hint in ("auto", "front", "side"):
                rec(f"te/full/{vname}/{stance}/{hint}", lambda: evaluate_video_full(
                    v[vname], pose_variant="full", stance=stance, view_hint=hint))
        rec(f"te/detail/{vname}", lambda: evaluate_video_detail(v[vname], pose_variant="full"))
        rec(f"te/assets/{vname}", lambda: evaluate_video_assets(v[vname], pose_variant="full")[1:])
        _lm, vs, _meta = extract_pose_and_view_scores(v[vname], pose_variant="full")
        rec(f"te/split/{vname}", lambda: split_front_side_segments(vs, 30.0))
        rec(f"te/split/{vname}/loose", lambda: split_front_side_segments(
            vs, 12.0, min_len_sec=0.2, min_mean_diff=0.0, side_thr=1.0, front_thr=0.0, max_transitions=50))


def _body_core(rec, v, t, work: Path) -> None:
    from core.services import create_body_core_template
    from core.evaluation import match_body_core_template

    (work / "bc").mkdir(parents=True, exist_ok=True)
    tpl = create_body_core_template(v["front"], backend="mediapipe", out_path=work / "bc" / "front.npz")
    t["body_core"] = tpl
    rec("bc/create", lambda: _template_dump(tpl))
    rec("bc/match", lambda: match_body_core_template(tpl, v["student"]))
    rec("bc/match_gappy", lambda: match_body_core_template(tpl, v["gappy"], backend="mediapipe"))
    rec("bc/pose33_tpl", lambda: match_body_core_template(t["front"], v["student"]))


@contextlib.contextmanager
def _argv(args):
    old, buf = sys.argv, io.StringIO()
    sys.argv = ["x", *map(str, args)]
    try:
        with contextlib.redirect_stdout(buf):
            yield buf
    finally:
        sys.argv = old


def _cli(rec, v, t, work: Path) -> None:
    from apps import make_template, match_template

    def run(module, args):
        with _argv(args) as buf:
            try:
                module.main()
            except BaseException as exc:  # noqa: BLE001 - 退出码与报错文本一并录下
                return buf.getvalue(), exc
        return buf.getvalue()

    out = work / "cli"
    out.mkdir(parents=True, exist_ok=True)
    rec("cli/make", lambda: (run(make_template, ["--video", v["student"], "--out", out / "m.npz"]),
                             _template_dump(out / "m.npz")))
    rec("cli/make_range", lambda: (run(make_template, ["--video", v["front"], "--pose", "lite", "--start", "3",
                                                       "--end", "50", "--out", out / "r.npz"]), _template_dump(out / "r.npz")))
    rec("cli/make_long", lambda: run(make_template, ["--video", v["long"], "--out", out / "l.npz"]))
    rec("cli/make_bc", lambda: (run(make_template, ["--video", v["side"], "--feature-layout", "body_core_v1",
                                                    "--out", out / "bc.npz"]), _template_dump(out / "bc.npz")))
    for key in ("front", "cycle", "legacy_v1", "legacy_v2", "legacy_v3", "alias", "no_meta", "body_core"):
        rec(f"cli/match/{key}", lambda: run(match_template, ["--template", t[key], "--video", v["gappy"]]))
    rec("cli/match/pose", lambda: run(match_template, ["--template", t["front"], "--video", v["student"], "--pose", "heavy"]))


def _read_outputs(out: Path):
    files = {}
    for p in sorted(out.rglob("*")):
        if p.suffix == ".csv":
            files[p.name] = list(csv.reader(io.StringIO(p.read_text(encoding="utf-8-sig"))))
        elif p.suffix == ".jsonl":
            files[p.name] = p.read_text(encoding="utf-8").splitlines()
    return files


def _batch(rec, work: Path) -> None:
    from batch import batch_dual_compare as bdc
    from tests import golden_harness as H

    std, stu = work / "batch" / "std", work / "batch" / "stu"
    std.mkdir(parents=True, exist_ok=True)
    stu.mkdir(parents=True, exist_ok=True)
    clips = {std / "标准_正面.mp4": "front_src_raw", std / "标准_侧面.mp4": "side_src_raw",
             stu / "张三_正面.mp4": "student_raw", stu / "张三_侧面.mp4": "side_src_raw",
             stu / "李四 front.mp4": "front_src_raw", stu / "李四 side.mp4": "student_raw", stu / "王五.mp4": "student_raw"}
    for path, raw in clips.items():
        path.write_bytes(b"x")
        H.register_video(path, synth.load_raw(raw), fps=synth.FPS)
    for name, extra in (("plain", []), ("rules", ["--rules", "--error-analysis", "--action", "stance"]),
                        ("paired", ["--paired", "--rules", "--error-analysis"]), ("paired_raw", ["--paired", "--export_raw"])):
        out = work / "batch" / f"out_{name}"
        args = ["--standard_dir", std, "--student_dir", stu, "--out_dir", out, *extra]
        with _argv(args) as buf:
            bdc.main()
        rec(f"batch/{name}", lambda: (buf.getvalue(), _read_outputs(out)))


def _offline_profile(rec, work: Path) -> None:
    from analysis import offline_matching_profile as omp

    res = omp.run_fixture_profile(work / "omp")
    for r in res["records"]:
        r["stages"] = [(s["stage"], s["count"]) for s in r["stages"]]
    rec("offline_profile", lambda: res)


def body_core(c) -> None:
    """body_core 模板要先建好，比对与 CLI 用例会用到。"""
    _body_core(c.rec, c.v, c.t, c.work)


def run(c) -> None:
    _templates_and_ranges(c.rec, c.v, c.work)
    _tech_eval(c.rec, c.v)
    if c.mode != "fixture":
        return
    _cli(c.rec, c.v, c.t, c.work)
    _batch(c.rec, c.work)
    _offline_profile(c.rec, c.work)  # 会清空 golden 注册表，放最后
