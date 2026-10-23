# -*- coding: utf-8 -*-
"""样本评估工具（``analysis/sample_eval``）：清单、样本口径、缓存往返、评估行、指标与对比报告，全部用合成小样本。

不读真实视频、不加载模型：姿态提取用假的 ``extract_pose_sequence`` 代替。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.sample_eval import cache as cache_mod  # noqa: E402
from analysis.sample_eval import metrics, report, runs  # noqa: E402
from analysis.sample_eval.evaluate import action_family, default_config, evaluate_pair, phase_row  # noqa: E402
from analysis.sample_eval.manifest import (  # noqa: E402
    ROLE_CORRECT,
    ROLE_ERROR,
    ROLE_EXCLUDED,
    ManifestError,
    pair_role,
    parse_verdict,
    read_manifest,
)
from core.pose import PoseSequence, SourceMeta  # noqa: E402

HEADER = "pair_id,repetition,segment_dir,category,person,action,rule_actions,error_item,expected_codes,teacher_verdict,front_rotation,note\n"


def _write(tmp_path: Path, body: str, header: str = HEADER) -> Path:
    path = tmp_path / "manifest.csv"
    path.write_text(header + body, encoding="utf-8-sig")
    return path


# ------------------------------------------------------------------ 清单


def test_manifest_batch1_rows_without_repetition(tmp_path):
    path = _write(tmp_path, "A_stance,,A/实战式,A,,实战式,stance,,,,90,\n"
                            "C7,,C  前半部分/C7,C,,实战式,stance,C7 站距过宽,stance_width,,0,\n")
    pairs = read_manifest(path)
    assert [p.pair_id for p in pairs] == ["A_stance", "C7"]
    assert pairs[0].front_rotation == 90 and pairs[0].rule_actions == ("stance",)
    assert pairs[1].segment_dir == "C  前半部分/C7"  # 目录名里的连续空格原样保留
    assert pairs[1].labels[0].item_id == "C7" and pairs[1].repetitions == ""
    assert pair_role(pairs[0]).role == ROLE_CORRECT
    role = pair_role(pairs[1])
    assert role.role == ROLE_ERROR and role.expected_codes == ("stance_width",) and not role.teacher_confirmed


def test_manifest_batch2_repetition_rows_merge_into_one_pair(tmp_path):
    header = "样本编号,第几次,片段目录,类别,人员编号,动作,规则动作,故意做错的项,预期检查项,教师判定\n"
    path = _write(tmp_path, "C7_S1,2,seg1,C,S1,实战式,stance,C7 站距过宽,stance_width,错\n"
                            "C7_S1,1,seg1,C,S1,实战式,stance,C7 站距过宽,stance_width,C7=错;C1=对\n", header)
    (pair,) = read_manifest(path)
    assert [lb.repetition for lb in pair.labels] == [1, 2] and pair.repetitions == "1|2"
    role = pair_role(pair)
    assert role.role == ROLE_ERROR and role.teacher_confirmed


@pytest.mark.parametrize("body, message", [
    ("X,1,seg,C,,,stance,,,,0,\nX,1,seg,C,,,stance,,,,0,\n", "第几次"),
    ("X,1,seg,C,,,stance,,,,0,\nX,2,other,C,,,stance,,,,0,\n", "segment_dir"),
    ("X,,seg,Z,,,stance,,,,0,\n", "类别"),
    ("X,,seg,C,,,stance,,,,45,\n", "正面旋转"),
    ("X,,seg,C,,,stance,,stance_width,,0,\n", "预期检查项"),
    ("X,,seg,C,,,stance,C7,stance_width,也许,0,\n", "教师判定"),
    (",,seg,C,,,stance,,,,0,\n", "pair_id"),
])
def test_manifest_rejects_malformed_rows(tmp_path, body, message):
    with pytest.raises(ManifestError, match=message):
        read_manifest(_write(tmp_path, body))


def test_verdict_grammar():
    assert parse_verdict("") == {}
    assert parse_verdict("对") == {"*": "对"}
    assert parse_verdict("C7=错；C1 = 对; *=看不清") == {"C7": "错", "C1": "对", "*": "看不清"}


def test_teacher_verdict_moves_pairs_between_roles(tmp_path):
    path = _write(tmp_path, "ok,,s1,C,,,stance,C7 站距过宽,stance_width,C7=对,0,\n"
                            "blur,,s2,C,,,stance,C7 站距过宽,stance_width,看不清,0,\n"
                            "side_err,,s3,B,,,stance,,,C3=错,0,\n"
                            "fine,,s4,B,,,stance,,,对,0,\n")
    roles = {p.pair_id: pair_role(p) for p in read_manifest(path)}
    assert roles["ok"].role == ROLE_EXCLUDED  # 教师判故意项做对了：两类指标都不计
    assert roles["blur"].role == ROLE_EXCLUDED
    assert roles["side_err"].role == ROLE_EXCLUDED  # 正确样本上教师判出别的错
    assert roles["fine"].role == ROLE_CORRECT and roles["fine"].teacher_confirmed


# ------------------------------------------------------------------ 缓存


def _standing_pose(frames: int = 40) -> np.ndarray:
    """左式站立的近似合成姿态（归一化坐标，全部可见），足够让规则引擎跑完。"""
    lm = np.zeros((frames, 33, 4), dtype=np.float64)
    lm[:, :, 0], lm[:, :, 1], lm[:, :, 3] = 0.5, 0.5, 0.99
    points = {0: (0.50, 0.20), 11: (0.46, 0.30), 12: (0.54, 0.30), 13: (0.42, 0.40), 14: (0.58, 0.40),
              15: (0.44, 0.25), 16: (0.55, 0.27), 23: (0.47, 0.55), 24: (0.53, 0.55), 25: (0.45, 0.70),
              26: (0.56, 0.70), 27: (0.43, 0.85), 28: (0.58, 0.85), 29: (0.42, 0.87), 30: (0.59, 0.87),
              31: (0.40, 0.88), 32: (0.61, 0.88)}
    for idx, (x, y) in points.items():
        lm[:, idx, 0], lm[:, idx, 1] = x, y
    return lm


def _sequence(width: int, height: int, frames: int = 40) -> PoseSequence:
    meta = SourceMeta(video="v.mp4", fps=30.0, fps_reported=30.0, frame_count=frames, width=width, height=height,
                      engine="heavy", profile="formal", tracking="sequential", validity_policy="mediapipe",
                      valid_conf_thr=0.5, frames_scanned=frames, inference_size=(width, height))
    return PoseSequence(landmarks=_standing_pose(frames), detected=np.ones(frames, bool), decoded=np.ones(frames, bool),
                        frame_index=np.arange(frames), t_ms=np.arange(frames) * 33,
                        source_wh=np.tile([width, height], (frames, 1)), meta=meta)


class _FakeEngine:
    def __init__(self) -> None:
        self.shapes, self.closed = [], False

    def infer(self, frame, *, timestamp_ms=None):
        self.shapes.append(frame.shape)

    def close(self) -> None:
        self.closed = True


def _pair_dir(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "samples"
    seg = root / "A" / "实战式"
    seg.mkdir(parents=True)
    for view in ("front", "side"):
        (seg / f"{view}.mp4").write_bytes(b"fake")
    manifest = _write(tmp_path, "A_stance,,A/实战式,A,,实战式,stance,,,,90,\nno_rules,,A/实战式,A,,勾拳,,,,,0,\n")
    return root, manifest


def test_cache_extracts_rotated_front_and_is_resumable(tmp_path, monkeypatch):
    root, manifest = _pair_dir(tmp_path)
    calls = []

    def fake_extract(video, spec, profile, *, engine_factory=None):
        calls.append((Path(video).name, type(engine_factory()).__name__))
        return _sequence(1920, 1080)

    monkeypatch.setattr(cache_mod, "extract_pose_sequence", fake_extract)
    monkeypatch.setattr(cache_mod, "create_engine", lambda spec, models_dir, download: _FakeEngine())
    pairs = read_manifest(manifest)
    cache_dir = tmp_path / "cache"
    jobs, missing = cache_mod.plan_jobs(pairs, root, cache_dir)
    assert missing == [] and [(j.pair_id, j.view, j.rotation) for j in jobs] == [("A_stance", "front", 90),
                                                                                 ("A_stance", "side", 0)]
    for job in jobs:  # 主进程里直接调用子进程入口，避开进程池
        cache_mod.extract_one(job)
    assert sorted(calls) == [("front.mp4", "RotatingEngine"), ("side.mp4", "_FakeEngine")]
    assert all(cache_mod.is_fresh(j) for j in jobs)
    assert cache_mod.build_cache(jobs, cache_dir, workers=1, echo=lambda _m: None) == {"reused": 2, "extracted": 0,
                                                                                         "failed": 0}
    front = cache_mod.load_sequence(cache_mod.cache_path(cache_dir, "A_stance", "front"))
    assert tuple(front.source_wh[-1]) == (1080, 1920) and (front.meta.width, front.meta.height) == (1080, 1920)
    views = cache_mod.load_views(pairs[0], cache_dir)
    assert set(views) == {"front", "side"}
    assert views["front"]["landmarks"][0, 0, 0] == pytest.approx(0.5 * 1080 / 1920)  # 宽高比修正按转后的尺寸
    assert views["side"]["valid_mask"].all()


def test_cache_key_change_marks_stale(tmp_path, monkeypatch):
    root, manifest = _pair_dir(tmp_path)
    monkeypatch.setattr(cache_mod, "extract_pose_sequence", lambda *a, **k: _sequence(1920, 1080))
    jobs, _ = cache_mod.plan_jobs(read_manifest(manifest), root, tmp_path / "cache")
    cache_mod.extract_one(jobs[1])
    assert cache_mod.is_fresh(jobs[1])
    (root / "A" / "实战式" / "side.mp4").write_bytes(b"replaced video")  # 视频被换掉
    assert not cache_mod.is_fresh(jobs[1])


def test_missing_model_stops_before_spawning_workers(tmp_path, monkeypatch):
    root, manifest = _pair_dir(tmp_path)
    jobs, _ = cache_mod.plan_jobs(read_manifest(manifest), root, tmp_path / "cache", tmp_path / "no_models")
    assert {j.models_dir for j in jobs} == {str(tmp_path / "no_models")}
    with pytest.raises(SystemExit, match="缺少 MediaPipe heavy"):
        cache_mod.build_cache(jobs, tmp_path / "cache", workers=2, echo=lambda _m: None)


def test_rotating_engine_rotates_before_inference():
    inner = _FakeEngine()
    engine = cache_mod.RotatingEngine(inner, 90)
    engine.infer(np.zeros((2, 3, 3), np.uint8), timestamp_ms=5)
    engine.close()
    assert inner.shapes == [(3, 2, 3)] and inner.closed


def test_missing_video_is_reported(tmp_path):
    root, manifest = _pair_dir(tmp_path)
    (root / "A" / "实战式" / "side.mp4").unlink()
    jobs, missing = cache_mod.plan_jobs(read_manifest(manifest), root, tmp_path / "cache")
    assert [j.view for j in jobs] == ["front"] and len(missing) == 1 and "A_stance/side" in missing[0]


# ------------------------------------------------------------------ 评估与运行目录


def test_evaluate_pair_rows_follow_production_results(tmp_path, monkeypatch):
    root, manifest = _pair_dir(tmp_path)
    monkeypatch.setattr(cache_mod, "extract_pose_sequence", lambda *a, **k: _sequence(1920, 1080))
    pairs = read_manifest(manifest)
    jobs, _ = cache_mod.plan_jobs(pairs, root, tmp_path / "cache")
    for job in jobs:
        cache_mod.extract_one(job)
    views = cache_mod.load_views(pairs[0], tmp_path / "cache")
    result, rows, phase = evaluate_pair(pairs[0], pair_role(pairs[0]), "stance", views, default_config())
    assert [r["check_id"] for r in rows] == [c["id"] for c in result["checks"]]
    assert all(r["fused_status"] == c["status"] for r, c in zip(rows, result["checks"]))
    assert {r["sample_role"] for r in rows} == {ROLE_CORRECT} and not any(r["expected"] for r in rows)
    assert phase["family"] == "stance" and phase["views"] == "front|side"
    runs.write_run(tmp_path / "run", {"label": "t"}, rows, [phase])
    meta, rows_back, phases_back = runs.read_run(tmp_path / "run")
    assert meta == {"label": "t"} and len(rows_back) == len(rows)
    assert rows_back[0]["expected"] is False and phases_back[0]["fusible"] is phase["fusible"]
    assert (tmp_path / "run" / "checks.csv").read_bytes().startswith(b"\xef\xbb\xbf")


@pytest.mark.parametrize("action, family", [("stance", "stance"), ("straight_combo", "combo"),
                                            ("rear_hook", "single"), ("front_low_kick", "kick")])
def test_action_family(action, family):
    assert action_family(action) == family


@pytest.mark.parametrize("side_reason, side_source, reliable, fusible", [
    ("已检测到完整动作", "own_view", True, True),
    ("未检测到完整离开并回收的动作", "front_synchronized_timing", True, True),  # 共用正面的同步动作时间段也算有窗口
    ("未检测到完整离开并回收的动作", "own_view", True, False),
    ("已检测到完整动作", "own_view", False, False),
])
def test_phase_row_fusible_counts_shared_timing(tmp_path, side_reason, side_source, reliable, fusible):
    (pair,) = read_manifest(_write(tmp_path, "C13,,seg,C,,,rear_straight,C13 护手向后拉,guard_back,,0,\n"))
    per_view = {"front": ((), {"reason": "已检测到完整动作", "phaseSource": "own_view"}),
                "side": ((), {"reason": side_reason, "phaseSource": side_source})}
    row = phase_row(pair, "rear_straight", {"alignment": {"reliable": reliable}}, per_view)
    assert row["family"] == "single" and row["front_found"] and row["side_found"] is (side_reason == "已检测到完整动作")
    assert row["side_window"] is (side_source != "own_view" or side_reason == "已检测到完整动作")
    assert row["fusible"] is fusible


# ------------------------------------------------------------------ 指标与报告


def _row(pair, role, code, fused, front, side, expected=False, item="", category="C", action="stance"):
    return {"pair_id": pair, "sample_role": role, "code": code, "fused_status": fused, "front_state": front,
            "side_state": side, "expected": expected, "error_item": item, "category": category, "action": action}


def _rows():
    return [
        _row("A1", ROLE_CORRECT, "stance_elbow", "candidate", "violation", "violation", category="A"),
        _row("A1", ROLE_CORRECT, "stance_width", "unable", "pass", "n/a", category="A"),
        _row("B1", ROLE_CORRECT, "stance_elbow", "not_observed", "pass", "violation", category="B"),
        _row("C7", ROLE_ERROR, "stance_width", "candidate", "violation", "n/a", True, "C7 站距过宽"),
        _row("C7", ROLE_ERROR, "stance_width", "unable", "n/a", "n/a", True, "C7 站距过宽"),
        _row("C7", ROLE_ERROR, "stance_elbow", "candidate", "violation", "violation", False, "C7 站距过宽"),
        _row("C15", ROLE_ERROR, "shoulder_level", "unable", "n/a", "violation", True, "C15 出拳不与肩齐平"),
        _row("C16", ROLE_ERROR, "guard_recover", "not_observed", "pass", "n/a", True, "C16 前手不回收"),
        _row("C12", ROLE_ERROR, "guard_low", "pending_rule", "n/a", "n/a", True, "C12 护手掉手"),
        _row("X", ROLE_EXCLUDED, "stance_elbow", "candidate", "violation", "violation"),
    ]


def _phases():
    base = {"views": "front|side", "front_found": True, "side_found": False, "alignment_reliable": True, "fusible": False}
    return [{**base, "category": "B", "family": "kick", "alignment_reliable": False},
            {**base, "category": "B", "family": "kick", "side_found": True, "fusible": True},
            {**base, "category": "C", "family": "stance", "front_found": False},
            {**base, "category": "C", "family": "single", "views": "front"}]


def test_metrics_follow_batch1_definitions():
    s = metrics.summarize(_rows(), _phases())
    fp = {e["code"]: e for e in s["false_positives"]}
    assert set(fp) == {"stance_elbow", "stance_width"}  # 不计入的对不进误报
    assert (fp["stance_elbow"]["fused_candidate"], fp["stance_elbow"]["fused_decided"]) == (1, 2)
    assert (fp["stance_elbow"]["front_violation"], fp["stance_elbow"]["front_decided"]) == (1, 2)
    assert (fp["stance_elbow"]["side_violation"], fp["stance_elbow"]["side_decided"]) == (2, 2)
    assert fp["stance_width"]["fused_unable"] == 1 and fp["stance_width"]["side_decided"] == 0
    assert s["detection_summary"] == {"detected": 1, "view_only": 1, "missed": 1, "unable": 0, "pending_rule": 1}
    assert [p["pair_id"] for p in s["detection_pairs"]] == ["C7", "C12", "C15", "C16"]  # 按错误项编号数值排序
    assert s["off_target"] == {"stance_elbow": 1}
    assert {(e["category"], e["family"]): (e["segments"], e["front_found"], e["side_found"], e["fusible"])
            for e in s["phases"]} == {("B", "kick"): (2, 2, 1, 1), ("C", "single"): (1, 1, 0, 0)}
    assert {(e["category"], e["family"]): (e["segments"], e["unreliable"]) for e in s["alignment"]} == {
        ("B", "kick"): (2, 1), ("C", "stance"): (1, 0)}


def test_report_has_fixed_sections_and_marks_changes():
    after = metrics.summarize(_rows(), _phases())
    improved = [dict(r, fused_status="not_observed", front_state="pass", side_state="pass") if r["pair_id"] == "A1"
                else r for r in _rows()]
    before = metrics.summarize(_rows(), _phases())
    after = metrics.summarize(improved, _phases())
    single = report.render_report({"label": "after"}, after)
    for heading in ("## 0. 运行信息", "## 1. 正确样本上的误报", "## 2. 错误样本上的检出", "### 2.1 逐对最好判定",
                    "## 3. 动作阶段检出", "## 4. 正侧对齐失败"):
        assert heading in single
    assert "→" not in single
    compare = report.render_report({"label": "after"}, after, {"label": "before"}, before)
    assert "| `stance_elbow` | 1/2（50%） → **0/2（0%）** |" in compare
    assert "| 标签 | after | before |" in compare
    assert compare.count("##") == single.count("##")


def test_report_rows_missing_on_one_side():
    before = metrics.summarize(_rows(), _phases())
    after = metrics.summarize([r for r in _rows() if r["code"] != "stance_width"], _phases())
    compare = report.render_report({}, after, {}, before)
    assert "| `stance_width` | 0/0 → **—** |" in compare
    assert report.ratio(0, 0) == "0/0" and report.ratio(1, 4) == "1/4（25%）"


def test_cli_compare_writes_report(tmp_path):
    from analysis.sample_eval.__main__ import main

    for name, rows in (("before", _rows()), ("after", _rows()[:3])):
        runs.write_run(tmp_path / name, {"label": name}, rows, _phases())
    out = tmp_path / "compare.md"
    assert main(["compare", "--before", str(tmp_path / "before"), "--after", str(tmp_path / "after"),
                 "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# 样本评估对比报告") and "| 标签 | after | before |" in text
    assert json.loads((tmp_path / "after" / "run.json").read_text(encoding="utf-8")) == {"label": "after"}
