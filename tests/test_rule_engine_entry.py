"""D-3、D-16：唯一的规则目录 ``sanda_v5`` 只经 ``evaluate_rules`` 执行，考试、练习、离线处理共用。"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from core import evaluation as ev
from core.evaluation import rule_checks
from core.evaluation.rules import engine
from core.evaluation.rules.spec import PhaseVerdict
from tests.eval_diff import synth_practice as sp

ROOT = Path(__file__).resolve().parents[1]


def _counting(monkeypatch, module, calls: list) -> None:
    real = engine.evaluate_rules

    def wrapper(*args, **kwargs):
        outcomes = real(*args, **kwargs)
        calls.append((args[2].name, len(outcomes)))
        return outcomes

    monkeypatch.setattr(module, "evaluate_rules", wrapper)


def test_practice_judges_every_measured_check_through_evaluate_rules(monkeypatch, tmp_path):
    calls: list = []
    _counting(monkeypatch, rule_checks, calls)
    cfg = ev.FeedbackConfig()
    views = {}
    for i, view in enumerate(("front", "side")):
        lm, detected = sp.landmarks("straight_combo", "left", view, 3 + i)
        video = sp.write_avi(tmp_path / f"{view}.avi")
        views[view] = ev.extract_series(video, cfg, lambda: False, sp.scripted_pipeline(lm, detected))
    result = ev.evaluate_series("straight_combo", "left", views, cfg)
    assert [name for name, _ in calls] == ["sanda_v5", "sanda_v5"]
    measured = [c for c in result["checks"] if c.get("measurements")]
    assert measured, "合成场景应至少有一项被测量"


def test_profile_is_the_single_catalog_with_phase_verdicts():
    profile = ev.rule_profile("hook_combo", "right", ev.FeedbackConfig())
    checks = ev.build_checks("hook_combo", "right")
    assert profile.name == "sanda_v5"
    assert [r.code for r in profile.rules] == [c["id"] for c in checks]
    assert all(isinstance(r.aggregation, PhaseVerdict) for r in profile.rules)
    assert {r.authorization for r in profile.rules} == {"candidate", "blocked"}
    measured = [r for r in profile.rules if r.authorization == "candidate"]
    assert all(r.required for r in measured)  # 能力契约：每个测量项都声明所需关键点


def test_formal_v1_is_gone_and_no_second_catalog_exists():
    """D-3：不允许两个 profile 并存。"""
    rules_dir = ROOT / "core" / "evaluation" / "rules"
    assert not (rules_dir / "formal_v1.py").exists() and not (rules_dir / "practice_v4.py").exists()
    profiles = []
    for path in (ROOT / "core").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "RuleProfile":
                profiles.append(path.relative_to(ROOT).as_posix())
    assert profiles == ["core/evaluation/rules/sanda_v5.py"]


def test_batch_and_exam_judge_through_the_same_entry(monkeypatch):
    """考试与 batch 的逐视角检查经 ``rule_checks.judge_view`` → ``evaluate_rules``，与练习同一目录。"""
    calls: list = []
    _counting(monkeypatch, rule_checks, calls)
    lm, detected = sp.landmarks("front_straight", "left", "front", 3)
    lm = lm.astype(np.float32)
    lm[~detected, :, 3] = 0.0
    series = ev.series_from_raw(lm, {"width": sp.W, "height": sp.H, "fps": sp.FPS, "start_frame": 0})
    checks, diagnostics = ev.judge_view(series, "front", "front_straight", "left", ev.FeedbackConfig())
    assert calls == [("sanda_v5", calls[0][1])]
    assert diagnostics["reason"] == "已检测到完整动作"
    assert {c.state for c in checks} <= {"pass", "violation", "unable", "pending_rule"}
    assert next(c for c in checks if c.code == "stance_elbow").state in {"pass", "violation"}


def test_rule_checks_are_called_only_by_the_engine():
    """规则的逐帧判定函数只由 ``rules/engine.py`` 调用；练习测量只经登记的 ``RuleSpec``。"""
    offenders = []
    for path in (ROOT / "core").rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "check" and rel != "core/evaluation/rules/engine.py":
                offenders.append(f"{rel}:{node.lineno} .check()")
            if isinstance(fn, ast.Name) and fn.id == "_measure" and rel != "core/evaluation/rules/sanda_v5.py":
                offenders.append(f"{rel}:{node.lineno} _measure()")
    assert offenders == []
