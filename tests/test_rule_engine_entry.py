"""D-16：``evaluate_rules`` 是正式（formal_v1）与练习（practice_v4）两个 profile 唯一的规则执行入口。"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from core import evaluation as ev
from core.evaluation import practice
from core.evaluation.rules import engine, formal_v1
from core.evaluation.rules.spec import PhaseVerdict, RatioDeduction
from tests.eval_diff import synth_practice as sp

ROOT = Path(__file__).resolve().parents[1]


def _counting(monkeypatch, module, calls: list) -> None:
    real = engine.evaluate_rules

    def wrapper(*args, **kwargs):
        outcomes = real(*args, **kwargs)
        calls.append((args[2].name, len(outcomes)))
        return outcomes

    monkeypatch.setattr(module, "evaluate_rules", wrapper)


def test_formal_v1_scores_through_evaluate_rules(monkeypatch):
    calls: list = []
    _counting(monkeypatch, formal_v1, calls)
    lm = np.load(ROOT / "tests" / "fixtures" / "pose33_v3" / "student_raw.npz", allow_pickle=True)["landmarks"]
    result = formal_v1.score_rules(np.asarray(lm, dtype=np.float32), view="front", action_scope="both")
    assert calls and calls[0][0] == "formal_v1"
    assert len(result.violations) == calls[0][1]


def test_practice_v4_judges_every_measured_check_through_evaluate_rules(monkeypatch, tmp_path):
    calls: list = []
    _counting(monkeypatch, practice, calls)
    cfg = ev.FeedbackConfig()
    views = {}
    for i, view in enumerate(("front", "side")):
        lm, detected = sp.landmarks("straight_combo", "left", view, 3 + i)
        video = sp.write_avi(tmp_path / f"{view}.avi")
        views[view] = ev.extract_series(video, cfg, lambda: False, sp.scripted_pipeline(lm, detected))
    result = ev.evaluate_series("straight_combo", "left", views, cfg)
    assert [name for name, _ in calls] == ["practice_v4", "practice_v4"]
    measured = [c for c in result["checks"] if c.get("measurements")]
    assert measured, "合成场景应至少有一项被测量"


def test_profiles_use_their_aggregations():
    assert all(isinstance(r.aggregation, RatioDeduction) for r in formal_v1.FORMAL_V1.rules)
    profile = ev.practice_profile("hook_combo", "right", ev.FeedbackConfig())
    checks = ev.build_checks("hook_combo", "right")
    assert [r.code for r in profile.rules] == [c["id"] for c in checks]
    assert all(isinstance(r.aggregation, PhaseVerdict) for r in profile.rules)
    assert {r.authorization for r in profile.rules} == {"candidate", "blocked"}


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
            if isinstance(fn, ast.Name) and fn.id == "_measure" and rel != "core/evaluation/rules/practice_v4.py":
                offenders.append(f"{rel}:{node.lineno} _measure()")
    assert offenders == []
