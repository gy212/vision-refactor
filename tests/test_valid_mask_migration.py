# -*- coding: utf-8 -*-
"""
valid_mask 契约迁移逐位等价回归（YOLO 迁移 Issue #5 / S1）。

为什么需要它
------------
Issue #5 把“此点是否可用”的判据从散落的 ``lm[idx, 3] >= thr`` 统一收口到集中式
``derive_valid_mask`` + ``valid_mask`` 形参。验收硬指标要求：**同一 fixture 下，三种
调用方式产出逐位一致**——

  ① 旧式调用：不传 ``valid_mask``，由各函数内部按 ``visibility >= 0.5`` 现场推导；
  ② 显式调用：显式传入 ``derive_valid_mask(landmarks, 0.5)`` 生成的 mask；
  ③ 改造前 golden：``tests/fixtures/pose33_v3/golden.json`` 冻结的迁移前行为
     （由 ``test_pose33_v3_golden.py`` 同源维护，本测试复用它作为“改造前”基线）。

三者必须在规则扣分、tech_eval 状态、关节误差上全部一致。本测试聚焦 ① == ②（迁移前后
API 两种调用方式逐位等价），③ 的等价由 ``test_pose33_v3_golden.py`` 保证（golden 不漂移
即说明迁移后行为 == 迁移前行为）。

确定性
------
与 golden 回归一致，通过 ``tests/golden_harness.py`` 回放已提交的 ``(T,33,4)`` landmark
fixture，不依赖 MediaPipe 实际推理。

验收命令
--------
    .\\.venv\\Scripts\\python.exe -m pytest tests\\test_valid_mask_migration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.pose_features import DEFAULT_VALID_CONF_THR, derive_valid_mask  # noqa: E402
from core.rule_scoring import score_rules  # noqa: E402
from analysis.tech_eval import (  # noqa: E402
    eval_cog_com,
    eval_cog_front,
    eval_cog_side,
    eval_force_sequence,
    eval_retract_speed_side,
    eval_wrist_angle,
    evaluate_video_full,
    to_jsonable,
)
from tests import golden_harness as H  # noqa: E402

FIX_DIR = Path(__file__).resolve().parent / "fixtures" / "pose33_v3"
STUDENT = "golden://student.mp4"
FPS = 30.0


def _load_raw(name: str) -> np.ndarray:
    d = np.load(FIX_DIR / name, allow_pickle=True)
    return np.asarray(d["landmarks"], dtype=np.float32)


@pytest.fixture(scope="module")
def student() -> np.ndarray:
    return _load_raw("student_raw.npz")


@pytest.fixture(scope="module")
def registered_student():
    H.clear_registry()
    H.register_video(STUDENT, _load_raw("student_raw.npz"), fps=FPS)
    yield
    H.clear_registry()


# --------------------------------------------------------------------------- #
# derive_valid_mask 自身契约
# --------------------------------------------------------------------------- #
def test_derive_valid_mask_matches_legacy_threshold(student):
    """derive_valid_mask 与旧式 ``lm[idx,3] >= 0.5`` 逐位一致（单帧 + 序列）。"""
    legacy = student[..., 3] >= DEFAULT_VALID_CONF_THR
    got = derive_valid_mask(student, DEFAULT_VALID_CONF_THR)
    assert got.shape == student.shape[:-1]
    assert got.dtype == bool
    assert np.array_equal(got, legacy)
    # 单帧路径
    for i in (0, student.shape[0] // 2, student.shape[0] - 1):
        row = derive_valid_mask(student[i], DEFAULT_VALID_CONF_THR)
        assert row.shape == (33,)
        assert np.array_equal(row, student[i, :, 3] >= DEFAULT_VALID_CONF_THR)


def test_derive_valid_mask_rejects_bad_shape():
    with pytest.raises(ValueError):
        derive_valid_mask(np.zeros((33, 3), dtype=np.float32))


# --------------------------------------------------------------------------- #
# 1) 规则扣分：① 旧式不传 mask == ② 显式传入 mask
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("view", ["front", "side"])
@pytest.mark.parametrize("action_scope", ["both", "stance", "punch"])
def test_score_rules_legacy_vs_explicit(student, view, action_scope):
    mask = derive_valid_mask(student, DEFAULT_VALID_CONF_THR)
    legacy = score_rules(student, view=view, action_scope=action_scope)
    explicit = score_rules(student, view=view, action_scope=action_scope, valid_mask=mask)

    assert legacy.score == explicit.score
    assert legacy.total_deduction == explicit.total_deduction
    assert len(legacy.violations) == len(explicit.violations)
    for a, b in zip(legacy.violations, explicit.violations):
        assert a.rule_id == b.rule_id
        assert a.name == b.name
        assert int(a.penalty) == int(b.penalty)
        assert int(a.valid_frames) == int(b.valid_frames)
        assert int(a.total_frames) == int(b.total_frames)
        assert a.detail == b.detail
        assert float(a.violation_ratio) == float(b.violation_ratio)


# --------------------------------------------------------------------------- #
# 2) tech_eval 单指标：① 旧式不传 mask == ② 显式传入 mask
# --------------------------------------------------------------------------- #
def _indicator_equal(a, b) -> None:
    assert a.status == b.status
    assert a.reason == b.reason
    # detail 经 to_jsonable 后整体逐位相等（含 numpy/np 标量归一化）。
    assert to_jsonable(a.detail) == to_jsonable(b.detail)


def test_eval_cog_side_legacy_vs_explicit(student):
    mask = derive_valid_mask(student, DEFAULT_VALID_CONF_THR)
    _indicator_equal(
        eval_cog_side(student, fps=FPS),
        eval_cog_side(student, fps=FPS, valid_mask=mask),
    )


def test_eval_cog_front_legacy_vs_explicit(student):
    mask = derive_valid_mask(student, DEFAULT_VALID_CONF_THR)
    _indicator_equal(
        eval_cog_front(student, fps=FPS, stance="left"),
        eval_cog_front(student, fps=FPS, stance="left", valid_mask=mask),
    )


def test_eval_cog_com_legacy_vs_explicit(student):
    mask = derive_valid_mask(student, DEFAULT_VALID_CONF_THR)
    _indicator_equal(
        eval_cog_com(student, fps=FPS),
        eval_cog_com(student, fps=FPS, valid_mask=mask),
    )


def test_eval_retract_speed_side_legacy_vs_explicit(student):
    mask = derive_valid_mask(student, DEFAULT_VALID_CONF_THR)
    _indicator_equal(
        eval_retract_speed_side(student, fps=FPS),
        eval_retract_speed_side(student, fps=FPS, valid_mask=mask),
    )


def test_eval_wrist_angle_legacy_vs_explicit(student):
    mask = derive_valid_mask(student, DEFAULT_VALID_CONF_THR)
    _indicator_equal(
        eval_wrist_angle(student, fps=FPS),
        eval_wrist_angle(student, fps=FPS, valid_mask=mask),
    )


def test_eval_force_sequence_legacy_vs_explicit(student):
    mask = derive_valid_mask(student, DEFAULT_VALID_CONF_THR)
    # front/side 用同一整段做等价性对照（不依赖视角切分，只验证 mask 传播）。
    _indicator_equal(
        eval_force_sequence(student, student, fps=FPS, stance="left"),
        eval_force_sequence(
            student,
            student,
            fps=FPS,
            stance="left",
            front_valid_mask=mask,
            side_valid_mask=mask,
        ),
    )


# --------------------------------------------------------------------------- #
# 3) 视频级入口：① 旧式不传 mask == ② 显式传入 mask（全指标逐位一致）
# --------------------------------------------------------------------------- #
_INDICATORS = ("cog_side", "cog_front", "cog_final", "cog_com", "retract_speed", "force_sequence", "wrist_angle")


def test_evaluate_video_full_legacy_vs_explicit(registered_student):
    # ① 旧式：不传 valid_mask（内部按 visibility>=0.5 推导）。
    # ② 显式：先取出整段 landmarks，按 derive_valid_mask 生成 mask 显式传入。
    from analysis.tech_eval import extract_pose_and_view_scores

    with H.replay_context():
        legacy = evaluate_video_full(STUDENT, pose_variant="full", stance="left", view_hint="auto")
        landmarks, _vs, meta = extract_pose_and_view_scores(STUDENT, pose_variant="full")
        thr = float(meta.get("valid_conf_thr") or DEFAULT_VALID_CONF_THR)
        mask = derive_valid_mask(landmarks, thr)
        explicit = evaluate_video_full(
            STUDENT, pose_variant="full", stance="left", view_hint="auto", valid_mask=mask
        )

    assert legacy.get("view_mode") == explicit.get("view_mode")
    assert legacy.get("front_segment") == explicit.get("front_segment")
    assert legacy.get("side_segment") == explicit.get("side_segment")
    for key in _INDICATORS:
        assert legacy.get(key) == explicit.get(key), f"指标 {key} 在两种调用方式下不一致"
