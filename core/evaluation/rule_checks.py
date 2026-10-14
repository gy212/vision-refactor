# -*- coding: utf-8 -*-
"""统一规则目录在一路视角上的判定：规则输入序列、相位窗口与经 ``evaluate_rules`` 的逐项结论（D-3、D-5）。

考试、学生练习与离线处理都从这里取规则输入与单视角结论，核心逻辑只有一套（设计文档 §1.3 第 5 条）：
坐标为宽高比修正坐标（D-5），有效掩码 = ``derive_valid_mask`` ∧ 坐标有限 ∧ 在画面内（补帧一律无效，D-2），
按相位窗口取帧、按左右式取手。练习另做正侧融合（``evaluation/practice.py``）；考试与 batch 逐视角报告。
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from core.evaluation.phases import phase_windows
from core.evaluation.rules.adapters import RuleCheck, to_rule_check
from core.evaluation.rules.engine import evaluate_rules
from core.evaluation.rules.sanda_v5 import ACTIONS, FeedbackConfig, build_checks, rule_profile
from core.evaluation.rules.spec import RuleContext, RuleOutcome, RuleProfile
from core.pose.validity import derive_valid_mask

# 模板名（考试池、batch 的动作名）→ 目录动作；不在目录里的动作没有规则检查。
ACTION_BY_NAME = {label: key for key, label in ACTIONS.items()}


def aspect_corrected(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """归一化坐标 → 宽高比修正坐标：x、z 乘 ``width / height``，统一为图像高度单位（float32）。"""
    lm = np.asarray(landmarks, dtype=np.float64).copy()
    lm[:, :, 0] *= float(width) / float(height)
    lm[:, :, 2] *= float(width) / float(height)
    return lm.astype(np.float32)


def rule_series(landmarks: np.ndarray, frames: Sequence[int], fps: float, aspect: float) -> dict:
    """规则输入序列：``landmarks`` 为宽高比修正坐标 ``(T,33,4)``，``aspect`` 为画面宽高比（x 的上界）。"""
    lm = np.asarray(landmarks)
    mask = derive_valid_mask(lm) & np.isfinite(lm[:, :, :3]).all(axis=2)
    mask &= (lm[:, :, 0] >= 0) & (lm[:, :, 0] <= aspect) & (lm[:, :, 1] >= 0) & (lm[:, :, 1] <= 1)
    frames = [int(i) for i in frames]
    return {"landmarks": lm, "valid_mask": mask, "frames": frames, "times": [i / fps for i in frames]}


def series_from_raw(landmarks: np.ndarray, meta: dict) -> dict:
    """正式原始序列（``extract_pose_raw``/``slice_pose_raw_series`` 的输出，归一化坐标）→ 规则输入序列。"""
    width, height = int(meta["width"]), int(meta["height"])
    start = int(meta.get("start_frame") or 0)
    fps = float(meta.get("fps") or 30.0)
    return rule_series(aspect_corrected(landmarks, width, height), range(start, start + len(landmarks)), fps,
                       width / height)


def view_outcomes(view: str, series: dict, view_windows: dict, profile: RuleProfile, action: str, stance: str,
                  cfg: FeedbackConfig, supported_capabilities=None) -> dict[str, RuleOutcome]:
    """一路视角上全部检查项经 ``evaluate_rules`` 的结论（以检查项 id 为键）；基线取起始实战式窗口，缺失时取开头几帧。"""
    baseline = view_windows.get(("stance", "start"), np.arange(min(cfg.min_valid, len(series["times"]))))
    ctx = RuleContext(view=view, action=action, stance=stance,
                      phase_windows={f"{segment}:{phase}": idx for (segment, phase), idx in view_windows.items()},
                      baseline=baseline, params=cfg)
    outcomes = evaluate_rules(series["landmarks"], series["valid_mask"], profile, context=ctx,
                              supported_capabilities=supported_capabilities)
    return {o.spec.code: o for o in outcomes}


def judge_view(series: dict, view: str, action: str, stance: str, cfg: FeedbackConfig,
               supported_capabilities=None) -> tuple[tuple[RuleCheck, ...], dict]:
    """一路视角的逐项结论（只列适用本视角的项）与相位诊断；考试与 batch 用，不做正侧融合。"""
    diagnostics: dict = {}
    windows = phase_windows(series, action, stance, cfg, diagnostics)
    outcomes = view_outcomes(view, series, windows, rule_profile(action, stance, cfg), action, stance, cfg,
                             supported_capabilities)
    checks = tuple(to_rule_check(c, view, outcomes[c["id"]], series["frames"], diagnostics.get("reason", ""))
                   for c in build_checks(action, stance) if c["id"] in outcomes)
    return checks, diagnostics
