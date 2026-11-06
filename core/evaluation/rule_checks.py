# -*- coding: utf-8 -*-
"""统一规则目录在一路视角上的判定：规则输入序列、相位窗口与经 ``evaluate_rules`` 的逐项结论（D-3、D-5）。

考试、学生练习与离线处理都从这里取规则输入与单视角结论，核心逻辑只有一套（设计文档 §1.3 第 5 条）：
坐标为宽高比修正坐标（D-5），有效掩码 = ``derive_valid_mask`` ∧ 坐标有限 ∧ 在画面内（补帧一律无效，D-2），
按相位窗口取帧、按左右式取手。同步的正侧两路先经 ``synchronized_windows`` 共用可见机位的动作时间段，
练习、考试与 batch paired 同一份证据得到同一阶段；练习另做正侧融合（``evaluation/practice.py``），考试与 batch 逐视角报告。
上步组合侧面计时策略的阶段只由侧面逐次切分（``step_reps``）确定，正面按同一时钟共用；没有侧面视频时不判定。
上步组合的躯干朝向另用 ``paired_torso`` 同步双机位投影，不将单路肩宽冒充身体朝向。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from core.evaluation.action_timing import shared_timeline
from core.evaluation.action_record import shared_action_record
from core.evaluation.paired_torso import paired_torso_measurements
from core.evaluation.rules.adapters import RuleCheck, to_rule_check
from core.evaluation.rules.engine import evaluate_rules
from core.evaluation.rules.sanda_v5 import (
    ACTIONS,
    FeedbackConfig,
    build_checks,
    rule_profile,
)
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
    return {
        "landmarks": lm,
        "valid_mask": mask,
        "frames": frames,
        "times": [i / fps for i in frames],
    }


def series_from_raw(landmarks: np.ndarray, meta: dict) -> dict:
    """正式原始序列（``extract_pose_raw``/``slice_pose_raw_series`` 的输出，归一化坐标）→ 规则输入序列。"""
    width, height = int(meta["width"]), int(meta["height"])
    start = int(meta.get("start_frame") or 0)
    fps = float(meta.get("fps") or 30.0)
    return rule_series(
        aspect_corrected(landmarks, width, height),
        range(start, start + len(landmarks)),
        fps,
        width / height,
    )


def view_outcomes(
    view: str,
    series: dict,
    view_windows: dict,
    profile: RuleProfile,
    action: str,
    stance: str,
    cfg: FeedbackConfig,
    supported_capabilities=None,
    *,
    paired_torso=None,
    phase_instances=(),
    phase_coverage: dict | None = None,
) -> dict[str, RuleOutcome]:
    """一路视角上全部检查项经 ``evaluate_rules`` 的结论（以检查项 id 为键）；基线取起始实战式窗口，缺失时取开头几帧。"""
    baseline = view_windows.get(
        ("stance", "start"), np.arange(min(cfg.min_valid, len(series["times"])))
    )
    ctx = RuleContext(
        view=view,
        action=action,
        stance=stance,
        phase_windows={f"{segment}:{phase}": idx for (segment, phase), idx in view_windows.items()},
        baseline=baseline,
        params=cfg,
        paired_torso=paired_torso,
        phase_instances=_phase_instance_frames(series["times"], phase_instances),
        phase_coverage=phase_coverage or {},
    )
    outcomes = evaluate_rules(
        series["landmarks"],
        series["valid_mask"],
        profile,
        context=ctx,
        supported_capabilities=supported_capabilities,
    )
    return {o.spec.code: o for o in outcomes}


def _phase_instance_frames(times, instances) -> dict:
    """把已检测阶段的原始时间转换为本路帧下标，保留明确的次数身份。"""
    times = np.asarray(times)
    grouped = {}
    for instance in instances:
        scope = f"{instance['segment']}:{instance['phase']}"
        frames = np.flatnonzero((times >= instance["start"]) & (times <= instance["end"]))
        grouped.setdefault(scope, []).append((instance["repetition"], frames))
    return {scope: tuple(values) for scope, values in grouped.items()}


def synchronized_windows(
    views: dict, action: str, stance: str, cfg: FeedbackConfig
) -> tuple[dict, dict, dict, dict]:
    """同步录像共用一次动作识别；各机位只按公共阶段读取自己的数据。"""
    return shared_timeline(views, action, stance, cfg)


def _view_checks(
    view: str,
    series: dict,
    view_windows: dict,
    phase_reason: str,
    action: str,
    stance: str,
    cfg: FeedbackConfig,
    supported_capabilities=None,
    *,
    paired_torso=None,
    phase_instances=(),
    evidence_views: dict,
    evidence_diagnostics: dict,
    phase_coverage: dict | None = None,
) -> tuple[RuleCheck, ...]:
    """一路视角在给定相位窗口上的逐项结论（只列适用本视角的项）。"""
    outcomes = view_outcomes(
        view,
        series,
        view_windows,
        rule_profile(action, stance, cfg),
        action,
        stance,
        cfg,
        supported_capabilities,
        paired_torso=paired_torso,
        phase_instances=phase_instances,
        phase_coverage=phase_coverage,
    )
    return tuple(
        to_rule_check(
            c, view, outcomes[c["id"]], evidence_views, evidence_diagnostics, phase_reason
        )
        for c in build_checks(action, stance, pose_variant=cfg.pose_variant)
        if c["id"] in outcomes
    )


def judge_view(
    series: dict,
    view: str,
    action: str,
    stance: str,
    cfg: FeedbackConfig,
    supported_capabilities=None,
) -> tuple[tuple[RuleCheck, ...], dict]:
    """一路视角的逐项结论与相位诊断；没有同步的另一路时用它，不做正侧融合。"""
    return judge_views({view: series}, action, stance, cfg, supported_capabilities)[view]


def judge_views(
    views: dict, action: str, stance: str, cfg: FeedbackConfig, supported_capabilities=None
) -> dict[str, tuple[tuple[RuleCheck, ...], dict]]:
    """同步录制的各路视角（``{"front"/"side": 规则输入序列}``）的逐项结论与相位诊断；考试与 batch paired 用。

    相位与练习一样经 ``synchronized_windows`` 准备，诊断另带 ``phaseSource``；逐视角报告。
    躯干朝向在正面项中引用双机位共同证据，其余项仍按各路独立测量。
    """
    windows, diagnostics, alignment, sources = synchronized_windows(views, action, stance, cfg)
    record = shared_action_record(action, stance, diagnostics, alignment, sources)
    profile = rule_profile(action, stance, cfg)
    paired = (
        paired_torso_measurements(views)
        if any(len(s.required_views) > 1 for s in profile.rules)
        else {}
    )
    return {
        view: (
            _view_checks(
                view,
                series,
                windows[view],
                diagnostics[view].get("reason", ""),
                action,
                stance,
                cfg,
                supported_capabilities,
                paired_torso=paired.get(view),
                evidence_views=views,
                evidence_diagnostics=diagnostics,
                phase_instances=(
                    diagnostics[view].get("phaseInstances")
                    or diagnostics[view].get("projectedPhaseInstances", ())
                ),
                phase_coverage=alignment["phases"],
            ),
            {**diagnostics[view], "phaseSource": sources[view], "sharedActionRecord": record},
        )
        for view, series in views.items()
    }
