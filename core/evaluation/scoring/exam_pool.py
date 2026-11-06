# -*- coding: utf-8 -*-
"""考试池计分方案 ``EXAM_POOL``：正对正池、侧对侧池逐模板子序列 DTW，踢腿几何加权、指定动作分数压缩、正侧权重合并。

迁移自 ``apps/recording_postprocess.py``（设计文档 §7.6）：Tk 考试与普通双摄的录后自动比对用它。
模板按 ``EXAM_POOL`` 契约（heavy + pose33_v3）读取与过滤；比对函数与原始序列提取由调用方注入（录后处理的测试会替换它们）。
模板名属于统一规则目录的动作，另按 ``sanda_v5`` 逐视角检查（固定左式，相位窗口取帧），结论只写进结果，不计入分数（D-3）；
正侧两池同一动作的匹配成对判定，相位与练习一样跨机位共用动作时间段。
按逐次评分策略登记的上步组合的模板按侧面切出的每一次分别匹配、子分取各次均值（D-25）；逐范围比对
``compare_windows`` 由调用方注入，未注入或切不出次数时按整段匹配，并在明细里写明；它们的规则检查取整段片段。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np

from core.evaluation.rule_checks import ACTION_BY_NAME, judge_views, series_from_raw
from core.evaluation.rules.sanda_v5 import DEFAULT_STANCE, FeedbackConfig, load_feedback_config
from core.evaluation.action_catalog import action_definition
from core.evaluation.scoring.kick_quality import score_kick_quality
from core.evaluation.step_reps import EXPECTED_REPETITIONS, step_combo_windows
from core.evaluation.templates import EXAM_POOL, TemplateError, TemplateStore, load_template
from core.evaluation.view_weights import normalized_pair
from core.pose.extract import pose_cache
from core.pose.raw_series import slice_pose_raw_series

POSE_VARIANT = "heavy"  # 黑盒检测统一走精度最高的 heavy + pose33_v3 链路
TEMPLATE_LAYOUT = "pose33_v3"

# 踢腿几何质量评分：动作名（模板 stem）→ kick_quality 类型；列出的动作并入几何分。
# 低鞭腿不列入——实测其低/高鞭高度差不稳定，几何常误判，直接用纯 DTW 分。
KICK_KINDS = {
    "高鞭腿": "high_whip",
    "侧踹腿": "kick_up",
    "正蹬腿": "kick_up",
}
# 踢腿动作子分 = 几何高度分 × w + DTW 相似度分 × (1-w)。校准旋钮。
KICK_GEOM_WEIGHT = 0.5

# DTW 分压缩映射（opt-in）：动作名 → (下限, 上限)。某些动作 DTW 定位不可靠（如背后七颠：低位移提踵，
# 归一化特征抹掉高度，开放边界退化匹配极短片段，虚高分与动作质量无关）。这类动作考察不严「是人就过」，
# 把 DTW 分线性压到高分窄区间：人人及格偏上，保留同一条视频恒定、可复现的微弱区分度。未登记动作不受影响。
SCORE_SQUEEZE: dict[str, tuple[float, float]] = {
    "背后七颠": (0.75, 0.90),
}

# 同一动作正侧两视角按权重合并成一个子分（opt-in，八段锦用）。key = 合并后动作名；value 自动归一。
# 触发条件：正池有 "<key>_正面"、侧池有 "<key>_侧面"，两者都在才合并；散打模板不在表内，行为不变。
VIEW_WEIGHTS: dict[str, dict[str, float]] = {
    "两手攀足": {"side": 0.75, "front": 0.25},  # 实测正面看不清前俯深度，侧面为主
    "攒拳怒目": {"side": 0.50, "front": 0.50},  # 正侧分接近，合并去重双计数
}
VIEW_FRONT_SUFFIX = "_正面"
VIEW_SIDE_SUFFIX = "_侧面"

EXAM_STANCE = DEFAULT_STANCE  # 考试固定左式（D-3）
# 规则检查取动作匹配窗口前后各放宽的时长：相位检测要看到动作前后的实战式。
RULE_MARGIN_SECONDS = 1.0
# 上步组合逐次匹配时每一次的范围前后各放宽的时长：让子序列 DTW 看到出拳前后的实战式。
REP_MARGIN_SECONDS = 0.5

PoolCompare = Callable[..., list]
# 逐范围比对（``match_video_windows``）：外层按模板、内层按范围的比对结果。
WindowCompare = Callable[..., list]
RawSeriesFn = Callable[..., Any]


def check_template(path: Path | str) -> Path:
    """按 ``EXAM_POOL`` 契约校验单个模板，返回 Path；违规抛 ``TemplateError``。"""
    load_template(Path(path), EXAM_POOL)
    return Path(path)


def usable_template(path: Path | str | None) -> Path | None:
    """模板路径可用则返回规范化 Path，否则 None（缺文件、无效都视为未填入）。"""
    usable = TemplateStore(EXAM_POOL).usable([path])
    return usable[0] if usable else None


def check_pair(front_path: Path, side_path: Path) -> None:
    """正式路径：正/侧模板必须都可用且布局一致。"""
    front = load_template(Path(front_path), EXAM_POOL)
    side = load_template(Path(side_path), EXAM_POOL)
    front_layout = str(front.raw_meta.get("feature_layout") or "") or TEMPLATE_LAYOUT
    side_layout = str(side.raw_meta.get("feature_layout") or "") or TEMPLATE_LAYOUT
    if front_layout != side_layout or front_layout != TEMPLATE_LAYOUT:
        raise TemplateError(
            "template_incompatible",
            f"正侧模板布局不兼容：front={front_layout or '?'} side={side_layout or '?'}",
        )


def usable_pools(front: Iterable | None, side: Iterable | None) -> tuple[list[Path], list[Path]]:
    """多模板池：各自过滤出可用模板（保序去重，单个坏模板跳过不整体失败）；两池都空抛 ``template_missing``。"""
    store = TemplateStore(EXAM_POOL)
    front_ok, side_ok = store.usable(front or []), store.usable(side or [])
    if not front_ok and not side_ok:
        raise TemplateError(
            "template_missing",
            "正/侧模板池均为空或不可用（需 heavy + pose33_v3）。请在主界面「录制」区至少为一侧添加模板。",
        )
    return front_ok, side_ok


def merge_view_weighted_actions(action_details: list[dict]) -> list[dict]:
    """把 ``VIEW_WEIGHTS`` 登记动作的正/侧子分按归一权重合并成一个子分；不成对或未登记的原样保序透传。"""
    by_name = {d.get("name"): d for d in action_details}
    merged: list[dict] = []
    consumed: set[int] = set()
    for idx, d in enumerate(action_details):
        if idx in consumed:
            continue
        name = str(d.get("name") or "")
        base = None
        if name.endswith(VIEW_FRONT_SUFFIX):
            base = name[: -len(VIEW_FRONT_SUFFIX)]
        elif name.endswith(VIEW_SIDE_SUFFIX):
            base = name[: -len(VIEW_SIDE_SUFFIX)]
        weights = VIEW_WEIGHTS.get(base) if base else None
        front_d = by_name.get(f"{base}{VIEW_FRONT_SUFFIX}") if base else None
        side_d = by_name.get(f"{base}{VIEW_SIDE_SUFFIX}") if base else None
        pair = normalized_pair(weights) if weights is not None else None
        if pair is None or front_d is None or side_d is None:
            merged.append(d)
            continue
        w_front, w_side = pair
        merged.append(
            {
                "name": base,
                "view": "combined",
                "score": float(w_front * float(front_d["score"]) + w_side * float(side_d["score"])),
                "front_score": float(front_d["score"]),
                "side_score": float(side_d["score"]),
                "view_weights": {"front": w_front, "side": w_side},
                "front_start": int(front_d["start"]),
                "front_end": int(front_d["end"]),
                "side_start": int(side_d["start"]),
                "side_end": int(side_d["end"]),
            }
        )
        consumed.add(idx)
        for j, o in enumerate(action_details):
            if o is front_d or o is side_d:
                consumed.add(j)
    return merged


def action_score(name: str, dtw_score: float, geom_score: float | None) -> float:
    """动作子分：踢腿有几何分时与 DTW 加权；登记了压缩区间的动作把 DTW 分线性压进区间。"""
    final = (
        KICK_GEOM_WEIGHT * geom_score + (1.0 - KICK_GEOM_WEIGHT) * dtw_score
        if geom_score is not None
        else dtw_score
    )
    squeeze = SCORE_SQUEEZE.get(name)
    if squeeze is not None:
        lo, hi = squeeze
        final = lo + (hi - lo) * float(np.clip(dtw_score, 0.0, 1.0))
    return float(final)


def rule_action_for(name: str) -> str | None:
    """模板名（可带 ``_正面``/``_侧面`` 后缀）对应的规则目录动作；目录外的动作为 None。"""
    return ACTION_BY_NAME.get(name.removesuffix(VIEW_FRONT_SUFFIX).removesuffix(VIEW_SIDE_SUFFIX))


def side_repetitions(series, cfg: FeedbackConfig) -> list[tuple[float, float]]:
    """侧面整段原始序列上上步组合每一次的 ``(起, 止)``（秒）；切不出时为空列表。"""
    landmarks, meta = slice_pose_raw_series(series)
    diagnostics: dict = {}
    step_combo_windows(series_from_raw(landmarks, meta), EXAM_STANCE, cfg, diagnostics)
    return [(r["start"], r["end"]) for r in diagnostics.get("repetitions", [])]


def rep_windows(reps: list[tuple[float, float]], fps: float) -> list[tuple[int, int]]:
    """各次时间 → 帧范围：前后最多放宽 0.5 s，在相邻两次之间的中点截断，避免重复匹配同一次动作。"""
    windows = [
        (
            max(0, int(round((start - REP_MARGIN_SECONDS) * fps))),
            int(round((end + REP_MARGIN_SECONDS) * fps)),
        )
        for start, end in reps
    ]
    for i in range(1, len(windows)):
        # 帧范围含两端：中点帧只归后一次，两个窗口不能共用同一帧。
        boundary = int(round((reps[i - 1][1] + reps[i][0]) * fps / 2))
        previous_start, previous_end = windows[i - 1]
        start, end = windows[i]
        windows[i - 1] = (previous_start, min(previous_end, boundary - 1))
        windows[i] = (max(start, boundary), end)
    return windows


def stepwise_detail(name: str, view: str, per_rep: list) -> dict:
    """一个上步组合模板的逐次匹配 → 动作明细：子分取各次均值，范围从第一次匹配起到最后一次匹配止。"""
    score = float(np.mean([float(r.score) for r in per_rep]))
    detail = {
        "name": name,
        "view": view,
        "score": score,
        "dtw_score": score,
        "geom_score": None,
        "geom_detail": "",
        "start": int(per_rep[0].start_frame),
        "end": int(per_rep[-1].end_frame),
        "repetitions": [
            {"start": int(r.start_frame), "end": int(r.end_frame), "score": float(r.score)}
            for r in per_rep
        ],
    }
    if len(per_rep) != EXPECTED_REPETITIONS:
        detail["repetition_note"] = (
            f"检测到 {len(per_rep)} 次动作（应为 {EXPECTED_REPETITIONS} 次），按检测到的次数取均值"
        )
    return detail


def rule_target(series, detail: dict, action: str, view: str) -> dict:
    """一个匹配到的规则目录动作的规则目标：匹配窗口前后各放宽 ``RULE_MARGIN_SECONDS`` 的规则输入序列与帧范围。

    上步组合取整段：逐次切分要从片段开头的实战式起算，与考试计次用的是同一段（匹配窗口可能从上步中途开始）。
    """
    if action_definition(action).rule_input == "whole_clip":
        landmarks, meta = slice_pose_raw_series(series)
    else:
        margin = int(round(RULE_MARGIN_SECONDS * float(series.meta.get("fps") or 30.0)))
        landmarks, meta = slice_pose_raw_series(
            series,
            start_frame=max(0, int(detail["start"]) - margin),
            end_frame=int(detail["end"]) + margin,
        )
    return {
        "name": detail["name"],
        "view": view,
        "action": action,
        "series": series_from_raw(landmarks, meta),
        "start": int(meta["start_frame"]),
        "end": int(meta["end_frame"]),
    }


def _report(target: dict, judged: tuple) -> dict:
    checks, diagnostics = judged
    return {
        "name": target["name"],
        "view": target["view"],
        "action": target["action"],
        "stance": EXAM_STANCE,
        "start": target["start"],
        "end": target["end"],
        "phase_reason": diagnostics.get("reason", ""),
        "checks": checks,
    }


def rule_reports(
    front_targets: list[dict], side_targets: list[dict], cfg: FeedbackConfig
) -> list[dict]:
    """两池的规则目录动作按左式判定；同一动作的正侧匹配按出现顺序成对，经 ``judge_views`` 共用同步相位。

    结论按正面池、侧面池的匹配顺序排列；没有同动作另一路的匹配单独判定。
    """
    # 考试始终提取 heavy；练习配置不能改变考试规则的模型资格。
    cfg = replace(cfg, pose_variant=POSE_VARIANT)
    judged: dict[int, tuple] = {}
    unpaired = list(side_targets)
    for front in front_targets:
        side = next((t for t in unpaired if t["action"] == front["action"]), None)
        views = (
            {"front": front["series"]}
            if side is None
            else {"front": front["series"], "side": side["series"]}
        )
        result = judge_views(views, front["action"], EXAM_STANCE, cfg)
        judged[id(front)] = result["front"]
        if side is not None:
            unpaired.remove(side)
            judged[id(side)] = result["side"]
    for side in unpaired:
        judged[id(side)] = judge_views({"side": side["series"]}, side["action"], EXAM_STANCE, cfg)[
            "side"
        ]
    return [_report(t, judged[id(t)]) for t in front_targets + side_targets]


def _check_stop(stop_evt: Event) -> None:
    if stop_evt.is_set():
        raise InterruptedError("后台比对已取消")


def _stepwise_matches(
    templates: list[Path],
    series,
    video: Path,
    reps: list[tuple[float, float]],
    compare_windows: WindowCompare,
    stop_evt: Event,
) -> dict[int, list]:
    """上步组合模板（``templates`` 下标 → 逐次比对结果）；一次也没匹配上的模板不列入，回到整段匹配。"""
    windows = rep_windows(reps, float(series.meta.get("fps") or 30.0))
    per_template = compare_windows(
        templates, video, windows, pose_variant=POSE_VARIANT, workers=1, stop_evt=stop_evt
    )
    return {i: per_rep for i, per_rep in enumerate(per_template) if per_rep}


@pose_cache()
def score_pool(
    templates: list[Path],
    video: Path,
    view: str,
    *,
    compare: PoolCompare,
    raw_series: RawSeriesFn,
    stop_evt: Event,
    rule_cfg: FeedbackConfig | None = None,
    reps: list[tuple[float, float]] | None = None,
    compare_windows: WindowCompare | None = None,
) -> tuple[list[dict], float | None, list[dict]]:
    """一个视角的池：视频姿态只提取一次复用给池内每个模板；含踢腿或规则目录动作时取原始序列算几何分与规则输入。

    在 ``pose_cache`` 作用域内执行：模板比对与原始序列的顺序提取共用同一次推理。返回 ``(逐动作明细, 池均分,
    规则目标)``；规则目标是规则目录动作的规则输入序列，由 ``rule_reports`` 与另一池成对判定。``rule_cfg=None`` 时不做规则检查。
    ``reps``（侧面切出的每一次，秒）非空且注入了 ``compare_windows`` 时，上步组合的模板逐次匹配、子分取均值；
    否则整段匹配，上步组合的明细写明原因。
    """
    if not templates:
        return [], None, []
    _check_stop(stop_evt)
    names = [Path(t).stem for t in templates]
    timed = [
        i
        for i, n in enumerate(names)
        if rule_action_for(n) is not None
        and action_definition(rule_action_for(n)).scoring == "per_repetition"
    ]
    rule_actions = [rule_action_for(n) if rule_cfg is not None else None for n in names]
    series = None
    stepped: dict[int, list] = {}
    if timed and reps and compare_windows is not None:
        series = raw_series(video, pose_variant=POSE_VARIANT)
        _check_stop(stop_evt)
        found = _stepwise_matches(
            [templates[i] for i in timed], series, video, reps, compare_windows, stop_evt
        )
        stepped = {timed[i]: per_rep for i, per_rep in found.items()}
    plain = [i for i in range(len(templates)) if i not in stepped]
    _check_stop(stop_evt)
    plain_results = (
        compare(
            [templates[i] for i in plain],
            video,
            pose_variant=POSE_VARIANT,
            workers=1,
            stop_evt=stop_evt,
        )
        if plain
        else []
    )
    results = dict(zip(plain, plain_results))
    if series is None and (any(n in KICK_KINDS for n in names) or any(rule_actions)):
        _check_stop(stop_evt)
        series = raw_series(video, pose_variant=POSE_VARIANT)

    details: list[dict] = []
    for i, name in enumerate(names):
        if i in stepped:
            details.append(stepwise_detail(name, view, stepped[i]))
            continue
        res = results[i]
        dtw_score = float(res.score)
        start, end = int(res.start_frame), int(res.end_frame)
        geom_score: float | None = None
        geom_detail = ""
        if series is not None and name in KICK_KINDS:
            lm_win, meta_win = slice_pose_raw_series(series, start_frame=start, end_frame=end)
            geom_score, geom_detail = score_kick_quality(
                lm_win,
                meta_win["valid_mask"],
                KICK_KINDS[name],
                width=meta_win.get("width"),
                height=meta_win.get("height"),
            )
        details.append(
            {
                "name": name,
                "view": view,
                "score": action_score(name, dtw_score, geom_score),
                "dtw_score": dtw_score,
                "geom_score": geom_score,
                "geom_detail": geom_detail,
                "start": start,
                "end": end,
            }
        )
        if i in timed:
            details[-1]["repetition_note"] = (
                "未接入逐次比对，按整段匹配一次"
                if compare_windows is None
                else "未切出动作次数，按整段匹配一次"
            )
    pool_mean = float(np.mean([d["score"] for d in details])) if details else None
    targets = [
        rule_target(series, d, action, view) for d, action in zip(details, rule_actions) if action
    ]
    return details, pool_mean, targets


@dataclass(frozen=True)
class ExamPoolResult:
    """front/side_score 是各池子分均值（成绩表正/侧列）；combined 是合并后全部动作子分的平均。"""

    front_score: float | None
    side_score: float | None
    combined_score: float
    combined_percent: int
    front_matches: tuple = ()
    side_matches: tuple = ()
    front_segment: tuple[int, int] | None = None
    side_segment: tuple[int, int] | None = None
    action_scores: tuple[dict, ...] = ()
    # 规则目录动作的逐视角规则结论（只报告，不计分，D-3）；规则配置无效时 ``rule_error`` 写明原因。
    rule_checks: tuple[dict, ...] = ()
    rule_error: str = ""


def score_exam_pools(
    front_tpls: list[Path],
    side_tpls: list[Path],
    front_video: Path,
    side_video: Path,
    *,
    compare: PoolCompare,
    raw_series: RawSeriesFn,
    stop_evt: Event,
    compare_windows: WindowCompare | None = None,
) -> ExamPoolResult:
    """正对正池、侧对侧池逐模板评分后合并（考试双机位视角固定，不做自动正侧切分）。

    池里有上步组合的模板且注入了 ``compare_windows`` 时，先在侧面视频上切出每一次，两池的上步组合模板都按这些时间
    逐次匹配；这时两池在同一个 ``pose_cache`` 作用域内评分，切分与侧面池共用同一次推理。
    """
    try:
        rule_cfg, rule_error = load_feedback_config(), ""
    except ValueError as exc:  # 规则只报告：配置无效不影响考试分数
        rule_cfg, rule_error = None, str(exc)
    stepwise = compare_windows is not None and any(
        rule_action_for(Path(t).stem) is not None
        and action_definition(rule_action_for(Path(t).stem)).scoring == "per_repetition"
        for t in [*front_tpls, *side_tpls]
    )
    with pose_cache() if stepwise else nullcontext():
        reps = None
        if stepwise:
            _check_stop(stop_evt)
            reps = side_repetitions(
                raw_series(side_video, pose_variant=POSE_VARIANT), rule_cfg or FeedbackConfig()
            )
        pool_args = {
            "compare": compare,
            "raw_series": raw_series,
            "stop_evt": stop_evt,
            "rule_cfg": rule_cfg,
            "reps": reps,
            "compare_windows": compare_windows,
        }
        front_details, front_score, front_targets = score_pool(
            front_tpls, front_video, "front", **pool_args
        )
        side_details, side_score, side_targets = score_pool(
            side_tpls, side_video, "side", **pool_args
        )
    reports = rule_reports(front_targets, side_targets, rule_cfg) if rule_cfg is not None else []
    # 合并只影响 combined 与逐动作列，避免正+侧双模板动作被双重计数。
    action_scores = merge_view_weighted_actions(front_details + side_details)
    all_scores = [d["score"] for d in action_scores]
    combined = float(np.mean(all_scores)) if all_scores else 0.0
    pct = int(np.clip(int(round(combined * 100.0)), 0, 100))
    # 兼容旧字段：front/side_segment 取各池首个匹配窗口
    front_seg = (front_details[0]["start"], front_details[0]["end"]) if front_details else None
    side_seg = (side_details[0]["start"], side_details[0]["end"]) if side_details else None
    return ExamPoolResult(
        front_score,
        side_score,
        combined,
        pct,
        front_segment=front_seg,
        side_segment=side_seg,
        action_scores=tuple(action_scores),
        rule_checks=tuple(reports),
        rule_error=rule_error,
    )
