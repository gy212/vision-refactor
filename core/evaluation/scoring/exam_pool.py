# -*- coding: utf-8 -*-
"""考试池计分方案 ``EXAM_POOL``：正对正池、侧对侧池逐模板子序列 DTW，踢腿几何加权、指定动作分数压缩、正侧权重合并。

迁移自 ``apps/recording_postprocess.py``（设计文档 §7.6）：Tk 考试与普通双摄的录后自动比对用它。
模板按 ``EXAM_POOL`` 契约（heavy + pose33_v3）读取与过滤；比对函数与原始序列提取由调用方注入（录后处理的测试会替换它们）。
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np

from core.evaluation.scoring.kick_quality import score_kick_quality
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

PoolCompare = Callable[..., list]
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
        merged.append({
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
        })
        consumed.add(idx)
        for j, o in enumerate(action_details):
            if o is front_d or o is side_d:
                consumed.add(j)
    return merged


def action_score(name: str, dtw_score: float, geom_score: float | None) -> float:
    """动作子分：踢腿有几何分时与 DTW 加权；登记了压缩区间的动作把 DTW 分线性压进区间。"""
    final = KICK_GEOM_WEIGHT * geom_score + (1.0 - KICK_GEOM_WEIGHT) * dtw_score if geom_score is not None else dtw_score
    squeeze = SCORE_SQUEEZE.get(name)
    if squeeze is not None:
        lo, hi = squeeze
        final = lo + (hi - lo) * float(np.clip(dtw_score, 0.0, 1.0))
    return float(final)


@pose_cache()
def score_pool(templates: list[Path], video: Path, view: str, *, compare: PoolCompare, raw_series: RawSeriesFn,
               stop_evt: Event) -> tuple[list[dict], float | None]:
    """一个视角的池：视频姿态只提取一次复用给池内每个模板；含踢腿动作时取原始序列算几何分。

    在 ``pose_cache`` 作用域内执行：模板比对与踢腿几何的顺序提取共用同一次推理。
    """
    if not templates:
        return [], None
    if stop_evt.is_set():
        raise InterruptedError("后台比对已取消")
    results = compare(templates, video, pose_variant=POSE_VARIANT, workers=1, stop_evt=stop_evt)
    names = [Path(t).stem for t in templates]
    series = None
    if any(n in KICK_KINDS for n in names):
        if stop_evt.is_set():
            raise InterruptedError("后台比对已取消")
        series = raw_series(video, pose_variant=POSE_VARIANT)

    details: list[dict] = []
    for name, res in zip(names, results):
        dtw_score = float(res.score)
        start, end = int(res.start_frame), int(res.end_frame)
        geom_score: float | None = None
        geom_detail = ""
        if series is not None and name in KICK_KINDS:
            lm_win, meta_win = slice_pose_raw_series(series, start_frame=start, end_frame=end)
            geom_score, geom_detail = score_kick_quality(
                lm_win, meta_win["valid_mask"], KICK_KINDS[name],
                width=meta_win.get("width"), height=meta_win.get("height"),
            )
        details.append({"name": name, "view": view, "score": action_score(name, dtw_score, geom_score),
                        "dtw_score": dtw_score, "geom_score": geom_score, "geom_detail": geom_detail,
                        "start": start, "end": end})
    pool_mean = float(np.mean([d["score"] for d in details])) if details else None
    return details, pool_mean


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


def score_exam_pools(front_tpls: list[Path], side_tpls: list[Path], front_video: Path, side_video: Path, *,
                     compare: PoolCompare, raw_series: RawSeriesFn, stop_evt: Event) -> ExamPoolResult:
    """正对正池、侧对侧池逐模板评分后合并（考试双机位视角固定，不做自动正侧切分）。"""
    front_details, front_score = score_pool(front_tpls, front_video, "front", compare=compare,
                                            raw_series=raw_series, stop_evt=stop_evt)
    side_details, side_score = score_pool(side_tpls, side_video, "side", compare=compare,
                                          raw_series=raw_series, stop_evt=stop_evt)
    # 合并只影响 combined 与逐动作列，避免正+侧双模板动作被双重计数。
    action_scores = merge_view_weighted_actions(front_details + side_details)
    all_scores = [d["score"] for d in action_scores]
    combined = float(np.mean(all_scores)) if all_scores else 0.0
    pct = int(np.clip(int(round(combined * 100.0)), 0, 100))
    # 兼容旧字段：front/side_segment 取各池首个匹配窗口
    front_seg = (front_details[0]["start"], front_details[0]["end"]) if front_details else None
    side_seg = (side_details[0]["start"], side_details[0]["end"]) if side_details else None
    return ExamPoolResult(front_score, side_score, combined, pct, front_segment=front_seg, side_segment=side_seg,
                          action_scores=tuple(action_scores))
