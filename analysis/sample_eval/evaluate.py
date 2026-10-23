# -*- coding: utf-8 -*-
"""一对样本在一个规则动作上的评估：练习融合 ``evaluate_series`` 加逐视角 ``judge_views``，展开成逐检查项行与相位行。

只调用 ``core.evaluation`` 的公共 API，规则、阈值与相位逻辑全部取生产实现，本模块不做任何判定。
逐检查项行的字段与第一批分析（``rule_checks.csv``）一致，另加样本口径列（``sample_role``、``expected``）。
"""
from __future__ import annotations

from analysis.sample_eval.manifest import PairRole, SamplePair
from core.evaluation import DEFAULT_STANCE, FeedbackConfig, evaluate_series, judge_views

VIEWS = ("front", "side")
PHASE_FOUND = "已检测到完整动作"
SHARED_TIMING = "_synchronized_timing"  # share_phase_timing 共用另一路动作时间段时的来源后缀


def action_family(action: str) -> str:
    """规则动作 → 指标分组：实战式 / 组合出拳 / 单拳 / 鞭腿。"""
    if action == "stance":
        return "stance"
    if action.endswith("_combo"):
        return "combo"
    if action.endswith("_kick"):
        return "kick"
    return "single"


def format_measurements(measurements) -> str:
    """测量值摘要：``名称=中位数[最小,最大]单位``，多项以分号分隔。"""
    parts = []
    for m in measurements:
        if isinstance(m.get("median"), (int, float)):
            parts.append(f"{m.get('name')}={m.get('median'):.3g}[{m.get('min'):.3g},{m.get('max'):.3g}]{m.get('unit', '')}")
        else:
            parts.append(str(m))
    return "; ".join(parts)


def _view_columns(view: str, check, diagnostics: dict) -> dict:
    row = {f"{view}_phase_source": diagnostics.get("phaseSource", ""), f"{view}_phase_reason": diagnostics.get("reason", "")}
    if check is None:
        row[f"{view}_state"] = "n/a"
        return row
    row.update({f"{view}_state": check.state, f"{view}_skip": check.skip_reason or "",
                f"{view}_viol_ratio": round(check.violation_ratio, 3),
                f"{view}_valid": f"{check.valid_frames}/{check.total_frames}",
                f"{view}_meas": format_measurements(check.measurements)})
    return row


def check_rows(pair: SamplePair, role: PairRole, action: str, result: dict, per_view: dict) -> list[dict]:
    """``evaluate_series`` 的每个检查项一行，带正侧逐视角结论。``per_view`` 是 ``judge_views`` 的返回值。"""
    checks_by_view = {view: {c.check_id: c for c in checks} for view, (checks, _d) in per_view.items()}
    diag = {view: d for view, (_c, d) in per_view.items()}
    alignment = result.get("alignment") or {}
    rows = []
    for c in result["checks"]:
        row = {"pair_id": pair.pair_id, "category": pair.category, "repetitions": pair.repetitions,
               "sample_role": role.role, "error_item": role.error_item, "action": action, "check_id": c["id"],
               "code": c["code"], "segment": c["segment"], "phase": c["phase"], "role": c["role"],
               "expected": c["code"] in role.expected_codes, "fused_status": c["status"],
               "fused_support": (c.get("fusion") or {}).get("support"), "fused_reason": c.get("reason", ""),
               "alignment_reliable": alignment.get("reliable"), "side_offset_s": alignment.get("sideOffsetSeconds")}
        for view in VIEWS:
            row.update(_view_columns(view, checks_by_view.get(view, {}).get(c["id"]), diag.get(view, {})))
        rows.append(row)
    return rows


def phase_row(pair: SamplePair, action: str, result: dict, per_view: dict) -> dict:
    """一对 × 动作的相位检出与正侧对齐：各视角是否自己找到完整动作、是否有动作窗口、对齐是否可靠、能否融合。

    动作窗口 = 自己找到，或经 ``share_phase_timing`` 共用了另一路的动作时间段（来源 ``*_synchronized_timing``）；
    可融合 = 两路都有动作窗口且对齐可靠。
    """
    diag = {view: d for view, (_c, d) in per_view.items()}
    reliable = (result.get("alignment") or {}).get("reliable")
    found = {view: diag.get(view, {}).get("reason", "") == PHASE_FOUND for view in VIEWS}
    source = {view: diag.get(view, {}).get("phaseSource", "") for view in VIEWS}
    window = {view: found[view] or source[view].endswith(SHARED_TIMING) for view in VIEWS}
    return {"pair_id": pair.pair_id, "category": pair.category, "action": action, "family": action_family(action),
            "views": "|".join(sorted(per_view)),
            **{f"{view}_reason": diag.get(view, {}).get("reason", "") for view in VIEWS},
            **{f"{view}_source": source[view] for view in VIEWS},
            "front_found": found["front"], "side_found": found["side"],
            "front_window": window["front"], "side_window": window["side"], "alignment_reliable": reliable,
            "fusible": window["front"] and window["side"] and reliable is True}


def evaluate_pair(pair: SamplePair, role: PairRole, action: str, views: dict,
                  cfg: FeedbackConfig) -> tuple[dict, list[dict], dict]:
    """返回 ``(练习结果字典, 逐检查项行, 相位行)``；左右式取 ``DEFAULT_STANCE``。"""
    result = evaluate_series(action, DEFAULT_STANCE, views, cfg)
    per_view = judge_views(views, action, DEFAULT_STANCE, cfg)
    return result, check_rows(pair, role, action, result, per_view), phase_row(pair, action, result, per_view)


def default_config() -> FeedbackConfig:
    """评估用配置：默认 ``FeedbackConfig``，模型按正式路径取 heavy（与姿态缓存一致）。"""
    return FeedbackConfig(pose_variant="heavy")
