# -*- coding: utf-8 -*-
"""样本评估指标：由逐检查项行与相位行算出误报率、检出率、动作阶段检出率与正侧对齐失败率（纯函数）。

口径沿用第一批样本分析（``report.md`` 第四节）：

- 误报：正确样本上每个检查项 code 的融合 ``candidate`` / 可判定（``candidate`` 或 ``not_observed``），
  以及逐视角 ``violation`` / 可判定（``violation`` 或 ``pass``），按检查项行计数；
- 检出：错误样本上预期检查项的逐对最好判定（``detected`` > ``view_only`` > ``missed`` > ``unable`` > ``pending_rule``），
  按错误项与预期 code 汇总；
- 动作阶段：每对 × 动作在各视角是否"已检测到完整动作"；两路都有动作窗口（含共用另一路的同步时间段）且对齐可靠才算可融合（实战式没有动作阶段，不计）；
- 正侧对齐失败：``alignment.reliable`` 为假的对 × 动作。
"""
from __future__ import annotations

from collections import Counter, defaultdict

from analysis.sample_eval.manifest import ROLE_CORRECT, ROLE_ERROR

VERDICTS = ("detected", "view_only", "missed", "unable", "pending_rule")
DECIDED_FUSED = ("candidate", "not_observed")
DECIDED_VIEW = ("violation", "pass")
FAMILY_LABELS = {"stance": "实战式", "combo": "组合出拳", "single": "单拳", "kick": "鞭腿"}
FAMILY_ORDER = tuple(FAMILY_LABELS)


def verdict(row: dict) -> str:
    """一条预期检查项行的判定（见模块说明）。"""
    states = {row.get("front_state"), row.get("side_state")}
    if row["fused_status"] == "pending_rule":
        return "pending_rule"
    if row["fused_status"] == "candidate":
        return "detected"
    if "violation" in states:
        return "view_only"
    if "pass" in states:
        return "missed"
    return "unable"


def _item_key(item: str) -> tuple:
    head = item.split()[0] if item else ""
    digits = "".join(ch for ch in head if ch.isdigit())
    return (head.rstrip("0123456789"), int(digits) if digits else 0, item)


def false_positives(rows: list[dict]) -> list[dict]:
    """正确样本上逐 code 的误报计数，按 code 排序。"""
    by_code: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["sample_role"] == ROLE_CORRECT:
            by_code[r["code"]].append(r)
    out = []
    for code in sorted(by_code):
        sub = by_code[code]
        fused = [r for r in sub if r["fused_status"] in DECIDED_FUSED]
        entry = {"code": code, "checks": len(sub), "fused_decided": len(fused),
                 "fused_candidate": sum(r["fused_status"] == "candidate" for r in fused),
                 "fused_unable": sum(r["fused_status"] == "unable" for r in sub),
                 "pending_rule": sum(r["fused_status"] == "pending_rule" for r in sub)}
        for view in ("front", "side"):
            decided = [r for r in sub if r.get(f"{view}_state") in DECIDED_VIEW]
            entry[f"{view}_decided"] = len(decided)
            entry[f"{view}_violation"] = sum(r[f"{view}_state"] == "violation" for r in decided)
        out.append(entry)
    return out


def detection_pairs(rows: list[dict]) -> list[dict]:
    """错误样本逐对的最好判定与各视角是否报 ``violation``，按错误项编号排序。"""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["sample_role"] == ROLE_ERROR and r["expected"]:
            grouped[r["pair_id"]].append(r)
    out = []
    for pair_id, sub in grouped.items():
        verdicts = [verdict(r) for r in sub]
        out.append({"pair_id": pair_id, "error_item": sub[0]["error_item"],
                    "codes": "|".join(sorted({r["code"] for r in sub})), "actions": "|".join(sorted({r["action"] for r in sub})),
                    "verdict": min(verdicts, key=VERDICTS.index), "verdicts": dict(Counter(verdicts)),
                    "front_violation": any(r.get("front_state") == "violation" for r in sub),
                    "side_violation": any(r.get("side_state") == "violation" for r in sub),
                    "front_decided": any(r.get("front_state") in DECIDED_VIEW for r in sub),
                    "side_decided": any(r.get("side_state") in DECIDED_VIEW for r in sub)})
    return sorted(out, key=lambda p: (_item_key(p["error_item"]), p["pair_id"]))


def verdict_counts(pairs: list[dict]) -> dict[str, int]:
    counts = Counter(p["verdict"] for p in pairs)
    return {v: counts.get(v, 0) for v in VERDICTS}


def detection_by_item(pairs: list[dict]) -> list[dict]:
    """按错误项：对数与各判定的对数。"""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        grouped[p["error_item"]].append(p)
    return [{"error_item": item, "codes": "|".join(sorted({c for p in sub for c in p["codes"].split("|")})),
             "pairs": len(sub), **verdict_counts(sub)}
            for item, sub in sorted(grouped.items(), key=lambda kv: _item_key(kv[0]))]


def detection_by_code(pairs: list[dict]) -> list[dict]:
    """按预期 code：融合检出对数 / 错误对数，逐视角 violation 对数 / 该视角可判定对数。"""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        for code in p["codes"].split("|"):
            grouped[code].append(p)
    out = []
    for code in sorted(grouped):
        sub = grouped[code]
        out.append({"code": code, "pairs": len(sub), "fused_detected": sum(p["verdict"] == "detected" for p in sub),
                    **{f"{view}_{k}": sum(p[f"{view}_{k}"] for p in sub)
                       for view in ("front", "side") for k in ("violation", "decided")}})
    return out


def off_target_candidates(rows: list[dict]) -> dict[str, int]:
    """错误样本上非预期检查项的融合 ``candidate`` 次数（按次数降序）。"""
    counts = Counter(r["code"] for r in rows
                     if r["sample_role"] == ROLE_ERROR and not r["expected"] and r["fused_status"] == "candidate")
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _group_key(row: dict) -> tuple:
    return (row["category"], FAMILY_ORDER.index(row["family"]) if row["family"] in FAMILY_ORDER else 99, row["family"])


def phase_detection(phase_rows: list[dict]) -> list[dict]:
    """按类别 × 动作分组（不含实战式）：段数、正面/侧面找到动作的段数、可融合段数。"""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in phase_rows:
        if r["family"] != "stance":
            grouped[_group_key(r)].append(r)
    return [{"category": key[0], "family": key[2], "segments": len(sub),
             "front_found": sum(bool(r["front_found"]) for r in sub),
             "side_found": sum(bool(r["side_found"]) for r in sub),
             "fusible": sum(bool(r["fusible"]) for r in sub)}
            for key, sub in sorted(grouped.items())]


def alignment_failures(phase_rows: list[dict]) -> list[dict]:
    """按类别 × 动作分组（含实战式）：正侧两路都在的段数与其中对齐不可靠的段数。"""
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in phase_rows:
        if r["views"] == "front|side":
            grouped[_group_key(r)].append(r)
    return [{"category": key[0], "family": key[2], "segments": len(sub),
             "unreliable": sum(r["alignment_reliable"] is False for r in sub)}
            for key, sub in sorted(grouped.items())]


def summarize(rows: list[dict], phase_rows: list[dict]) -> dict:
    """报告用的全部指标。"""
    pairs = detection_pairs(rows)
    return {"false_positives": false_positives(rows), "detection_pairs": pairs,
            "detection_summary": verdict_counts(pairs), "detection_by_item": detection_by_item(pairs),
            "detection_by_code": detection_by_code(pairs), "off_target": off_target_candidates(rows),
            "phases": phase_detection(phase_rows), "alignment": alignment_failures(phase_rows)}
