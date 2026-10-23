# -*- coding: utf-8 -*-
"""样本评估的 Markdown 报告：固定章节与列，单次运行出结果表，给了基线运行时每格写"基线 → 本次"。

规则 PR 把修改前（基线）和修改后各跑一次，用 ``compare`` 生成对比报告附在 PR 里（规则检出修复计划第四节）。
章节与列固定，两份报告可以直接 diff；某行只在一边出现时，另一边写"—"。
"""
from __future__ import annotations

from collections.abc import Callable

from analysis.sample_eval.metrics import FAMILY_LABELS, VERDICTS

CATEGORY_LABELS = {"A": "A 标准示范", "B": "B 标定", "C": "C 错误样本", "D": "D 边界"}
MISSING = "—"

Cell = Callable[[dict], str]


def ratio(hit: int, total: int) -> str:
    """``hit/total（百分比）``；分母为 0 时只写 ``0/0``。"""
    return f"{hit}/{total}（{hit / total:.0%}）" if total else f"{hit}/{total}"


def _join(before: str | None, after: str | None, compare: bool) -> str:
    after = MISSING if after is None else after
    if not compare:
        return after
    before = MISSING if before is None else before
    return after if before == after else f"{before} → **{after}**"


def _table(headers: list[str], key: Callable[[dict], tuple], label: Callable[[tuple], list[str]],
           cells: list[Cell], after: list[dict], before: list[dict] | None) -> list[str]:
    """按行键对齐两次运行的同一张表；``label`` 给出键列，``cells`` 给出指标列。"""
    after_by = {key(e): e for e in after}
    before_by = {key(e): e for e in before or []}
    keys = list(after_by) + [k for k in before_by if k not in after_by]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for k in keys:
        a, b = after_by.get(k), before_by.get(k)
        values = [_join(fn(b) if b else None, fn(a) if a else None, before is not None) for fn in cells]
        lines.append("| " + " | ".join(label(k) + values) + " |")
    if not keys:
        lines.append("| " + " | ".join(["（无）"] + [""] * (len(headers) - 1)) + " |")
    return lines


def _run_info(after: dict, before: dict | None) -> list[str]:
    fields = [("标签", "label"), ("规则版本", "rule_version"), ("代码提交", "commit"), ("清单", "manifest"),
              ("清单 sha256", "manifest_sha256"), ("样本根目录", "samples_root"), ("姿态模型", "pose_variant"),
              ("清单中的对", "pairs"), ("评估的对", "evaluated_pairs"), ("评估的对 × 动作", "evaluations"),
              ("缺失视频", "missing_videos"), ("姿态缓存（复用/提取/失败）", "cache")]
    headers = ["项目", "本次"] + (["基线"] if before else [])
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for name, key in fields:
        row = [name, str(after.get(key, MISSING))] + ([str(before.get(key, MISSING))] if before else [])
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _false_positive_section(after: dict, before: dict | None) -> list[str]:
    cells: list[Cell] = [
        lambda e: ratio(e["fused_candidate"], e["fused_decided"]),
        lambda e: ratio(e["front_violation"], e["front_decided"]),
        lambda e: ratio(e["side_violation"], e["side_decided"]),
        lambda e: str(e["fused_unable"]), lambda e: str(e["pending_rule"]), lambda e: str(e["checks"]),
    ]
    return ["## 1. 正确样本上的误报", "",
            "每个检查项 code 按检查项行计数。融合：`candidate` / 可判定（`candidate`、`not_observed`）；"
            "逐视角：`violation` / 可判定（`violation`、`pass`）。", "",
            *_table(["检查项", "融合 candidate", "正面 violation", "侧面 violation", "融合 unable", "pending_rule", "检查项行"],
                    lambda e: (e["code"],), lambda k: [f"`{k[0]}`"], cells,
                    after["false_positives"], before and before["false_positives"]), ""]


def _detection_section(after: dict, before: dict | None) -> list[str]:
    summary_after = [{"verdict": v, "n": after["detection_summary"][v]} for v in VERDICTS]
    summary_before = before and [{"verdict": v, "n": before["detection_summary"][v]} for v in VERDICTS]
    verdict_cells: list[Cell] = [lambda e, v=v: str(e[v]) for v in VERDICTS]
    code_cells: list[Cell] = [lambda e: ratio(e["fused_detected"], e["pairs"]),
                              lambda e: ratio(e["front_violation"], e["front_decided"]),
                              lambda e: ratio(e["side_violation"], e["side_decided"])]
    off_after = [{"code": c, "n": n} for c, n in after["off_target"].items()]
    off_before = before and [{"code": c, "n": n} for c, n in before["off_target"].items()]
    return ["## 2. 错误样本上的检出", "",
            "每对取预期检查项各相位中最好的一条：detected（融合 `candidate`）> view_only（有视角 `violation`，融合没给出）"
            "> missed（有视角 `pass`，没有 `violation`）> unable > pending_rule。", "",
            "### 2.1 逐对最好判定", "",
            *_table(["判定", "对数"], lambda e: (e["verdict"],), lambda k: [k[0]], [lambda e: str(e["n"])],
                    summary_after, summary_before), "",
            "### 2.2 按错误项", "",
            *_table(["错误项", "预期检查项", "对数", *VERDICTS], lambda e: (e["error_item"], e["codes"]),
                    lambda k: [k[0], f"`{k[1]}`"], [lambda e: str(e["pairs"]), *verdict_cells],
                    after["detection_by_item"], before and before["detection_by_item"]), "",
            "### 2.3 按预期检查项", "",
            "融合检出 = detected 对数 / 错误对数；逐视角 = 报 `violation` 的对数 / 该视角可判定的对数。", "",
            *_table(["检查项", "融合检出", "正面 violation", "侧面 violation"], lambda e: (e["code"],),
                    lambda k: [f"`{k[0]}`"], code_cells, after["detection_by_code"], before and before["detection_by_code"]), "",
            "### 2.4 错误样本上非预期检查项的融合 candidate 次数", "",
            *_table(["检查项", "次数"], lambda e: (e["code"],), lambda k: [f"`{k[0]}`"], [lambda e: str(e["n"])],
                    off_after, off_before), "",
            "### 2.5 逐对明细", "",
            *_table(["对", "错误项", "规则动作", "判定", "各相位判定"], lambda e: (e["pair_id"], e["error_item"], e["actions"]),
                    lambda k: list(k), [lambda e: e["verdict"],
                                        lambda e: ", ".join(f"{v} {n}" for v, n in sorted(e["verdicts"].items()))],
                    after["detection_pairs"], before and before["detection_pairs"]), ""]


def _group_label(k: tuple) -> list[str]:
    return [CATEGORY_LABELS.get(k[0], k[0]), FAMILY_LABELS.get(k[1], k[1])]


def _phase_sections(after: dict, before: dict | None) -> list[str]:
    phase_cells: list[Cell] = [lambda e: str(e["segments"]), lambda e: ratio(e["front_found"], e["segments"]),
                               lambda e: ratio(e["side_found"], e["segments"]), lambda e: ratio(e["fusible"], e["segments"])]
    return ["## 3. 动作阶段检出", "",
            "每对 × 规则动作一段（实战式没有动作阶段，不计）。找到 = 该视角\"已检测到完整动作\"；"
            "可融合 = 两路都有动作窗口（自己找到，或共用另一路的同步动作时间段）且正侧对齐可靠。", "",
            *_table(["类别", "动作", "段数", "正面找到", "侧面找到", "可融合"], lambda e: (e["category"], e["family"]),
                    _group_label, phase_cells, after["phases"], before and before["phases"]), "",
            "## 4. 正侧对齐失败", "",
            "只计正侧两路都有缓存的段；失败 = `alignment.reliable` 为假（融合因此给不出结论）。", "",
            *_table(["类别", "动作", "段数", "对齐失败"], lambda e: (e["category"], e["family"]), _group_label,
                    [lambda e: str(e["segments"]), lambda e: ratio(e["unreliable"], e["segments"])],
                    after["alignment"], before and before["alignment"]), ""]


def render_report(after_meta: dict, after: dict, before_meta: dict | None = None, before: dict | None = None) -> str:
    """单次运行（``before`` 为 None）或对比报告的 Markdown 文本。``after``/``before`` 是 ``metrics.summarize`` 的结果。"""
    title = "# 样本评估对比报告" if before is not None else "# 样本评估报告"
    note = ("每格写\"基线 → 本次\"，没有变化时只写一个值；某行只在一边出现时另一边写\"—\"。"
            if before is not None else "单次运行。规则 PR 用 `compare` 生成修改前后的对比报告。")
    lines = [title, "", note, "", "## 0. 运行信息", "", *_run_info(after_meta, before_meta), "",
             "样本口径：没有故意做错的项、教师未判错的对是正确样本（未判定时按推定正确计）；"
             "有故意做错的项、教师未判对或看不清的对是错误样本。逐次切分落地前按整段评估。", "",
             *_false_positive_section(after, before), *_detection_section(after, before),
             *_phase_sections(after, before)]
    return "\n".join(lines).rstrip() + "\n"
