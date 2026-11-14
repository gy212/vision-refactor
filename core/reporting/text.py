# -*- coding: utf-8 -*-
"""学生练习结果的纯文本报告：问题说明全文与单项正侧融合说明（原 ``action_feedback.format_report``/``format_fusion``）。"""

from __future__ import annotations

from core.evaluation.action_catalog import SEGMENT_LABELS
from core.evaluation.rules.sanda_v5 import ACTIONS, PHASES, STANCES, STATE_LABELS
from core.reporting.timing_diagnostics import format_timing_diagnostics
from core.reporting.saved_results import MISSING_CHECKS_NOTE, saved_checks


def format_fusion(check: dict) -> str:
    fusion = check.get("fusion")
    if not fusion:
        return ""
    labels = {"front": "正面", "side": "侧面"}
    base = " / ".join(f"{labels[v]}{w:.0%}" for v, w in fusion["baseWeights"].items())
    lines = [f"视角基础权重：{base}（工程初值，待标定）"]
    for m in check.get("measurements", []):
        ratio = (
            "不可用" if m["violationRatio"] is None else f"{m['violationRatio']:.0%}"
        )
        lines.append(
            f"{labels[m['view']]}：有效帧{m['validFrames']}/{m['totalFrames']}，问题帧占比{ratio}"
        )
        for value in m.get("values", []):
            lines.append(
                f"  {value['name']}：中位{value['median']:.2f}，范围{value['min']:.2f}～{value['max']:.2f} {value['unit']}"
            )
    if fusion["effectiveWeights"]:
        actual = " / ".join(
            f"{labels[v]}{w:.0%}" for v, w in fusion["effectiveWeights"].items()
        )
        lines.append(
            f"实际权重：{actual}；问题支持度{fusion['support']:.0%}（非成绩、非准确率）"
        )
    if fusion["method"] == "primary_view":
        lines.append(
            "本项只使用指定画面的检查结果，不要求另一面也单独判一次；不能称为双视角一致。"
        )
    return "\n".join(lines)


def _diagnostic_lines(result: dict) -> list[str]:
    """格式化录像采集、身体可见程度与动作时段的诊断说明。"""
    lines: list[str] = []
    lines += [
        "",
        f"采集与分段诊断（离线分析模型：{result.get('poseVariant', '未知')} / CPU）：",
    ]
    joints = {
        0: "鼻",
        9: "左口角",
        10: "右口角",
        11: "左肩",
        12: "右肩",
        13: "左肘",
        14: "右肘",
        15: "左腕",
        16: "右腕",
        23: "左髋",
        24: "右髋",
        25: "左膝",
        26: "右膝",
        27: "左踝",
        28: "右踝",
        29: "左跟",
        30: "右跟",
        31: "左脚尖",
        32: "右脚尖",
    }
    for view, label in (("front", "正面"), ("side", "侧面")):
        diagnostic = result["diagnostics"].get(view)
        if not diagnostic:
            lines.append(label + "：缺少录像")
            continue
        capture = result.get("capture", {}).get(view, {})
        if capture:
            lines.append(
                f"{label}：录像{capture['width']}×{capture['height']} / {capture['sourceFps']:.2f}fps（文件标称帧率），"
                f"分析{capture['analyzedFrames']}/{capture['sourceFrames']}帧，推理输入{capture['inferenceWidth']}×{capture['inferenceHeight']}，"
                f"耗时{capture['analysisSeconds']:.1f}秒"
            )
        source = diagnostic.get("phaseSource", "own_view")
        source_label = (
            "本视角检测"
            if source == "own_view"
            else ("正面" if source.startswith("front") else "侧面")
            + "提供同步时间段，各自测量"
        )
        lines.append(
            f"{label}：可见肩髋链{diagnostic['bodyValidFrames']}/{diagnostic['totalFrames']}帧；{source_label}；本路检测：{diagnostic['reason']}"
        )
        ratios = diagnostic["jointValidRatios"]
        lines.append(
            "关键点有效比例："
            + "、".join(f"{name}{ratios[index]:.0%}" for index, name in joints.items())
        )
        for phase in result.get("phaseWindows", {}).get(view, []):
            spans = "、".join(f"{lo:.2f}～{hi:.2f}s" for lo, hi in phase["ranges"])
            lines.append(
                f"  {SEGMENT_LABELS.get(phase['segment'], phase['segment'])}/{PHASES[phase['phase']]}：{spans}"
            )
    lines.extend(
        format_timing_diagnostics(result["diagnostics"], result.get("alignment", {}))
    )
    return lines


def _check_description(check: dict) -> str:
    """按保存时的启用状态显示检查名称与试用说明，兼容没有该字段的旧记录。"""
    availability = check.get("availability")
    if not isinstance(availability, dict) or availability.get("state") != "trial":
        return check["name"]
    name = check["name"] + "（试用）"
    note = availability.get("reason", "")
    return f"{name}；试用说明：{note}" if note else name


def format_report(record: dict) -> str:
    result = record.get("result") or {}
    checks = saved_checks(result)
    action = result.get("action", record.get("action", ""))
    stance = result.get("stance", record.get("stance", ""))
    method = (
        "MediaPipe关键点规则检查；二维投影与工程阈值仍需教师标定，不作为正式成绩。"
        if result.get("backend") == "mediapipe"
        else "旧版结果；可重新分析生成MediaPipe规则检查结果。"
    )
    lines = [
        "散打动作问题说明（不打分）",
        f"学生：{record.get('studentId', '未登记学号')} {record.get('studentName', '')}",
        f"动作：{ACTIONS.get(action, action)}；{STANCES.get(stance, stance)}",
        f"记录时间：{record.get('createdAt', '未记录')}；保存至：{record.get('expiresAt', '未记录')}",
        f"规则版本：{result.get('ruleVersion', '未记录')}",
        method,
        "",
        result.get("summary", ""),
    ]
    if not checks:
        lines.append(MISSING_CHECKS_NOTE)
    if result.get("fusionMethod"):
        lines += [
            "分析方式：正侧面按动作/检查项及有效帧比例加权；仅适用单视角公式的项目单独注明。",
            "；".join(
                f"{label}：{sum(c['status'] == status for c in checks)}项"
                for status, label in STATE_LABELS.items()
            ),
        ]
    if result.get("diagnostics"):
        lines += _diagnostic_lines(result)
    lines += ["", "动作问题："]
    issues = [
        c for c in checks if c["status"] == "candidate" and c["review"] != "revoked"
    ]
    for i, c in enumerate(issues, 1):
        state = "教师已确认" if c["review"] == "confirmed" else "待教师确认"
        lines.append(
            f"{i}. {c['segmentLabel']} / {c['phaseLabel']} / {c['bodyPart']}：{_check_description(c)}（{state}）"
        )
        if c.get("fusion"):
            lines.append(format_fusion(c))
    if not issues:
        lines.append("当前没有未撤销的候选问题；不代表所有项目均合格。")
    incomplete = [c for c in checks if c["status"] in {"unable", "pending_rule"}]
    if incomplete:
        lines += ["", "未完成判断的项目："]
        for c in incomplete:
            lines.append(
                f"- {c['segmentLabel']} / {c['phaseLabel']} / {c['bodyPart']}：{_check_description(c)}；{c['reason']}"
            )
            if c.get("measurements"):
                lines.append(format_fusion(c))
    if result.get("needsRerecord"):
        lines += [
            "",
            "请重新录制：全身及拳脚入画、光线充足、减少遮挡，保留开始和结束实战式。",
        ]
    if result.get("fusionMethod"):
        lines += ["", "已检查未发现的问题项（不代表全部动作合格）："]
        for c in checks:
            if c["status"] == "not_observed":
                lines += [
                    f"- {c['segmentLabel']} / {c['phaseLabel']}：{_check_description(c)}",
                    format_fusion(c),
                ]
    return "\n".join(lines) + "\n"
