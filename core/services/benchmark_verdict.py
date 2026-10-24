# -*- coding: utf-8 -*-
"""录制性能测试的判定：每段是否达标（``phase_verdict``）、"按当前设置能否稳定录像"的总结论（``conclusion``）与给测试人员看的摘要。

总结论只看与正式录制负载相同的那一段（``HEADLINE_PHASE``，没有测这一段时看最后一段并写明）；其余各段是分项诊断，
单独一段正常不代表能录像。能录时再按每帧写盘耗时的 p95 与帧间隔比较，余量不足 ``TIGHT_RATIO`` 时说"余量小"。
本机在用电池、开着节电或电源模式是最佳能效时附一句提示（这会让电脑降频，测试结果偏慢）。
缺少测试要用的模型时（``report["missing_models"]``，如考试与练习都离不开的 Lite 模型），结论判"测试不完整"，
缺模型的提示放在结论最前面，已测部分的结论接在后面。
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

TARGET_FPS = 30.0
FRAME_BUDGET_MS = 1000.0 / TARGET_FPS
PRODUCTION_QUEUE = 8            # 第 6 层录像队列的容量（D-27）
PRODUCTION_OPEN_BUDGET_S = 1.5  # 第 6 层后台打开 writer 的上限（``WRITER_OPEN_TIMEOUT_S``），超过即开录失败
BACKLOG_LIMIT = math.ceil(PRODUCTION_OPEN_BUDGET_S * TARGET_FPS)  # 生产允许的开录积压帧数
MIN_CAPTURE_FPS = 28.5          # 判定"采集达到 30 fps"的下限
STOP_TOLERANCE_S = 0.5          # 最后一帧离段末超过它即判该路中途停止（甲方实测最大帧间隔 85 ms）
TIGHT_RATIO = 0.6               # 每帧写盘 p95 超过帧间隔的这个比例时算"余量小"
HEADLINE_PHASE = "production"

_VIEW_NAMES = {"front": "正面", "side": "侧面"}
_SLOW_POWER = {"best_efficiency": "电源模式是「最佳能效」", "better_battery": "电源模式是「更长续航」"}


def phase_verdict(result: Mapping[str, Any]) -> dict[str, Any]:
    """这一段是否达标：各路采集一直持续到段末、整段不低于 ``MIN_CAPTURE_FPS``，写盘时没有一帧遇到 8 帧排队，
    打开 writer 不超过 ``PRODUCTION_OPEN_BUDGET_S``、开录积压不超过 ``BACKLOG_LIMIT``（否则生产配置会判录制失败），
    各负载线程没有出错。"""
    problems: list[str] = []
    for key, entry in result["roles"].items():
        view = _VIEW_NAMES.get(key, key)
        capture = entry["capture"]
        fps = capture["fps"]
        if capture.get("error"):
            problems.append(f"{view} 摄像头在第 {capture['covered_s']} 秒后停止出帧：{capture['error']}")
        elif capture["covered_s"] < result["seconds"] - STOP_TOLERANCE_S:
            problems.append(f"{view} 摄像头在第 {capture['covered_s']} 秒后不再出帧")
        if fps is None or fps < MIN_CAPTURE_FPS:
            problems.append(f"{view} 采集 {fps} fps，低于 {MIN_CAPTURE_FPS}")
        write = entry.get("write")
        if write is not None and write.get("error"):
            problems.append(f"{view} {write['error']}")
        elif write is not None and (write["over_production_queue"] or write["dropped"]):
            problems.append(f"{view} 写盘跟不上：{write['over_production_queue']} 帧遇到 {PRODUCTION_QUEUE} 帧排队")
        if write is not None:
            problems.extend(_open_problems(view, write))
    for key in ("lite", "skeleton", "session"):
        if result.get(key, {}).get("error"):
            problems.append(result[key]["error"])
    return {"ok": not problems, "problems": problems}


def _open_problems(view: str, write: Mapping[str, Any]) -> list[str]:
    """打开 writer 超过生产上限、开录积压超过生产允许的帧数时，生产会判开录失败。"""
    problems = []
    open_ms = write.get("open_ms")
    if open_ms is not None and open_ms > PRODUCTION_OPEN_BUDGET_S * 1000.0:
        problems.append(f"{view} 打开录像文件用了 {open_ms:g} ms（{write.get('codec')}），"
                        f"超过生产上限 {PRODUCTION_OPEN_BUDGET_S * 1000.0:g} ms，录制开始时会失败")
    if write.get("backlog_dropped"):
        problems.append(f"{view} 开录积压超过 {BACKLOG_LIMIT} 帧（{write['backlog_dropped']} 帧放不下）")
    return problems


def _headline(phases: list[Mapping[str, Any]]) -> tuple[Mapping[str, Any] | None, bool]:
    """定结论用的那一段与它是否就是与正式录制相同负载的一段。"""
    for phase in phases:
        if phase["key"] == HEADLINE_PHASE:
            return phase, True
    return (phases[-1], False) if phases else (None, False)


def _worst_write_p95(phase: Mapping[str, Any]) -> float | None:
    values = [entry["write"]["write_ms"]["p95"] for entry in phase["roles"].values()
              if entry.get("write") and entry["write"]["write_ms"]["p95"] is not None]
    return max(values) if values else None


def power_hint(host: Mapping[str, Any] | None) -> str | None:
    """本机在用电池、开着节电或电源模式偏省电时的提示；都不是或取不到时为 None。"""
    power = (host or {}).get("power") or {}
    reasons = []
    if power.get("ac_power") is False:
        reasons.append("没有接电源（在用电池）")
    if power.get("battery_saver"):
        reasons.append("节电模式开着")
    if power.get("power_mode") in _SLOW_POWER:
        reasons.append(_SLOW_POWER[power["power_mode"]])
    if not reasons:
        return None
    return f"测试时电脑{'、'.join(reasons)}，会自动降速；请接上电源、关闭节电、电源模式选「最佳性能」后再测一次。"


def conclusion(report: Mapping[str, Any]) -> dict[str, Any]:
    """总结论 ``{"level", "ok", "text"}``：level 是 ok（能稳定录像）、tight（能录但余量小）、fail（不能稳定录像）
    或 incomplete（测试没做完、或缺模型没测全）；``ok`` 只在 ok/tight 时为 True。"""
    measured = _measured_conclusion(report)
    missing = [hint for hint in (report.get("missing_models") or {}).values() if hint]
    if not missing:
        return measured
    return {"level": "incomplete", "ok": False, "measured_level": measured["level"],
            "text": "".join(missing) + f"已测的部分：{measured['text']}"}


def _measured_conclusion(report: Mapping[str, Any]) -> dict[str, Any]:
    phases = list(report.get("phases", []))
    headline, same_load = _headline(phases)
    if report.get("error") or headline is None:
        reason = report.get("error") or "没有测完任何一段"
        return {"level": "incomplete", "ok": False, "text": f"测试没有做完（{reason}），不能判断能否稳定录像。"}
    basis = "" if same_load else f"（没有测与正式录制相同的负载，按「{headline['label']}」判断）"
    verdict = headline["verdict"]
    if not verdict["ok"]:
        return {"level": "fail", "ok": False,
                "text": f"按当前设置不能稳定录像{basis}：{'；'.join(verdict['problems'])}。"}
    worst = _worst_write_p95(headline)
    if worst is not None and worst > TIGHT_RATIO * FRAME_BUDGET_MS:
        return {"level": "tight", "ok": True,
                "text": f"按当前设置能录像，但余量小{basis}：写一帧最慢要 {worst:g} ms（每帧只有 "
                        f"{FRAME_BUDGET_MS:.0f} ms），电脑再忙一点就可能失败。"}
    timing = "" if worst is None else f"写一帧最慢 {worst:g} ms，每帧有 {FRAME_BUDGET_MS:.0f} ms"
    return {"level": "ok", "ok": True,
            "text": f"按当前设置可以稳定录像{basis}{f'（{timing}）' if timing else ''}。"}


def _phase_line(phase: Mapping[str, Any]) -> str:
    parts = []
    for view, entry in phase["roles"].items():
        text = f"{_VIEW_NAMES.get(view, view)} {entry['capture']['fps']} fps"
        write = entry.get("write")
        if write:
            ms = write["write_ms"]
            text += f"，{write['codec']} 打开 {write['open_ms']} ms，写一帧 p50 {ms['p50']} / p95 {ms['p95']} ms"
        parts.append(text)
    session = phase.get("session")
    if session and session.get("fps") is not None:
        parts.append(f"预览处理 {session['fps']} 次/秒")
    state = "本段正常" if phase["verdict"]["ok"] else "本段有问题：" + "；".join(phase["verdict"]["problems"])
    return f"· {phase['label']}：{'；'.join(parts)}（{state}）"


def summary_lines(report: Mapping[str, Any]) -> list[str]:
    """给测试人员看的摘要：第一行是总结论，必要时一行电源提示，之后是各段明细。"""
    lines = [f"结论：{report.get('conclusion', conclusion(report))['text']}"]
    hint = power_hint(report.get("host"))
    if hint:
        lines.append(hint)
    phases = report.get("phases", [])
    if phases:
        lines += ["", "各段明细（给技术人员看；单独一段正常不代表能录像，以上面的结论为准）："]
        lines += [_phase_line(phase) for phase in phases]
    if report.get("error"):
        lines.append(f"测试中断：{report['error']}")
    return lines


__all__ = [
    "BACKLOG_LIMIT",
    "FRAME_BUDGET_MS",
    "HEADLINE_PHASE",
    "MIN_CAPTURE_FPS",
    "PRODUCTION_OPEN_BUDGET_S",
    "PRODUCTION_QUEUE",
    "STOP_TOLERANCE_S",
    "TARGET_FPS",
    "TIGHT_RATIO",
    "conclusion",
    "phase_verdict",
    "power_hint",
    "summary_lines",
]
