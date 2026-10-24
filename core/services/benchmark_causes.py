# -*- coding: utf-8 -*-
"""录制性能测试的"可能的原因"：只按报告里实测到的整机数据（电源、内存、换页、内存复制、录像盘读写）给出白话原因，
没有数据的项不猜；以及这些整机项在摘要明细里的一行说明。总结论（``benchmark_verdict.conclusion``）把原因接在结论后面。

阈值是工程初值（甲方报告回收后再校）：
- 可用内存低于 ``LOW_MEMORY_GB``：两路 1080p 录像与识别的工作集约 1～1.5 GB，再少系统就要换页；
- 换页率高于 ``PAGING_LIMIT`` 页/秒（约 4 MB/s 持续读写页面文件）：内存不够在换页；
- 复制一帧画面的带宽低于 ``MIN_COPY_GBPS``：处理循环每帧要复制、转色、缩放几次画面，带宽太低会拖慢；
- 录像盘写入低于本机实测录像写入量的 ``DISK_MARGIN`` 倍，或写入中卡顿超过 ``DISK_STALL_MS``
  （录像 writer 送帧队列约能盖住 1 秒，再长就会判写盘跟不上）。
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

LOW_MEMORY_GB = 2.0
PAGING_LIMIT = 1000.0
MIN_COPY_GBPS = 2.0
DISK_MARGIN = 2.0
DISK_STALL_MS = 500.0
FALLBACK_RECORD_MB_S = 24.0     # 没测到录像写入量时按两路 × 30 帧 × 约 0.4 MB 估计

_SLOW_POWER = {"best_efficiency": "电源模式是「最佳能效」", "better_battery": "电源模式是「更长续航」"}
_MODE_NAMES = {"best_efficiency": "最佳能效", "better_battery": "更长续航", "balanced": "平衡", "best_performance": "最佳性能"}
_PLAN_NAMES = {"balanced": "平衡", "high_performance": "高性能", "power_saver": "节能", "ultimate": "卓越性能"}


def power_hint(host: Mapping[str, Any] | None) -> str | None:
    """本机在用电池、开着节电、电源模式偏省电或电源计划是"节能"时的原因与建议；都不是或取不到时为 None。"""
    power = (host or {}).get("power") or {}
    reasons = []
    if power.get("ac_power") is False:
        reasons.append("没有接电源（在用电池）")
    if power.get("battery_saver"):
        reasons.append("节电模式开着")
    if power.get("power_mode") in _SLOW_POWER:
        reasons.append(_SLOW_POWER[power["power_mode"]])
    if (power.get("plan") or {}).get("key") == "power_saver":
        reasons.append("电源计划是「节能」")
    if not reasons:
        return None
    return (f"电脑在省电状态（{'、'.join(reasons)}），会自动降速；请接上电源、关闭节电、电源模式选「最佳性能」"
            f"后再测一次")


def likely_causes(report: Mapping[str, Any]) -> list[str]:
    """按实测数据给出的可能原因（白话，已按电源、内存、录像盘排序）；没有可疑项时为空列表。"""
    causes = [power_hint(report.get("host")), _memory_cause(report), _paging_cause(report), _copy_cause(report)]
    causes += _disk_causes(report)
    return [cause for cause in causes if cause]


def _memory_cause(report: Mapping[str, Any]) -> str | None:
    values = [((report.get("host") or {}).get("memory") or {}).get("available_gb")]
    values += [(phase.get("memory") or {}).get("min_available_gb") for phase in report.get("phases", [])]
    measured = [float(value) for value in values if value is not None]
    if not measured or min(measured) >= LOW_MEMORY_GB:
        return None
    return f"可用内存不足 {LOW_MEMORY_GB:g} GB（测试中最低 {min(measured):g} GB），请关掉其它程序后再测"


def _paging_cause(report: Mapping[str, Any]) -> str | None:
    rates = [(phase.get("memory") or {}).get("pages_per_s") for phase in report.get("phases", [])]
    measured = [float(rate) for rate in rates if rate is not None]
    if not measured or max(measured) <= PAGING_LIMIT:
        return None
    return f"内存不够用，系统在频繁把内存换到硬盘上（最多每秒 {max(measured):.0f} 页）"


def _copy_cause(report: Mapping[str, Any]) -> str | None:
    copy = report.get("memory_bandwidth") or {}
    speed = copy.get("gb_per_s")
    if speed is None or speed >= MIN_COPY_GBPS:
        return None
    return f"内存读写慢：复制一帧画面每秒只有 {speed:g} GB（每次最慢 {copy['copy_ms']['p95']} ms）"


def record_mb_per_s(report: Mapping[str, Any]) -> float:
    """本机实测的录像写入量（各写盘段两路合计 MB/s 的最大值）；没有时用 ``FALLBACK_RECORD_MB_S``。"""
    totals = []
    for phase in report.get("phases", []):
        rates = [(entry.get("write") or {}).get("mb_per_s") for entry in phase.get("roles", {}).values()]
        rates = [rate for rate in rates if rate]
        if rates:
            totals.append(sum(rates))
    return round(max(totals), 1) if totals else FALLBACK_RECORD_MB_S


def _drive_text(volume: Mapping[str, Any]) -> str:
    root = (volume.get("root") or "").rstrip("\\") or "录像盘"
    kinds = []
    if volume.get("bus") == "usb":
        kinds.append("USB 盘（U 盘或移动硬盘）")
    elif volume.get("drive_type") == "removable":
        kinds.append("可移动盘")
    elif volume.get("drive_type") == "remote":
        kinds.append("网络盘")
    if volume.get("seek_penalty"):
        kinds.append("机械硬盘")
    return f"录像盘 {root}" + (f"是{'、'.join(kinds)}，" if kinds else "")


def _disk_causes(report: Mapping[str, Any]) -> list[str]:
    disk = report.get("disk") or {}
    write = disk.get("write") or {}
    speed = write.get("mb_per_s")
    if speed is None:
        return []
    where, causes = _drive_text(disk.get("volume") or {}), []
    need = record_mb_per_s(report)
    if speed < need * DISK_MARGIN:
        causes.append(f"{where}写入只有 {speed:g} MB/s（录像要写约 {need:g} MB/s，余量不到 {DISK_MARGIN:g} 倍），"
                      f"请把主界面的「保存目录」改到电脑自带的硬盘上")
    stall = write.get("stall_ms")
    if stall is not None and stall > DISK_STALL_MS:
        causes.append(f"{where}写入时卡住过 {stall / 1000:.1f} 秒")
    return causes


def detail_lines(report: Mapping[str, Any]) -> list[str]:
    """摘要明细里的整机项：电源、内存、录像盘各一行（没有数据的项不写）。"""
    lines = []
    power = (report.get("host") or {}).get("power")
    if power:
        plan = power.get("plan") or {}
        source = {True: "接着电源", False: "用电池"}.get(power.get("ac_power"), "电源未知")
        saver = "节电开着" if power.get("battery_saver") else "节电关着"
        lines.append(f"· 电源：{source}，{saver}，电源模式 {_MODE_NAMES.get(power.get('power_mode'), power.get('power_mode') or '未知')}，"
                     f"电源计划 {plan.get('name') or '未知'}（{_PLAN_NAMES.get(plan.get('key'), plan.get('key') or '-')}）")
    copy = report.get("memory_bandwidth")
    memory = (report.get("host") or {}).get("memory") or {}
    if copy or memory:
        text = f"· 内存：共 {memory.get('total_gb', '?')} GB，开始时可用 {memory.get('available_gb', '?')} GB"
        if copy and copy.get("gb_per_s") is not None:
            text += (f"；复制一帧（{copy['frame_mb']} MB）p50 {copy['copy_ms']['p50']} / p95 {copy['copy_ms']['p95']} ms，"
                     f"每秒 {copy['gb_per_s']} GB")
        lines.append(text)
    disk = report.get("disk")
    if disk:
        lines.append(_disk_line(disk))
    return lines


def _disk_line(disk: Mapping[str, Any]) -> str:
    volume = disk.get("volume") or {}
    facts = [value for value in (volume.get("filesystem"), volume.get("bus"), volume.get("product")) if value]
    if volume.get("free_gb") is not None:
        facts.append(f"剩 {volume['free_gb']} GB")
    head = f"· 录像盘 {volume.get('root') or disk.get('directory')}（{'，'.join(facts) or '信息取不到'}）"
    if disk.get("skipped") or disk.get("error"):
        return f"{head}：{disk.get('skipped') or disk.get('error')}"
    write, read = disk.get("write") or {}, disk.get("read") or {}
    text = (f"{head}：写 {write.get('mb_per_s')} MB/s（每块 {write.get('chunk_kb')} KB，p95 "
            f"{(write.get('write_ms') or {}).get('p95')} ms，刷盘 p95 {(write.get('flush_ms') or {}).get('p95')} ms，"
            f"最长卡顿 {write.get('stall_ms')} ms）")
    if read:
        text += f"，读 {read.get('mb_per_s')} MB/s" + ("" if read.get("uncached") else "（可能读自缓存）")
    return text


__all__ = ["DISK_MARGIN", "DISK_STALL_MS", "LOW_MEMORY_GB", "MIN_COPY_GBPS", "PAGING_LIMIT", "detail_lines",
           "likely_causes", "power_hint", "record_mb_per_s"]
