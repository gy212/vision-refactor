# -*- coding: utf-8 -*-
"""``capture.json``：一个录制片段的采集记录（D-18，设计文档 §8.8），由各路 ``Timeline`` 生成，原子写出，按需读取。

内容是采集时钟（``capture_clock_ns``）上的录制区间、每路的采集统计和每个输出帧对应的源帧，
用来区分"视频时长正确"和"真实采集率达标"：补帧后的帧数不能当作真实采集率。
旧录像没有这个文件，读取时返回 None，调用方照旧按视频本身处理，不推造采集时间。
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.foundation.atomic_io import atomic_write_json
from core.media.pacing import Timeline

SCHEMA_VERSION = 1
CLOCK = "perf_counter_ns"


def capture_document(
    timelines: Mapping[str, Timeline],
    extra: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """各路（键为视角名）的采集记录；``extra`` 按视角补充写出方的字段（如录像队列峰值）。"""
    if not timelines:
        raise ValueError("采集记录至少需要一路")
    rates = {timeline.fps for timeline in timelines.values()}
    if len(rates) != 1:
        raise ValueError(f"各路输出帧率不一致：{sorted(rates)}")
    views = {}
    for view, timeline in timelines.items():
        record = timeline.record()
        record.update((extra or {}).get(view, {}))
        views[view] = record
    return {"schema": SCHEMA_VERSION, "clock": CLOCK, "fps": rates.pop(), "views": views}


def write_capture_record(path: Path | str, document: Mapping[str, Any]) -> None:
    """原子写出（UTF-8，不缩进：每个输出帧一项，60 秒约 1800 项）。"""
    atomic_write_json(path, dict(document), indent=None)


def read_capture_record(path: Path | str) -> dict[str, Any] | None:
    """读取采集记录；文件不存在、不是这个格式或版本不认识时返回 None。"""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict) or document.get("schema") != SCHEMA_VERSION:
        return None
    if not isinstance(document.get("views"), dict):
        return None
    return document


__all__ = ["CLOCK", "SCHEMA_VERSION", "capture_document", "read_capture_record", "write_capture_record"]
