# -*- coding: utf-8 -*-
"""诊断日志的唯一出口：``event(name, **fields)`` 写一行 ``key=value``，字段集见设计文档 §4.5。

- 一行以 ``ts=`` 开头、``event=`` 其次，其余字段按传入顺序；值为 ``None`` 的字段省略。
- 字符串不含空白、``=``、引号时原样写出，否则写成 JSON 字符串；其余类型写成紧凑 JSON，
  结果含空白时再整体写成 JSON 字符串。因此每个值要么是不含空白的单词，要么是一个 JSON 字符串。
- Python 关键字字段用尾随下划线传入（``from_=`` 写成 ``from``）。
- 源码运行写 stderr；冻结包里 ``desktop_launcher`` 已把 stderr 重定向到 ``desktop.log``，
  没有可用的 stderr 时直接追加 ``<repo_root>/desktop.log``。写失败静默，诊断不影响业务。
"""
from __future__ import annotations

import json
import sys
import threading
from datetime import datetime
from typing import Any, TextIO

from core.foundation import paths

_lock = threading.Lock()
_UNSAFE = set(" \t\r\n=\"'")


def _value(value: Any) -> str:
    if isinstance(value, str):
        if value and not (_UNSAFE & set(value)):
            return value
        return json.dumps(value, ensure_ascii=False)
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        text = str(value)
    if not text or any(ch.isspace() for ch in text):
        return json.dumps(text, ensure_ascii=False)
    return text


def format_line(name: str, fields: dict[str, Any], ts: datetime | None = None) -> str:
    """把一个事件格式化成一行（不含换行）。"""
    stamp = (ts or datetime.now()).isoformat(timespec="milliseconds")
    parts = [f"ts={stamp}", f"event={_value(name)}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key.rstrip('_')}={_value(value)}")
    return " ".join(parts)


def _write(line: str, stream: TextIO | None) -> None:
    if stream is not None:
        stream.write(line + "\n")
        stream.flush()
        return
    with (paths.repo_root() / "desktop.log").open("a", encoding="utf-8") as log:
        log.write(line + "\n")


def event(name: str, **fields: Any) -> None:
    """记录一个诊断事件，如 ``event("camera.ready", role="front", device=0, elapsed_ms=812)``。"""
    try:
        line = format_line(name, fields)
        with _lock:
            _write(line, sys.stderr)
    except Exception:
        pass
