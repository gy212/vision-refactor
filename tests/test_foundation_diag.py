# -*- coding: utf-8 -*-
"""core.foundation.diag：行格式、取值转义、关键字字段、stderr 与 desktop.log 两种出口、写失败静默。"""
from __future__ import annotations

import io
import json
import re
import sys
from datetime import datetime

from core.foundation import diag, paths


_FIELD = re.compile(r'([^\s=]+)=("(?:\\.|[^"\\])*"|\S+)(?:\s|$)')


def _parse(line: str) -> dict[str, str]:
    """按"单词或 JSON 字符串"规则切分；整行必须被完整覆盖。"""
    fields, pos = {}, 0
    while pos < len(line):
        match = _FIELD.match(line, pos)
        assert match, f"无法解析：{line[pos:]!r}"
        fields[match.group(1)] = match.group(2)
        pos = match.end()
    return fields


def test_format_line_orders_fields_and_skips_none():
    line = diag.format_line(
        "device.state",
        {"role": "front", "device": 0, "from_": "opening", "to": "ready", "lease": None, "elapsed_ms": 12.5},
        ts=datetime(2026, 10, 4, 9, 30, 1, 234000),
    )
    assert line == (
        "ts=2026-10-04T09:30:01.234 event=device.state role=front device=0 "
        "from=opening to=ready elapsed_ms=12.5"
    )


def test_values_with_spaces_or_quotes_are_json_strings():
    line = diag.format_line("camera.error", {
        "device": "USB Camera 2",
        "error": 'open "failed"=1',
        "requested": {"w": 1280, "h": 720},
        "actual": {"name": "USB Cam", "fps": 30},
        "ok": False,
        "empty": "",
        "名称": "前置",
    })
    fields = _parse(line)
    assert json.loads(fields["device"]) == "USB Camera 2"
    assert json.loads(fields["error"]) == 'open "failed"=1'
    assert json.loads(fields["requested"]) == {"w": 1280, "h": 720}
    assert json.loads(json.loads(fields["actual"])) == {"name": "USB Cam", "fps": 30}
    assert fields["ok"] == "false" and fields["empty"] == '""' and fields["名称"] == "前置"
    assert "\n" not in diag.format_line("x", {"error": "a\nb"})


def test_event_writes_one_line_to_stderr(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)
    diag.event("session.end", session="s1", cancel="user_stop")
    lines = buffer.getvalue().splitlines()
    assert len(lines) == 1
    fields = _parse(lines[0])
    assert fields["event"] == "session.end" and fields["cancel"] == "user_stop"


def test_event_appends_desktop_log_without_stderr(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    diag.event("camera.ready", role="side")
    diag.event("camera.ready", role="front")
    lines = (tmp_path / "desktop.log").read_text(encoding="utf-8").splitlines()
    assert [_parse(line)["role"] for line in lines] == ["side", "front"]


def test_event_never_raises(monkeypatch):
    class Broken(io.StringIO):
        def write(self, _text):
            raise OSError("disk full")

    monkeypatch.setattr(sys, "stderr", Broken())
    diag.event("finalize.state", stage="write")
