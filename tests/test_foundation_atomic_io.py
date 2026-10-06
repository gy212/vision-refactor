# -*- coding: utf-8 -*-
"""core.foundation.atomic_io：原子替换、目标被占用时有界重试、失败不留半截文件。"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from core.foundation import atomic_io
from core.foundation.atomic_io import (
    atomic_target,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    replace_file,
)


def _leftovers(directory: Path, keep: set[str]) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name not in keep)


def test_writers_roundtrip_and_leave_no_temporary(tmp_path):
    atomic_write_bytes(tmp_path / "a.bin", b"\x00\x01")
    atomic_write_text(tmp_path / "b.txt", "中文", encoding="utf-8-sig")
    atomic_write_json(tmp_path / "c.json", {"键": [1, 2]})
    atomic_write_json(tmp_path / "d.json", {"k": 1}, trailing_newline=True)
    assert (tmp_path / "a.bin").read_bytes() == b"\x00\x01"
    assert (tmp_path / "b.txt").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (tmp_path / "c.json").read_text(encoding="utf-8") == json.dumps({"键": [1, 2]}, ensure_ascii=False, indent=2)
    assert (tmp_path / "d.json").read_text(encoding="utf-8").endswith("}\n")
    assert _leftovers(tmp_path, {"a.bin", "b.txt", "c.json", "d.json"}) == []


def test_overwrite_replaces_whole_file(tmp_path):
    target = tmp_path / "t.json"
    target.write_text("x" * 10_000, encoding="utf-8")
    atomic_write_json(target, {"short": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"short": True}


def test_permission_error_is_retried_a_bounded_number_of_times(tmp_path, monkeypatch):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.write_bytes(b"new")
    calls = []
    real_replace = os.replace

    def flaky(a, b):
        calls.append((Path(a).name, Path(b).name))
        if len(calls) < 3:
            raise PermissionError("busy")
        real_replace(a, b)

    monkeypatch.setattr(atomic_io.os, "replace", flaky)
    replace_file(src, dst, delay_s=0)
    assert calls == [("src", "dst")] * 3
    assert dst.read_bytes() == b"new" and not src.exists()


def test_persistent_lock_raises_and_leaves_target_and_no_temporary(tmp_path, monkeypatch):
    target = tmp_path / "成绩.xlsx"
    target.write_bytes(b"old")
    attempts = []

    def locked(a, b):
        attempts.append(Path(b).name)
        raise PermissionError("locked by Excel")

    def no_copy(*_a, **_k):
        raise AssertionError("旧文件挪不开时不得就地覆盖")

    monkeypatch.setattr(atomic_io.os, "replace", locked)
    monkeypatch.setattr(atomic_io.shutil, "copy2", no_copy)
    with pytest.raises(PermissionError):
        atomic_write_bytes(target, b"new")
    # retries+1 次直接替换，再试一次挪 .bak；之后放弃
    assert attempts.count("成绩.xlsx") == atomic_io.REPLACE_RETRIES + 1
    assert attempts.count("成绩.xlsx.bak") == 1
    assert target.read_bytes() == b"old"
    assert _leftovers(tmp_path, {"成绩.xlsx"}) == []


def _block_temporary_replace(monkeypatch):
    """临时文件 → 目标的替换一直被占用，其它 os.replace（挪 .bak、还原）照常。"""
    real_replace = os.replace

    def fake(a, b):
        if Path(a).name.startswith("_tmp_"):
            raise PermissionError("source held by scanner")
        real_replace(a, b)

    monkeypatch.setattr(atomic_io.os, "replace", fake)


def _partial_copy(src, dst, **_kwargs):
    Path(dst).write_bytes(b"NE")
    raise OSError(28, "No space left on device")


def test_partial_copy_after_moving_old_aside_restores_old(tmp_path, monkeypatch):
    target = tmp_path / "成绩.xlsx"
    target.write_bytes(b"OLD-GOOD")
    _block_temporary_replace(monkeypatch)
    monkeypatch.setattr(atomic_io.shutil, "copy2", _partial_copy)
    with pytest.raises(PermissionError):
        atomic_write_bytes(target, b"NEW" * 100)
    assert target.read_bytes() == b"OLD-GOOD"
    assert _leftovers(tmp_path, {"成绩.xlsx"}) == []


def test_partial_copy_without_old_file_leaves_nothing(tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    _block_temporary_replace(monkeypatch)
    monkeypatch.setattr(atomic_io.shutil, "copy2", _partial_copy)
    with pytest.raises(PermissionError):
        atomic_write_bytes(target, b"{}")
    assert _leftovers(tmp_path, set()) == []


def test_copy_fallback_succeeds_and_removes_bak(tmp_path, monkeypatch):
    target = tmp_path / "record.json"
    target.write_bytes(b"old")
    _block_temporary_replace(monkeypatch)
    atomic_write_bytes(target, b"new")
    assert target.read_bytes() == b"new"
    assert _leftovers(tmp_path, {"record.json"}) == []


def test_text_writers_match_path_write_text_bytes(tmp_path):
    """换行随平台转换，与原先的 write_text / 文本模式临时文件逐字节一致（Windows 为 CRLF）。"""
    text = "第一行\n第二行\n"
    (tmp_path / "ref.txt").write_text(text, encoding="utf-8-sig")
    atomic_write_text(tmp_path / "new.txt", text, encoding="utf-8-sig")
    assert (tmp_path / "new.txt").read_bytes() == (tmp_path / "ref.txt").read_bytes()
    data = {"键": [1, 2]}
    (tmp_path / "ref.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    atomic_write_json(tmp_path / "new.json", data, trailing_newline=True)
    assert (tmp_path / "new.json").read_bytes() == (tmp_path / "ref.json").read_bytes()


def test_exception_inside_atomic_target_keeps_old_file(tmp_path):
    target = tmp_path / "record.json"
    target.write_text("old", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with atomic_target(target) as tmp:
            tmp.write_text("half", encoding="utf-8")
            raise RuntimeError("boom")
    assert target.read_text(encoding="utf-8") == "old"
    assert _leftovers(tmp_path, {"record.json"}) == []


def test_atomic_target_without_output_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        with atomic_target(tmp_path / "x.txt"):
            pass
    assert not (tmp_path / "x.txt").exists()


def test_temporary_is_ascii_and_invisible_to_suffix_scans(tmp_path):
    """残片不能匹配目标后缀，否则 ``*.npz`` 模板库等扫描会把半截文件当正式文件读。"""
    with atomic_target(tmp_path / "直拳_左手.npz") as tmp:
        assert tmp.parent == tmp_path and tmp.suffix == ".tmp" and tmp.name.isascii()
        tmp.write_bytes(b"x")
        assert list(tmp_path.glob("*.npz")) == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows 下打开中的文件不能被 os.replace 覆盖")
def test_real_open_handle_released_during_retry(tmp_path, monkeypatch):
    target = tmp_path / "busy.json"
    target.write_text("old", encoding="utf-8")
    handle = open(target, "rb")
    monkeypatch.setattr(atomic_io, "REPLACE_RETRY_DELAY_S", 0.05)
    threading.Timer(0.08, handle.close).start()
    started = time.monotonic()
    replace_file(_write(tmp_path / "_new", b'"new"'), target, retries=20, delay_s=0.05)
    assert time.monotonic() - started < 2.0
    assert target.read_text(encoding="utf-8") == '"new"'
    assert _leftovers(tmp_path, {"busy.json"}) == []


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path
