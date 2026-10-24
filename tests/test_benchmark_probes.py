# -*- coding: utf-8 -*-
"""录制性能测试的整机项：内存复制带宽、录像盘读写（临时目录 + 小数据量真实读写：写读字节数、刷盘、删除测试文件、
空间不够时跳过、读写出错与取消）、每段的内存监测（假内存与换页读数）；按实测数据给出的可能原因与摘要明细；
以及 L0 的系统查询（电源计划与电源模式、内存、换页计数器、卷信息、绕过缓存的读）在本机 Windows 上能取到值、
API 失败时写明原因不抛出。"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

from core.foundation import CancelToken, Cancelled, PagingCounter, TaskGroup, UnbufferedReader, memory_snapshot
from core.foundation import power as power_mod
from core.foundation import volume_info
from core.services import benchmark_causes as causes
from core.services import benchmark_verdict as verdict
from core.services.benchmark_probes import DISK_FILE, MemoryWatch, SystemProbes, disk_test, memory_bandwidth

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="系统查询只在 Windows 上有")


class FakePaging:
    def __init__(self, rate=12.0, error=None):
        self.rate, self.error, self.closed = rate, error, False

    def read(self):
        return {"pages_per_s": self.rate, "error": self.error}

    def close(self):
        self.closed = True


def _probes(**changes) -> SystemProbes:
    base = dict(memory=lambda: {"total_gb": 16.0, "available_gb": 8.0}, paging=FakePaging,
                volume=lambda _d: {"root": "T:\\", "free_gb": 500.0, "errors": {}},
                memory_seconds=0.05, frame_shape=(108, 192, 3), disk_total_bytes=1 * 2**20,
                disk_chunk_bytes=32 * 2**10, disk_read_chunk_bytes=64 * 2**10, disk_max_write_s=2.0,
                disk_max_read_s=2.0, disk_spare_bytes=0, sample_s=0.02)
    base.update(changes)
    return SystemProbes(**base)


# --------------------------------------------------------------------------- 内存


def test_memory_bandwidth_reports_rate_and_per_copy_latency():
    result = memory_bandwidth(_probes(), CancelToken())
    assert result["frame_mb"] == pytest.approx(108 * 192 * 3 / 2**20, abs=0.01)
    assert result["copies"] > 10 and result["gb_per_s"] > 0
    assert set(result["copy_ms"]) == {"p50", "p95", "max"} and result["copy_ms"]["p95"] >= result["copy_ms"]["p50"]


def test_memory_watch_tracks_the_lowest_available_memory_and_paging():
    readings = iter([8.0, 3.5, 1.2, 4.0] + [4.0] * 1000)
    paging = FakePaging(rate=1500.0)
    tasks = TaskGroup("test-memory")
    watch = MemoryWatch(_probes(memory=lambda: {"available_gb": next(readings)}, paging=lambda: paging),
                        tasks).start()
    threading.Event().wait(0.2)
    result = watch.stop()
    tasks.cancel(reason="test")
    tasks.join_until(5.0)
    assert result["before"] == {"available_gb": 8.0} and result["min_available_gb"] == 1.2
    assert result["pages_per_s"] == 1500.0 and result["paging_error"] is None and paging.closed


def test_memory_watch_without_readings_reports_none_and_the_paging_error():
    tasks = TaskGroup("test-memory")
    watch = MemoryWatch(_probes(memory=lambda: None, paging=lambda: FakePaging(None, "PdhOpenQueryW 返回 0x1")),
                        tasks).start()
    result = watch.stop()
    tasks.cancel(reason="test")
    tasks.join_until(5.0)
    assert result["min_available_gb"] is None and result["pages_per_s"] is None
    assert result["paging_error"] == "PdhOpenQueryW 返回 0x1"


# --------------------------------------------------------------------------- 录像盘


def test_disk_test_writes_flushes_reads_and_removes_its_file(tmp_path, monkeypatch):
    import core.services.benchmark_probes as probes_mod

    flushes = []
    real_fsync = probes_mod.os.fsync
    monkeypatch.setattr(probes_mod.os, "fsync", lambda fd: flushes.append(fd) or real_fsync(fd))
    messages: list[str] = []
    result = disk_test(tmp_path / "录制", _probes(), CancelToken(), messages.append)
    write, read = result["write"], result["read"]
    assert result["error"] is None and result["skipped"] is None and result["volume"]["root"] == "T:\\"
    assert write["bytes"] == 2**20 and write["chunks"] == 32 and write["complete"] and write["mb_per_s"] > 0
    assert len(flushes) == 32 // 30 + 1                       # 每 30 块刷一次盘，最后再刷一次
    assert write["stall_ms"] >= write["write_ms"]["max"] and write["flush_ms"]["max"] is not None
    assert read["bytes"] == 2**20 and read["mb_per_s"] > 0 and read["uncached"] is (sys.platform == "win32")
    assert not (tmp_path / "录制" / DISK_FILE).exists()
    assert any("正在测录像盘写入：已写" in text for text in messages)
    assert any("正在测录像盘读取" in text for text in messages)


def test_disk_test_is_skipped_when_the_drive_is_nearly_full(tmp_path):
    result = disk_test(tmp_path, _probes(volume=lambda _d: {"root": "F:\\", "free_gb": 0.5},
                                         disk_spare_bytes=2**30), CancelToken(), lambda _text: None)
    assert result["write"] is None and "只剩 0.5 GB" in result["skipped"]
    assert not (tmp_path / DISK_FILE).exists()


def test_disk_errors_are_reported_and_the_file_is_still_removed(tmp_path):
    def broken_reader(_path, _chunk):
        raise PermissionError(13, "拒绝访问")

    result = disk_test(tmp_path, _probes(reader=broken_reader), CancelToken(), lambda _text: None)
    assert result["write"]["complete"] and result["read"] is None
    assert result["error"].startswith("录像盘读写失败") and "拒绝访问" in result["error"]
    assert not (tmp_path / DISK_FILE).exists()


def test_cancel_during_the_disk_test_removes_the_file(tmp_path):
    token = CancelToken()
    messages = []

    def progress(text):
        messages.append(text)
        token.cancel("test")

    with pytest.raises(Cancelled):
        disk_test(tmp_path, _probes(), token, progress)
    assert not (tmp_path / DISK_FILE).exists()


# --------------------------------------------------------------------------- 原因与明细


def _phase(mb_per_s=5.0, min_available=8.0, paging=10.0):
    write = {"write_ms": {"p50": 5.0, "p95": 8.0, "max": 9.0}, "codec": "MJPG", "open_ms": 5.0, "mb_per_s": mb_per_s}
    return {"key": "production", "label": "按当前设置录像", "seconds": 30.0,
            "roles": {"front": {"capture": {"fps": 30.0}, "write": write},
                      "side": {"capture": {"fps": 30.0}, "write": dict(write)}},
            "memory": {"min_available_gb": min_available, "pages_per_s": paging},
            "verdict": {"ok": True, "problems": []}}


def _disk(mb_per_s=300.0, stall_ms=40.0, **volume):
    return {"directory": "F:\\10 测试视频\\_临时", "skipped": None, "error": None,
            "volume": {"root": "F:\\", "drive_type": "fixed", "filesystem": "exFAT", "free_gb": 100.0, **volume},
            "write": {"mb_per_s": mb_per_s, "chunk_kb": 384, "stall_ms": stall_ms,
                      "write_ms": {"p95": 1.0}, "flush_ms": {"p95": 20.0}},
            "read": {"mb_per_s": 400.0, "uncached": True}}


def test_no_causes_when_the_measurements_are_fine():
    report = {"phases": [_phase()], "disk": _disk(), "memory_bandwidth": {"gb_per_s": 8.0, "copy_ms": {"p95": 1.0}},
              "host": {"memory": {"available_gb": 8.0}, "power": {"ac_power": True, "plan": {"key": "balanced"}}}}
    assert causes.likely_causes(report) == []
    result = verdict.conclusion(report)
    assert result["level"] == "ok" and result["causes"] == [] and "需要注意" not in result["text"]


def test_slow_usb_drive_low_memory_paging_and_power_saver_plan_are_named_from_the_data():
    report = {"phases": [_phase(mb_per_s=6.0, min_available=1.4, paging=2400.0)],
              "disk": _disk(mb_per_s=15.0, stall_ms=1800.0, bus="usb"),
              "memory_bandwidth": {"gb_per_s": 1.1, "copy_ms": {"p95": 7.5}},
              "host": {"memory": {"available_gb": 1.8}, "power": {"ac_power": True, "plan": {"key": "power_saver"}}}}
    found = causes.likely_causes(report)
    assert found[0].startswith("电脑在省电状态（电源计划是「节能」）")
    assert "可用内存不足 2 GB（测试中最低 1.4 GB）" in found[1]
    assert "每秒 2400 页" in found[2] and "每秒只有 1.1 GB" in found[3]
    assert "录像盘 F:是USB 盘（U 盘或移动硬盘），写入只有 15 MB/s（录像要写约 12 MB/s" in found[4]
    assert found[5] == "录像盘 F:是USB 盘（U 盘或移动硬盘），写入时卡住过 1.8 秒"
    failing = dict(report, phases=[dict(_phase(), verdict={"ok": False, "problems": ["正面 写盘跟不上"]})])
    result = verdict.conclusion(failing)
    assert result["level"] == "fail" and "可能的原因：电脑在省电状态" in result["text"]


def test_disk_cause_needs_a_measured_shortfall_not_just_a_usb_drive():
    """USB 盘本身不算原因：写入够快、不卡顿时不提。"""
    report = {"phases": [_phase(mb_per_s=6.0)], "disk": _disk(mb_per_s=120.0, bus="usb")}
    assert causes.likely_causes(report) == []
    assert causes.record_mb_per_s({"phases": []}) == causes.FALLBACK_RECORD_MB_S


def test_detail_lines_describe_power_memory_and_drive():
    report = {"disk": _disk(bus="usb", product="SanDisk Ultra"),
              "memory_bandwidth": {"frame_mb": 5.93, "gb_per_s": 6.2, "copy_ms": {"p50": 0.9, "p95": 1.3}},
              "host": {"memory": {"total_gb": 16.0, "available_gb": 1.8},
                       "power": {"ac_power": True, "battery_saver": False, "power_mode": "best_performance",
                                 "plan": {"name": "平衡", "key": "balanced"}}}}
    lines = causes.detail_lines(report)
    assert lines[0] == "· 电源：接着电源，节电关着，电源模式 最佳性能，电源计划 平衡（平衡）"
    assert lines[1].startswith("· 内存：共 16.0 GB，开始时可用 1.8 GB；复制一帧（5.93 MB）p50 0.9 / p95 1.3 ms")
    assert lines[2].startswith("· 录像盘 F:\\（exFAT，usb，SanDisk Ultra，剩 100.0 GB）：写 300.0 MB/s")
    skipped = dict(_disk(), skipped="录像盘只剩 0.5 GB")
    assert causes.detail_lines({"disk": skipped})[0].endswith("：录像盘只剩 0.5 GB")


# --------------------------------------------------------------------------- L0 系统查询


@windows_only
def test_power_status_reads_plan_and_mode_on_this_machine():
    status = power_mod.power_status()
    assert status["errors"] == {} and isinstance(status["ac_power"], bool)
    plan = status["plan"]
    assert len(plan["guid"]) == 36 and plan["name"] and plan["key"] in {*power_mod.POWER_PLANS.values(), "other"}
    assert status["power_mode"] is not None


def test_power_status_records_api_failures_instead_of_raising(monkeypatch):
    monkeypatch.setattr(power_mod, "is_windows", lambda: True)

    def broken(name):
        raise OSError(f"{name} 不可用")

    monkeypatch.setattr(power_mod, "library", broken)
    status = power_mod.power_status()
    assert status["plan"] is None and status["power_mode"] is None and status["ac_power"] is None
    assert set(status["errors"]) == {"status", "power_mode", "plan"} and "不可用" in status["errors"]["plan"]


@windows_only
def test_memory_paging_and_volume_queries_work_on_this_machine(tmp_path):
    memory = memory_snapshot()
    assert memory["total_gb"] >= memory["available_gb"] > 0 and 0 <= memory["load_percent"] <= 100
    counter = PagingCounter().start()
    try:
        threading.Event().wait(0.3)
        paging = counter.read()
    finally:
        counter.close()
    assert paging["error"] is None and paging["pages_per_s"] >= 0
    volume = volume_info(tmp_path)
    assert volume["root"] and volume["root"].endswith("\\") and volume["filesystem"]
    assert volume["drive_type"] in {"fixed", "removable", "remote", "ramdisk"} and volume["free_gb"] > 0


@windows_only
def test_unbuffered_reader_reads_a_file_that_is_not_sector_aligned(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(bytes(range(256)) * 4097)            # 1048832 字节，不是 4096 的整数倍
    total = 0
    with UnbufferedReader(path, 100_000) as reader:
        assert reader.uncached and reader.size % 4096 == 0
        while (got := reader.read()):
            total += got
    assert total == path.stat().st_size
