# -*- coding: utf-8 -*-
"""录制性能测试的整机项：内存复制带宽、录像盘顺序读写，以及每段的内存与换页监测（判定见 ``benchmark_verdict``）。

- 内存：反复 ``copy()`` 一帧 1080p BGR 大小的数组（约 6 MB，处理循环每帧都要复制画面），记每次耗时与带宽；
  可用内存少到系统换页时，这一项会明显变慢。每段另记段前、段后与段内最低（约每 0.5 s 采样）的可用内存和全系统换页率。
- 录像盘：在用户设置的录制目录里（本次测试的临时目录就在它下面）顺序写一个测试文件，块大小接近录像每帧的 JPEG
  （约 370 KB），每写满一秒的帧（30 块）就 ``fsync``（Windows 上即 ``FlushFileBuffers``），系统写缓存不让结果虚高；
  记写入速度与每块、每次刷盘的耗时分位。再绕过系统缓存顺序读回（``UnbufferedReader``），记读取速度。
  测试文件在 ``finally`` 里删除；录像盘剩余空间不够时跳过并写明原因。同时记录卷的类型、文件系统、总线（USB 等）。

系统查询经 ``SystemProbes`` 注入（测试用假实现、临时目录与小数据量，不依赖本机）。
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.foundation import (
    CancelToken,
    PagingCounter,
    TaskGroup,
    UnbufferedReader,
    host_summary,
    memory_snapshot,
    ms_summary,
    volume_info,
)

FRAME_SHAPE = (1080, 1920, 3)
MEMORY_SECONDS = 2.0
MEMORY_SAMPLE_S = 0.5
DISK_TOTAL_BYTES = 256 * 2**20
DISK_CHUNK_BYTES = 384 * 2**10      # 录像 writer 质量 85 时 1080p 每帧实测约 367 KB
DISK_FLUSH_EVERY = 30               # 一路一秒的帧数
DISK_READ_CHUNK_BYTES = 2**20
DISK_MAX_WRITE_S = 20.0
DISK_MAX_READ_S = 10.0
DISK_SPARE_BYTES = 2**30            # 写测试文件后至少还要留下的空间
DISK_FILE = "_录像盘读写测试.bin"
_DATA_BANK = 8                      # 轮流写的随机数据块数（随机内容，磁盘或文件系统压缩不会让结果虚高）
_MB = 2**20
_GB = 2**30


def _start_paging() -> Any:
    return PagingCounter().start()


@dataclass(frozen=True)
class SystemProbes:
    """整机项用到的系统查询与数据量；默认是真实查询，测试注入假实现与小数据量。
    ``paging()`` 返回已开始计数的对象（``read() -> {"pages_per_s", "error"}``、``close()``）；
    ``reader(path, chunk)`` 是 ``UnbufferedReader`` 同接口的读取器。"""

    host: Callable[[], Mapping[str, Any]] = host_summary
    memory: Callable[[], Mapping[str, Any] | None] = memory_snapshot
    paging: Callable[[], Any] = _start_paging
    volume: Callable[[Path], Mapping[str, Any]] = volume_info
    reader: Callable[[Path, int], Any] = UnbufferedReader
    memory_seconds: float = MEMORY_SECONDS
    frame_shape: tuple[int, int, int] = FRAME_SHAPE
    disk_total_bytes: int = DISK_TOTAL_BYTES
    disk_chunk_bytes: int = DISK_CHUNK_BYTES
    disk_read_chunk_bytes: int = DISK_READ_CHUNK_BYTES
    disk_max_write_s: float = DISK_MAX_WRITE_S
    disk_max_read_s: float = DISK_MAX_READ_S
    disk_spare_bytes: int = DISK_SPARE_BYTES
    sample_s: float = MEMORY_SAMPLE_S

    @property
    def seconds(self) -> float:
        """整机项最多用时（写进进度说明与总时长）。"""
        return self.memory_seconds + self.disk_max_write_s + self.disk_max_read_s


# --------------------------------------------------------------------------- #
# 内存
# --------------------------------------------------------------------------- #


def memory_bandwidth(probes: SystemProbes, cancel: CancelToken) -> dict[str, Any]:
    """反复复制一帧大小的数组 ``probes.memory_seconds`` 秒：``frame_mb``、``copies``、``gb_per_s``、``copy_ms``。"""
    source = np.random.default_rng(0).integers(0, 256, probes.frame_shape, dtype=np.uint8)
    times: list[float] = []
    started = time.perf_counter()
    while time.perf_counter() - started < probes.memory_seconds:
        if len(times) % 16 == 0:
            cancel.raise_if_cancelled()
        begin = time.perf_counter()
        copy = source.copy()          # 与处理循环一样每次新分配：可用内存紧张时缺页、换页的代价也算在里面
        times.append((time.perf_counter() - begin) * 1000.0)
        del copy
    spent_s = sum(times) / 1000.0
    return {
        "frame_mb": round(source.nbytes / _MB, 2),
        "copies": len(times),
        "gb_per_s": round(source.nbytes * len(times) / spent_s / 1e9, 2) if spent_s > 0 else None,
        "copy_ms": ms_summary(times),
    }


class MemoryWatch:
    """一段的内存监测：``start`` 记段前快照、开始换页计数并在 ``tasks`` 里定时采样；``stop`` 返回
    ``before``、``after``、``min_available_gb``（段内最低可用）、``pages_per_s``（段内平均换页率）与 ``paging_error``。"""

    def __init__(self, probes: SystemProbes, tasks: TaskGroup) -> None:
        self._probes = probes
        self._tasks = tasks
        self._samples: list[float] = []
        self._before: Mapping[str, Any] | None = None
        self._paging: Any = None

    def start(self) -> "MemoryWatch":
        self._before = self._snapshot()
        self._paging = self._probes.paging()
        self._tasks.spawn("memory", self._sample, owner="benchmark")
        return self

    def _snapshot(self) -> Mapping[str, Any] | None:
        snapshot = self._probes.memory()
        if snapshot is not None and snapshot.get("available_gb") is not None:
            self._samples.append(float(snapshot["available_gb"]))
        return snapshot

    def _sample(self, token: CancelToken) -> None:
        while not token.wait(self._probes.sample_s):
            self._snapshot()

    def stop(self) -> dict[str, Any]:
        after = self._snapshot()
        paging = {"pages_per_s": None, "error": "换页计数器没有打开"}
        if self._paging is not None:
            try:
                paging = dict(self._paging.read())
            finally:
                self._paging.close()
        return {
            "before": self._before,
            "after": after,
            "min_available_gb": min(self._samples) if self._samples else None,
            "pages_per_s": paging.get("pages_per_s"),
            "paging_error": paging.get("error"),
        }


# --------------------------------------------------------------------------- #
# 录像盘
# --------------------------------------------------------------------------- #


def disk_test(directory: Path, probes: SystemProbes, cancel: CancelToken,
              progress: Callable[[str], None]) -> dict[str, Any]:
    """在 ``directory``（录制目录下）测顺序写与读：``volume``、``write``、``read``，跳过时 ``skipped`` 写明原因，
    读写出错时 ``error`` 写明原因（不抛出；取消照常抛出）。测试文件在结束时删除。"""
    directory.mkdir(parents=True, exist_ok=True)
    volume = dict(probes.volume(directory))
    result: dict[str, Any] = {"directory": str(directory), "volume": volume, "write": None, "read": None,
                              "skipped": None, "error": None}
    need = probes.disk_total_bytes + probes.disk_spare_bytes
    free_gb = volume.get("free_gb")
    if free_gb is not None and free_gb * _GB < need:
        result["skipped"] = f"录像盘只剩 {free_gb} GB，读写测试需要 {need / _GB:.1f} GB 空闲，没有测"
        return result
    path = directory / DISK_FILE
    try:
        result["write"] = _write_test(path, probes, cancel, progress)
        result["read"] = _read_test(path, probes, cancel, progress)
    except OSError as exc:
        result["error"] = f"录像盘读写失败：{exc}"
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            result["error"] = result["error"] or f"删除测试文件失败：{exc}"
    return result


def _write_test(path: Path, probes: SystemProbes, cancel: CancelToken,
                progress: Callable[[str], None]) -> dict[str, Any]:
    chunk = probes.disk_chunk_bytes
    bank = [os.urandom(chunk) for _ in range(_DATA_BANK)]
    write_ms: list[float] = []
    flush_ms: list[float] = []
    written = 0
    total_mb = probes.disk_total_bytes // _MB
    started = time.perf_counter()
    with open(path, "wb", buffering=0) as stream:
        while written < probes.disk_total_bytes and time.perf_counter() - started < probes.disk_max_write_s:
            begin = time.perf_counter()
            stream.write(bank[len(write_ms) % _DATA_BANK])
            write_ms.append((time.perf_counter() - begin) * 1000.0)
            written += chunk
            if len(write_ms) % DISK_FLUSH_EVERY == 0:
                flush_ms.append(_flush(stream))
                cancel.raise_if_cancelled()
                progress(f"正在测录像盘写入：已写 {written // _MB} / {total_mb} MB"
                         f"（最多约 {probes.disk_max_write_s:g} 秒）")
        flush_ms.append(_flush(stream))
    seconds = time.perf_counter() - started
    return {
        "bytes": written,
        "seconds": round(seconds, 2),
        "mb_per_s": round(written / _MB / seconds, 1) if seconds > 0 else None,
        "chunk_kb": chunk // 2**10,
        "chunks": len(write_ms),
        "flush_every": DISK_FLUSH_EVERY,
        "complete": written >= probes.disk_total_bytes,
        "write_ms": ms_summary(write_ms),
        "flush_ms": ms_summary(flush_ms),
        "stall_ms": round(max(write_ms + flush_ms), 1) if write_ms else None,
    }


def _flush(stream: Any) -> float:
    begin = time.perf_counter()
    os.fsync(stream.fileno())
    return (time.perf_counter() - begin) * 1000.0


def _read_test(path: Path, probes: SystemProbes, cancel: CancelToken,
               progress: Callable[[str], None]) -> dict[str, Any]:
    progress(f"正在测录像盘读取（最多约 {probes.disk_max_read_s:g} 秒）")
    read = 0
    started = time.perf_counter()
    with probes.reader(path, probes.disk_read_chunk_bytes) as reader:
        while time.perf_counter() - started < probes.disk_max_read_s:
            got = reader.read()
            if not got:
                break
            read += got
            cancel.raise_if_cancelled()
        uncached = bool(reader.uncached)
    seconds = time.perf_counter() - started
    return {
        "bytes": read,
        "seconds": round(seconds, 2),
        "mb_per_s": round(read / _MB / seconds, 1) if seconds > 0 else None,
        "uncached": uncached,
    }


__all__ = ["DISK_FILE", "MemoryWatch", "SystemProbes", "disk_test", "memory_bandwidth"]
