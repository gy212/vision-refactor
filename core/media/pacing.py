# -*- coding: utf-8 -*-
"""``Timeline``：把一路带采集时刻的源帧换算成恒定帧率的输出帧序列，是录像时间轴的唯一实现（D-18，设计文档 §8.8）。

规则：

- 输出第 k 帧的时刻是 ``k / fps``（输出时间，扣除暂停区间），它显示该时刻**之前**（含）最近的一帧源帧。
- 一个输出槽要等到确认再也不会有更早的源帧时才写出：收到采集时刻晚于它的源帧，或到达暂停、结束边界。
  因此新到的画面只写进它自己及之后的时间槽，不会复制到过去（PR #34 的缺陷）。
- 开录、继续时的第一个槽用边界之前最近的源帧（``start`` 的 ``seed``）；没有这一帧就不能开录。
- 各路共用同一组开录、暂停、继续、结束边界，因此正常结束时各路的输出帧数相同，
  时长与有效录制时长相差不超过一个输出帧周期。同帧数不等于硬件同步：各路仍按自己的采集时刻取帧。

调用方的约定（写出方按此排序命令与帧）：源帧按采集顺序推入；``pause``/``stop`` 之前推完边界之前采到的帧，
且不推采集于边界之后的帧（推了就会写出边界之后的输出槽）。暂停期间推入的帧只作为继续时的首帧候选，
采集于继续边界之后的会在 ``resume`` 时补推。
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from core.media.frames import CapturedFrame

NS_PER_S = 1_000_000_000
PAUSED_BUFFER = 64  # 暂停中保留的源帧数：继续边界之后才处理到的帧在继续时补推


@dataclass
class TimelineStats:
    """一路录像的采集统计（写进 ``capture.json``）：录制区间内的源帧、输出槽、重复与跳过、最大源帧间隔。"""

    source_frames: int = 0       # 录制区间内收到的源帧，加上被用作区间首帧的边界前源帧
    used_frames: int = 0         # 至少写进一个输出槽的源帧
    slots: int = 0               # 输出帧数
    repeated_slots: int = 0      # 与上一槽是同一源帧的输出槽（补帧）
    skipped_frames: int = 0      # 收到但没写进任何槽的源帧（快于输出帧率时）
    max_source_gap_ns: int = 0   # 同一录制区间内相邻源帧的最大间隔

    def as_dict(self) -> dict[str, int]:
        return {
            "source_frames": self.source_frames,
            "used_frames": self.used_frames,
            "slots": self.slots,
            "repeated_slots": self.repeated_slots,
            "skipped_frames": self.skipped_frames,
            "max_source_gap_ns": self.max_source_gap_ns,
        }


@dataclass(frozen=True)
class Run:
    """一个录制区间（开录或继续到暂停或结束），采集时钟上的纳秒。"""

    start_ns: int
    end_ns: int | None = None


@dataclass
class _State:
    runs: list[Run] = field(default_factory=list)
    output_offset_ns: float = 0.0     # 当前区间开始时已经过的输出时间
    held: CapturedFrame | None = None  # 已收到、可能还要填后面槽的最新源帧
    held_used: bool = False
    held_active: bool = False          # held 是否在录制区间内收到（已计入 source_frames）
    last_active_ns: int | None = None  # 当前区间内上一帧源帧的采集时刻
    paused: deque = field(default_factory=lambda: deque(maxlen=PAUSED_BUFFER))  # 暂停中推入的帧


class Timeline:
    """一路录像的时间轴。状态：未开始 → ``start`` → 录制中 ⇄（``pause``/``resume``）→ ``stop`` → 结束。

    ``push`` 返回此刻确认的输出帧（按输出顺序，同一源帧可能连续出现多次）；``pause``/``stop`` 返回边界之前
    还没写出的输出帧。``slot_sources`` 是每个输出槽对应的源帧 ``(seq, captured_at_ns)``。
    """

    def __init__(self, fps: float = 30.0) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"帧率必须为正数：{fps}")
        self.fps = float(fps)
        self._period_ns = NS_PER_S / self.fps
        self._state = _State()
        self._recording = False
        self._stopped = False
        self.stats = TimelineStats()
        self.slot_sources: list[tuple[int, int]] = []

    # ---- 状态 ----
    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def runs(self) -> tuple[Run, ...]:
        return tuple(self._state.runs)

    # ---- 边界 ----
    def start(self, at_ns: int, seed: CapturedFrame | None) -> None:
        """在 ``at_ns`` 开始第一个录制区间；``seed`` 是边界之前（含）最近的源帧。"""
        if self._state.runs or self._stopped:
            raise RuntimeError("时间轴已经开始过")
        self._begin_run(at_ns, seed)

    def resume(self, at_ns: int, seed: CapturedFrame | None = None) -> list[CapturedFrame]:
        """暂停后在 ``at_ns`` 继续。第一个槽用边界之前最近的源帧：暂停期间推入的帧与 ``seed`` 中较新的那个；
        暂停期间推入、但采集于边界之后的帧在继续后补推，返回因此确认的输出帧。"""
        state = self._state
        if self._recording or self._stopped or not state.runs:
            raise RuntimeError("只有暂停中的时间轴可以继续")
        candidates = [frame for frame in (*state.paused, seed) if frame is not None]
        before = [frame for frame in candidates if frame.captured_at_ns <= at_ns]
        after = sorted((frame for frame in state.paused if frame.captured_at_ns > at_ns),
                       key=lambda frame: frame.captured_at_ns)
        state.paused.clear()
        best = max(before, key=lambda frame: frame.captured_at_ns, default=None)
        if best is not None and (state.held is None or best.captured_at_ns > state.held.captured_at_ns):
            self._replace_held(best, active=False)
        self._begin_run(at_ns, None)
        out: list[CapturedFrame] = []
        for frame in after:
            out.extend(self.push(frame))
        return out

    def pause(self, at_ns: int) -> list[CapturedFrame]:
        """在 ``at_ns`` 暂停：写出边界之前的输出槽；之后推入的帧只作为继续时的候选首帧。"""
        if not self._recording:
            raise RuntimeError("只有录制中的时间轴可以暂停")
        out = self._close_run(at_ns)
        self._recording = False
        return out

    def stop(self, at_ns: int) -> list[CapturedFrame]:
        """在 ``at_ns`` 结束（暂停中结束时以暂停边界为准）：写出剩余的输出槽。"""
        if self._stopped:
            return []
        out = self._close_run(at_ns) if self._recording else []
        self._recording = False
        self._stopped = True
        self._finish_held()
        return out

    # ---- 源帧 ----
    def push(self, frame: CapturedFrame) -> list[CapturedFrame]:
        """推入一帧源帧，返回因它而确认的输出帧。早于已收到的帧、或结束后的帧被忽略。"""
        state = self._state
        if self._stopped or not state.runs:
            return []
        if not self._recording:  # 暂停中：留作继续时的候选首帧
            if not state.paused or frame.captured_at_ns > state.paused[-1].captured_at_ns:
                state.paused.append(frame)
            return []
        held = state.held
        if held is not None and frame.captured_at_ns <= held.captured_at_ns:
            return []
        run_start = state.runs[-1].start_ns
        in_run = frame.captured_at_ns >= run_start
        if in_run:
            self.stats.source_frames += 1
            if state.last_active_ns is not None:
                gap = frame.captured_at_ns - state.last_active_ns
                self.stats.max_source_gap_ns = max(self.stats.max_source_gap_ns, gap)
            state.last_active_ns = frame.captured_at_ns
        out = self._emit_before(frame.captured_at_ns)
        self._replace_held(frame, active=in_run)
        return out

    # ---- 内部 ----
    def _begin_run(self, at_ns: int, seed: CapturedFrame | None) -> None:
        state = self._state
        if seed is not None:
            if seed.captured_at_ns > at_ns:
                raise ValueError("首帧必须采集于开录边界之前")
            self._replace_held(seed, active=False)
        if state.held is None or state.held.captured_at_ns > at_ns:
            raise ValueError("没有开录边界之前的源帧，不能开录")
        state.runs.append(Run(int(at_ns)))
        state.last_active_ns = None
        self._recording = True

    def _close_run(self, at_ns: int) -> list[CapturedFrame]:
        state = self._state
        run = state.runs[-1]
        end = max(int(at_ns), run.start_ns)
        out = self._emit_before(end)
        state.output_offset_ns += end - run.start_ns
        state.runs[-1] = Run(run.start_ns, end)
        return out

    def _slot_wall_ns(self, slot: int) -> float:
        """输出槽 ``slot`` 在当前录制区间里对应的采集时刻。"""
        state = self._state
        return state.runs[-1].start_ns + (slot * self._period_ns - state.output_offset_ns)

    def _emit_before(self, limit_ns: int) -> list[CapturedFrame]:
        """用 ``held`` 填满采集时刻早于 ``limit_ns`` 的输出槽（等于 ``limit_ns`` 的槽留给采集于该时刻的帧）。"""
        state = self._state
        out: list[CapturedFrame] = []
        held = state.held
        if held is None:
            return out
        while self._slot_wall_ns(self.stats.slots) < limit_ns:
            if state.held_used:
                self.stats.repeated_slots += 1
            else:
                state.held_used = True
                self.stats.used_frames += 1
                if not state.held_active:  # 边界前的首帧：用上了才计入源帧
                    self.stats.source_frames += 1
            self.slot_sources.append((held.seq, held.captured_at_ns))
            self.stats.slots += 1
            out.append(held)
        return out

    def _replace_held(self, frame: CapturedFrame, *, active: bool) -> None:
        self._finish_held()
        state = self._state
        state.held = frame
        state.held_used = False
        state.held_active = active

    def _finish_held(self) -> None:
        state = self._state
        if state.held is not None and state.held_active and not state.held_used:
            self.stats.skipped_frames += 1
        state.held_active = False

    def record(self) -> dict:
        """``capture.json`` 中这一路的内容：录制区间、统计、每个输出槽的源帧。"""
        return {
            "fps": self.fps,
            "runs": [[run.start_ns, run.end_ns] for run in self._state.runs],
            "stats": self.stats.as_dict(),
            "slot_sources": [list(item) for item in self.slot_sources],
        }


__all__ = ["NS_PER_S", "Run", "Timeline", "TimelineStats"]
