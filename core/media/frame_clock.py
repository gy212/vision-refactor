# -*- coding: utf-8 -*-
"""``FrameClock``：从一路源帧的到达时刻估计它们的采集时刻，去掉投递抖动，供录像时间轴使用（D-18，设计文档 §8.8）。

为什么需要：``CapturedFrame.captured_at_ns`` 是主机读到帧的时刻。摄像头按自己的节拍曝光，但 USB、驱动与解码让帧成批到达
（甲方机器实测：帧数完整，帧间隔中位 32 ms、p95 50～55 ms、最大 85 ms）。时间轴直接用到达时刻时，30 fps 网格与成批
到达错开，会出现"重复一帧、跳过下一帧"（仿真 5%～24% 的输出槽），帧都采到了，回放却一顿一顿。

模型：到达 = 曝光 + 延迟，延迟 ≥ 0；曝光落在周期 ``P`` 的网格上，第 k 帧的网格号 ``n_k`` 随帧递增，摄像头掉帧时多跳。
因为延迟非负，最近约 1 秒内 ``到达 - n·P`` 的最小值跟踪摄像头自己的时钟 ``b``，估计的采集时刻是 ``b + n_k·P``。
``b`` 遇到更小的值立即下调，上调则每帧不超过 ``CLOCK_RISE`` 个周期：窗口淘汰旧的最小值时 ``b`` 会抬高几毫秒，
直接跟上会让估计值跳变，帧恰好落在输出槽边界附近时又会重复一帧、跳过一帧。

- 网格号：相邻帧按 ``seq`` 之差递增（读帧队列满了丢帧时 ``seq`` 会跳）；摄像头自己掉的帧（``seq`` 连续但时间空出来）
  靠前瞻判定：等到其后 ``LOOKAHEAD`` 帧都到了，若这几帧相对当前时钟都晚了半个周期以上，就是掉了帧，按晚的周期数补上网格号；
  只是这一帧送晚了的话，后面总有帧按时到，最小残差接近 0。因此每帧要晚 ``LOOKAHEAD`` 帧（约 130 ms）才放出。
- 性质：估计值不晚于到达时刻，且严格递增。摄像头不掉帧时也不早于真实曝光时刻（``b`` 不低于真实时钟）；
  掉帧时，紧挨空档、或刚起算时的个别帧可能比真实曝光早，但不到一个周期。
  摄像头真实帧率低于标称（如弱光下 15 fps）表现为每帧都掉一帧，时间轴照实重复；快于标称时估计值贴着到达时刻，
  退化为不去抖，不会把帧排到到达之后。
- 输出相位：去抖后的帧落在固定相位上；若它恰在输出槽边界附近，亚毫秒的估计误差也会让它在相邻两槽间来回，
  同样重复一帧、跳过一帧。``segment_shift`` 在一段录制开始时为这一路算一次平移量（不到一个周期），
  这一段的开录、暂停、继续、结束边界都减去同一个量，使输出槽落在两帧之间的中点：仍只用边界之前的帧；
  每路各自的平移在段内不变，所以各路的录制区间长度相同、帧数相同（不能每个边界各算一次：
  各路相位不同时，起止平移不同会让区间长度相差近一个周期，帧数差一帧）。
- ``generation`` 变化（设备重开）时重新起算。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

from core.media.frames import CapturedFrame

NS_PER_S = 1_000_000_000
LOOKAHEAD = 4          # 放出一帧前等它之后的帧数，用来区分"掉帧"与"这一帧送晚了"
CLOCK_WINDOW_S = 1.0   # 跟踪摄像头时钟的回看时长
CLOCK_RISE = 0.005     # 时钟每帧最多上调的周期比例：跟得上 ±0.5% 的帧率偏差，又不让窗口淘汰最小值时估计值跳变


@dataclass
class FrameClockStats:
    """一路去抖的统计（写进 ``capture.json``）。"""

    frames: int = 0                # 放出的帧数
    inferred_drops: int = 0        # 判定摄像头掉的帧（seq 连续、时间空出来的周期数）
    seq_gaps: int = 0              # seq 跳过的帧（读帧队列满了丢的）
    max_correction_ns: int = 0     # 到达时刻减估计时刻的最大值
    resets: int = 0                # 因设备重开（generation 变化）重新起算的次数

    def as_dict(self) -> dict[str, int]:
        return {
            "frames": self.frames,
            "inferred_drops": self.inferred_drops,
            "seq_gaps": self.seq_gaps,
            "max_correction_ns": self.max_correction_ns,
            "resets": self.resets,
        }


@dataclass(frozen=True)
class BoundaryShift:
    """一路在一段录制内固定的边界平移量：本段开录、暂停、继续、结束都用 ``apply`` 换算，区间长度因此不变。"""

    ns: int

    def apply(self, at_ns: int) -> int:
        return at_ns - self.ns


class FrameClock:
    """一路源帧的去抖：``push`` 推入到达顺序的帧，返回已能确定采集时刻的帧（``captured_at_ns`` 换成估计值）；
    ``flush`` 放出剩下的帧（前瞻不足 ``LOOKAHEAD`` 帧时按已到的帧判定），之后可以继续推入同一路的帧。"""

    def __init__(self, fps: float = 30.0, *, lookahead: int = LOOKAHEAD, window_s: float = CLOCK_WINDOW_S) -> None:
        if fps <= 0:
            raise ValueError("fps 必须为正")
        self._period = NS_PER_S / float(fps)
        self._lookahead = max(0, int(lookahead))
        self._window = max(2, int(round(window_s * fps)))   # 回看窗口，按网格周期数计
        self.stats = FrameClockStats()
        self._pending: deque[CapturedFrame] = deque()
        self._last_arrival: int | None = None
        self._restart(None)

    def _restart(self, generation: int | None) -> None:
        self._generation = generation
        self._last_seq = 0
        self._last_n = -1
        self._last_est: int | None = None
        self._offsets: deque[tuple[int, float]] = deque()   # 窗口内 (n, 到达 - n·P)，偏移递增，队首是最小值
        self._clock: float | None = None                     # 摄像头时钟 b：网格号 0 的曝光时刻

    def push(self, frame: CapturedFrame) -> list[CapturedFrame]:
        out: list[CapturedFrame] = []
        current = self._pending[-1].generation if self._pending else self._generation
        if current is not None and frame.generation != current:   # 设备重开：放出旧代的帧，重新起算
            out += self.flush()
            self._restart(frame.generation)
            self._last_arrival = None
            self.stats.resets += 1
        if self._last_arrival is not None and frame.captured_at_ns <= self._last_arrival:
            return out                                       # 乱序或重复的帧：到达时刻不前进就丢弃
        self._last_arrival = frame.captured_at_ns
        self._pending.append(frame)
        while len(self._pending) > self._lookahead:
            out.append(self._release())
        return out

    def flush(self) -> list[CapturedFrame]:
        return [self._release() for _ in range(len(self._pending))]

    def segment_shift(self, start_ns: int) -> BoundaryShift:
        """在一段录制开始时调用一次：这一路本段所有边界要提前的量（``0 <= ns < P``），使 ``start_ns - ns`` 落在
        最近一帧网格曝光之后半个周期，输出槽离帧时刻最远。还没有时钟（尚未放出帧）时为 0。"""
        if self._clock is None:
            return BoundaryShift(0)
        age = (start_ns - self._clock) % self._period
        return BoundaryShift(int((age - self._period / 2) % self._period))

    def _release(self) -> CapturedFrame:
        frame = self._pending[0]
        drops = 0
        if self._last_n < 0:                                  # 本代第一帧：网格号 0，时钟从它起算
            self._generation, n = frame.generation, 0
        else:
            n = self._last_n + max(1, frame.seq - self._last_seq)
            self.stats.seq_gaps += max(0, frame.seq - self._last_seq - 1)
            drops = self._camera_drops(n)
            self.stats.inferred_drops += drops
            n += drops
        self._pending.popleft()
        self._track(n, frame.captured_at_ns, may_lower=drops == 0)
        est = int(min(frame.captured_at_ns, self._clock + n * self._period))
        if self._last_est is not None:
            est = max(est, self._last_est + 1)
        self._last_seq, self._last_n, self._last_est = frame.seq, n, est
        self.stats.frames += 1
        self.stats.max_correction_ns = max(self.stats.max_correction_ns, frame.captured_at_ns - est)
        return replace(frame, captured_at_ns=est)

    def _camera_drops(self, n: int) -> int:
        """当前帧（网格号暂定为 ``n``）之前摄像头掉了几个周期。

        取当前帧与其后已到的帧相对时钟的最小残差：真掉了帧时这几帧都晚一个周期以上；只是当前帧送晚了的话，
        后面总有按时到的帧，最小残差接近 0。另要求补上网格号后当前帧不落到时钟之前（容差 ``P/8``）：
        送晚了大半个周期的帧后面紧跟一次掉帧时，最小残差也会超过半个周期，这一条把掉帧留给真正跟在空档之后的帧。
        """
        clock = self._clock
        first = self._pending[0]
        residuals = [
            frame.captured_at_ns - (clock + (n + max(0, frame.seq - first.seq)) * self._period)
            for frame in self._pending
        ]
        lowest = min(residuals)
        drops = int(round(lowest / self._period)) if lowest > self._period / 2 else 0
        while drops > 0 and residuals[0] - drops * self._period < -self._period / 8:
            drops -= 1
        return drops

    def _track(self, n: int, arrived_ns: int, *, may_lower: bool) -> None:
        """把 ``(n, 到达 - n·P)`` 放进回看窗口，淘汰窗口外的项，再按窗口内的最小偏移更新时钟（下调立即、上调限速）。

        判定了掉帧的那一帧不能把时钟拉低：掉了几个周期是推断出来的，推多了会让之后约一秒的估计都偏早。
        """
        offset = arrived_ns - n * self._period
        if not may_lower and self._clock is not None:
            offset = max(offset, self._clock)
        while self._offsets and self._offsets[-1][1] >= offset:
            self._offsets.pop()
        self._offsets.append((n, offset))
        while self._offsets[0][0] <= n - self._window:
            self._offsets.popleft()
        lowest = self._offsets[0][1]
        if self._clock is None:
            self._clock = lowest
        else:
            step = max(1, n - self._last_n) * CLOCK_RISE * self._period
            self._clock = min(lowest, self._clock + step)


__all__ = ["BoundaryShift", "CLOCK_RISE", "CLOCK_WINDOW_S", "FrameClock", "FrameClockStats", "LOOKAHEAD"]
