# -*- coding: utf-8 -*-
"""Tk 桌面唯一的物理摄像头所有者（与 Tk 无关、线程安全）。

每个角色（primary/secondary）至多一个后台 reader，按 :class:`CaptureSpec`
（编号 + 请求尺寸）打开设备，并持续覆盖单槽 latest 帧。会话用 ``claim_*``
取得 :class:`CameraLease` 后独占读帧；lease 成功释放前该编号仍记为占用，
新的 reader 只在后台等待，绝不重叠打开同一设备。

``cancel()`` 只请求停止；“可以重新打开”以设备真实释放为准。释放失败的设备
保持隔离，下一次打开同编号时有间隔地重试释放，而不是遗弃句柄后重开。
"""
from __future__ import annotations

import inspect
import threading
import time
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Callable, Final, Mapping

import numpy as np

from apps.camera_enum import open_camera


PRIMARY: Final = "primary"
SECONDARY: Final = "secondary"
ROLES: Final = (PRIMARY, SECONDARY)
DEFAULT_CAPTURE_SIZE: Final = (1280, 720)
_RELEASE_RETRY_INTERVAL_S: Final = 1.0
_FACTORY_KEYWORDS: Final = ("stop_event", "width", "height")

CaptureFactory = Callable[..., Any]
CameraLog = Callable[[dict], None]


class CameraWarmupError(RuntimeError):
    """Base error raised by :class:`CameraWarmupPool`."""


class CameraWarmupTimeout(CameraWarmupError, TimeoutError):
    """The requested cameras did not become ready before their deadline."""


class CameraWarmupStopped(CameraWarmupError):
    """The caller cancelled a wait through its stop event."""


@dataclass(frozen=True)
class CaptureSpec:
    """一次打开请求的完整采集配置；请求时冻结，打开过程中不再读取模式标志。"""

    index: int
    width: int = DEFAULT_CAPTURE_SIZE[0]
    height: int = DEFAULT_CAPTURE_SIZE[1]

    def __post_init__(self) -> None:
        for name in ("index", "width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"camera {name} must be an integer")
        if self.index < 0:
            raise ValueError("camera index must be a non-negative integer")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera size must be positive")

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


@dataclass(frozen=True)
class WarmFrame:
    """Latest valid frame captured for one camera role."""

    frame: np.ndarray
    captured_at: float
    sequence: int
    generation: int


@dataclass(eq=False)
class _WarmupSlot:
    role: str
    spec: CaptureSpec
    generation: int
    stop_event: threading.Event = field(default_factory=threading.Event)
    cap: Any | None = None
    latest: WarmFrame | None = None
    error: BaseException | None = None
    thread: threading.Thread | None = None
    claim_token: object | None = None
    sequence: int = 0
    release_lock: threading.Lock = field(default_factory=threading.Lock)
    released: bool = False
    reaper_started: bool = False
    # 会话持有 lease 期间只能由 lease.release() 释放，池不得代为释放。
    leased: bool = False
    release_attempted_at: float | None = None
    open_started_at: float | None = None

    @property
    def index(self) -> int:
        return self.spec.index

    @property
    def claimed(self) -> bool:
        return self.claim_token is not None


class CameraLease:
    """会话独占的已就绪采集；release() 成功前设备保持占用，失败可再次调用重试。"""

    def __init__(self, pool: "CameraWarmupPool", slot: _WarmupSlot, capture: Any) -> None:
        self._pool = pool
        self._slot = slot
        self._capture = capture

    @property
    def spec(self) -> CaptureSpec:
        return self._slot.spec

    @property
    def capture(self) -> Any:
        return self._capture

    def read(self):
        return self._capture.read()

    def get(self, prop):
        return self._capture.get(prop)

    def isOpened(self) -> bool:
        is_opened = getattr(self._capture, "isOpened", None)
        return bool(is_opened()) if callable(is_opened) else True

    def __getattr__(self, name: str):
        return getattr(self._capture, name)

    def release(self) -> None:
        self._pool._return_lease(self._slot, self._capture)


class CameraWarmupPool:
    """Latest-only background readers plus lease ownership for every device."""

    def __init__(
        self,
        capture_factory: CaptureFactory | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        stop_poll_interval: float = 0.02,
        join_timeout: float = 1.0,
        release_retry_interval: float = _RELEASE_RETRY_INTERVAL_S,
        log: CameraLog | None = None,
    ) -> None:
        self._capture_factory = capture_factory or open_camera
        self._factory_keywords = self._supported_keywords(self._capture_factory)
        self._clock = clock
        self._stop_poll_interval = max(0.001, float(stop_poll_interval))
        self._join_timeout = max(0.0, float(join_timeout))
        self._release_retry_interval = max(0.0, float(release_retry_interval))
        self._log = log
        self._condition = threading.Condition(threading.RLock())
        self._slots: dict[str, _WarmupSlot | None] = {
            PRIMARY: None,
            SECONDARY: None,
        }
        self._generations = {PRIMARY: 0, SECONDARY: 0}
        # 可能仍占有设备的 slot：退役 reader、会话 lease、释放失败的隔离设备。
        self._retired: dict[int, _WarmupSlot] = {}
        self._closed = False

    # ---- 期望状态 ----

    def warm(self, role: str, spec: CaptureSpec | int) -> None:
        """Schedule one role; the reader waits for any old device owner to exit."""

        role = self._validate_role(role)
        self._apply({role: self._validate_spec(spec)}, reject_conflict=True)

    def sync(
        self,
        primary: CaptureSpec | int | None,
        secondary: CaptureSpec | int | None,
    ) -> None:
        """声明两路期望配置；None 表示该角色不预热。配置一致的 reader 直接复用。"""

        desired = {
            PRIMARY: None if primary is None else self._validate_spec(primary),
            SECONDARY: None if secondary is None else self._validate_spec(secondary),
        }
        if (
            desired[PRIMARY] is not None
            and desired[SECONDARY] is not None
            and desired[PRIMARY].index == desired[SECONDARY].index
        ):
            raise ValueError("primary and secondary cameras must be different")
        self._apply(desired)

    def cancel(self, role: str, *, generation: int | None = None) -> None:
        """Request one role to stop; release completes asynchronously."""

        role = self._validate_role(role)
        with self._condition:
            slot = self._slots[role]
            if slot is None or (
                generation is not None and slot.generation != generation
            ):
                return
            self._slots[role] = None
            self._generations[role] += 1
            slot.claim_token = None
            self._retire_locked(slot)
            self._condition.notify_all()
        self._emit("cancel", slot)
        self._detach_slot(slot)

    def release_all(self) -> None:
        for role in ROLES:
            self.cancel(role)

    def device_busy(self, index: int) -> bool:
        """同编号是否仍有 reader、lease 或隔离中的句柄未释放。"""

        index = self._validate_index(index)
        with self._condition:
            if any(
                slot is not None and slot.index == index
                for slot in self._slots.values()
            ):
                return True
            return self._busy_slot_locked(index) is not None

    def wait_released(self, timeout: float) -> bool:
        """等待所有退役/隔离设备释放完成；超时返回 False，不代替释放失败的诊断。"""

        deadline = self._clock() + self._validate_timeout(timeout)
        with self._condition:
            while any(self._holds_device(slot) for slot in self._retired.values()):
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return False
                self._condition.wait(min(remaining, self._stop_poll_interval))
            return True
    # ---- 等待与接管 ----

    def wait_pair(
        self,
        primary: CaptureSpec | int,
        secondary: CaptureSpec | int,
        *,
        timeout: float,
        stop_event: threading.Event,
    ) -> tuple[WarmFrame, WarmFrame]:
        """Wait until both roles have valid frames.

        Missing or differently configured roles are (re)started concurrently,
        after any old owner releases that device. The timeout includes this
        drain. Any error, timeout, stop, or replacement cancels only the pair
        observed by this call.
        """

        specs = self._validate_pair(primary, secondary)
        return self._wait_roles(dict(zip(ROLES, specs)), timeout, stop_event)

    def wait_one(
        self,
        spec: CaptureSpec | int,
        *,
        timeout: float,
        stop_event: threading.Event,
    ) -> WarmFrame:
        """单摄：primary 按 spec 就绪，secondary 不保留任何设备。"""

        desired = {PRIMARY: self._validate_spec(spec), SECONDARY: None}
        return self._wait_roles(desired, timeout, stop_event)[0]

    def snapshot_pair(
        self,
        primary: CaptureSpec | int,
        secondary: CaptureSpec | int,
        *,
        after: tuple[int, int] | None = None,
    ) -> tuple[WarmFrame, WarmFrame] | None:
        """Return the current pair if ready and newer than ``after``.

        ``after`` is a ``(primary_sequence, secondary_sequence)`` cursor. A new
        snapshot is returned when either role has advanced beyond its cursor.
        """

        wants = self._validate_pair(primary, secondary)
        cursor = self._validate_after(after)
        with self._condition:
            slots = self._matching(dict(zip(ROLES, wants)))
            if slots is None or not all(
                not slot.claimed and slot.error is None and slot.latest is not None
                for slot in slots
            ):
                return None
            result = (slots[0].latest, slots[1].latest)
            if cursor is not None and (
                result[0].sequence <= cursor[0]
                and result[1].sequence <= cursor[1]
            ):
                return None
            return result

    def claim_pair(
        self,
        primary: CaptureSpec | int,
        secondary: CaptureSpec | int,
        *,
        timeout: float,
        expected_generations: tuple[int, int] | None = None,
        stop_event: threading.Event | None = None,
    ) -> tuple[CameraLease, CameraLease]:
        """Stop ready readers and atomically turn both captures into leases.

        ``claim_pair`` never starts or replaces readers. The timeout covers
        reader shutdown only. If either capture cannot be transferred, both are
        released by the pool and no partial lease escapes.
        """

        wants = self._validate_pair(primary, secondary)
        generations = self._validate_generations(expected_generations, 2)
        return self._claim(dict(zip(ROLES, wants)), timeout, generations, stop_event)

    def claim_one(
        self,
        spec: CaptureSpec | int,
        *,
        timeout: float,
        expected_generation: int | None = None,
        stop_event: threading.Event | None = None,
    ) -> CameraLease:
        generations = self._validate_generations(
            None if expected_generation is None else (expected_generation,), 1
        )
        wants = {PRIMARY: self._validate_spec(spec)}
        return self._claim(wants, timeout, generations, stop_event)[0]

    def close(self) -> None:
        """Stop every reader and permanently release pool-owned captures.

        Leased captures stay owned by their session; they are only interrupted
        so a blocked read returns and the session can release them.
        """

        with self._condition:
            if not self._closed:
                self._closed = True
                for role in ROLES:
                    slot = self._slots[role]
                    self._slots[role] = None
                    self._generations[role] += 1
                    if slot is not None:
                        slot.claim_token = None
                        self._retire_locked(slot)
            slots = list(self._retired.values())
            self._condition.notify_all()
        for slot in slots:
            if slot.leased:
                self._request_capture_interrupt(slot)
        self._interrupt_slots(
            [slot for slot in slots if not slot.leased], timeout=self._join_timeout
        )

    def __enter__(self) -> "CameraWarmupPool":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    # ---- 内部实现 ----

    def _apply(
        self,
        desired: Mapping[str, CaptureSpec | None],
        *,
        reject_conflict: bool = False,
    ) -> None:
        previous: list[_WarmupSlot] = []
        created: list[_WarmupSlot] = []
        rollback: list[_WarmupSlot] = []
        start_error: BaseException | None = None
        with self._condition:
            self._ensure_open()
            for role, spec in desired.items():
                current = self._slots[role]
                if current is None or (spec is not None and self._reusable(current, spec)):
                    continue
                if current.claimed:
                    raise CameraWarmupError(
                        f"{role} camera is being claimed and cannot be replaced"
                    )
            if reject_conflict:
                for role, spec in desired.items():
                    other = self._slots[SECONDARY if role == PRIMARY else PRIMARY]
                    if spec is not None and other is not None and other.index == spec.index:
                        raise ValueError(
                            "the same camera index cannot be warmed for both roles"
                        )
            for role, spec in desired.items():
                current = self._slots[role]
                if current is not None and spec is not None and self._reusable(current, spec):
                    continue
                if current is not None:
                    self._slots[role] = None
                    if spec is None:
                        self._generations[role] += 1
                    self._retire_locked(current)
                    previous.append(current)
                if spec is not None:
                    slot = self._new_slot_locked(role, spec)
                    self._slots[role] = slot
                    created.append(slot)
            for slot in created:
                try:
                    slot.thread.start()
                except BaseException as exc:
                    start_error = exc
                    for role in desired:
                        current = self._slots[role]
                        if current is None:
                            continue
                        self._slots[role] = None
                        self._generations[role] += 1
                        self._retire_locked(current)
                        rollback.append(current)
                    break
            self._condition.notify_all()

        # 旧 generation 已脱离，不等待可能阻塞的旧 open；它返回后由 reader 自行释放。
        for slot in previous:
            self._emit("replace", slot)
        for slot in [*previous, *rollback]:
            self._detach_slot(slot)
        for slot in created:
            if slot not in rollback:
                self._emit("warm", slot)
        if start_error is not None:
            raise CameraWarmupError(
                "failed to start camera warmup reader"
            ) from start_error

    def _wait_roles(
        self,
        desired: Mapping[str, CaptureSpec | None],
        timeout: float,
        stop_event: threading.Event,
    ) -> tuple[WarmFrame, ...]:
        timeout = self._validate_timeout(timeout)
        if not hasattr(stop_event, "is_set"):
            raise TypeError("stop_event must provide is_set()")
        if stop_event.is_set():
            raise CameraWarmupStopped("camera warmup was stopped")
        self._apply(desired)
        with self._condition:
            expected = self._matching(
                {role: spec for role, spec in desired.items() if spec is not None}
            )
            if expected is None:
                raise CameraWarmupError("camera warmup was replaced before waiting")
        try:
            return self._wait_expected(expected, timeout, stop_event)
        except BaseException:
            self._cancel_expected(expected)
            raise

    def _wait_expected(
        self,
        expected: tuple[_WarmupSlot, ...],
        timeout: float,
        stop_event: threading.Event,
    ) -> tuple[WarmFrame, ...]:
        deadline = self._clock() + timeout
        with self._condition:
            while True:
                if stop_event.is_set():
                    raise CameraWarmupStopped("camera warmup was stopped")
                if self._closed:
                    raise CameraWarmupError("camera warmup pool is closed")
                if not all(self._slots[slot.role] is slot for slot in expected):
                    raise CameraWarmupError("camera warmup pair was replaced")
                for slot in expected:
                    if slot.claimed:
                        raise CameraWarmupError(
                            "camera warmup pair is already being claimed"
                        )
                    if slot.error is not None:
                        raise CameraWarmupError(str(slot.error)) from slot.error
                if all(slot.latest is not None for slot in expected):
                    return tuple(slot.latest for slot in expected)

                remaining = deadline - self._clock()
                if remaining <= 0:
                    retiring = [
                        str(slot.index)
                        for slot in expected
                        if self._busy_slot_locked(slot.index, exclude=slot) is not None
                    ]
                    detail = (
                        f"; 摄像头 {', '.join(retiring)} 的上次采集仍未释放"
                        if retiring else ""
                    )
                    raise CameraWarmupTimeout(
                        "timed out waiting for cameras to warm" + detail
                    )
                self._condition.wait(min(remaining, self._stop_poll_interval))
    def _claim(
        self,
        wants: Mapping[str, CaptureSpec | int],
        timeout: float,
        generations: tuple[int, ...] | None,
        stop_event: threading.Event | None,
    ) -> tuple[CameraLease, ...]:
        timeout = self._validate_timeout(timeout)
        if stop_event is not None and not hasattr(stop_event, "is_set"):
            raise TypeError("stop_event must provide is_set()")
        if stop_event is not None and stop_event.is_set():
            raise CameraWarmupStopped("camera warmup was stopped")
        deadline = self._clock() + timeout
        claim_token = object()

        with self._condition:
            expected = self._matching(wants)
            if expected is None:
                raise CameraWarmupError("camera warmup pair is not ready for claim")
            if generations is not None and tuple(
                slot.generation for slot in expected
            ) != generations:
                raise CameraWarmupError(
                    "camera warmup pair generation changed before claim"
                )
            if stop_event is not None and stop_event.is_set():
                raise CameraWarmupStopped("camera warmup was stopped")
            claim_error = None
            if not all(self._ready_for_claim(slot) for slot in expected):
                claim_error = CameraWarmupError(
                    "camera warmup pair is not transferable"
                )
            else:
                for slot in expected:
                    slot.claim_token = claim_token
                    slot.stop_event.set()
            self._condition.notify_all()

        if claim_error is not None:
            self._cancel_expected(expected)
            raise claim_error

        try:
            for slot in expected:
                thread = slot.thread
                while thread is not None and thread.is_alive():
                    if stop_event is not None and stop_event.is_set():
                        raise CameraWarmupStopped("camera warmup was stopped")
                    remaining = deadline - self._clock()
                    if remaining <= 0:
                        raise CameraWarmupTimeout(
                            "timed out stopping camera readers"
                        )
                    thread.join(min(remaining, self._stop_poll_interval))

            with self._condition:
                if stop_event is not None and stop_event.is_set():
                    raise CameraWarmupStopped("camera warmup was stopped")
                if (
                    self._closed
                    or not all(self._slots[slot.role] is slot for slot in expected)
                    or not all(
                        self._completed_claim(slot, claim_token) for slot in expected
                    )
                ):
                    raise CameraWarmupError(
                        "camera warmup pair changed while being claimed"
                    )
                leases = []
                for slot in expected:
                    self._slots[slot.role] = None
                    self._generations[slot.role] += 1
                    slot.claim_token = None
                    slot.leased = True
                    # lease 期间仍计为设备占用，直到会话释放成功。
                    self._retired[id(slot)] = slot
                    leases.append(CameraLease(self, slot, slot.cap))
                self._condition.notify_all()
        except BaseException:
            self._abort_claim(expected, claim_token)
            raise
        for slot in expected:
            self._emit("lease", slot)
        return tuple(leases)

    def _return_lease(self, slot: _WarmupSlot, cap: Any) -> None:
        with self._condition:
            if slot.released and not slot.leased:
                return
        released = self._release_once(slot, cap)
        with self._condition:
            slot.leased = False
            if released:
                slot.cap = None
                self._retired.pop(id(slot), None)
            else:
                # 释放失败：所有权回到池并隔离，同编号重开前按间隔重试释放。
                slot.cap = cap
                self._retired[id(slot)] = slot
            self._condition.notify_all()
        if not released:
            raise CameraWarmupError(
                f"摄像头 {slot.index} 释放失败，已隔离；重新开始时会再次尝试释放"
            ) from slot.error

    def _reader_loop(self, slot: _WarmupSlot) -> None:
        cap: Any | None = None
        preserve_for_claim = False
        phase = "wait_release"
        try:
            self._await_device(slot)
            phase = "open"
            with self._condition:
                slot.open_started_at = self._clock()
            cap = self._open(slot)
            if cap is None:
                raise CameraWarmupError(
                    f"{slot.role} camera {slot.index} returned no capture"
                )
            is_opened = getattr(cap, "isOpened", None)
            if callable(is_opened) and not bool(is_opened()):
                raise CameraWarmupError(
                    f"{slot.role} camera {slot.index} could not be opened"
                )

            with self._condition:
                if not self._slot_is_current(slot) or slot.stop_event.is_set():
                    return
                slot.cap = cap
                self._condition.notify_all()

            phase = "first_frame"
            while not slot.stop_event.is_set():
                ok, frame = cap.read()
                if slot.stop_event.is_set():
                    break
                if not ok or not self._valid_frame(frame):
                    raise CameraWarmupError(
                        f"{slot.role} camera {slot.index} failed to read a frame"
                    )
                captured_at = self._clock()
                with self._condition:
                    if not self._slot_is_current(slot) or slot.stop_event.is_set():
                        break
                    slot.sequence += 1
                    slot.latest = WarmFrame(
                        frame=frame,
                        captured_at=captured_at,
                        sequence=slot.sequence,
                        generation=slot.generation,
                    )
                    opened_at = slot.open_started_at
                    self._condition.notify_all()
                if phase == "first_frame":
                    phase = "read"
                    self._emit(
                        "ready",
                        slot,
                        actual=self._frame_size(frame),
                        open_ms=(
                            round((captured_at - opened_at) * 1000.0, 1)
                            if opened_at is not None
                            else None
                        ),
                    )
        except BaseException as exc:
            with self._condition:
                current = self._slot_is_current(slot)
                if current:
                    slot.error = self._normalize_error(slot, exc)
                    slot.stop_event.set()
                    self._condition.notify_all()
            if current:
                self._emit("error", slot, phase=phase, error=str(exc))
        finally:
            with self._condition:
                preserve_for_claim = (
                    cap is not None
                    and self._slot_is_current(slot)
                    and slot.claimed
                    and slot.cap is cap
                )
                if not preserve_for_claim and slot.cap is cap:
                    slot.cap = None
                self._condition.notify_all()
            released = True
            if cap is not None and not preserve_for_claim:
                released = self._release_once(slot, cap)
            if not preserve_for_claim:
                with self._condition:
                    if released:
                        self._retired.pop(id(slot), None)
                    else:
                        self._retired[id(slot)] = slot
                    self._condition.notify_all()

    def _await_device(self, slot: _WarmupSlot) -> None:
        """在 reader 线程等待同编号旧 owner 真正释放；到期时重试释放隔离设备。"""

        while True:
            retry: _WarmupSlot | None = None
            with self._condition:
                if slot.stop_event.is_set() or not self._slot_is_current(slot):
                    raise CameraWarmupStopped("camera warmup was stopped")
                busy = self._busy_slot_locked(slot.index, exclude=slot)
                if busy is None:
                    return
                if self._release_retry_due_locked(busy):
                    busy.release_attempted_at = self._clock()
                    retry = busy
                else:
                    self._condition.wait(self._stop_poll_interval)
            if retry is not None:
                cap = retry.cap
                if cap is not None and self._release_once(retry, cap):
                    with self._condition:
                        if retry.cap is cap:
                            retry.cap = None
                        self._retired.pop(id(retry), None)
                        self._condition.notify_all()

    def _open(self, slot: _WarmupSlot) -> Any:
        kwargs: dict[str, Any] = {}
        if "stop_event" in self._factory_keywords:
            kwargs["stop_event"] = slot.stop_event
        if "width" in self._factory_keywords:
            kwargs["width"] = slot.spec.width
        if "height" in self._factory_keywords:
            kwargs["height"] = slot.spec.height
        return self._capture_factory(slot.index, **kwargs)
    def _cancel_expected(self, expected: tuple[_WarmupSlot, ...]) -> None:
        retired: list[_WarmupSlot] = []
        with self._condition:
            for slot in expected:
                if self._slots[slot.role] is slot:
                    self._slots[slot.role] = None
                    self._generations[slot.role] += 1
                    slot.claim_token = None
                    self._retire_locked(slot)
                    retired.append(slot)
            self._condition.notify_all()
        for slot in retired:
            self._detach_slot(slot)

    def _abort_claim(
        self,
        expected: tuple[_WarmupSlot, ...],
        claim_token: object,
    ) -> None:
        retired: list[_WarmupSlot] = []
        with self._condition:
            for slot in expected:
                if slot.claim_token is claim_token:
                    slot.claim_token = None
                if self._slots[slot.role] is slot:
                    self._slots[slot.role] = None
                    self._generations[slot.role] += 1
                    self._retire_locked(slot)
                    retired.append(slot)
            self._condition.notify_all()
        # Completed readers are released by the reaper; blocked readers release
        # their own capture after open/read returns and observes the detached slot.
        for slot in retired:
            self._detach_slot(slot)

    def _retire_locked(self, slot: _WarmupSlot) -> None:
        slot.stop_event.set()
        self._retired[id(slot)] = slot

    def _release_once(self, slot: _WarmupSlot, cap: Any) -> bool:
        with slot.release_lock:
            if slot.released:
                return True
            try:
                cap.release()
            except Exception as exc:
                with self._condition:
                    slot.error = CameraWarmupError(
                        f"{slot.role} camera {slot.index} release failed: {exc}"
                    )
                    slot.release_attempted_at = self._clock()
                    if slot.cap is None:
                        slot.cap = cap
                    self._retired[id(slot)] = slot
                    self._condition.notify_all()
                self._emit("release_failed", slot, error=str(exc))
                return False
            slot.released = True
        with self._condition:
            if slot.cap is cap:
                slot.cap = None
            self._condition.notify_all()
        self._emit("released", slot)
        return True

    def _request_capture_interrupt(self, slot: _WarmupSlot) -> None:
        with self._condition:
            cap = slot.cap
        interrupt = getattr(cap, "interrupt", None) if cap is not None else None
        if callable(interrupt):
            try:
                interrupt()
            except Exception:
                pass

    def _detach_slot(self, slot: _WarmupSlot) -> None:
        """只发停止与中断信号，释放交给 reader/reaper，调用线程（可能是 Tk）不碰驱动。"""

        slot.stop_event.set()
        thread = slot.thread
        if thread is not None and thread.is_alive():
            self._request_capture_interrupt(slot)
        self._reap_slot_async(slot)

    def _interrupt_slot(self, slot: _WarmupSlot, *, join: bool) -> None:
        slot.stop_event.set()
        thread = slot.thread
        current = threading.current_thread()
        if join and thread is not None and thread is not current and thread.is_alive():
            thread.join(self._join_timeout)
        if thread is not None and thread.is_alive():
            self._request_capture_interrupt(slot)
            if join and thread is not current:
                thread.join(self._join_timeout)

        # Never release while the owner thread may still be inside open/read,
        # and never release a capture that a session currently leases.
        if (thread is None or not thread.is_alive()) and not slot.leased:
            with self._condition:
                cap = slot.cap
            released = True if cap is None else self._release_once(slot, cap)
            if released:
                with self._condition:
                    self._retired.pop(id(slot), None)
                    self._condition.notify_all()

    def _reap_slot_async(self, slot: _WarmupSlot) -> None:
        """Finish a cancelled slot without blocking a UI caller."""

        with self._condition:
            if slot.reaper_started:
                return
            slot.reaper_started = True

        def _reap() -> None:
            thread = slot.thread
            if (
                thread is not None
                and thread is not threading.current_thread()
                and thread.is_alive()
            ):
                thread.join(self._join_timeout)
            if thread is not None and thread.is_alive():
                self._request_capture_interrupt(slot)
                thread.join(self._join_timeout)
            self._interrupt_slot(slot, join=False)

        reaper = threading.Thread(
            target=_reap,
            name=f"camera-warmup-reaper-{slot.role}-{slot.generation}",
            daemon=True,
        )
        try:
            reaper.start()
        except BaseException:
            # Keep cancellation bounded. The reader remains the only owner
            # allowed to release a native capture after open/read returns.
            self._interrupt_slot(slot, join=True)

    def _interrupt_slots(self, slots: list[_WarmupSlot], *, timeout: float) -> None:
        deadline = self._clock() + max(0.0, timeout)
        for slot in slots:
            slot.stop_event.set()
            thread = slot.thread
            if (
                thread is not None
                and thread is not threading.current_thread()
                and thread.is_alive()
            ):
                thread.join(max(0.0, deadline - self._clock()))
        for slot in slots:
            thread = slot.thread
            if thread is not None and thread.is_alive():
                self._request_capture_interrupt(slot)
            self._interrupt_slot(slot, join=False)

    @staticmethod
    def _holds_device(slot: _WarmupSlot) -> bool:
        thread = slot.thread
        return bool(
            slot.leased
            or (thread is not None and thread.is_alive())
            or (slot.cap is not None and not slot.released)
        )

    def _busy_slot_locked(
        self, index: int, *, exclude: _WarmupSlot | None = None
    ) -> _WarmupSlot | None:
        for slot in self._retired.values():
            if slot is not exclude and slot.index == index and self._holds_device(slot):
                return slot
        for slot in self._slots.values():
            if (
                slot is not None
                and slot is not exclude
                and slot.index == index
                and self._holds_device(slot)
            ):
                return slot
        return None

    def _release_retry_due_locked(self, slot: _WarmupSlot) -> bool:
        thread = slot.thread
        if slot.leased or slot.cap is None or slot.released:
            return False
        if thread is not None and thread.is_alive():
            return False
        attempted = slot.release_attempted_at
        return attempted is None or (
            self._clock() - attempted >= self._release_retry_interval
        )

    def _new_slot_locked(self, role: str, spec: CaptureSpec) -> _WarmupSlot:
        self._generations[role] += 1
        slot = _WarmupSlot(role=role, spec=spec, generation=self._generations[role])
        slot.thread = threading.Thread(
            target=self._reader_loop,
            args=(slot,),
            name=f"camera-warmup-{role}-{slot.generation}",
            daemon=True,
        )
        return slot

    def _matching(
        self, wants: Mapping[str, CaptureSpec]
    ) -> tuple[_WarmupSlot, ...] | None:
        slots = []
        for role, spec in wants.items():
            slot = self._slots[role]
            if slot is None or slot.spec != spec:
                return None
            slots.append(slot)
        return tuple(slots)

    @staticmethod
    def _reusable(slot: _WarmupSlot, spec: CaptureSpec) -> bool:
        return (
            slot.spec == spec
            and slot.error is None
            and not slot.stop_event.is_set()
            and not slot.claimed
        )

    def _slot_is_current(self, slot: _WarmupSlot) -> bool:
        return not self._closed and self._slots[slot.role] is slot

    @staticmethod
    def _ready_for_claim(slot: _WarmupSlot) -> bool:
        return (
            slot.error is None
            and slot.cap is not None
            and slot.latest is not None
            and slot.thread is not None
            and slot.thread.is_alive()
            and not slot.stop_event.is_set()
            and not slot.claimed
        )

    @staticmethod
    def _completed_claim(slot: _WarmupSlot, claim_token: object) -> bool:
        return (
            slot.error is None
            and slot.cap is not None
            and slot.latest is not None
            and slot.thread is not None
            and not slot.thread.is_alive()
            and slot.claim_token is claim_token
        )

    def _emit(self, event: str, slot: _WarmupSlot, **fields: Any) -> None:
        if self._log is None:
            return
        payload = {
            "event": event,
            "role": slot.role,
            "index": slot.index,
            "requested": f"{slot.spec.width}x{slot.spec.height}",
            "generation": slot.generation,
            **fields,
        }
        try:
            self._log(payload)
        except Exception:
            pass

    @staticmethod
    def _frame_size(frame: Any) -> str | None:
        shape = getattr(frame, "shape", None)
        if not shape or len(shape) < 2:
            return None
        return f"{int(shape[1])}x{int(shape[0])}"

    def _ensure_open(self) -> None:
        if self._closed:
            raise CameraWarmupError("camera warmup pool is closed")

    @staticmethod
    def _validate_role(role: str) -> str:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES!r}")
        return role

    @staticmethod
    def _validate_index(index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise ValueError("camera index must be a non-negative integer")
        normalized = int(index)
        if normalized < 0:
            raise ValueError("camera index must be a non-negative integer")
        return normalized

    @classmethod
    def _validate_spec(cls, spec: CaptureSpec | int) -> CaptureSpec:
        if isinstance(spec, CaptureSpec):
            return spec
        return CaptureSpec(cls._validate_index(spec))


    def _validate_pair(
        self, primary: CaptureSpec | int, secondary: CaptureSpec | int
    ) -> tuple[CaptureSpec, CaptureSpec]:
        first = self._validate_spec(primary)
        second = self._validate_spec(secondary)
        if first.index == second.index:
            raise ValueError("primary and secondary cameras must be different")
        return first, second

    @staticmethod
    def _validate_timeout(timeout: float) -> float:
        try:
            normalized = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout must be a non-negative number") from exc
        if normalized < 0:
            raise ValueError("timeout must be a non-negative number")
        return normalized

    @staticmethod
    def _validate_after(after: tuple[int, int] | None) -> tuple[int, int] | None:
        if after is None:
            return None
        if not isinstance(after, tuple) or len(after) != 2:
            raise ValueError("after must be a two-item sequence cursor")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in after):
            raise ValueError("after must contain integer sequences")
        return after

    @staticmethod
    def _validate_generations(
        generations: tuple[int, ...] | None, size: int
    ) -> tuple[int, ...] | None:
        if generations is None:
            return None
        if not isinstance(generations, tuple) or len(generations) != size:
            raise ValueError(f"expected_generations must be a {size}-item tuple")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in generations
        ):
            raise ValueError("expected_generations must contain positive integers")
        return generations

    @staticmethod
    def _valid_frame(frame: Any) -> bool:
        if frame is None:
            return False
        try:
            return int(getattr(frame, "size", 0)) > 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _supported_keywords(factory: CaptureFactory) -> frozenset[str]:
        try:
            parameters = list(inspect.signature(factory).parameters.values())
        except (TypeError, ValueError):
            return frozenset()
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
            return frozenset(_FACTORY_KEYWORDS)
        return frozenset(
            p.name
            for p in parameters
            if p.name in _FACTORY_KEYWORDS
            and p.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        )

    @staticmethod
    def _normalize_error(slot: _WarmupSlot, exc: BaseException) -> BaseException:
        if isinstance(exc, CameraWarmupError):
            return exc
        return CameraWarmupError(f"{slot.role} camera {slot.index} failed: {exc}")


__all__ = [
    "CameraLease",
    "CameraWarmupError",
    "CameraWarmupPool",
    "CameraWarmupStopped",
    "CameraWarmupTimeout",
    "CaptureSpec",
    "PRIMARY",
    "SECONDARY",
    "WarmFrame",
]
