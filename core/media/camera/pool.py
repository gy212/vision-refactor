# -*- coding: utf-8 -*-
"""``CameraPool``：Tk 与 CLI 唯一的物理摄像头所有者（与 Tk 无关、线程安全），即 PR #75 的 ``CameraPool``。

每个角色（primary/secondary）至多一个后台 reader，按 :class:`CaptureSpec`
（编号 + 请求尺寸）打开设备，并持续覆盖单槽 latest 帧。会话用 ``claim_*``
取得 :class:`CameraLease` 后独占读帧；lease 成功释放前该编号仍记为占用，
新的 reader 只在后台等待，绝不重叠打开同一设备。会话正常结束时 lease 可以
归还池（``CameraLease.restore``），已打开的设备转回预热 reader（V-1）。

reader 线程、设备释放与隔离重试见 :mod:`slot_lifecycle`，参数校验见 :mod:`pool_checks`。

``cancel()`` 只请求停止；“可以重新打开”以设备真实释放为准。释放失败的设备
保持隔离，下一次打开同编号时有间隔地重试释放，而不是遗弃句柄后重开。
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Final, Mapping

from core.media.camera.capture_process import ProcessCamera
from core.media.camera.pool_checks import CaptureFactory, PoolArgumentChecks
from core.media.camera.slot import WarmupSlot
from core.media.camera.slot_lifecycle import SlotLifecycle
from core.media.camera.spec import (
    PRIMARY,
    ROLES,
    SECONDARY,
    CameraWarmupError,
    CameraWarmupStopped,
    CameraWarmupTimeout,
    CaptureSpec,
)

_RELEASE_RETRY_INTERVAL_S: Final = 1.0

CameraLog = Callable[[dict], None]


class CameraLease:
    """会话独占的已就绪采集；release() 成功前设备保持占用，失败可再次调用重试。"""

    def __init__(self, pool: "CameraPool", slot: WarmupSlot, capture: Any) -> None:
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

    def restore(self) -> bool:
        """会话正常结束：把仍打开的设备交回池继续预热（V-1）；池不接收时返回 False，调用方改为 release()。"""
        return self._pool._restore_lease(self._slot, self._capture)


class CameraPool(SlotLifecycle, PoolArgumentChecks):
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
        self._capture_factory = capture_factory or ProcessCamera
        self._factory_keywords = self._supported_keywords(self._capture_factory)
        self._clock = clock
        self._stop_poll_interval = max(0.001, float(stop_poll_interval))
        self._join_timeout = max(0.0, float(join_timeout))
        self._release_retry_interval = max(0.0, float(release_retry_interval))
        self._log = log
        self._condition = threading.Condition(threading.RLock())
        self._slots: dict[str, WarmupSlot | None] = {
            PRIMARY: None,
            SECONDARY: None,
        }
        self._generations = {PRIMARY: 0, SECONDARY: 0}
        # 可能仍占有设备的 slot：退役 reader、会话 lease、释放失败的隔离设备。
        self._retired: dict[int, WarmupSlot] = {}
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
    ) -> tuple[CapturedFrame, CapturedFrame]:
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
    ) -> CapturedFrame:
        """单摄：primary 按 spec 就绪，secondary 不保留任何设备。"""

        desired = {PRIMARY: self._validate_spec(spec), SECONDARY: None}
        return self._wait_roles(desired, timeout, stop_event)[0]

    def snapshot_pair(
        self,
        primary: CaptureSpec | int,
        secondary: CaptureSpec | int,
        *,
        after: tuple[int, int] | None = None,
    ) -> tuple[CapturedFrame, CapturedFrame] | None:
        """Return the current pair if ready and newer than ``after``, a
        ``(primary_sequence, secondary_sequence)`` cursor; either role advancing counts."""
        wants = self._validate_pair(primary, secondary)
        return self._snapshot(dict(zip(ROLES, wants)), self._validate_after(after))

    def snapshot_one(self, spec: CaptureSpec | int, *, after: int | None = None) -> CapturedFrame | None:
        """单摄：primary 的最新预热帧，比 ``after`` 新时才返回（模型加载期的裸帧预览，D-8）。"""
        result = self._snapshot({PRIMARY: self._validate_spec(spec)}, None if after is None else (int(after),))
        return None if result is None else result[0]

    def _snapshot(self, wants: Mapping[str, CaptureSpec], cursor: tuple[int, ...] | None):
        with self._condition:
            slots = self._matching(wants)
            if slots is None or not all(
                not slot.claimed and slot.error is None and slot.latest is not None for slot in slots
            ):
                return None
            result = tuple(slot.latest for slot in slots)
            if cursor is not None and all(frame.seq <= seen for frame, seen in zip(result, cursor)):
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

    def __enter__(self) -> "CameraPool":
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
        previous: list[WarmupSlot] = []
        created: list[WarmupSlot] = []
        rollback: list[WarmupSlot] = []
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
    ) -> tuple[CapturedFrame, ...]:
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
        expected: tuple[WarmupSlot, ...],
        timeout: float,
        stop_event: threading.Event,
    ) -> tuple[CapturedFrame, ...]:
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

    def _return_lease(self, slot: WarmupSlot, cap: Any) -> None:
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

    def _restore_lease(self, old: WarmupSlot, cap: Any) -> bool:
        """租约转回预热 reader：只在池未关闭、该角色没有新的预热声明、设备无其他占用时接收。"""
        with self._condition:
            if (
                self._closed
                or not old.leased
                or old.released
                or self._slots[old.role] is not None
                or self._busy_slot_locked(old.index, exclude=old) is not None
            ):
                return False
            slot = self._new_slot_locked(old.role, old.spec)
            slot.cap = cap
            # 句柄移交新 slot；旧 slot 视为已释放，之后的 lease.release() 不再碰设备。
            old.leased, old.released, old.cap = False, True, None
            self._retired.pop(id(old), None)
            self._slots[old.role] = slot
            try:
                slot.thread.start()
            except BaseException:
                self._slots[old.role] = None
                self._generations[old.role] += 1
                old.leased, old.released, old.cap = True, False, cap
                self._retired[id(old)] = old
                slot.cap = None
                return False
            finally:
                self._condition.notify_all()
        self._emit("restore", slot)
        return True

    def _cancel_expected(self, expected: tuple[WarmupSlot, ...]) -> None:
        retired: list[WarmupSlot] = []
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
        expected: tuple[WarmupSlot, ...],
        claim_token: object,
    ) -> None:
        retired: list[WarmupSlot] = []
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

    def _new_slot_locked(self, role: str, spec: CaptureSpec) -> WarmupSlot:
        self._generations[role] += 1
        slot = WarmupSlot(role=role, spec=spec, generation=self._generations[role])
        slot.thread = threading.Thread(
            target=self._reader_loop,
            args=(slot,),
            name=f"camera-warmup-{role}-{slot.generation}",
            daemon=True,
        )
        return slot

    def _matching(
        self, wants: Mapping[str, CaptureSpec]
    ) -> tuple[WarmupSlot, ...] | None:
        slots = []
        for role, spec in wants.items():
            slot = self._slots[role]
            if slot is None or slot.spec != spec:
                return None
            slots.append(slot)
        return tuple(slots)

    @staticmethod
    def _reusable(slot: WarmupSlot, spec: CaptureSpec) -> bool:
        return (
            slot.spec == spec
            and slot.error is None
            and not slot.stop_event.is_set()
            and not slot.claimed
        )

    def _slot_is_current(self, slot: WarmupSlot) -> bool:
        return not self._closed and self._slots[slot.role] is slot

    @staticmethod
    def _ready_for_claim(slot: WarmupSlot) -> bool:
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
    def _completed_claim(slot: WarmupSlot, claim_token: object) -> bool:
        return (
            slot.error is None
            and slot.cap is not None
            and slot.latest is not None
            and slot.thread is not None
            and not slot.thread.is_alive()
            and slot.claim_token is claim_token
        )

    def _emit(self, event: str, slot: WarmupSlot, **fields: Any) -> None:
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

__all__ = [
    "CameraLease",
    "CameraWarmupError",
    "CameraPool",
    "CameraWarmupStopped",
    "CameraWarmupTimeout",
    "CaptureSpec",
    "PRIMARY",
    "SECONDARY",
]
