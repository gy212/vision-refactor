# -*- coding: utf-8 -*-
"""``SlotLifecycle``：摄像头池槽位的 reader 线程、设备释放与隔离重试，由 ``pool.py`` 按职责拆出（D-15）。

作为 :class:`~core.media.camera.pool.CameraPool` 的基类使用，方法依赖池在
``__init__`` 里建立的 ``_condition``、``_slots``、``_retired``、``_clock`` 等状态，
以及池上的 ``_emit``、``_slot_is_current`` 等方法；代码从 ``pool.py`` 原样搬来，逻辑不变。
"""
from __future__ import annotations

import threading
from typing import Any

from core.media.camera.slot import WarmupSlot
from core.media.camera.spec import CameraWarmupError, CameraWarmupStopped, WarmFrame
from core.media.frames import is_valid_bgr


class SlotLifecycle:
    """池槽位的 reader 循环、等待旧 owner 释放、打开设备，以及停止、中断、回收与释放重试。"""

    def _reader_loop(self, slot: WarmupSlot) -> None:
        cap: Any | None = slot.cap  # 归还的租约带着已打开的设备（V-1），不再重新打开
        preserve_for_claim = False
        phase = "wait_release"
        try:
            self._await_device(slot)
            phase = "open"
            with self._condition:
                slot.open_started_at = self._clock()
            if cap is None:
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
                if not ok or not is_valid_bgr(frame):
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

    def _await_device(self, slot: WarmupSlot) -> None:
        """在 reader 线程等待同编号旧 owner 真正释放；到期时重试释放隔离设备。"""

        while True:
            retry: WarmupSlot | None = None
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

    def _open(self, slot: WarmupSlot) -> Any:
        kwargs: dict[str, Any] = {}
        if "stop_event" in self._factory_keywords:
            kwargs["stop_event"] = slot.stop_event
        if "width" in self._factory_keywords:
            kwargs["width"] = slot.spec.width
        if "height" in self._factory_keywords:
            kwargs["height"] = slot.spec.height
        if "expected_name" in self._factory_keywords and slot.spec.expected_name:
            kwargs["expected_name"] = slot.spec.expected_name
        if "policy" in self._factory_keywords and slot.spec.policy != "prefer":
            kwargs["policy"] = slot.spec.policy
        return self._capture_factory(slot.index, **kwargs)

    def _retire_locked(self, slot: WarmupSlot) -> None:
        slot.stop_event.set()
        self._retired[id(slot)] = slot

    def _release_once(self, slot: WarmupSlot, cap: Any) -> bool:
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

    def _request_capture_interrupt(self, slot: WarmupSlot) -> None:
        with self._condition:
            cap = slot.cap
        interrupt = getattr(cap, "interrupt", None) if cap is not None else None
        if callable(interrupt):
            try:
                interrupt()
            except Exception:
                pass

    def _detach_slot(self, slot: WarmupSlot) -> None:
        """只发停止与中断信号，释放交给 reader/reaper，调用线程（可能是 Tk）不碰驱动。"""

        slot.stop_event.set()
        thread = slot.thread
        if thread is not None and thread.is_alive():
            self._request_capture_interrupt(slot)
        self._reap_slot_async(slot)

    def _interrupt_slot(self, slot: WarmupSlot, *, join: bool) -> None:
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

    def _reap_slot_async(self, slot: WarmupSlot) -> None:
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

    def _interrupt_slots(self, slots: list[WarmupSlot], *, timeout: float) -> None:
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
    def _holds_device(slot: WarmupSlot) -> bool:
        thread = slot.thread
        return bool(
            slot.leased
            or (thread is not None and thread.is_alive())
            or (slot.cap is not None and not slot.released)
        )

    def _busy_slot_locked(
        self, index: int, *, exclude: WarmupSlot | None = None
    ) -> WarmupSlot | None:
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

    def _release_retry_due_locked(self, slot: WarmupSlot) -> bool:
        thread = slot.thread
        if slot.leased or slot.cap is None or slot.released:
            return False
        if thread is not None and thread.is_alive():
            return False
        attempted = slot.release_attempted_at
        return attempted is None or (
            self._clock() - attempted >= self._release_retry_interval
        )


__all__ = ["SlotLifecycle"]
