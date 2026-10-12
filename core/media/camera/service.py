# -*- coding: utf-8 -*-
"""``CameraService`` 是 Tk 与 CLI 获取摄像头帧的唯一入口（池之上的门面），``CameraStream`` 是单次会话的设备句柄。

- 两者都属 L1，不认识会话；``RoleSpecs`` 是"角色 → ``CaptureSpec``"的映射，单摄只有 primary。
- ``CameraStream`` 把 ``CancelToken`` 适配成池要的 ``stop_event``，把 ``CameraWarmupStopped`` 转成 ``Cancelled``；
  阻塞调用（``wait_ready``、``claim``、``read``）只在后台线程执行。
- 池的生命周期事件写成 ``diag`` 诊断行（``device.state``、``camera.ready``、``camera.error``）。
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

from core.foundation import diag
from core.foundation.tasks import Cancelled, CancelToken, TaskGroup, TaskHandle
from core.media.camera.devices import CatalogSnapshot, DeviceCatalog
from core.media.camera.pool import CameraLease, CameraPool
from core.media.camera.spec import (
    PRIMARY,
    ROLES,
    SECONDARY,
    CameraWarmupError,
    CameraWarmupStopped,
    CaptureSpec,
    WarmFrame,
)
from core.media.frames import FrameSet

RoleSpecs = Mapping[str, CaptureSpec]

DEFAULT_READY_TIMEOUT_S = 15.0  # 覆盖 ProcessCamera 12 秒首帧预算与旧设备释放等待
DEFAULT_CLAIM_TIMEOUT_S = 5.0  # 接管只需停掉预热 reader
DEFAULT_RELEASE_WAIT_S = 3.0

_STATE_OF_EVENT = {
    "warm": "opening",
    "replace": "releasing",
    "cancel": "releasing",
    "lease": "leased",
    "restore": "warm",
    "released": "free",
    "release_failed": "quarantined",
}
_STAGE_OF_PHASE = {"wait_release": "acquire_wait"}


def log_pool_event(payload: dict) -> None:
    """把池的事件（PR #75 的 ``[camera]`` JSON 行）写成 §4.5 的诊断行。"""
    event = payload.get("event")
    fields: dict[str, Any] = {
        "role": payload.get("role"),
        "device": payload.get("index"),
        "generation": payload.get("generation"),
        "requested": payload.get("requested"),
        "actual": payload.get("actual"),
        "elapsed_ms": payload.get("open_ms"),
        "stage": _STAGE_OF_PHASE.get(payload.get("phase"), payload.get("phase")),
        "error": payload.get("error"),
    }
    if event == "ready":
        diag.event("camera.ready", **fields)
    elif event == "error":
        diag.event("camera.error", **fields)
    else:
        diag.event("device.state", to=_STATE_OF_EVENT.get(event, event), cause=event, **fields)


class CameraReadError(RuntimeError):
    """会话读帧失败（设备断开或采集进程已关闭）。"""

    def __init__(self, role: str, *, numbered: bool) -> None:
        prefix = f"第{ROLES.index(role) + 1}路" if numbered else ""
        super().__init__(f"{prefix}摄像头读帧失败，请检查连接后重新开始")
        self.role = role


class _TokenEvent:
    """池的 ``stop_event`` 只需要 ``is_set()``；由 token 的取消状态提供。"""

    def __init__(self, token: CancelToken | None) -> None:
        self._token = token

    def is_set(self) -> bool:
        return self._token is not None and self._token.cancelled


def _cancelled(token: CancelToken | None) -> Cancelled:
    return Cancelled(token.reason if token is not None and token.reason else "user_stop")


class CameraStream:
    """一次会话的摄像头句柄：等每路有效首帧 → 接管租约 → 读帧 → 归还或释放。"""

    def __init__(
        self,
        pool: CameraPool,
        specs: RoleSpecs,
        *,
        ready_timeout: float = DEFAULT_READY_TIMEOUT_S,
        claim_timeout: float = DEFAULT_CLAIM_TIMEOUT_S,
    ) -> None:
        if PRIMARY not in specs or set(specs) - set(ROLES):
            raise ValueError("camera stream needs a primary role and only primary/secondary")
        self._pool = pool
        self._specs = {role: specs[role] for role in ROLES if role in specs}
        self._ready_timeout = ready_timeout
        self._claim_timeout = claim_timeout
        self._generations: dict[str, int] = {}
        self._leases: dict[str, CameraLease] = {}
        self._closed = False

    @property
    def specs(self) -> dict[str, CaptureSpec]:
        return dict(self._specs)

    @property
    def dual(self) -> bool:
        return SECONDARY in self._specs

    @property
    def claimed(self) -> bool:
        return bool(self._leases) and not self._closed

    def wait_ready(self, cancel: CancelToken | None = None) -> dict[str, WarmFrame]:
        """等每路都收到有效首帧（``wait_one``/``wait_pair``）；无预热时池按规格冷启动。"""
        stop = _TokenEvent(cancel)
        try:
            if self.dual:
                frames = self._pool.wait_pair(
                    self._specs[PRIMARY], self._specs[SECONDARY],
                    timeout=self._ready_timeout, stop_event=stop,
                )
            else:
                frames = (self._pool.wait_one(
                    self._specs[PRIMARY], timeout=self._ready_timeout, stop_event=stop,
                ),)
        except CameraWarmupStopped as exc:
            raise _cancelled(cancel) from exc
        ready = dict(zip(self._specs, frames))
        self._generations = {role: frame.generation for role, frame in ready.items()}
        return ready

    def snapshot(self, after: Mapping[str, int] | None = None) -> dict[str, WarmFrame] | None:
        """模型加载期间的裸帧预览（D-8，单双摄都有）；``after`` 是各路已显示的帧序号，没有更新的帧时返回 None。"""
        if not self.dual:
            frame = self._pool.snapshot_one(
                self._specs[PRIMARY], after=None if after is None else after[PRIMARY],
            )
            return None if frame is None else {PRIMARY: frame}
        cursor = None if after is None else (after[PRIMARY], after[SECONDARY])
        pair = self._pool.snapshot_pair(self._specs[PRIMARY], self._specs[SECONDARY], after=cursor)
        return None if pair is None else dict(zip(ROLES, pair))

    def claim(self, cancel: CancelToken | None = None) -> None:
        """接管 ``wait_ready`` 观察到的那一代预热（``claim_one``/``claim_pair``），取得租约。"""
        stop = _TokenEvent(cancel)
        generations = tuple(self._generations.get(role) for role in self._specs)
        expected = None if None in generations else generations
        try:
            if self.dual:
                leases = self._pool.claim_pair(
                    self._specs[PRIMARY], self._specs[SECONDARY],
                    timeout=self._claim_timeout, expected_generations=expected, stop_event=stop,
                )
            else:
                leases = (self._pool.claim_one(
                    self._specs[PRIMARY], timeout=self._claim_timeout,
                    expected_generation=None if expected is None else expected[0], stop_event=stop,
                ),)
        except CameraWarmupStopped as exc:
            raise _cancelled(cancel) from exc
        self._leases = dict(zip(self._specs, leases))

    def actual_sizes(self) -> dict[str, tuple[int, int]]:
        """驱动实际给出的宽高（未旋转）；拿不到时回退为请求尺寸。"""
        sizes = {}
        for role, lease in self._leases.items():
            spec = self._specs[role]
            sizes[role] = (int(lease.get(3) or spec.width), int(lease.get(4) or spec.height))
        return sizes

    def fps(self, role: str = PRIMARY) -> float:
        lease = self._leases.get(role)
        return float(lease.get(5) or 0.0) if lease is not None else 0.0

    def read(self, cancel: CancelToken | None = None) -> FrameSet:
        """各路各读一帧；任一路失败抛 ``CameraReadError``，驱动异常原样抛出。"""
        if cancel is not None:
            cancel.raise_if_cancelled()
        frames: FrameSet = {}
        for role, lease in self._leases.items():
            ok, frame = lease.read()
            if not ok:
                raise CameraReadError(role, numbered=self.dual)
            frames[role] = frame
        return frames

    def close(self, *, keep_warm: bool = False) -> None:
        """结束会话。已接管时：``keep_warm`` 且池接收则把设备交回预热（V-1），否则 ``lease.release()``；
        未接管时只取消本次观察到的那一代预热，不误伤之后重新声明的预热。释放失败的设备由池隔离，
        各路都处理完后抛出第一个错误。"""
        if self._closed:
            return
        self._closed = True
        if not self._leases:
            for role, generation in self._generations.items():
                try:
                    self._pool.cancel(role, generation=generation)
                except Exception:
                    pass
            return
        errors: list[BaseException] = []
        for lease in self._leases.values():
            try:
                if keep_warm and lease.restore():
                    continue
                lease.release()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]


class CameraService:
    """设备目录、预热声明、会话句柄与关闭；Tk 与 CLI 只经这里使用摄像头。"""

    def __init__(
        self,
        pool: CameraPool | None = None,
        *,
        catalog: DeviceCatalog | None = None,
        capture_factory: Callable[..., Any] | None = None,
        join_timeout: float = 1.0,
        ready_timeout: float = DEFAULT_READY_TIMEOUT_S,
        claim_timeout: float = DEFAULT_CLAIM_TIMEOUT_S,
    ) -> None:
        self._pool = pool if pool is not None else CameraPool(
            capture_factory, join_timeout=join_timeout, log=log_pool_event,
        )
        self._catalog = catalog if catalog is not None else DeviceCatalog()
        self._ready_timeout = ready_timeout
        self._claim_timeout = claim_timeout
        self._tasks = TaskGroup("camera-service")

    def catalog(self) -> CatalogSnapshot:
        return self._catalog.snapshot()

    def refresh(self, on_done: Callable[[CatalogSnapshot], None] | None = None) -> TaskHandle:
        """后台重新枚举；兜底探测跳过池占用中的设备（V-6）。结果是任务的 ``result``，也交给 ``on_done``。"""

        def run(_token: CancelToken) -> CatalogSnapshot:
            snapshot = self._catalog.refresh(busy=self._pool.device_busy)
            if on_done is not None:
                on_done(snapshot)
            return snapshot

        return self._tasks.spawn("refresh", run, owner="catalog")

    def prepare(self, specs: RoleSpecs) -> None:
        """声明各角色的预热需求（转调 ``pool.sync``）；空映射表示放弃所有预热。"""
        if not specs:
            self._pool.release_all()
            return
        self._pool.sync(specs.get(PRIMARY), specs.get(SECONDARY))

    def device_busy(self, index: int) -> bool:
        return self._pool.device_busy(index)

    def open_stream(self, specs: RoleSpecs) -> CameraStream:
        """本次会话的设备句柄；不阻塞，等待与接管在 ``CameraStream`` 上进行。"""
        return CameraStream(
            self._pool, specs, ready_timeout=self._ready_timeout, claim_timeout=self._claim_timeout,
        )

    def close(self, timeout: float = DEFAULT_RELEASE_WAIT_S) -> bool:
        """关窗：停止全部 reader，在 ``timeout`` 内等采集进程真实退出；超时写诊断并返回 False。"""
        self._pool.close()
        released = self._pool.wait_released(timeout)
        if not released:
            diag.event("camera.release_timeout", elapsed_ms=round(timeout * 1000.0))
        return released


__all__ = [
    "CameraReadError",
    "CameraService",
    "CameraStream",
    "DEFAULT_CLAIM_TIMEOUT_S",
    "DEFAULT_READY_TIMEOUT_S",
    "DEFAULT_RELEASE_WAIT_S",
    "RoleSpecs",
    "log_pool_event",
]
