# -*- coding: utf-8 -*-
"""``WarmupSlot``：摄像头池里一个角色的 reader 槽位状态（规格、代号、句柄、最新帧、占用与释放标记），由 ``pool.py`` 拆出。"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from core.media.camera.spec import CaptureSpec, WarmFrame


@dataclass(eq=False)
class WarmupSlot:
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


__all__ = ["WarmupSlot"]
