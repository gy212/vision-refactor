# -*- coding: utf-8 -*-
"""迟到结果过滤的唯一原语 ``Epoch``：所有者推进代次，事件携带 token，消费方丢弃过期 token。

每个异步边界只有一个所有者持有 ``Epoch``（设计文档 §4.3）；取代 App 上手写的
``_session_generation``、``_active_feedback_token`` 等代次字段（P6、P8 替换）。
"""
from __future__ import annotations

import threading


class Epoch:
    """单调递增的代次计数器，线程安全。初始代次为 0。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current = 0

    @property
    def current(self) -> int:
        with self._lock:
            return self._current

    def advance(self) -> int:
        """开始新一代并返回其 token；此前发出的 token 全部过期。"""
        with self._lock:
            self._current += 1
            return self._current

    def is_current(self, token: int) -> bool:
        with self._lock:
            return token == self._current
