# -*- coding: utf-8 -*-
"""异步语音播报队列：考试叫号与学生练习的"开始/结束"共用；播报失败时静默降级（界面另行显示文案）。

P7 从 ``core/exam_announcer.py`` 迁入，后台线程改由 ``foundation.tasks.TaskGroup`` 管理。
"""

from __future__ import annotations

import queue
import subprocess
import sys
from typing import Callable

from core.foundation.tasks import CancelToken, TaskGroup

# 各播报器的后台朗读线程
_SPEAKERS = TaskGroup("announcer")


class Announcer:
    def __init__(
        self,
        *,
        speak_fn: Callable[[str], None] | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> None:
        self._speak_fn = speak_fn or _default_windows_speak
        self._on_text = on_text
        self._q: queue.Queue[str | None] = queue.Queue()
        self._task = _SPEAKERS.spawn("speak", self._loop, owner=f"announcer-{id(self):x}")

    def announce(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if self._on_text is not None:
            try:
                self._on_text(text)
            except Exception:
                pass
        self._q.put(text)

    def close(self) -> None:
        self._task.token.cancel("closed")
        self._q.put(None)
        self._task.join(timeout=2.0)

    def _loop(self, token: CancelToken) -> None:
        while not token.cancelled:
            try:
                item = self._q.get(timeout=0.3)
            except queue.Empty:
                continue
            if item is None:
                return
            try:
                self._speak_fn(item)
            except Exception:
                pass


def _default_windows_speak(text: str) -> None:
    """尽量用 Windows System.Speech；失败则忽略。"""
    # 转义单引号
    safe = text.replace("'", "''")
    ps = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Speak('{safe}')"
    )
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-WindowStyle",
            "Hidden",
            "-Command",
            ps,
        ],
        check=False,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        timeout=60,
    )
