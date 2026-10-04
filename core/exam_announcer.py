# -*- coding: utf-8 -*-
"""异步考试播报队列。失败时静默降级（由 UI 展示文案）。"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from typing import Callable


class ExamAnnouncer:
    def __init__(
        self,
        *,
        speak_fn: Callable[[str], None] | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> None:
        self._speak_fn = speak_fn or _default_windows_speak
        self._on_text = on_text
        self._q: queue.Queue[str | None] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="exam-announcer", daemon=True
        )
        self._thread.start()

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
        self._stop.set()
        self._q.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
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
