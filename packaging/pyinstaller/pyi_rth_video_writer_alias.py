"""Runtime alias for the frozen tech_eval debug-video import."""

from __future__ import annotations

import importlib
import sys

if "video_writer" not in sys.modules:
    sys.modules["video_writer"] = importlib.import_module("core.video_writer")
