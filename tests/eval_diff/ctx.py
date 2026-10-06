"""差分用例共享的运行上下文：录制函数、视频与模板、工作目录，以及 fixture/真实视频两种模式的取数方式。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass
class Ctx:
    rec: Callable[[str, Callable[[], Any]], None]
    v: dict[str, Any]  # 视频名 → 路径或 golden key：student/front/side/gappy/mirror/long/short
    t: dict[str, Path]  # 模板名 → 路径（synth.write_templates）
    work: Path
    mode: str  # "fixture" | "real"
    place: Callable[[Path, str], None]  # 把视频 name 放到 dst：fixture 注册回放，real 复制文件
    raw: Callable[[str], np.ndarray]  # 视频 name 的 Pose33 原始序列 (T,33,4)
