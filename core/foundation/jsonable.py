# -*- coding: utf-8 -*-
"""把结果对象转成可 ``json.dumps`` 的结构：Path → str、numpy 标量与数组 → Python 值、dataclass → dict、tuple → list。"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj
