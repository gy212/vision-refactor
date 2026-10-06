# -*- coding: utf-8 -*-
"""core.foundation.jsonable：结果对象转 JSON 结构；tech_eval 沿用同一实现。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis import tech_eval
from core.foundation import to_jsonable


@dataclass
class _Inner:
    score: np.float32
    frames: tuple[int, ...]


def test_nested_structures_become_plain_json():
    payload = {
        1: _Inner(np.float32(0.5), (1, 2)),
        "path": Path("a") / "b.mp4",
        "arr": np.arange(3, dtype=np.int64),
        "n": np.int16(7),
        "keep": None,
    }
    out = to_jsonable(payload)
    assert out == {
        "1": {"score": 0.5, "frames": [1, 2]},
        "path": str(Path("a") / "b.mp4"),
        "arr": [0, 1, 2],
        "n": 7,
        "keep": None,
    }
    json.dumps(out)
    assert type(out["n"]) is int and type(out["1"]["score"]) is float


def test_tech_eval_reexports_the_single_implementation():
    assert tech_eval.to_jsonable is to_jsonable
