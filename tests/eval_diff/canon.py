"""把评估输出规整成可 JSON 往返、可逐位比对的结构；类型（dtype、tuple、dataclass 名）一并记录。"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePath
from types import SimpleNamespace
from typing import Any

import numpy as np

WORKDIR_MARK = "<WORK>"
DIGEST_OVER = 256  # 元素数超过它的数组、序列化后超过 8 倍长度的列表只记摘要


class Canon:
    """``Canon(workdir)(obj)``：路径与字符串中的临时目录替换成 ``<WORK>``，其余原样保留。"""

    def __init__(self, workdir: Path | None = None) -> None:
        self._roots = []
        if workdir is not None:
            w = Path(workdir)
            self._roots = sorted({str(w), str(w.resolve()), w.as_posix(), str(w).replace("\\", "\\\\")},
                                 key=len, reverse=True)

    def text(self, s: str) -> str:
        for root in self._roots:
            s = s.replace(root, WORKDIR_MARK)
        return s.replace("\\", "/")

    def __call__(self, obj: Any) -> Any:
        if obj is None or isinstance(obj, bool):
            return obj
        if isinstance(obj, (int, float)) and type(obj) in (int, float):
            return obj
        if isinstance(obj, str):
            return self.text(obj)
        if isinstance(obj, np.generic):
            return {"np": obj.dtype.str, "v": obj.item() if obj.dtype.kind != "O" else self(obj.item())}
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind == "O":
                return {"nd": "O", "shape": list(obj.shape), "v": [self(x) for x in obj.ravel().tolist()]}
            if obj.size > DIGEST_OVER:  # 大数组只记字节摘要：任何一位变化都会改变摘要
                data = np.ascontiguousarray(obj).tobytes()
                return {"nd": obj.dtype.str, "shape": list(obj.shape), "sha256": hashlib.sha256(data).hexdigest()}
            return {"nd": obj.dtype.str, "shape": list(obj.shape), "v": obj.ravel().tolist()}
        if isinstance(obj, PurePath):
            return {"path": self.text(str(obj))}
        if isinstance(obj, BaseException):
            return {"exc": type(obj).__name__, "msg": self.text(str(obj)),
                    "code": self(getattr(obj, "code", None))}
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {"dc": type(obj).__name__,
                    "f": {f.name: self(getattr(obj, f.name)) for f in dataclasses.fields(obj)}}
        if isinstance(obj, SimpleNamespace):
            return {"ns": {k: self(v) for k, v in sorted(vars(obj).items())}}
        if isinstance(obj, dict):
            return {"dict": [[self(k), self(v)] for k, v in obj.items()]}
        if isinstance(obj, tuple):
            return {"tuple": [self(x) for x in obj]}
        if isinstance(obj, list):
            items = [self(x) for x in obj]
            if _numeric(items):
                text = json.dumps(items, allow_nan=True)
                if len(text) > DIGEST_OVER * 8:  # 证据帧关键点等纯数值长列表只记摘要
                    return {"list_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "len": len(items)}
            return items
        if isinstance(obj, (set, frozenset)):
            return {"set": sorted(repr(x) for x in obj)}
        raise TypeError(f"canon 不认识的类型：{type(obj)!r}")


def _numeric(items) -> bool:
    return all(isinstance(x, (int, float, bool)) or (isinstance(x, list) and _numeric(x)) for x in items)


def diff(a: Any, b: Any, path: str = "$", out: list[str] | None = None, limit: int = 50) -> list[str]:
    """逐项比较两份 canon 输出；浮点逐位相等（NaN 视为相等，区分 ±0）。"""
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if type(a) is not type(b):
        out.append(f"{path}: 类型 {type(a).__name__} != {type(b).__name__}")
    elif isinstance(a, dict):
        if list(a) != list(b):
            out.append(f"{path}: 键 {list(a)} != {list(b)}")
        else:
            for k in a:
                diff(a[k], b[k], f"{path}.{k}", out, limit)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: 长度 {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out, limit)
    elif isinstance(a, float):
        same = (math.isnan(a) and math.isnan(b)) or (a == b and math.copysign(1, a) == math.copysign(1, b))
        if not same:
            out.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def dump(path: Path, data: dict) -> None:
    raw = json.dumps(data, ensure_ascii=False, allow_nan=True, sort_keys=False).encode("utf-8")
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(raw)


def load(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))
