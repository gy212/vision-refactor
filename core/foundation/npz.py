# -*- coding: utf-8 -*-
"""非模板 npz 归档（原始骨架导出、标定缓存、测试 fixture）的读取：一次读出全部数组并关闭文件。

模板不经这里：读模板用 ``core.evaluation.load_template``，它带契约校验（设计文档 §13.2 原语单一入口）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """读出 npz 里的全部数组（允许 pickle 的 ``meta`` 对象），返回键到数组的字典。"""
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def npz_meta(arrays: dict[str, np.ndarray]) -> dict:
    """``meta`` 对象数组转成 dict；缺 ``meta`` 键或值为 None 时返回空 dict。"""
    if "meta" not in arrays:
        return {}
    return dict(arrays["meta"].item() or {})
