# -*- coding: utf-8 -*-
"""
一次性脚本：从本地（gitignored）outputs/ 下已提取的骨架序列裁剪出小体积的
原始 (T,33,4) landmark fixture，纳入版本控制。

这些 fixture 是 golden 的“真值输入”。生成后即锁定；除非源数据或裁剪范围有意调整，
否则不应重跑本脚本。日常重生成 golden 只需 regen_pose33_v3_golden.py（仅依赖本目录
已提交的 *_raw.npz）。

运行：
    .\\.venv\\Scripts\\python.exe tests\\fixtures\\_build_raw_fixtures.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

FIX_DIR = Path(__file__).resolve().parent / "pose33_v3"


def _load_one(pattern: str) -> np.ndarray:
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"找不到源骨架序列：{pattern}")
    d = np.load(matches[0], allow_pickle=True)
    return np.asarray(d["landmarks"], dtype=np.float32)


def main() -> None:
    FIX_DIR.mkdir(parents=True, exist_ok=True)

    base = "outputs/**/直拳"

    front_src = _load_one(f"{base} 正面_pose33_heavy.npz")[:90]
    side_src = _load_one(f"{base} 右侧_pose33_heavy.npz")[:90]

    # 学员长视频：前段=正面 clip，后段=侧面 clip（不同于模板源，避免完全退化）。
    stu_front = _load_one(f"{base} 正面_2_pose33_heavy.npz")[20:110]
    stu_side = _load_one(f"{base} 左侧_pose33_heavy.npz")[:90]
    student = np.concatenate([stu_front, stu_side], axis=0)

    np.savez_compressed(FIX_DIR / "front_src_raw.npz", landmarks=front_src.astype(np.float32))
    np.savez_compressed(FIX_DIR / "side_src_raw.npz", landmarks=side_src.astype(np.float32))
    np.savez_compressed(FIX_DIR / "student_raw.npz", landmarks=student.astype(np.float32))

    print("front_src_raw:", front_src.shape)
    print("side_src_raw :", side_src.shape)
    print("student_raw  :", student.shape)
    print("已写入：", FIX_DIR)


if __name__ == "__main__":
    main()
