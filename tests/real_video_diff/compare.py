"""逐元素比对 record.py 的两次输出（dtype、形状一并比较，NaN 视为相等）。用法：python compare.py <旧输出> <新输出>"""
import sys
from pathlib import Path
import numpy as np
a, b = Path(sys.argv[1]), Path(sys.argv[2])
bad = 0
names = sorted(p.name for p in a.glob("*.npz"))
assert names == sorted(p.name for p in b.glob("*.npz")), "文件集合不同"
for name in names:
    x, y = np.load(a / name), np.load(b / name)
    assert sorted(x.files) == sorted(y.files), name
    for k in x.files:
        u, v = x[k], y[k]
        same = u.dtype == v.dtype and u.shape == v.shape and np.array_equal(u, v, equal_nan=u.dtype.kind == "f")
        if not same:
            bad += 1
            print("DIFF", name, k, u.dtype, v.dtype, u.shape, v.shape)
print(f"{len(names)} files, {bad} diffs")
sys.exit(1 if bad else 0)
