"""连贯套路视频 × 分段模板 互抢验证探针。

用法:
  PYTHONPATH=. .venv/Scripts/python.exe scripts/match_segments_probe.py <连贯视频.mp4> [动作1 动作2 ...]

第2个参数起可只指定实际录了的动作名(不带.npz),缺省则用 templates/segments 下全部模板。

对每个模板跑一次 subsequence DTW,打印得分与锁定窗口。视频姿态**只提取一次**并复用给所有模板
(避免上次 724帧×N 模板逐个重复提取导致的超时)。评分公式与比对入口一致
(``core.evaluation.score_from_avg_cost``,baseline 取布局注册表的 pose33_v3 值)。

开头会打印视频分辨率与模板源分辨率——**画幅方向(横/竖)必须一致**,否则归一化畸变导致匹配全垮
(见记忆 sanda-aspect-ratio-normalize-bug)。

ponytail: 一次性诊断脚本,非生产路径;不落盘、不改模板。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from core.evaluation import POSE33_BASELINE, score_from_avg_cost
from core.media import VideoOpenError
from core.pose import FORMAL, POSE33_V3, extract_pose_sequence, formal_engine, normalize_pose_xy_v3
from core.pose_features import subsequence_dtw

TEMPL_DIR = Path("templates/segments")

# 视角分组：侧面路(横拍)配这7个;正面路(竖拍)配摇闪。画幅+视角须与视频一致。
SIDE_ACTIONS = ["直拳", "摆拳", "勾拳", "高鞭腿", "低鞭腿", "侧踹腿", "正蹬腿"]
FRONT_ACTIONS = ["摇闪"]


def extract_features(video: Path) -> tuple[np.ndarray, float, int, int]:
    """提取整段视频的 pose33_v3 归一化特征 (T,22,2),返回 (feats, fps, w, h)。"""
    try:
        seq = extract_pose_sequence(video, formal_engine("heavy"), FORMAL)
    except VideoOpenError:
        raise SystemExit(f"无法打开视频: {video}") from None
    if len(seq) == 0:
        raise SystemExit("未从视频读到任何帧")
    feats = seq.features(normalize_pose_xy_v3, tuple(POSE33_V3.shape))
    return feats, seq.meta.fps, seq.meta.width, seq.meta.height


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("用法: match_segments_probe.py <连贯视频> [动作名 ...]")
    video = Path(sys.argv[1])
    if not video.exists():
        raise SystemExit(f"视频不存在: {video}")

    wanted = set(sys.argv[2:])
    tpls = sorted(TEMPL_DIR.glob("*.npz"))
    if wanted:
        tpls = [t for t in tpls if t.stem in wanted]
        missing = wanted - {t.stem for t in tpls}
        if missing:
            print(f"⚠ 指定的动作无对应模板: {', '.join(sorted(missing))}")
    if not tpls:
        raise SystemExit(f"无模板可比对: {TEMPL_DIR}")

    t0 = time.time()
    seq_xy, fps, w, h = extract_features(video)
    seq = seq_xy.reshape(seq_xy.shape[0], -1)  # (T,44)
    orient = "竖拍" if h > w else "横拍"
    print(f"视频: {video.name}  分辨率={w}×{h} ({orient})  {seq.shape[0]}帧  提取用时{time.time()-t0:.1f}s")
    print("模板源分辨率=1280×720(横拍)  —— 画幅方向须一致,否则归一化畸变!")
    print()

    rows = []
    for tpl in tpls:
        q = np.load(tpl, allow_pickle=True)["features"].reshape(-1, 44)
        cost, s, e = subsequence_dtw(q, seq)
        avg = cost / max(1, q.shape[0])
        score = score_from_avg_cost(avg, POSE33_BASELINE)
        rows.append((tpl.stem, score, s, e, q.shape[0]))

    rows.sort(key=lambda x: (x[2] if x[2] is not None else 1 << 30))
    print(f"{'模板':<8}{'得分':>7}{'窗口(帧)':>14}{'窗口秒':>12}{'模板长':>7}")
    print("-" * 50)
    for n, sc, s, e, ql in rows:
        win = f"{s}..{e}"
        sec = f"{s/fps:.1f}-{e/fps:.1f}s"
        flag = "  ⚠短" if (e - s) < ql * 0.4 else ""
        print(f"{n:<8}{sc:7.3f}{win:>14}{sec:>12}{ql:>7}{flag}")

    # 重叠检测
    spans = [(n, s, e) for n, _, s, e, _ in rows]
    overlaps = []
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            n1, s1, e1 = spans[i]
            n2, s2, e2 = spans[j]
            if s1 <= e2 and s2 <= e1:
                overlaps.append((n1, n2, max(s1, s2), min(e1, e2)))
    print("-" * 50)
    if overlaps:
        print("⚠ 窗口重叠(疑似互抢):")
        for n1, n2, a, b in overlaps:
            print(f"  {n1} × {n2} 重叠帧 {a}..{b}")
    else:
        print("✓ 无窗口重叠")
    print("提示: 若窗口标'⚠短'(远小于模板长)或全挤一处,多为画幅/归一化不匹配,而非动作不像。")


if __name__ == "__main__":
    main()
