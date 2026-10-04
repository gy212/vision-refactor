# -*- coding: utf-8 -*-
r"""S3 标定与基准报告：body_core_v1 三路对照 + 阈值标定（YOLO 迁移 Issue #10）。

用途
====
对照 Issue #10（S3）：在同一批样本上跑三路特征提取与模板匹配——
  1. MediaPipe ``pose33_v3``（旧默认布局，22 点）；
  2. MediaPipe ``body_core_v1``（12 点共享布局）；
  3. YOLO ``body_core_v1``（12 点共享布局，COCO17 映射）。

并据此产出**对照数字**，用于：
  - 判定 ``body_core_v1`` 是否可用于评分（结论四选一）；
  - 重标定 ``body_core_v1`` DTW baseline（替换全局占位 2.0）；
  - 标定 YOLO 侧 ``valid_conf_thr``（替换 #7 占位值 0.5）。

方法论（与报告头部预注册口径一致）
================================
**模板分数必须在「不同视频」之间产生**，否则 self-match（query 是 seq 的子段）
恒为满分、无方差、相关性无意义。本脚本因此构建「成对匹配矩阵」：
  - 在每个视角组（front / side）内，取每段样本的活跃区间作为模板 query，
    与组内其它样本整段做 subsequence DTW，得到该 (模板, 视频) 对的 avg_cost / 分数。
  - 三路（pose33_v3 / MP body_core / YOLO body_core）在**完全相同的 (模板, 视频) 对**
    上各自打分，因此可比较：
      * corr(MP body_core 分数, pose33 分数)：body_core 布局是否跟随 pose33（同后端）；
      * corr(YOLO body_core 分数, MP body_core 分数)：YOLO 后端是否跟随 MediaPipe（同布局）；
      * MAE(YOLO body_core 分数, MP body_core 分数)；
      * pass/fail 一致率（以 pose33 分数阈值为参照口径）。

baseline 标定
=============
pose33_v3 已标定 baseline=2.0。body_core_v1 的 avg_cost 尺度与 pose33 不同，故按
**尺度对齐**标定：``baseline_bodycore = 2.0 * median(avg_cost_bodycore) / median(avg_cost_pose33)``
（在跨样本对上统计，self 对 avg_cost≈0 不计入），使 body_core 分数与 pose33 分数同尺度，
共享阈值即可保持 pass/fail 一致。

强约束（来自迁移计划 / AGENTS.md）
================================
- 本脚本**不改主代码默认行为**：只调用既有 ``extract_pose_raw`` /
  ``extract_yolo_landmark_series`` / 共享 normalizer / ``subsequence_dtw`` 等生产函数。
- **先预注册阈值、后对照数据**：阈值/判定口径写死在 ``docs/yolo_body_core_calibration.md``
  头部（跑数据前），本脚本只产出数字供报告逐条引用。
- **未标定不得对外评分**：本脚本所有分数仅供 S3 标定/审计。
- **缺失点不伪造**：YOLO 路径有效性一律走 conf 阈值派生的 valid_mask。

效率设计
========
每段样本每后端**只推理一次**并缓存到磁盘（``raw_cache/*.npz``）；重跑分析时若缓存存在
则直接加载，不重复推理。YOLO ``valid_conf_thr`` 扫描只对缓存 conf 重新阈值化，不重复推理。

运行示例
========
::

    .\.venv\Scripts\python.exe -m analysis.calibrate_body_core \
        --samples docs/yolo_eval_samples.json \
        --yolo-model models/yolo11n-pose.pt \
        --pose-variant full \
        --out outputs/calib_body_core
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np

# 降低原生日志噪声，保持与主链路一致的安静度。
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

# 允许以脚本或模块方式运行。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.feature_layout import BODY_CORE_V1, POSE33_V3  # noqa: E402
from core.pose_features import (  # noqa: E402
    find_active_range,
    motion_energy,
    normalize_pose_body_core_v1,
    normalize_pose_xy_v3,
    subsequence_dtw,
)
from core.yolo_adapter import evaluate_multi_person_gate  # noqa: E402

# pose33_v3 已标定 baseline（固定参照）。
POSE33_BASELINE: float = 2.0

# YOLO valid_conf_thr 扫描网格（标定用）。
YOLO_CONF_THR_GRID: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

# pass/fail 参照阈值（预注册口径，见报告头部）：pose33 分数 >= 该值视为 "pass(相似)"。
PASS_SCORE_THR: float = 0.55
YOLO_ARTIFACT_METADATA: dict[str, object] = {
    "backend": "yolo",
    "raw_layout": "pose33_like_coco17",
    "feature_layout": "body_core_v1",
    "capability": "body_only",
    "score_authorized": False,
    "calibration_status": "unvalidated",
    "display_scope": "internal",
}


# --------------------------------------------------------------------------- #
# 原始关键点缓存（每后端每视频只推理一次；可持久化到磁盘）
# --------------------------------------------------------------------------- #
@dataclass
class RawCache:
    sample_id: str
    video_path: str
    view: str = "mixed"
    multi_person: bool = False
    # MediaPipe
    mp_landmarks: np.ndarray | None = None  # (T,33,4)
    mp_valid_mask: np.ndarray | None = None  # (T,33) bool
    mp_fps: float = 30.0
    mp_infer_sec: float = 0.0
    mp_frames: int = 0
    # YOLO（第 4 通道保留原始 conf）
    yolo_landmarks: np.ndarray | None = None  # (T,33,4)
    yolo_fps: float = 30.0
    yolo_infer_sec: float = 0.0
    yolo_frames: int = 0
    yolo_num_persons_per_frame: list = field(default_factory=list)


def _cache_file(out_dir: Path, sample_id: str) -> Path:
    return out_dir / "raw_cache" / f"{sample_id}.npz"


def extract_or_load_cache(
    sample: dict,
    *,
    pose_variant: str,
    yolo_model: str | Path | None,
    out_dir: Path,
    end_frame: int | None,
    force: bool,
) -> RawCache:
    sid = sample["id"]
    vp = Path(sample["path"])
    cf = _cache_file(out_dir, sid)
    if cf.exists() and not force:
        d = np.load(cf, allow_pickle=True)
        meta = dict(d["meta"].item())
        print(f"[cache] {sid} <- {cf.name}")
        return RawCache(
            sample_id=sid,
            video_path=str(vp),
            view=str(sample.get("view", "mixed")),
            multi_person=(sample.get("multi_person") is True),
            mp_landmarks=d["mp_landmarks"],
            mp_valid_mask=d["mp_valid_mask"],
            mp_fps=float(meta["mp_fps"]),
            mp_infer_sec=float(meta["mp_infer_sec"]),
            mp_frames=int(meta["mp_frames"]),
            yolo_landmarks=d["yolo_landmarks"],
            yolo_fps=float(meta["yolo_fps"]),
            yolo_infer_sec=float(meta["yolo_infer_sec"]),
            yolo_frames=int(meta["yolo_frames"]),
            yolo_num_persons_per_frame=list(meta.get("yolo_num_persons_per_frame", [])),
        )

    from core.rule_scoring import extract_pose_raw
    from core.yolo_adapter import extract_yolo_landmark_series

    print(f"[extract] {sid} <- {vp}")
    t0 = time.perf_counter()
    mp_landmarks, mp_meta = extract_pose_raw(vp, pose_variant=pose_variant, end_frame=end_frame)
    mp_infer = time.perf_counter() - t0

    t0 = time.perf_counter()
    yolo_landmarks, _v, yolo_meta = extract_yolo_landmark_series(
        vp, yolo_model=yolo_model, valid_conf_thr=0.0, end_frame=end_frame
    )
    yolo_infer = time.perf_counter() - t0

    cache = RawCache(
        sample_id=sid,
        video_path=str(vp),
        view=str(sample.get("view", "mixed")),
        multi_person=(sample.get("multi_person") is True),
        mp_landmarks=mp_landmarks,
        mp_valid_mask=np.asarray(mp_meta.get("valid_mask"), dtype=bool),
        mp_fps=float(mp_meta.get("fps") or 30.0),
        mp_infer_sec=float(mp_infer),
        mp_frames=int(mp_landmarks.shape[0]),
        yolo_landmarks=yolo_landmarks,
        yolo_fps=float(yolo_meta.get("fps") or 30.0),
        yolo_infer_sec=float(yolo_infer),
        yolo_frames=int(yolo_landmarks.shape[0]),
        yolo_num_persons_per_frame=list(yolo_meta.get("num_persons_per_frame", [])),
    )

    cf.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cf,
        mp_landmarks=cache.mp_landmarks,
        mp_valid_mask=cache.mp_valid_mask,
        yolo_landmarks=cache.yolo_landmarks,
        meta=np.array(
            {
                "mp_fps": cache.mp_fps,
                "mp_infer_sec": cache.mp_infer_sec,
                "mp_frames": cache.mp_frames,
                "yolo_fps": cache.yolo_fps,
                "yolo_infer_sec": cache.yolo_infer_sec,
                "yolo_frames": cache.yolo_frames,
                "yolo_num_persons_per_frame": cache.yolo_num_persons_per_frame,
                "yolo_artifact_metadata": dict(YOLO_ARTIFACT_METADATA),
                **{f"yolo_{key}": value for key, value in YOLO_ARTIFACT_METADATA.items()},
            },
            dtype=object,
        ),
    )
    return cache


# --------------------------------------------------------------------------- #
# 特征派生
# --------------------------------------------------------------------------- #
class _RowView:
    """把 (33,4) numpy 行包装成 normalize_pose_xy_v3 期望的 ``lm[i].x/.y`` 接口。"""

    __slots__ = ("_a",)

    def __init__(self, arr: np.ndarray) -> None:
        self._a = np.asarray(arr, dtype=np.float32)

    def __getitem__(self, i: int) -> "_Pt":
        return _Pt(self._a[i])


class _Pt:
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, row: np.ndarray) -> None:
        self.x = float(row[0])
        self.y = float(row[1])
        self.z = float(row[2]) if row.shape[0] > 2 else 0.0
        self.visibility = float(row[3]) if row.shape[0] > 3 else 0.0


def _derive_valid_mask_from_conf(landmarks: np.ndarray, thr: float) -> np.ndarray:
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.shape[0] == 0:
        return np.zeros((0, 33), dtype=bool)
    return arr[..., 3] >= float(thr)


def derive_pose33_features(landmarks: np.ndarray) -> np.ndarray:
    feats: list[np.ndarray] = []
    zero = np.zeros(tuple(int(x) for x in POSE33_V3.shape), dtype=np.float32)
    for t in range(int(landmarks.shape[0])):
        f = normalize_pose_xy_v3(_RowView(landmarks[t]))
        if f is None:
            f = feats[-1].copy() if feats else zero.copy()
        feats.append(f)
    if not feats:
        return np.zeros((0, *POSE33_V3.shape), dtype=np.float32)
    return np.stack(feats, axis=0).astype(np.float32)


def derive_body_core_features(
    landmarks: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    feats: list[np.ndarray] = []
    zero = np.zeros(tuple(int(x) for x in BODY_CORE_V1.shape), dtype=np.float32)
    core_idx = np.asarray(BODY_CORE_V1.source_indices, dtype=int)
    valid_core_frames = 0
    for t in range(int(landmarks.shape[0])):
        core_valid = bool(np.all(valid_mask[t, core_idx]))
        f = normalize_pose_body_core_v1(landmarks[t]) if core_valid else None
        if f is None:
            f = feats[-1].copy() if feats else zero.copy()
        else:
            valid_core_frames += 1
        feats.append(f)
    if not feats:
        return np.zeros((0, *BODY_CORE_V1.shape), dtype=np.float32), 0
    return np.stack(feats, axis=0).astype(np.float32), int(valid_core_frames)


def _active_query(features: np.ndarray) -> np.ndarray:
    if features.shape[0] < 2:
        return features
    energy = motion_energy(features.reshape(features.shape[0], -1))
    a_s, a_e = find_active_range(energy, pad=10)
    a_s = max(0, min(int(a_s), int(features.shape[0] - 1)))
    a_e = max(a_s, min(int(a_e), int(features.shape[0] - 1)))
    return features[a_s : a_e + 1]


def _match(query_feats: np.ndarray, seq_feats: np.ndarray) -> float | None:
    """以 query 活跃区间匹配 seq 整段，返回 avg_cost。"""
    if query_feats.shape[0] == 0 or seq_feats.shape[0] == 0:
        return None
    q = _active_query(query_feats)
    cost, _s, _e = subsequence_dtw(q, seq_feats)
    return float(cost) / max(1, int(q.shape[0]))


def _score(avg_cost: float | None, baseline: float) -> float | None:
    if avg_cost is None:
        return None
    return float(baseline / (baseline + float(avg_cost)))


# --------------------------------------------------------------------------- #
# 特征束（每视频每路一份；YOLO body_core 随 conf_thr 重算）
# --------------------------------------------------------------------------- #
@dataclass
class SampleFeatures:
    sample_id: str
    view: str
    pose33: np.ndarray
    mp_bodycore: np.ndarray
    mp_bodycore_valid_ratio: float
    yolo_bodycore: np.ndarray
    yolo_bodycore_valid_ratio: float
    mp_fail_ratio: float
    yolo_fail_ratio: float


def build_features(cache: RawCache, *, yolo_conf_thr: float) -> SampleFeatures:
    pose33 = derive_pose33_features(cache.mp_landmarks)
    mp_bc, mp_bc_valid = derive_body_core_features(cache.mp_landmarks, cache.mp_valid_mask)
    yolo_valid = _derive_valid_mask_from_conf(cache.yolo_landmarks, yolo_conf_thr)
    yolo_bc, yolo_bc_valid = derive_body_core_features(cache.yolo_landmarks, yolo_valid)

    n_mp = max(1, cache.mp_frames)
    n_yolo = max(1, cache.yolo_frames)
    mp_fail = (
        float(np.mean(~np.any(cache.mp_valid_mask, axis=1)))
        if cache.mp_valid_mask.shape[0]
        else 0.0
    )
    yolo_any = (
        np.any(cache.yolo_landmarks[..., 3] > 0.0, axis=1)
        if cache.yolo_landmarks.shape[0]
        else np.zeros((0,), bool)
    )
    yolo_fail = float(np.mean(~yolo_any)) if yolo_any.shape[0] else 0.0

    return SampleFeatures(
        sample_id=cache.sample_id,
        view=cache.view,
        pose33=pose33,
        mp_bodycore=mp_bc,
        mp_bodycore_valid_ratio=float(mp_bc_valid) / n_mp,
        yolo_bodycore=yolo_bc,
        yolo_bodycore_valid_ratio=float(yolo_bc_valid) / n_yolo,
        mp_fail_ratio=mp_fail,
        yolo_fail_ratio=yolo_fail,
    )


# --------------------------------------------------------------------------- #
# 成对匹配矩阵（视角组内）
# --------------------------------------------------------------------------- #
def pairwise_matrix(
    feats: dict[str, SampleFeatures],
    *,
    baseline_bodycore: float,
    include_cross_view: bool = False,
) -> list[dict]:
    """对 (模板, 视频) 对（视角组内）三路打分。返回逐对记录。"""
    ids = list(feats.keys())
    rows: list[dict] = []
    for tpl_id, vid_id in product(ids, ids):
        tf = feats[tpl_id]
        vf = feats[vid_id]
        same_view = tf.view == vf.view
        if not include_cross_view and not same_view:
            continue
        pose33_cost = _match(tf.pose33, vf.pose33)
        mpbc_cost = _match(tf.mp_bodycore, vf.mp_bodycore)
        yolobc_cost = _match(tf.yolo_bodycore, vf.yolo_bodycore)
        rows.append(
            {
                "template": tpl_id,
                "video": vid_id,
                "self": tpl_id == vid_id,
                "view": tf.view,
                "pose33_avg_cost": pose33_cost,
                "pose33_score": _score(pose33_cost, POSE33_BASELINE),
                "mp_bodycore_avg_cost": mpbc_cost,
                "mp_bodycore_score": _score(mpbc_cost, baseline_bodycore),
                "yolo_bodycore_avg_cost": yolobc_cost,
                "yolo_bodycore_score": _score(yolobc_cost, baseline_bodycore),
                **{f"yolo_{key}": value for key, value in YOLO_ARTIFACT_METADATA.items()},
            }
        )
    return rows


def _pearson(a: list, b: list) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    va = np.asarray([p[0] for p in pairs], dtype=np.float64)
    vb = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if np.std(va) < 1e-9 or np.std(vb) < 1e-9:
        return None
    return float(np.corrcoef(va, vb)[0, 1])


def _mae(a: list, b: list) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    return float(np.mean([abs(x - y) for x, y in pairs]))


def _passfail_consistency(ref_scores: list, test_scores: list, thr: float) -> float | None:
    pairs = [(x, y) for x, y in zip(ref_scores, test_scores) if x is not None and y is not None]
    if not pairs:
        return None
    agree = sum(1 for x, y in pairs if (x >= thr) == (y >= thr))
    return float(agree) / float(len(pairs))


def summarize_pairwise_metrics(rows: list[dict], *, include_self: bool = False) -> dict:
    """Summarize pairwise score metrics using the pre-registered pair scope.

    The S3 go/no-go criteria are defined on cross-video pairs. Self matches are
    useful as a sanity check, but including them in correlation/MAE/pass-fail
    metrics injects four trivial 1.0 scores and overstates calibration quality.
    """
    metric_rows = rows if include_self else [r for r in rows if not r["self"]]
    pose33_s = [r["pose33_score"] for r in metric_rows]
    mpbc_s = [r["mp_bodycore_score"] for r in metric_rows]
    yolobc_s = [r["yolo_bodycore_score"] for r in metric_rows]
    return {
        "corr_mp_bodycore_vs_pose33": _pearson(mpbc_s, pose33_s),
        "corr_yolo_bodycore_vs_mp_bodycore": _pearson(yolobc_s, mpbc_s),
        "mae_yolo_bodycore_vs_mp_bodycore": _mae(yolobc_s, mpbc_s),
        "mae_mp_bodycore_vs_pose33": _mae(mpbc_s, pose33_s),
        "passfail_consistency_mp_bodycore_vs_pose33": _passfail_consistency(
            pose33_s, mpbc_s, PASS_SCORE_THR
        ),
        "passfail_consistency_yolo_bodycore_vs_pose33": _passfail_consistency(
            pose33_s, yolobc_s, PASS_SCORE_THR
        ),
        "passfail_consistency_yolo_bodycore_vs_mp_bodycore": _passfail_consistency(
            mpbc_s, yolobc_s, PASS_SCORE_THR
        ),
        "pass_score_thr": PASS_SCORE_THR,
        "n_pairs": len(metric_rows),
        "n_total_pairs": len(rows),
        "n_cross_pairs": sum(1 for r in rows if not r["self"]),
        "self_matches_included": bool(include_self),
    }


# --------------------------------------------------------------------------- #
# baseline 标定（尺度对齐）
# --------------------------------------------------------------------------- #
def calibrate_bodycore_baseline(rows_with_pose33: list[dict]) -> dict:
    """按跨样本对的 avg_cost 尺度比标定 body_core baseline（使其分数与 pose33 同尺度）。"""
    cross = [r for r in rows_with_pose33 if not r["self"]]
    pose33_costs = [r["pose33_avg_cost"] for r in cross if r["pose33_avg_cost"] is not None]
    bc_costs = [r["mp_bodycore_avg_cost"] for r in cross if r["mp_bodycore_avg_cost"] is not None]
    if not pose33_costs or not bc_costs:
        return {"baseline": POSE33_BASELINE, "scale_ratio": 1.0, "note": "insufficient data"}
    med_pose33 = float(np.median(pose33_costs))
    med_bc = float(np.median(bc_costs))
    ratio = med_bc / med_pose33 if med_pose33 > 1e-9 else 1.0
    baseline = float(POSE33_BASELINE * ratio)
    return {
        "baseline": round(baseline, 4),
        "scale_ratio": round(ratio, 4),
        "median_pose33_avg_cost": round(med_pose33, 4),
        "median_bodycore_avg_cost": round(med_bc, 4),
        "n_cross_pairs": len(cross),
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def _env_table() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version()}
    for mod, key in (("numpy", "numpy"), ("cv2", "opencv"), ("mediapipe", "mediapipe"), ("ultralytics", "ultralytics")):
        try:
            m = __import__(mod)
            info[key] = getattr(m, "__version__", "?")
        except Exception:
            pass
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    return info


def run(args: argparse.Namespace) -> int:
    samples_cfg = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    samples = samples_cfg.get("samples", [])
    if args.only:
        only = {s.strip() for s in args.only.split(",")}
        samples = [s for s in samples if s.get("id") in only]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 提取/加载缓存。
    caches: dict[str, RawCache] = {}
    for s in samples:
        if not Path(s["path"]).exists():
            print(f"[skip] 样本不存在：{s['id']} -> {s['path']}")
            continue
        caches[s["id"]] = extract_or_load_cache(
            s,
            pose_variant=args.pose_variant,
            yolo_model=args.yolo_model,
            out_dir=out_dir,
            end_frame=(args.max_frames - 1) if args.max_frames else None,
            force=args.force_extract,
        )

    # 多人闸门判定（实测）：review_required=True 的样本**不得进入评分/标定**
    # （多人 = 正确性风险，闸门契约要求拒绝出分）。因此标定统计集 = 声明非多人
    # 且闸门未触发的样本；被闸门标记的样本（含 multi_person="unknown" 的学员视频）
    # 仅用于多人闸门展示，不进入相关性/baseline/阈值统计。
    gate_review: dict[str, bool] = {}
    for sid, c in caches.items():
        g = evaluate_multi_person_gate(c.yolo_num_persons_per_frame)
        gate_review[sid] = bool(g["review_required"])

    single_ids = [
        sid
        for sid, c in caches.items()
        if (not c.multi_person) and (not gate_review[sid])
    ]
    # 被多人闸门排除出标定集的样本（声明多人 或 实测多人）。
    excluded_ids = [sid for sid in caches if sid not in single_ids]
    student_ids = excluded_ids

    # 2) baseline 标定（在占位 baseline 下先建矩阵拿 avg_cost，再标定）。
    feats_calib = {sid: build_features(caches[sid], yolo_conf_thr=args.calib_conf_thr) for sid in single_ids}
    rows_for_calib = pairwise_matrix(feats_calib, baseline_bodycore=POSE33_BASELINE)
    calib = calibrate_bodycore_baseline(rows_for_calib)
    baseline_bodycore = float(calib["baseline"])

    # 3) 用标定后的 baseline 重建单人矩阵。
    rows = pairwise_matrix(feats_calib, baseline_bodycore=baseline_bodycore)

    # 4) 相关性 / MAE / pass-fail 一致率。
    # 主指标严格使用跨视频对；self-match 仅作为 sanity 指标单独保留。
    metrics = summarize_pairwise_metrics(rows, include_self=False)
    metrics_with_self = summarize_pairwise_metrics(rows, include_self=True)

    # 5) YOLO valid_conf_thr 扫描（用标定 baseline；只重算 YOLO body_core 特征）。
    sweep: list[dict] = []
    for thr in YOLO_CONF_THR_GRID:
        feats_thr = {sid: build_features(caches[sid], yolo_conf_thr=thr) for sid in single_ids}
        rows_thr = pairwise_matrix(feats_thr, baseline_bodycore=baseline_bodycore)
        cross_thr = [r for r in rows_thr if not r["self"]]
        yb = [r["yolo_bodycore_score"] for r in cross_thr]
        mb = [r["mp_bodycore_score"] for r in cross_thr]
        pose33 = [r["pose33_score"] for r in cross_thr]
        valid_ratios = [f.yolo_bodycore_valid_ratio for f in feats_thr.values()]
        sweep.append(
            {
                "yolo_conf_thr": float(thr),
                "corr_yolo_vs_mp_bodycore": _pearson(yb, mb),
                "mae_yolo_vs_mp_bodycore": _mae(yb, mb),
                "passfail_consistency_yolo_vs_pose33": _passfail_consistency(pose33, yb, PASS_SCORE_THR),
                "passfail_consistency_yolo_vs_mp_bodycore": _passfail_consistency(mb, yb, PASS_SCORE_THR),
                "n_cross_pairs": len(cross_thr),
                "yolo_bodycore_valid_ratio_mean": float(np.mean(valid_ratios)) if valid_ratios else None,
                "yolo_bodycore_valid_ratio_min": float(np.min(valid_ratios)) if valid_ratios else None,
                "yolo_bodycore_skip_ratio_max": float(1.0 - np.min(valid_ratios)) if valid_ratios else None,
                **{f"yolo_{key}": value for key, value in YOLO_ARTIFACT_METADATA.items()},
            }
        )

    # 6) 多人闸门展示（全部样本）。
    gate_rows = []
    for sid in caches:
        c = caches[sid]
        gate = evaluate_multi_person_gate(c.yolo_num_persons_per_frame)
        gate_rows.append(
            {
                "sample_id": sid,
                "declared_multi_person": (c.multi_person if c.multi_person else "unknown_or_false"),
                "max_persons": gate["max_persons"],
                "multi_person_frames": gate["multi_person_frames"],
                "review_required": gate["review_required"],
                "gate_status": gate["gate_status"],
                "in_calibration_set": sid in single_ids,
                **{f"yolo_{key}": value for key, value in YOLO_ARTIFACT_METADATA.items()},
            }
        )

    # 7) 逐样本基础统计（有效帧率/失败帧率/FPS/初始化）。
    per_sample = []
    for sid in caches:
        c = caches[sid]
        f = feats_calib.get(sid) or build_features(c, yolo_conf_thr=args.calib_conf_thr)
        per_sample.append(
            {
                "sample_id": sid,
                "view": c.view,
                "frames_mp": c.mp_frames,
                "frames_yolo": c.yolo_frames,
                "extract_fps_mp": round(c.mp_frames / c.mp_infer_sec, 2) if c.mp_infer_sec > 0 else None,
                "extract_fps_yolo": round(c.yolo_frames / c.yolo_infer_sec, 2) if c.yolo_infer_sec > 0 else None,
                "mp_infer_sec": round(c.mp_infer_sec, 3),
                "yolo_infer_sec": round(c.yolo_infer_sec, 3),
                "mp_fail_ratio": round(f.mp_fail_ratio, 4),
                "yolo_fail_ratio": round(f.yolo_fail_ratio, 4),
                "mp_bodycore_valid_ratio": round(f.mp_bodycore_valid_ratio, 4),
                "yolo_bodycore_valid_ratio": round(f.yolo_bodycore_valid_ratio, 4),
                **{f"yolo_{key}": value for key, value in YOLO_ARTIFACT_METADATA.items()},
            }
        )

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": _env_table(),
        "pose_variant": args.pose_variant,
        "yolo_model": str(args.yolo_model),
        "yolo_artifact_metadata": dict(YOLO_ARTIFACT_METADATA),
        "pose33_baseline": POSE33_BASELINE,
        "calib_conf_thr_used": float(args.calib_conf_thr),
        "single_person_sample_ids": single_ids,
        "student_sample_ids": student_ids,
        "baseline_calibration": calib,
        "metrics": metrics,
        "metrics_with_self_match_sanity": metrics_with_self,
        "yolo_conf_thr_sweep": sweep,
        "multi_person_gate": gate_rows,
        "per_sample": per_sample,
        "pairwise_rows": rows,
    }

    (out_dir / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(out_dir / "calibration_pairwise.csv", rows)
    _write_csv(out_dir / "calibration_conf_sweep.csv", sweep)
    _write_csv(out_dir / "calibration_per_sample.csv", per_sample)
    _write_csv(out_dir / "calibration_multi_person_gate.csv", gate_rows)

    # 控制台摘要。
    print(f"\n[done] 输出目录：{out_dir}")
    print(f"  baseline 标定: {calib}")
    print(f"  metrics（跨视频，不含 self-match）: corr(mp_bc,pose33)={metrics['corr_mp_bodycore_vs_pose33']}, "
          f"corr(yolo_bc,mp_bc)={metrics['corr_yolo_bodycore_vs_mp_bodycore']}, "
          f"MAE(yolo_bc,mp_bc)={metrics['mae_yolo_bodycore_vs_mp_bodycore']}")
    print(f"  pass/fail 一致率(yolo_bc vs pose33)={metrics['passfail_consistency_yolo_bodycore_vs_pose33']}")
    print("  conf sweep:")
    for r in sweep:
        print(f"    thr={r['yolo_conf_thr']:.2f}  corr={r['corr_yolo_vs_mp_bodycore']}  "
              f"MAE={r['mae_yolo_vs_mp_bodycore']}  valid_min={r['yolo_bodycore_valid_ratio_min']}")
    print("  多人闸门:")
    for g in gate_rows:
        print(f"    {g['sample_id']}: max_persons={g['max_persons']} "
              f"multi_frames={g['multi_person_frames']} review_required={g['review_required']}")
    return 0


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="S3 body_core_v1 三路对照 + 阈值标定（Issue #10）")
    p.add_argument("--samples", default="docs/yolo_eval_samples.json")
    p.add_argument("--yolo-model", default="models/yolo11n-pose.pt")
    p.add_argument("--pose-variant", default="full")
    p.add_argument("--out", default="outputs/calib_body_core")
    p.add_argument("--calib-conf-thr", type=float, default=0.6, help="逐样本明细/标定使用的 YOLO valid_conf_thr")
    p.add_argument("--only", default=None, help="只跑指定样本 id（逗号分隔）")
    p.add_argument("--max-frames", type=int, default=None, help="每段最多帧数（调试用）")
    p.add_argument("--force-extract", action="store_true", help="忽略磁盘缓存，强制重新推理")
    return p


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
