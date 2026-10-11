"""差分用的确定性输入：fixture 回放视频、模板（直接由特征写出，不经被测代码）、合成练习序列。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tests import golden_harness as H

FIX_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "pose33_v3"
FPS = 30.0
REAL_CAPTURE = cv2.VideoCapture  # 练习用真实 AVI 解码，模块导入时记下（golden harness 会替换它）


def load_raw(name: str) -> np.ndarray:
    return np.asarray(np.load(FIX_DIR / f"{name}.npz", allow_pickle=True)["landmarks"], dtype=np.float32)


class GapPipeline(H._FakePipeline):
    """NaN 帧回放为未检出（``pose_landmarks=None``）。"""

    def infer(self, frame, *, timestamp_ms=None):
        if np.isnan(frame.landmarks).all():
            return None, []
        return super().infer(frame, timestamp_ms=timestamp_ms)


def fixture_videos() -> dict[str, str]:
    """注册回放视频，返回 名字 → golden key。"""
    student, front, side = load_raw("student_raw"), load_raw("front_src_raw"), load_raw("side_src_raw")
    gappy = student.copy()
    gappy[[0, 1, 40, 41, 42, 43, 44, 90]] = np.nan
    gappy[60:70, 13:17, 3] = 0.2  # 手臂低可见度：规则有效帧与关节误差的可见掩码
    mirror = student.copy()
    mirror[:, :, 0] = 1.0 - mirror[:, :, 0]
    still = np.repeat(student[:1], 150, axis=0)
    long = np.concatenate([still, student, still, student[::-1], still, student, still, still], axis=0)
    out = {}
    for name, arr in (("student", student), ("front", front), ("side", side), ("gappy", gappy),
                      ("mirror", mirror), ("long", long), ("short", student[:8])):
        out[name] = H.register_video(f"golden://p3_{name}.mp4", arr, fps=FPS)
    return out


def template_meta(base: dict | None = None, **over) -> dict:
    meta = dict(np.load(FIX_DIR / "front_template.npz", allow_pickle=True)["meta"].item()) if base is None else dict(base)
    meta.update(over)
    return {k: v for k, v in meta.items() if v is not _DROP}


_DROP = object()


def save_template(path: Path, features: np.ndarray, meta: dict | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=np.asarray(features), meta=np.array(meta, dtype=object))
    return path


def v3_features(raw: np.ndarray) -> np.ndarray:
    """模板特征：逐帧 v3 归一化，未检出帧沿用上一帧（与模板生成同口径，但不经被测入口）。"""
    from core.pose import normalize_pose_xy_v3

    rows, last = [], np.zeros((22, 2), dtype=np.float32)
    for frame in raw:
        lm = [H._FakeLandmark(*map(float, frame[i])) for i in range(33)]
        f = normalize_pose_xy_v3(lm)
        last = last if f is None else f
        rows.append(last)
    return np.stack(rows).astype(np.float32)


def write_templates(work: Path) -> dict[str, Path]:
    """比对、考试池、在线库与契约校验用的模板，全部由 fixture 特征直接写出。"""
    tdir = work / "tpl"
    front = np.load(FIX_DIR / "front_template.npz", allow_pickle=True)
    side = np.load(FIX_DIR / "side_template.npz", allow_pickle=True)
    f_feat, s_feat = front["features"], side["features"]
    student = v3_features(load_raw("student_raw"))
    m = template_meta()
    heavy = template_meta(pose_variant="heavy", model_name="pose_landmarker_heavy")
    t = {
        "front": save_template(tdir / "front.npz", f_feat, front["meta"].item()),
        "side": save_template(tdir / "side.npz", s_feat, side["meta"].item()),
        "cycle": save_template(tdir / "cycle.npz", student[20:170], template_meta(start_frame=20, end_frame=169)),
        "legacy_v1": save_template(tdir / "legacy_v1.npz", f_feat, template_meta(
            m, feature_layout=_DROP, normalizer_version=_DROP, backend=_DROP, validity_policy=_DROP)),
        "legacy_v2": save_template(tdir / "legacy_v2.npz", f_feat, template_meta(
            m, feature_layout="pose_indices_11_32_xy_rot_scale_norm_v2", normalizer_version="v2")),
        "legacy_v3": save_template(tdir / "legacy_v3.npz", s_feat, template_meta(
            m, feature_layout="pose_indices_11_32_xy_rot_scale_norm_v3")),
        "alias": save_template(tdir / "alias.npz", f_feat, template_meta(
            m, feature_layout=_DROP, feature_layout_name="pose33_v3")),
        "image_mode": save_template(tdir / "image_mode.npz", f_feat, template_meta(
            m, running_mode="image", cfg=_DROP)),
        "no_meta": save_template(tdir / "no_meta.npz", f_feat, None),
    }
    pool = tdir / "pool"
    for name, feat in (("直拳", f_feat), ("高鞭腿", student[100:140]), ("侧踹腿", student[30:60]),
                       ("背后七颠", s_feat), ("两手攀足_正面", f_feat), ("两手攀足_侧面", s_feat),
                       ("攒拳怒目_正面", student[60:90])):
        t[f"pool:{name}"] = save_template(pool / f"{name}.npz", feat, heavy)
    bad = tdir / "bad"
    t["bad:full_variant"] = save_template(bad / "full_variant.npz", f_feat, m)
    t["bad:shape"] = save_template(bad / "shape.npz", f_feat[:, :12], heavy)
    t["bad:nan"] = save_template(bad / "nan.npz", np.where(np.arange(22)[None, :, None] == 3, np.nan, f_feat), heavy)
    t["bad:layout"] = save_template(bad / "layout.npz", f_feat, template_meta(heavy, feature_layout="body_core_v1"))
    t["bad:empty_layout"] = save_template(bad / "empty_layout.npz", f_feat, template_meta(heavy, feature_layout=""))
    t["bad:no_meta"] = save_template(bad / "no_meta.npz", f_feat, None)
    (bad / "garbage.npz").write_bytes(b"not a zip")
    (bad / "empty.npz").write_bytes(b"")
    t["bad:garbage"], t["bad:empty"], t["bad:missing"] = bad / "garbage.npz", bad / "empty.npz", bad / "missing.npz"
    online = tdir / "online"
    t["online:good"] = online / "good"
    save_template(online / "good" / "直拳.npz", f_feat, m)
    save_template(online / "good" / "侧踹.npz", student[30:60], m)
    save_template(online / "bad_layout" / "a.npz", f_feat, template_meta(m, feature_layout="pose33_v2"))
    save_template(online / "bad_norm" / "a.npz", f_feat, template_meta(m, normalizer_version="v2"))
    save_template(online / "bad_shape" / "a.npz", f_feat[:, :12], m)
    save_template(online / "bad_meta" / "a.npz", f_feat, None)
    for k in ("bad_layout", "bad_norm", "bad_shape", "bad_meta"):
        t[f"online:{k}"] = online / k
    t["online:missing"] = online / "missing"
    return t
