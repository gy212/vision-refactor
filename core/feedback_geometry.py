"""学生练习的 MediaPipe CPU 逐帧推理及几何/时序规则。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from core.paths import models_dir, repo_root
from core.pose_features import derive_valid_mask
from core.rule_scoring import _angle_deg, _rule_feet_parallel, _rule_knee_slight_bend
from core.vision_pipeline import MediaPipePipeline, PipelineConfig
from core.feedback_phases import (phase_windows, align_views, share_phase_timing, contiguous_runs,
                                  torso_length as _torso, limb_indices as _indices)


def _default_front_weights() -> dict:
    # 工程初值，非准确率/概率；后续按教师标注的双摄样本标定。
    return {
        "stance": {"default": .5, "stance_elbow": .2, "stance_knee": .2,
                   "stance_rear_guard": .6},
        "straight": {"default": .4, "guard_low": .7, "guard_elbow": .7, "shoulder_level": .7},
        "hook": {"default": .6, "guard_low": .75, "guard_elbow": .7, "shoulder_level": .6},
        "kick": {"default": .4, "guard_low": .65, "guard_elbow": .65,
                 "foot_flexed": .15, "hip_turn": .7, "arm_swing": .6},
    }


# 这些现有公式依赖固定观察方向，不能把另一视角同一公式的数值硬凑进来。
_PRIMARY_VIEW = {"guard_back": "side", "stance_toes": "front", "stance_torso": "front"}


@dataclass(frozen=True)
class FeedbackConfig:
    """工程初始阈值，保留标定入口；除肘角范围外不冒充教师确认的标准。"""
    sample_fps: float = 0.0  # 0=原帧率逐帧，不主动抽帧
    max_edge: int = 1920
    pose_variant: str = "full"
    min_valid: int = 3
    valid_ratio: float = .6
    trigger_ratio: float = .3
    motion_distance: float = .25  # 相对躯干长度
    guard_drop: float = .25
    guard_back: float = .18
    shoulder_height: float = .20
    elbow_raise: float = .10
    torso_front_ratio: float = .75
    foot_angle: float = 145.0
    hip_rotation: float = 20.0
    fusion_support: float = .7
    phase_tolerance_seconds: float = .4
    stance_window_seconds: float = .2
    min_motion_seconds: float = .06
    return_distance: float = .15
    max_gap_seconds: float = .10
    finish_window_seconds: float = .05
    front_weights: dict = field(default_factory=_default_front_weights)

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for k, v in asdict(self).items()
               if k not in {"front_weights", "pose_variant", "sample_fps"}):
            raise ValueError("规则参数必须是有限正数")
        if not math.isfinite(self.sample_fps) or not 0 <= self.sample_fps <= 60 or self.max_edge < 64 or self.min_valid < 2 or not isinstance(self.min_valid, int):
            raise ValueError("采样频率为0（逐帧）或不超过60，分辨率至少64，有效帧数至少2")
        if self.pose_variant not in {"lite", "full", "heavy"}:
            raise ValueError("模型只能选择lite/full/heavy")
        if self.return_distance >= self.motion_distance:
            raise ValueError("回收阈值必须小于动作起动阈值")
        if self.valid_ratio > 1 or self.trigger_ratio > 1:
            raise ValueError("比例参数不能大于1")
        if not .5 < self.fusion_support <= 1:
            raise ValueError("融合支持阈值必须大于0.5且不超过1")
        from core.action_feedback import RULES
        if not isinstance(self.front_weights, dict) or set(self.front_weights) != set(_default_front_weights()):
            raise ValueError("视角权重必须包含 stance/straight/hook/kick 四类")
        for weights in self.front_weights.values():
            if not isinstance(weights, dict) or "default" not in weights or set(weights) - {"default", *RULES}:
                raise ValueError("视角权重必须包含default且使用已知检查项名称")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   or not 0 < v < 1 for v in weights.values()):
                raise ValueError("双视角的正面权重必须大于0且小于1，侧面权重为1减正面权重")


def load_feedback_config() -> FeedbackConfig:
    path = repo_root() / "feedback_config.json"
    if not path.exists():
        return FeedbackConfig()
    try:
        values = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(values, dict):
            raise ValueError("配置应为JSON对象")
        return FeedbackConfig(**values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"学生练习配置无效：{path}；{exc}") from exc


def _view_weights(check: dict, cfg: FeedbackConfig) -> dict[str, float]:
    if check["code"] in _PRIMARY_VIEW:
        return {view: float(view == _PRIMARY_VIEW[check["code"]]) for view in ("front", "side")}
    segment = check["segment"]
    family = "stance" if segment == "stance" else "kick" if "kick" in segment else "hook" if "hook" in segment else "straight"
    weights = cfg.front_weights[family]
    front = weights.get(check["code"], weights["default"])
    return {"front": front, "side": 1 - front}


def _pipeline(cfg: FeedbackConfig | None = None):
    cfg = cfg or FeedbackConfig()
    directory = models_dir()
    if not (directory / f"pose_landmarker_{cfg.pose_variant}.task").is_file():
        raise FileNotFoundError(f"缺少 MediaPipe {cfg.pose_variant}，请在设置的模型管理中安装。")
    return MediaPipePipeline(models_dir=directory, cfg=PipelineConfig(
        pose_variant=cfg.pose_variant, enable_hands=False, delegate="cpu", running_mode="video"))


def extract_series(path: Path, cfg: FeedbackConfig, stopped: Callable[[], bool], pipeline_factory=None) -> dict:
    """默认原帧率逐帧推理，只保留关键点。丢失帧全无效，绝不复用上一帧。"""
    started = time.monotonic()
    cap = cv2.VideoCapture(str(path))
    pipe = None
    try:
        if not cap.isOpened():
            raise ValueError(f"视频无法读取，请重新录制：{path.name}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("视频帧率不可用，请重新录制")
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total = int(total) if math.isfinite(total) else 0
        stride = max(1, math.ceil(fps / cfg.sample_fps)) if cfg.sample_fps else 1
        points, frames, times = [], [], []
        index = 0
        timestamp = -1
        while cap.grab():
            if stopped():
                raise InterruptedError("分析已取消")
            if index % stride == 0 or index == total - 1:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    raise ValueError("视频存在不可解码的帧，请重新录制")
                h, w = frame.shape[:2]
                if max(h, w) > cfg.max_edge:
                    scale = cfg.max_edge / max(h, w)
                    frame = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))))
                if pipe is None:
                    pipe = pipeline_factory() if pipeline_factory else _pipeline(cfg)
                timestamp = max(timestamp + 1, round(index * 1000 / fps))
                landmarks, _ = pipe.infer(frame, timestamp_ms=timestamp)
                arr = np.zeros((33, 4), dtype=np.float32)
                if landmarks is not None:
                    for j, lm in enumerate(landmarks[:33]):
                        # x和z统一为图像高度单位，防止非正方形画面把角度算歪。
                        arr[j] = (lm.x * w / h, lm.y, lm.z * w / h, getattr(lm, "visibility", 0))
                points.append(arr)
                frames.append(index)
                times.append(index / fps)
            index += 1
        if not points:
            raise ValueError("视频没有可读取的帧，请重新录制")
        lm = np.stack(points)
        mask = derive_valid_mask(lm) & np.isfinite(lm[:, :, :3]).all(axis=2)
        mask &= (lm[:, :, 0] >= 0) & (lm[:, :, 0] <= w / h) & (lm[:, :, 1] >= 0) & (lm[:, :, 1] <= 1)
        return {"landmarks": lm, "valid_mask": mask, "frames": frames, "times": times,
                "capture": {"sourceFps": fps, "sourceFrames": index, "analyzedFrames": len(frames),
                            "stride": stride, "width": w, "height": h,
                            "inferenceWidth": frame.shape[1], "inferenceHeight": frame.shape[0],
                            "analysisSeconds": round(time.monotonic() - started, 3)}}
    finally:
        cap.release()
        if pipe is not None:
            pipe.close()



def _measure(check: dict, lm: np.ndarray, mask: np.ndarray, baseline: np.ndarray, baseline_mask: np.ndarray,
             stance: str, cfg: FeedbackConfig, details: dict | None = None):
    details = details if details is not None else {}
    code = check["code"]
    sh, el, wr, hip, knee, ankle, heel, toe = _indices(check["role"], stance)
    torso = _torso(lm, mask)
    required = []
    if code == "stance_knee":
        bad, valid, _ = _rule_knee_slight_bend(lm, mask)
        for label, joints in (("左膝角", (23, 25, 27)), ("右膝角", (24, 26, 28))):
            details[label] = (np.array([_angle_deg(p[joints[0], :2], p[joints[1], :2], p[joints[2], :2]) for p in lm]), "度")
        for a, b in ((23, 25), (25, 27), (24, 26), (26, 28)):
            valid &= np.linalg.norm(lm[:, a, :2] - lm[:, b, :2], axis=1) > 1e-4
        return bad, valid
    if code == "stance_toes":
        bad, valid, _ = _rule_feet_parallel(lm, mask)
        valid &= (np.linalg.norm(lm[:, 31, :2] - lm[:, 29, :2], axis=1) > 1e-4)
        valid &= (np.linalg.norm(lm[:, 32, :2] - lm[:, 30, :2], axis=1) > 1e-4)
        return bad, valid
    if code == "stance_elbow":
        required += [sh, el, wr]
        angles = np.array([_angle_deg(p[sh, :2], p[el, :2], p[wr, :2]) for p in lm])
        details["前手肘角"] = (angles, "度")
        bad = (angles < 90) | (angles > 135)
        mask = mask.copy()
        mask[:, el] &= (np.linalg.norm(lm[:, sh, :2] - lm[:, el, :2], axis=1) > 1e-4)
        mask[:, wr] &= (np.linalg.norm(lm[:, wr, :2] - lm[:, el, :2], axis=1) > 1e-4)
    elif code in {"stance_rear_guard", "guard_low", "guard_recover"}:
        required += [wr, 9, 10]
        mouth = (lm[:, 9, :2] + lm[:, 10, :2]) / 2
        if code == "guard_low":
            details["手腕低于口部"] = ((lm[:, wr, 1] - mouth[:, 1]) / np.maximum(torso, 1e-4), "躯干长度")
            bad = lm[:, wr, 1] - mouth[:, 1] > cfg.guard_drop * torso
        else:
            details["手腕距口部"] = (np.linalg.norm(lm[:, wr, :2] - mouth, axis=1) / np.maximum(torso, 1e-4), "躯干长度")
            bad = np.linalg.norm(lm[:, wr, :2] - mouth, axis=1) > (.45 if code == "stance_rear_guard" else .6) * torso
            if code == "stance_rear_guard":
                required += [el, sh, hip]
                center = (lm[:, sh, :2] + lm[:, hip, :2]) / 2
                bad |= np.linalg.norm(lm[:, el, :2] - center, axis=1) > .5 * torso
    elif code in {"guard_elbow", "shoulder_level"}:
        joint = el if code == "guard_elbow" else wr
        required += [sh, joint]
        bad = (lm[:, sh, 1] - lm[:, el, 1] > cfg.elbow_raise * torso) if code == "guard_elbow" else (np.abs(lm[:, wr, 1] - lm[:, sh, 1]) > cfg.shoulder_height * torso)
        details["肘高于肩" if code == "guard_elbow" else "拳肩高度差"] = (
            (lm[:, sh, 1] - lm[:, el, 1]) / np.maximum(torso, 1e-4) if code == "guard_elbow" else
            np.abs(lm[:, wr, 1] - lm[:, sh, 1]) / np.maximum(torso, 1e-4), "躯干长度")
    elif code == "stance_torso":
        required += [11, 12]
        details["投影肩宽"] = (np.linalg.norm(lm[:, 11, :2] - lm[:, 12, :2], axis=1) / np.maximum(torso, 1e-4), "躯干长度")
        bad = np.linalg.norm(lm[:, 11, :2] - lm[:, 12, :2], axis=1) > cfg.torso_front_ratio * torso
    elif code == "foot_flexed":
        required += [knee, ankle, toe]
        angles = np.array([_angle_deg(p[knee, :2], p[ankle, :2], p[toe, :2]) for p in lm])
        details["膝踝脚尖角"] = (angles, "度")
        bad = angles < cfg.foot_angle
        mask = mask.copy()
        mask[:, toe] &= np.linalg.norm(lm[:, toe, :2] - lm[:, ankle, :2], axis=1) > 1e-4
    elif code in {"guard_back", "arm_swing", "hip_turn"}:
        required += [wr, sh, hip, 0] if code == "guard_back" else [wr, sh] if code == "arm_swing" else [23, 24]
        baseline_torso = _torso(baseline, baseline_mask)
        if not baseline_mask[:, required].all() or np.any(baseline_torso < 1e-4):
            return np.zeros(len(lm), bool), np.zeros(len(lm), bool)
        if code == "hip_turn":
            axis = lm[:, 24, [0, 2]] - lm[:, 23, [0, 2]]
            initial = baseline[:, 24, [0, 2]] - baseline[:, 23, [0, 2]]
            base_angle = math.atan2(float(np.median(initial[:, 1])), float(np.median(initial[:, 0])))
            delta = (np.arctan2(axis[:, 1], axis[:, 0]) - base_angle + np.pi) % (2 * np.pi) - np.pi
            bad = np.abs(np.degrees(delta)) < cfg.hip_rotation
            details["估计髋轴转角"] = (np.abs(np.degrees(delta)), "度（估计深度）")
        elif code == "arm_swing":
            initial = np.median((baseline[:, wr, 1] - baseline[:, sh, 1]) / baseline_torso)
            bad = (lm[:, wr, 1] - lm[:, sh, 1]) / np.maximum(torso, 1e-4) - initial < cfg.motion_distance
            details["手腕下摆位移"] = ((lm[:, wr, 1] - lm[:, sh, 1]) / np.maximum(torso, 1e-4) - initial, "躯干长度")
        else:
            direction = np.median(baseline[:, 0, 0] - baseline[:, hip, 0])
            if abs(direction) < .1 * np.median(baseline_torso):
                return np.zeros(len(lm), bool), np.zeros(len(lm), bool)
            initial = np.median((baseline[:, wr, 0] - baseline[:, sh, 0]) / baseline_torso)
            change = ((lm[:, wr, 0] - lm[:, sh, 0]) / np.maximum(torso, 1e-4) - initial) * np.sign(direction)
            bad = change < -cfg.guard_back
            details["护手前后位移"] = (change, "躯干长度")
    else:
        return np.zeros(len(lm), bool), np.zeros(len(lm), bool)
    return bad, mask[:, required].all(axis=1) & (torso > 1e-4)


def evaluate_series(action: str, stance: str, views: dict, cfg: FeedbackConfig | None = None) -> dict:
    from core.action_feedback import build_checks, RULE_VERSION

    cfg = cfg or FeedbackConfig()
    checks = build_checks(action, stance)
    diagnostics = {view: {} for view in views}
    windows = {view: phase_windows(s, action, stance, cfg, diagnostics[view]) for view, s in views.items()}
    alignment = align_views(views, windows, cfg)
    phase_sources = share_phase_timing(views, windows, alignment, diagnostics, cfg)
    evidence = {}
    for c in checks:
        c.update(status="unable", reason="关键点、视角或动作阶段不足，请重新录制", evidence=[], review="pending")
        if c["blockedReason"]:
            c.update(status="pending_rule", reason=c["blockedReason"])
            continue
        weights = _view_weights(c, cfg)
        c["fusion"] = {"method": "weighted_support" if all(weights.values()) else "primary_view",
                       "baseWeights": weights, "effectiveWeights": {}, "support": None,
                       "threshold": cfg.fusion_support, "calibrationStatus": "unvalidated"}
        c["measurements"] = []
        c["viewProblems"] = {}
        outcomes = {}
        ranges = {}
        for view in ("front", "side"):
            if view not in views:
                continue
            series = views[view]
            indices = windows[view].get((c["segment"], c["phase"]))
            if indices is None or len(indices) < cfg.min_valid:
                c["viewProblems"][view] = "未找到该阶段；" + diagnostics[view].get("reason", "阶段不足")
                continue
            # 共用的只能是同步时间段；每路仍须独立提供有效人体和规则所需的点。
            body_valid = _torso(series["landmarks"][indices], series["valid_mask"][indices]) > 1e-4
            if body_valid.mean() >= cfg.valid_ratio and body_valid.sum() >= cfg.min_valid:
                offset = alignment["sideOffsetSeconds"] if view == "side" else 0
                ranges[view] = [(float(series["times"][run[0]]) + offset, float(series["times"][run[-1]]) + offset)
                                for run in contiguous_runs(indices)]
            else:
                c["viewProblems"][view] = f"该阶段可见肩髋链不足（{body_valid.sum()}/{len(indices)}帧）"
            if weights[view] == 0:
                continue
            baseline = windows[view].get(("stance", "start"), np.arange(min(cfg.min_valid, len(series["times"]))))
            details = {}
            bad, valid = _measure(c, series["landmarks"][indices], series["valid_mask"][indices],
                series["landmarks"][baseline], series["valid_mask"][baseline], stance, cfg, details)
            count = int(valid.sum())
            quality = count / len(indices)
            violation = float((bad & valid).sum() / count) if count else None
            measurement = {"view": view, "validFrames": count, "totalFrames": len(indices),
                           "validRatio": quality, "violationRatio": violation, "baseWeight": weights[view], "values": []}
            for name, (values, unit) in details.items():
                observed = values[valid & np.isfinite(values)]
                if observed.size:
                    measurement["values"].append({"name": name, "unit": unit, "median": float(np.median(observed)),
                                                  "min": float(observed.min()), "max": float(observed.max())})
            c["measurements"].append(measurement)
            if count < cfg.min_valid or count / len(indices) < cfg.valid_ratio:
                c["viewProblems"][view] = f"该项测量有效帧不足（{count}/{len(indices)}帧），见关键点有效比例"
                continue
            # “未做某动作”只能在整个阶段均未出现该动作时成立。
            issue = bool(bad[valid].all()) if c["code"] in {"hip_turn", "arm_swing"} else violation >= cfg.trigger_ratio
            measurement["issueObserved"] = bool(issue)
            outcomes[view] = (issue, weights[view] * quality)
            # 保留首/中/末代表帧及原始骨架，支持教师直接核对，避免保存所有视频图像。
            candidates = indices[valid & bad] if issue else indices[valid]
            for index in candidates[np.unique([0, len(candidates) // 2, len(candidates) - 1])]:
                ref = f"{view}:{series['frames'][index]}"
                evidence[ref] = {"id": ref, "view": view, "frame": series["frames"][index], "timeSeconds": series["times"][index],
                                 "landmarks": np.round(series["landmarks"][index], 5).tolist(),
                                 "validMask": series["valid_mask"][index].tolist()}
                c["evidence"].append(ref)
        if set(views) != {"front", "side"}:
            c["reason"] = "缺少正面或侧面视频；仅保留单视角观测，不能给出联合结论"
            continue
        if set(ranges) != {"front", "side"}:
            c["reason"] = "；".join(("正面" if v == "front" else "侧面") + "：" + c["viewProblems"].get(v, "阶段不足")
                                  for v in ("front", "side") if v not in ranges)
            continue
        if not alignment["reliable"]:
            c["reason"] = alignment.get("reason", "正侧面时间无法对应")
            continue
        if len(ranges["front"]) != len(ranges["side"]) or any(
            max(abs(a - b) for a, b in zip(front, side)) > cfg.phase_tolerance_seconds
            for front, side in zip(ranges["front"], ranges["side"])
        ):
            c["reason"] = "正侧面动作阶段时间不对应，请使用同一次同步录制的视频"
            continue
        if any(weight > 0 and view not in outcomes for view, weight in weights.items()):
            c["reason"] = "；".join(("正面" if v == "front" else "侧面") + "：" + problem for v, problem in c["viewProblems"].items())
            continue
        total = sum(value[1] for value in outcomes.values())
        effective = {view: value[1] / total for view, value in outcomes.items()}
        support = sum(effective[view] * value[0] for view, value in outcomes.items())
        c["fusion"].update(effectiveWeights=effective, support=float(support))
        if support >= cfg.fusion_support - 1e-9:
            c["status"] = "candidate"
        elif support <= 1 - cfg.fusion_support + 1e-9:
            c["status"] = "not_observed"
        else:
            c["reason"] = "正侧加权证据接近，暂不能确定，请教师复核"
            continue
        c["reason"] = ("按动作与检查项权重、有效帧比例融合；权重待标定"
                       if c["fusion"]["method"] == "weighted_support" else
                       "该公式仅适用" + ("正面" if weights["front"] else "侧面") + "，另一视角只核对阶段；不作为双视角一致结论")
    unable = any(c["status"] == "unable" for c in checks)
    issues = any(c["status"] == "candidate" for c in checks)
    return {"schemaVersion": 1, "ruleVersion": RULE_VERSION, "action": action, "stance": stance,
            "backend": "mediapipe", "poseVariant": cfg.pose_variant, "delegate": "cpu", "actionSource": "user_selected",
            "fusionMethod": "action_rule_quality_weighted_v1", "inputViews": sorted(views),
            "capture": {view: series.get("capture", {}) for view, series in views.items()}, "alignment": alignment,
            "diagnostics": {view: {**diagnostics[view], "phaseSource": phase_sources[view],
                                    "bodyValidFrames": int((_torso(s["landmarks"], s["valid_mask"]) > 1e-4).sum()),
                                    "totalFrames": len(s["times"]),
                                    "jointValidRatios": s["valid_mask"].mean(axis=0).tolist()}
                            for view, s in views.items()},
            "phaseWindows": {view: [{"segment": segment, "phase": phase,
                                      "ranges": [[float(series["times"][run[0]]), float(series["times"][run[-1]])]
                                                 for run in contiguous_runs(indices)]}
                                     for (segment, phase), indices in windows[view].items()]
                             for view, series in views.items()},
            "config": asdict(cfg), "status": "partial" if any(c["status"] in {"unable", "pending_rule"} for c in checks) else "completed",
            "needsRerecord": unable, "scoreAuthorized": False, "calibrationStatus": "unvalidated",
            "summary": "规则检查发现待教师确认的问题" if issues else "证据不足，请重新录制" if unable else "已检查项目未发现明显问题，不代表全部项目合格",
            "checks": checks, "evidence": list(evidence.values()), "reviewHistory": []}


def analyze_geometry(action, stance, front, side, *, stopped=lambda: False, config=None, pipeline_factory=None):
    from core.action_feedback import build_checks
    build_checks(action, stance)
    cfg = config or load_feedback_config()
    if front is not None and side is not None and Path(front).samefile(Path(side)):
        raise ValueError("正侧面不能使用同一个视频，请选择同次录制的两路视频")
    views = {}
    for view, path in (("front", front), ("side", side)):
        if stopped():
            raise InterruptedError("分析已取消")
        if path is not None:
            views[view] = extract_series(Path(path), cfg, stopped, pipeline_factory)
    if stopped():
        raise InterruptedError("分析已取消")
    return evaluate_series(action, stance, views, cfg)
