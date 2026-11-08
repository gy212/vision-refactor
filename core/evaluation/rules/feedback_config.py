"""练习评估参数的读取与外部配置校验。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math

from core.evaluation.rules.rule_descriptions import PARAMS, RULES
from core.evaluation.view_weights import check_front_weights
from core.foundation.paths import repo_root


def default_front_weights() -> dict:
    """各动作族、检查项的正面权重初值（工程初值，非准确率/概率；后续按教师标注的双摄样本标定）。"""
    return {
        "stance": {
            "default": 0.5,
            "stance_elbow": 0.2,
            "stance_knee": 0.2,
            "stance_rear_guard": 0.6,
        },
        "straight": {"default": 0.4, "guard_low": 0.7, "guard_elbow": 0.7, "shoulder_level": 0.7},
        "hook": {"default": 0.6, "guard_low": 0.75, "guard_elbow": 0.7, "shoulder_level": 0.6},
        "kick": {
            "default": 0.4,
            "guard_low": 0.65,
            "guard_elbow": 0.65,
            "foot_flexed": 0.15,
            "hip_turn": 0.7,
            "arm_swing": 0.6,
        },
    }


@dataclass(frozen=True)
class FeedbackConfig:
    """工程初始阈值，保留标定入口；除肘角范围外不冒充教师确认的标准。"""

    sample_fps: float = 0.0  # 0=原帧率逐帧，不主动抽帧
    max_edge: int = 1920
    pose_variant: str = "full"
    min_valid: int = 3
    valid_ratio: float = 0.6
    trigger_ratio: float = 0.3
    motion_distance: float = 0.25  # 相对躯干长度
    guard_drop: float = 0.25
    guard_back: float = 0.18
    shoulder_height: float = 0.20
    elbow_raise: float = 0.10  # 肘高于肩超过该值（躯干长度）即算抬肘；可为负，见 straight_policy
    torso_front_ratio: float = 0.75
    foot_angle: float = 145.0
    hip_rotation: float = 20.0
    fusion_support: float = 0.7
    phase_tolerance_seconds: float = 0.4  # 仅兼容旧配置；共享阶段覆盖改按真实采样时间逐次检查
    stance_window_seconds: float = 0.2
    min_motion_seconds: float = 0.06
    return_distance: float = 0.15
    max_gap_seconds: float = 0.10
    finish_window_seconds: float = 0.05
    fist_height: float = 0.08  # 拳峰与鼻尖纵向差上限（躯干长度，原正式规则，暂定）
    stance_width_min: float = PARAMS["stance_width_min"]  # 较窄脚间距/较完整肩宽下限
    stance_width_max: float = PARAMS["stance_width_max"]  # 较窄脚间距/较完整肩宽上限
    punch_elbow: float = PARAMS["punch_elbow"]  # 每次直拳最伸展时的肘角下限（度）
    guard_wrist_height: float = 0.5  # 实战式护位：手腕不低于肩下 0.5 倍躯干长，排除直立垂手的帧
    guard_shoulder_drop: float = (
        0.08  # 护手手腕低于同侧肩超过该值（躯干长度）算掉到肩膀以下（第二批样本标定）
    )
    punch_drop: float = 0.15  # 出拳前手腕相对起始护位向下偏移（待独立错误样本验证）
    punch_pull: float = 0.22  # 出拳前手腕相对同侧肩向后偏移（躯干长度）
    punch_elbow_raise: float = -0.10  # 出拳前肘高于肩下0.1倍躯干长（侧面）
    punch_recovery: float = 0.34  # 收回后手腕距口部上限（正面，躯干长度）
    front_weights: dict = field(default_factory=default_front_weights)
    stance_elbow_min: float = 90.0  # 上步直拳容错由 straight_policy 单独覆盖
    stance_knee_max: float = 170.0
    punch_level_deviation: float = 5.0

    def __post_init__(self):
        if any(
            not math.isfinite(v) or v <= 0
            for k, v in asdict(self).items()
            if k
            not in {
                "front_weights",
                "pose_variant",
                "sample_fps",
                "elbow_raise",
                "punch_elbow_raise",
            }
        ):
            raise ValueError("规则参数必须是有限正数")
        if not math.isfinite(self.elbow_raise) or not -1 < self.elbow_raise < 1:
            raise ValueError(
                "抬肘阈值必须在-1～1倍躯干长之间（负值表示肘低于肩不足该距离即算抬肘）"
            )
        if not math.isfinite(self.punch_elbow_raise) or not -1 < self.punch_elbow_raise < 1:
            raise ValueError("出拳前抬肘阈值必须在-1～1倍躯干长之间")
        if (
            not math.isfinite(self.sample_fps)
            or not 0 <= self.sample_fps <= 60
            or self.max_edge < 64
            or self.min_valid < 2
            or not isinstance(self.min_valid, int)
        ):
            raise ValueError("采样频率为0（逐帧）或不超过60，分辨率至少64，有效帧数至少2")
        if self.pose_variant not in {"lite", "full", "heavy"}:
            raise ValueError("模型只能选择lite/full/heavy")
        if self.return_distance >= self.motion_distance:
            raise ValueError("回收阈值必须小于动作起动阈值")
        if self.stance_width_min >= self.stance_width_max:
            raise ValueError("站距下限必须小于上限")
        if (
            self.stance_elbow_min >= 135
            or self.stance_knee_max >= 180
            or self.punch_level_deviation >= 90
        ):
            raise ValueError("站姿肘角下限须小于135°，膝角上限须小于180°，齐肩余量须小于90°")
        if self.valid_ratio > 1 or self.trigger_ratio > 1:
            raise ValueError("比例参数不能大于1")
        if not 0.5 < self.fusion_support <= 1:
            raise ValueError("融合支持阈值必须大于0.5且不超过1")
        check_front_weights(self.front_weights, RULES)


def load_feedback_config() -> FeedbackConfig:
    """练习 profile 的阈值：``repo_root()/feedback_config.json`` 存在时按其字段覆盖默认值。"""
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
