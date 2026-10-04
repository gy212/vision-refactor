# -*- coding: utf-8 -*-
"""
特征布局注册表（YOLO 迁移 Issue #4 / S1：FeatureLayoutSpec + layout shape 参数化）。

目的
----
把散落在热路径里的 ``(22, 2)`` 硬编码收敛到一处显式的「布局注册」对象，
让缺帧补零、mirror、关节误差统计、周期裁切等步骤都按 layout 取值，而不是
写死 22 点 / BlazePose 11..32。

范围（见 docs/yolo_migration_issues.md Issue #4 / #8）
----------------------------------------------------------------
- ``pose33_v3``：生产默认布局，默认行为完全不变（Issue #4 注册）。
- ``body_core_v1``：YOLO / MediaPipe 共享的躯干四肢核心 12 点布局，
  随 YOLO 离线闭环在 **Issue #8（S2）** 引入——此时才真正有消费者，避免提前建模。
- **不引入** S4 才消费的 ``required_landmarks`` 字段（同样避免提前建模）。

设计要点
--------
- 该模块**不依赖** ``pose_features`` / ``action_compare``，避免循环 import；
  上层（``pose_features.mirror_pose_features`` / ``action_compare``）反向依赖它。
- ``mirror_pairs`` 是「J 维度内」需要左右互换的下标对，与 ``source_indices``
  的具体取值无关，因此任意布局都能用同一套 mirror 逻辑驱动。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureLayoutSpec:
    """一个特征布局的显式定义。

    字段
    ----
    name:
        布局名（如 ``"pose33_v3"``），作为注册表 key 与模板 metadata 标识。
    source_indices:
        该布局从源关键点（BlazePose 33 点）里取用的下标，顺序即布局 J 维顺序。
        ``len(source_indices) == shape[0]``。
    shape:
        归一化特征张量的单帧形状 ``(J, D)``，本项目 ``D == 2``（仅 x/y）。
    mirror_pairs:
        镜像（左右翻转）时需要互换的「J 维下标对」集合。与 source_indices 的
        具体取值无关，只描述布局内部的 L/R 配对。
    joint_names:
        每个 J 维下标对应的关节名（用于误差统计输出），``len == shape[0]``。
    default_baseline:
        该布局 DTW 打分的默认 baseline；``None`` 表示尚未标定。
    """

    name: str
    source_indices: tuple[int, ...]
    shape: tuple[int, int]
    mirror_pairs: tuple[tuple[int, int], ...]
    joint_names: tuple[str, ...]
    default_baseline: float | None = None

    def __post_init__(self) -> None:
        j = int(self.shape[0])
        if len(self.shape) != 2:
            raise ValueError(f"layout {self.name!r} shape 必须为 (J, D)，实际 {self.shape}")
        if len(self.source_indices) != j:
            raise ValueError(
                f"layout {self.name!r}: source_indices 长度 {len(self.source_indices)} "
                f"与 shape[0] {j} 不一致"
            )
        if len(self.joint_names) != j:
            raise ValueError(
                f"layout {self.name!r}: joint_names 长度 {len(self.joint_names)} "
                f"与 shape[0] {j} 不一致"
            )
        for a, b in self.mirror_pairs:
            if not (0 <= a < j and 0 <= b < j):
                raise ValueError(
                    f"layout {self.name!r}: mirror_pairs ({a},{b}) 越界，J={j}"
                )

    @property
    def num_joints(self) -> int:
        return int(self.shape[0])


# --------------------------------------------------------------------------- #
# pose33_v3：旧 MediaPipe 模板布局（BlazePose 11..32，22 点，去掉面部）。
# 关节名 / 顺序与 core/action_compare.py 原 joint_names_11_32 完全一致（L/R 交替）。
# mirror_pairs 即相邻对 (0,1),(2,3),...,(20,21)，与原 mirror_pose_features 行为一致。
# --------------------------------------------------------------------------- #
_POSE33_V3_JOINT_NAMES: tuple[str, ...] = (
    "L_SHOULDER",
    "R_SHOULDER",
    "L_ELBOW",
    "R_ELBOW",
    "L_WRIST",
    "R_WRIST",
    "L_PINKY",
    "R_PINKY",
    "L_INDEX",
    "R_INDEX",
    "L_THUMB",
    "R_THUMB",
    "L_HIP",
    "R_HIP",
    "L_KNEE",
    "R_KNEE",
    "L_ANKLE",
    "R_ANKLE",
    "L_HEEL",
    "R_HEEL",
    "L_FOOT_INDEX",
    "R_FOOT_INDEX",
)

POSE33_V3 = FeatureLayoutSpec(
    name="pose33_v3",
    source_indices=tuple(range(11, 33)),  # BlazePose 11..32 -> 22 点
    shape=(22, 2),
    mirror_pairs=tuple((i, i + 1) for i in range(0, 22, 2)),  # (0,1),(2,3),...,(20,21)
    joint_names=_POSE33_V3_JOINT_NAMES,
    # baseline 从 2.0 → 3.0（2026-07-11 校准）：4 组「同一人连打同一套两遍」的
    # avg_cost 中位数 ≈ 0.333，把这一「人类自然复现上限」锚到 ~0.90 分（3.0/(3.0+0.333)）。
    default_baseline=3.0,
)


# --------------------------------------------------------------------------- #
# body_core_v1：YOLO / MediaPipe 共享的躯干四肢核心布局（12 点，YOLO 迁移 Issue #8 / S2）。
#
# 为什么在此引入（而非 S1）
# ------------------------
# 该布局**只在真接 YOLO 离线闭环时才有消费者**（见 docs/yolo_migration_plan_optimized.md
# 三审拆层说明）。放进 S1 属提前建模，故下移到 S2（#8）随离线闭环一起引入。
#
# 索引（BlazePose33）
# ------------------
# 11/12 肩、13/14 肘、15/16 腕、23/24 髋、25/26 膝、27/28 踝——全部落在 COCO17 可映射点，
# 因此 YOLO 与 MediaPipe 都能诚实产出该布局（不含嘴角/手指/脚跟脚尖等 COCO17 缺失点）。
#
# mirror_pairs
# ------------
# joint_names 按 L/R 交替排列，故 mirror_pairs 即相邻对 (0,1),(2,3),...,(10,11)，
# 与 pose33_v3 同构，可被同一套 mirror 逻辑驱动。
#
# baseline
# --------
# **已在 S3（#10）标定**：default_baseline=1.2826。
# 标定方法（见 docs/yolo_body_core_calibration.md）：在单人样本的跨视频成对匹配上，
# 按 avg_cost 尺度比 baseline = pose33_baseline(2.0) * median(bodycore_avg_cost) /
# median(pose33_avg_cost) = 2.0 * 0.6413 ≈ 1.2826，使 body_core_v1 分数与 pose33_v3
# 分数同尺度、可共享 pass/fail 阈值。标定前的占位 2.0 已移除。
# --------------------------------------------------------------------------- #
_BODY_CORE_V1_JOINT_NAMES: tuple[str, ...] = (
    "L_SHOULDER",
    "R_SHOULDER",
    "L_ELBOW",
    "R_ELBOW",
    "L_WRIST",
    "R_WRIST",
    "L_HIP",
    "R_HIP",
    "L_KNEE",
    "R_KNEE",
    "L_ANKLE",
    "R_ANKLE",
)

# BlazePose33 源下标（肩/肘/腕 11..16 + 髋/膝/踝 23..28），顺序即布局 J 维顺序。
_BODY_CORE_V1_SOURCE_INDICES: tuple[int, ...] = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)

BODY_CORE_V1 = FeatureLayoutSpec(
    name="body_core_v1",
    source_indices=_BODY_CORE_V1_SOURCE_INDICES,
    shape=(12, 2),
    mirror_pairs=tuple((i, i + 1) for i in range(0, 12, 2)),  # (0,1),(2,3),...,(10,11)
    joint_names=_BODY_CORE_V1_JOINT_NAMES,
    # 已在 S3（#10）标定：按尺度对齐反推（见上方说明与 docs/yolo_body_core_calibration.md）。
    default_baseline=1.2826,
)


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, FeatureLayoutSpec] = {}

# 周期裁切等「按 shape 泛化」的步骤接受的最小关节数下限。
# 低于该值的 (J,2) 视为非法布局，不做周期裁切（避免对退化输入做无意义切分）。
MIN_LAYOUT_JOINTS: int = 4


def register_layout(spec: FeatureLayoutSpec) -> FeatureLayoutSpec:
    """注册一个布局；重复注册同名且不同定义时报错。"""
    existing = _REGISTRY.get(spec.name)
    if existing is not None and existing != spec:
        raise ValueError(f"layout {spec.name!r} 已注册且定义不同，拒绝覆盖")
    _REGISTRY[spec.name] = spec
    return spec


def get_layout(name: str) -> FeatureLayoutSpec:
    """按名取布局；未注册时报清晰错误。"""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"未注册的 feature layout：{name!r}（已注册：{sorted(_REGISTRY)}）"
        ) from None


def has_layout(name: str) -> bool:
    return name in _REGISTRY


def resolve_layout_by_shape(shape: tuple[int, ...]) -> FeatureLayoutSpec | None:
    """按单帧 shape ``(J, D)`` 反查唯一匹配的已注册布局。

    - 命中唯一布局：返回该布局。
    - 无匹配 / 多个布局共享同一 shape（本期不会发生）：返回 ``None``，
      由调用方决定是否报错或要求显式传入 layout。
    """
    shape = tuple(int(x) for x in shape)
    matches = [spec for spec in _REGISTRY.values() if tuple(spec.shape) == shape]
    if len(matches) == 1:
        return matches[0]
    return None


def is_valid_feature_shape(shape: tuple[int, ...], *, min_joints: int = MIN_LAYOUT_JOINTS) -> bool:
    """判断单帧 shape 是否为合法 ``(J, 2)`` 布局（``J >= min_joints``）。

    供 ``_select_representative_cycle`` 等「按 shape 泛化」的热路径使用——
    它只需知道输入是否为合法 ``(J,2)``，无需知道该布局是否已在注册表登记，
    因此能被任意 ``(T,J,2)`` 输入驱动（含测试期构造的 dummy 布局）。
    """
    return len(shape) == 2 and int(shape[1]) == 2 and int(shape[0]) >= int(min_joints)


register_layout(POSE33_V3)
register_layout(BODY_CORE_V1)
