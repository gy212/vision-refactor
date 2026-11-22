"""检查标准的名称、身体部位、文字依据与已批准参数（纯数据，按公共域与动作私有域分区）。"""

from __future__ import annotations

DOC = "武术散打得分点.docx 表1"
SUMMARY = "扣分点说明功能需求汇总.md"
TEACHER_SOURCE = "教师确认 #33 issuecomment-6235417498；卷三第7题更正：后手必须收回下颚"
KICK_SOURCE = "低鞭腿动作规则标准 2026-11-14"
HOOK_STEP_SOURCE = "用户2026-11-22确认摆拳与公共步法/姿态/衔接规则标准"
USER_20261105_SOURCE = "用户2026-11-05逐项意见及核对回复"
USER_20261106_SOURCE = "用户2026-11-06确认报告动作正确并要求增加容错；docs/直拳容错调整_20261106.md"
USER_20261108_SOURCE = (
    "用户2026-11-08授权按样本调整站距、优先减少误报；docs/实战式站距调整_20261108.md"
)

PARAMS = {
    "elbow_raise": -0.18,
    "shoulder_height": 0.17,
    "punch_elbow": 168.0,
    "stance_elbow_min": 80.0,
    "stance_knee_max": 179.0,
    "stance_width_min": 0.7,
    "stance_width_max": 1.35,
    "punch_level_deviation": 15.0,
}

HOOK_GUARD_PARAMS = {
    "elbow_raise": PARAMS["elbow_raise"],
    "guard_shoulder_drop": 0.08,
    "guard_back": 0.18,
    "punch_drop": 0.15,
    "punch_pull": 0.22,
    "punch_recovery": 0.34,
}

HOOK_SHOULDER_PARAMS = {
    "shoulder_height": 0.20,
}

KICK_PARAMS = {
    "pad_step_min_disp": 0.45,
    "pad_step_overtake_max": 0.20,
    "pad_step_pause_max": 0.60,
    "kick_height_hip_margin": 0.06,
    "kick_height_knee_margin": 0.08,
    "kick_retract_max_x": 0.40,
    "foot_flexed_min_angle": 165.0,
    "hip_turn_width_ratio": 0.80,
    "head_down_drop": 0.030,
    "body_lean_angle": 35.0,
    "arm_swing_drop": 0.45,
    "arm_swing_min_angle": 115.0,
    "kick_guard_drop_side": 0.35,
    "kick_guard_drop_front": 0.25,
    "kick_guard_back_side": 0.32,
    "kick_guard_back_front": 0.18,
    "kick_guard_elbow_flare": 45.0,
    "kick_guard_jaw_dist": 0.34,
}

# 公共域元数据：(默认名称, 身体部位键, 默认标准描述)
STANCE_CHECK_META: dict[str, tuple[str, str, str]] = {
    "stance_elbow": (
        f"前手肘角不在{PARAMS['stance_elbow_min']:g}°～135°",
        "arm",
        f"前手大小臂夹角{PARAMS['stance_elbow_min']:g}°～135°；已给较小角度增加10°余量",
    ),
    "stance_height": (
        "前手手腕低于肩",
        "hand",
        "前手手腕不低于同侧肩膀",
    ),
    "stance_rear_guard": (
        "后手肘离身体过远",
        "arm",
        "看肘与身体的距离；护下巴另查",
    ),
    "stance_knee": (
        "单腿或双腿伸得过直",
        "knees",
        f"两膝微屈；任一膝角超过{PARAMS['stance_knee_max']:g}°才提醒",
    ),
    "stance_width": (
        "两脚站距过宽或过窄",
        "feet",
        "较窄的脚尖或脚跟间距，与两面中较完整肩宽相比，"
        f"应在{PARAMS['stance_width_min']:.1f}到{PARAMS['stance_width_max']:g}倍",
    ),
    "stance_toes": (
        "后脚外撇过度（脚尖落后于脚跟）",
        "step_foot",
        "实战式后脚斜向前方，侧面看后脚尖不落后于后脚跟（不查前脚）",
    ),
    "stance_torso": (
        "躯干正向前方",
        "torso",
        "同步双视角归一肩宽比超过√3；正交机位近似检查，不表示真实三维角度",
    ),
}

GUARD_CHECK_META: dict[str, tuple[str, str, str]] = {
    "guard_shoulder": (
        "护手掉到肩膀以下",
        "hand",
        "出拳时护脸的手不低于肩膀",
    ),
    "guard_back": (
        "护手肘明显退到身体后方",
        "hand",
        "侧面看护手肘到肩与腿根连线的后方距离；允许从身前收回到身旁",
    ),
    "guard_elbow": (
        "向上抬肘",
        "arm",
        "护手侧保持护位，不向上抬肘",
    ),
    "guard_jaw": (
        "后手未护住下颚",
        "hand",
        "实战式后手靠近下颚；此项不判断夹臂和贴肋",
    ),
}

PUNCH_CHECK_META: dict[str, tuple[str, str, str]] = {
    "punch_predrop": (
        "出拳前先下沉",
        "hand",
        "出拳前不先把拳头向下放；允许收拳时从下方绕回",
    ),
    "punch_prepull": (
        "出拳前先向后拉手",
        "hand",
        "从护位直接出拳，不先向后预摆",
    ),
    "punch_recover": (
        "后手收回后未护住下颚",
        "hand",
        "后手收回后回到下颚，不能停在胸口；不限制收拳路径",
    ),
    "shoulder_level": (
        "出拳未与肩部齐平",
        "arm",
        "出拳与肩部齐平",
    ),
    "punch_unrecovered": (
        "出拳后未及时收回",
        "hand",
        "出拳打到峰值后应及时收回护位，不得停在外面超过0.45秒或保持不收",
    ),
    "combo_punch_pause": (
        "前后手衔接停顿过长",
        "hand",
        "组合拳第一拳收完到第二拳起动之间的停顿空档不超过0.25秒，允许边收边出",
    ),
}

BODY_POSTURE_CHECK_META: dict[str, tuple[str, str, str]] = {
    "head_down": (
        "踢腿时低头",
        "head",
        "侧面观察鼻相对双肩中点垂向位置相对起始护位下压不超过"
        f"{KICK_PARAMS['head_down_drop']:.3f}倍躯干长",
    ),
    "body_lean": (
        "踢腿时上体明显侧倾",
        "torso",
        f"正面双肩连线与水平夹角不超过{KICK_PARAMS['body_lean_angle']:g}°",
    ),
    "trunk_lean_sagittal": (
        "出拳时上体前倾或后仰",
        "torso",
        "侧面看肩中点到髋中点连线相对竖直线前倾或后仰不超过10°",
    ),
    "hip_turn": (
        "未充分翻胯",
        "hip",
        f"正面两髋水平宽相对起始护位需明显收窄（比值不高于{KICK_PARAMS['hip_turn_width_ratio']:.2f}）",
    ),
    "pivot_heel": (
        "支撑脚未碾脚内旋",
        "step_foot",
        "踢腿击打过程中支撑脚应以前脚掌为轴充分内旋碾转，脚跟转至领先于脚尖",
    ),
}

STEP_CHECK_META: dict[str, tuple[str, str, str]] = {
    "step_feet_together": (
        "上步时两脚并拢过窄",
        "feet",
        "双摄确认为侧身帧时，侧面两脚跟水平距离不小于1.0倍侧面肩宽投影",
    ),
    "step_late_landing": (
        "脚步落稳晚于出拳",
        "step_foot",
        "上步或跟步的脚步落稳时间不晚于对应出拳打到瞬间超过0.20秒",
    ),
    "step_follow_missing": (
        "后手出拳时后脚未跟步",
        "step_foot",
        "上步组合拳过程中后脚向攻击方向跟步推进不少于0.25倍躯干长",
    ),
    "step_overtake": (
        "跟步时后脚超过前脚",
        "step_foot",
        "上步组合拳跟步过程中后脚踝不得超越前脚踝超过0.02倍躯干长",
    ),
}

# 动作独有私有域元数据
STRAIGHT_PRIVATE_META: dict[str, tuple[str, str, str]] = {
    "punch_extension": (
        "直拳出拳时肘未伸直",
        "arm",
        f"侧面看每次出拳最伸展时的肘角；{PARAMS['punch_elbow']:g}°及以上不提示弯肘"
        f"（距180°留{180 - PARAMS['punch_elbow']:g}°余量）",
    ),
    "punch_preelbow": (
        "出拳前先抬肘",
        "arm",
        "后手直拳起动前不先将肘抬至肩高附近",
    ),
}

HOOK_PRIVATE_META: dict[str, tuple[str, str, str]] = {
    "hook_arm_bend": (
        "摆拳大小臂弯屈角度不当",
        "arm",
        "侧面看打到瞬间小臂投影与大臂长度之比：腕在肘后不超过0.50（防弯太紧），"
        "腕在肘前不超过0.70（防直臂甩大臂）",
    ),
    "hook_forearm_level": (
        "摆拳小臂未保持水平",
        "arm",
        "正面看打到瞬间小臂（肘到腕）与水平线夹角不超过10°",
    ),
    "hook_over_center": (
        "摆拳摆过身体中线过多",
        "hand",
        "正面看打到瞬间手腕越过两肩中点到达对侧方向不超过0.30倍正面肩宽",
    ),
}

KICK_PRIVATE_META: dict[str, tuple[str, str, str]] = {
    "pad_step_missing": (
        "未做后脚垫步",
        "step_foot",
        f"上步前低鞭腿起腿前，后脚向攻击方向推进不少于{KICK_PARAMS['pad_step_min_disp']:.2f}倍躯干长",
    ),
    "pad_step_overtake": (
        "垫步后脚超越前脚",
        "step_foot",
        "垫步结束临起腿时，允许并拢到与前脚并齐，但后脚踝不得超越前脚踝"
        f"超过{KICK_PARAMS['pad_step_overtake_max']:.2f}倍躯干长",
    ),
    "pad_step_pause": (
        "垫步落稳后停顿过长",
        "step_foot",
        f"完成垫步落稳后应在{KICK_PARAMS['pad_step_pause_max']:.2f}秒内衔接起腿，避免先定步再起脚",
    ),
    "kick_height": (
        "踢击高度过低或过高",
        "leg",
        "踢腿脚尖高度应不高过支撑腿髋，且不明显低于支撑膝高度",
    ),
    "kick_retract": (
        "踢后未及时收腿直接落地",
        "leg",
        f"踢击最高点后落地前应及时回屈小腿，踝相对髋最小前伸不超过{KICK_PARAMS['kick_retract_max_x']:.2f}倍躯干长",
    ),
    "foot_flexed": (
        "脚背未绷直（勾脚尖）",
        "foot",
        f"踢腿过程与发力端膝-踝-脚尖背伸角应达到{KICK_PARAMS['foot_flexed_min_angle']:g}°及以上",
    ),
    "arm_swing": (
        "未顺势向下甩手",
        "arm",
        f"踢腿同侧手臂顺势向下甩臂发力（腕低于肩{KICK_PARAMS['arm_swing_drop']:.2f}倍躯干长"
        f"且臂伸展角不小于{KICK_PARAMS['arm_swing_min_angle']:g}°）",
    ),
}

RULES: dict[str, tuple[str, str, str]] = {
    **STANCE_CHECK_META,
    **GUARD_CHECK_META,
    **PUNCH_CHECK_META,
    **BODY_POSTURE_CHECK_META,
    **STEP_CHECK_META,
    **STRAIGHT_PRIVATE_META,
    **HOOK_PRIVATE_META,
    **KICK_PRIVATE_META,
}

PARTS = {
    "arm": "臂肘",
    "hand": "手",
    "foot": "脚背",
    "step_foot": "脚",
    "leg": "腿",
    "hip": "髋",
    "knees": "双膝",
    "feet": "双脚",
    "torso": "躯干",
    "head": "头部",
}

PRIMARY_VIEW = {
    "guard_back": "side",
    "stance_toes": "side",
    "stance_torso": "front",
    "stance_width": "front",
}
