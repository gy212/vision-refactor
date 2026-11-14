"""检查标准的名称、文字依据与已批准参数。"""

from __future__ import annotations

TEACHER_SOURCE = "教师确认 #33 issuecomment-6235417498；卷三第7题更正：后手必须收回下颚"

PARAMS = {
    "elbow_raise": -0.18,
    "shoulder_height": 0.17,
    "punch_elbow": 168.0,
    "stance_elbow_min": 80.0,
    "stance_knee_max": 175.0,
    "stance_width_min": 0.7,
    "stance_width_max": 1.35,
    "punch_level_deviation": 15.0,
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
    "kick_guard_elbow_flare": 30.0,
    "kick_guard_jaw_dist": 0.34,
}

KICK_STANDARDS = {
    "pad_step_missing": (
        "未做后脚垫步",
        f"上步前低鞭腿起腿前，后脚向攻击方向推进不少于{KICK_PARAMS['pad_step_min_disp']:.2f}倍躯干长",
    ),
    "pad_step_overtake": (
        "垫步后脚超越前脚",
        "垫步结束临起腿时，允许并拢到与前脚并齐，但后脚踝不得超越前脚踝"
        f"超过{KICK_PARAMS['pad_step_overtake_max']:.2f}倍躯干长",
    ),
    "pad_step_pause": (
        "垫步落稳后停顿过长",
        f"完成垫步落稳后应在{KICK_PARAMS['pad_step_pause_max']:.2f}秒内衔接起腿，避免先定步再起脚",
    ),
    "pivot_heel": (
        "支撑脚未碾脚外旋",
        "踢腿击打过程中支撑脚应充分外旋拧转，脚尖转至不领先于脚跟",
    ),
    "kick_height": (
        "踢击高度过低或过高",
        "踢腿脚尖高度应不高过同侧髋，且不明显低于支撑膝高度",
    ),
    "kick_retract": (
        "踢后未及时收腿直接落地",
        f"踢击最高点后落地前应及时回屈小腿，踝相对髋最小前伸不超过{KICK_PARAMS['kick_retract_max_x']:.2f}倍躯干长",
    ),
    "foot_flexed": (
        "脚背未绷直（勾脚尖）",
        f"踢腿过程与发力端膝-踝-脚尖背伸角应达到{KICK_PARAMS['foot_flexed_min_angle']:g}°及以上",
    ),
    "hip_turn": (
        "未充分翻胯",
        f"正面两髋水平宽相对起始护位需明显收窄（比值不高于{KICK_PARAMS['hip_turn_width_ratio']:.2f}）",
    ),
    "head_down": (
        "踢腿时低头",
        "侧面观察鼻相对双肩中点垂向位置相对起始护位下压不超过"
        f"{KICK_PARAMS['head_down_drop']:.3f}倍躯干长",
    ),
    "body_lean": (
        "踢腿时上体明显侧倾",
        f"正面双肩连线与水平夹角不超过{KICK_PARAMS['body_lean_angle']:g}°",
    ),
    "arm_swing": (
        "未顺势向下甩手",
        f"踢腿同侧手臂顺势向下甩臂发力（腕低于肩{KICK_PARAMS['arm_swing_drop']:.2f}倍躯干长"
        f"且臂伸展角不小于{KICK_PARAMS['arm_swing_min_angle']:g}°）",
    ),
    "guard_shoulder": (
        "护手掉到肩膀以下",
        "非踢腿侧护手保持高位防护，手腕不明显低于同侧肩部",
    ),
    "guard_back": (
        "护手肘明显退到身体后方",
        "非踢腿侧护手保持身前防护，肘部不明显退到躯干后方",
    ),
    "guard_elbow": (
        "护手开肘亮肋",
        f"正面非踢腿侧大臂与躯干夹角不超过{KICK_PARAMS['kick_guard_elbow_flare']:g}°",
    ),
    "guard_jaw": (
        "前手未护住下颚",
        f"原地后低鞭腿起腿过程中前手腕距口部不超过{KICK_PARAMS['kick_guard_jaw_dist']:.2f}倍躯干长",
    ),
}

STRAIGHT_RULES = {
    "guard_jaw": (
        "后手未护住下颚",
        "hand",
        "实战式后手靠近下颚；此项不判断夹臂和贴肋",
        "",
    ),
    "punch_predrop": (
        "出拳前先下沉",
        "hand",
        "出拳前不先把拳头向下放；允许收拳时从下方绕回",
        "",
    ),
    "punch_prepull": (
        "出拳前先向后拉手",
        "hand",
        "从护位直接出拳，不先向后预摆",
        "",
    ),
    "punch_preelbow": (
        "出拳前先抬肘",
        "arm",
        "后手直拳起动前不先将肘抬至肩高附近",
        "",
    ),
    "punch_recover": (
        "后手收回后未护住下颚",
        "hand",
        "后手收回后回到下颚，不能停在胸口；不限制收拳路径",
        "",
    ),
}

PREFERRED_VIEW = {
    "stance_height": "front",
    "stance_elbow": "side",
    "stance_toes": "front",
    "stance_width": "front",
    "stance_knee": "side",
    "stance_rear_guard": "front",
    "guard_back": "side",
    "guard_shoulder": "side",
    "guard_elbow": "side",
    "punch_extension": "side",
    "shoulder_level": "side",
    "punch_predrop": "side",
}

REVISED_STANDARDS = {
    "stance_width": (
        "两脚站距过宽或过窄",
        "较窄的脚尖或脚跟间距，与两面中较完整肩宽相比，"
        f"应在{PARAMS['stance_width_min']:.1f}到{PARAMS['stance_width_max']:g}倍",
    ),
    "stance_elbow": (
        f"前手肘角不在{PARAMS['stance_elbow_min']:g}°～135°",
        f"前手大小臂夹角{PARAMS['stance_elbow_min']:g}°～135°；已给较小角度增加10°余量",
    ),
    "stance_knee": (
        "单腿或双腿伸得过直",
        f"两膝微屈；任一膝角超过{PARAMS['stance_knee_max']:g}°才提醒",
    ),
    "stance_height": ("前手手腕低于肩", "前手手腕不低于同侧肩膀"),
    "stance_rear_guard": ("后手肘离身体过远", "看肘与身体的距离；护下巴另查"),
    "punch_extension": (
        "直拳出拳时肘未伸直",
        f"侧面看每次出拳最伸展时的肘角；{PARAMS['punch_elbow']:g}°及以上不提示弯肘"
        f"（距180°留{180 - PARAMS['punch_elbow']:g}°余量）",
    ),
    "guard_back": (
        "护手肘明显退到身体后方",
        "侧面看护手肘到肩与腿根连线的后方距离；允许从身前收回到身旁",
    ),
}

DOC = "武术散打得分点.docx 表1"

SUMMARY = "扣分点说明功能需求汇总.md"

RULES = {
    "stance_elbow": (
        "前手肘角不在90°～135°",
        "front_arm",
        "前手大小臂夹角90°～135°",
        "",
    ),
    "stance_height": (
        "前手拳峰未与鼻尖同高",
        "front_hand",
        "拳峰与鼻尖同高：纵向差不超过0.08倍躯干长（暂定，现场微调）",
        "",
    ),
    "stance_rear_guard": (
        "后手未夹紧并贴近肋骨与下颚",
        "rear_arm",
        "后手大小臂夹紧、紧贴肋骨与下颚",
        "",
    ),
    "stance_knee": ("单腿或双腿伸直", "knees", "两膝微屈", ""),
    "stance_width": (
        "两脚站距过宽或过窄",
        "feet",
        REVISED_STANDARDS["stance_width"][1],
        "",
    ),
    "stance_toes": ("两脚内八或外八", "feet", "两脚大致平行、朝斜前方", ""),
    "stance_torso": ("躯干正向前方", "torso", "躯干侧向前方", ""),
    "stance_head": (
        "挺胸抬头",
        "head_torso",
        "下颚微收、含胸收腹",
        "Pose33不足以可靠判定含胸收腹，需教师查看",
    ),
    "guard_low": ("向下掉手", "hand", "护手保持在下颚附近", ""),
    "guard_back": ("向后拉手", "hand", "护手保持原护位，不向后拉", ""),
    "guard_elbow": ("向上抬肘", "arm", "护手侧保持护位，不向上抬肘", ""),
    "guard_shoulder": ("护手掉到肩膀以下", "hand", "出拳时护脸的手不低于肩膀", ""),
    "vertical_fist": (
        "出拳成立拳",
        "hand",
        "前手直拳出拳结束时拳背朝上",
        "轻量Pose关键点不能可靠判断握拳朝向，需教师查看",
    ),
    "shoulder_level": ("出拳未与肩部齐平", "arm", "出拳与肩部齐平", ""),
    "punch_extension": (
        "直拳出拳时肘未伸直",
        "arm",
        REVISED_STANDARDS["punch_extension"][1],
        "",
    ),
    "guard_recover": ("未回收护住下颚", "hand", "前手回收护住下颚", ""),
    "foot_flexed": ("脚背未绷直（勾脚尖）", "foot", "低鞭腿脚背绷直", ""),
    "hip_turn": ("未翻胯", "hip", "低鞭腿翻胯", ""),
    "arm_swing": ("未顺势向下甩手", "arm", "踢腿同侧手顺势向下甩手", ""),
    "pad_step_missing": (
        KICK_STANDARDS["pad_step_missing"][0],
        "step_foot",
        KICK_STANDARDS["pad_step_missing"][1],
        "",
    ),
    "pad_step_overtake": (
        KICK_STANDARDS["pad_step_overtake"][0],
        "step_foot",
        KICK_STANDARDS["pad_step_overtake"][1],
        "",
    ),
    "pad_step_pause": (
        KICK_STANDARDS["pad_step_pause"][0],
        "step_foot",
        KICK_STANDARDS["pad_step_pause"][1],
        "",
    ),
    "pivot_heel": (
        KICK_STANDARDS["pivot_heel"][0],
        "step_foot",
        KICK_STANDARDS["pivot_heel"][1],
        "",
    ),
    "kick_height": (
        KICK_STANDARDS["kick_height"][0],
        "leg",
        KICK_STANDARDS["kick_height"][1],
        "",
    ),
    "kick_retract": (
        KICK_STANDARDS["kick_retract"][0],
        "leg",
        KICK_STANDARDS["kick_retract"][1],
        "",
    ),
    "head_down": (
        KICK_STANDARDS["head_down"][0],
        "head",
        KICK_STANDARDS["head_down"][1],
        "",
    ),
    "body_lean": (
        KICK_STANDARDS["body_lean"][0],
        "torso",
        KICK_STANDARDS["body_lean"][1],
        "",
    ),
    **STRAIGHT_RULES,
}

STANCE_RULES = tuple(code for code in RULES if code.startswith("stance_"))

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
    "head_torso": "头部与胸腹",
}

PRIMARY_VIEW = {
    "guard_back": "side",
    "stance_toes": "front",
    "stance_torso": "front",
    "stance_width": "front",
}

FORMAL = "原正式规则 formal_v1（D-3 并入）"
