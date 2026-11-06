"""检查标准的名称、文字依据与已批准参数。"""

from __future__ import annotations

TEACHER_SOURCE = "教师确认 #33 issuecomment-6235417498；卷三第7题更正：后手必须收回下颚"

PARAMS = {
    "elbow_raise": -0.18,
    "shoulder_height": 0.17,
    "punch_elbow": 177.0,
    "stance_elbow_min": 80.0,
    "stance_knee_max": 175.0,
    "punch_level_deviation": 15.0,
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
    "punch_extension": ("直拳出拳时肘未伸直", "侧面看出拳肘角；177°及以上不提示弯肘"),
    "guard_back": (
        "护手肘明显退到身体后方",
        "侧面看护手肘到肩与腿根连线的后方距离；允许从身前收回到身旁",
    ),
}

DOC = "武术散打得分点.docx 表1"

SUMMARY = "扣分点说明功能需求汇总.md"

RULES = {
    "stance_elbow": ("前手肘角不在90°～135°", "front_arm", "前手大小臂夹角90°～135°", ""),
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
        "两脚略与肩同宽：踝距/肩宽在0.8～1.2（暂定，现场微调）",
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
        "直拳出拳结束时肘接近伸直（肘角不小于160°）",
        "",
    ),
    "guard_recover": ("未回收护住下颚", "hand", "前手回收护住下颚", ""),
    "foot_flexed": ("脚背未绷直（勾脚尖）", "foot", "低鞭腿脚背绷直", ""),
    "hip_turn": ("未翻胯", "hip", "低鞭腿翻胯", ""),
    "arm_swing": ("未顺势向下甩手", "arm", "踢腿同侧手顺势向下甩手", ""),
    **STRAIGHT_RULES,
}

STANCE_RULES = tuple(code for code in RULES if code.startswith("stance_"))

PARTS = {
    "arm": "臂肘",
    "hand": "手",
    "foot": "脚背",
    "hip": "髋",
    "knees": "双膝",
    "feet": "双脚",
    "torso": "躯干",
    "head_torso": "头部与胸腹",
}

PRIMARY_VIEW = {
    "guard_back": "side",
    "stance_toes": "front",
    "stance_torso": "front",
    "stance_width": "front",
}

FORMAL = "原正式规则 formal_v1（D-3 并入）"
