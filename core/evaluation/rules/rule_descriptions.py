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
        f"前手自然架起微屈，大小臂夹角保持在 {PARAMS['stance_elbow_min']:g}°～135°；"
        f"折得太死（小于 {PARAMS['stance_elbow_min']:g}°）或伸得太直（大于 135°）都算错",
    ),
    "stance_height": (
        "前手手腕低于肩",
        "hand",
        "前手抬高护脸，手腕不能掉到同侧肩膀下面",
    ),
    "stance_rear_guard": (
        "后手肘离身体过远",
        "arm",
        "后手手肘往肋部收紧，不能向外张开离上身中段超过半个上身长（护下巴单独另查）",
    ),
    "stance_knee": (
        "单腿或双腿伸得过直",
        "knees",
        f"双腿保持微屈弹性；只要有一条腿几乎蹬直（膝盖夹角超过 {PARAMS['stance_knee_max']:g}°）就算错",
    ),
    "stance_width": (
        "两脚站距过宽或过窄",
        "feet",
        f"两脚站距大约与肩同宽；窄于 {PARAMS['stance_width_min']:.1f} 倍肩宽"
        f"或宽于 {PARAMS['stance_width_max']:g} 倍肩宽都算错",
    ),
    "stance_toes": (
        "后脚外撇过度（脚尖落后于脚跟）",
        "step_foot",
        "实战式后脚保持斜向前方，不能过度向外撇；侧面看后脚尖落后到后脚跟后面就算错（不查前脚）",
    ),
    "stance_torso": (
        "躯干正向前方",
        "torso",
        "实战式要侧身站，胸口不能正朝前方；同一时刻正面看到的肩宽超过侧面的 1.73 倍（约侧身不足 30°）就算错",
    ),
}

GUARD_CHECK_META: dict[str, tuple[str, str, str]] = {
    "guard_shoulder": (
        "护手掉到肩膀以下",
        "hand",
        "出拳时另一只护脸的手要举在肩膀以上；手腕掉到肩下超过约一成上身长（0.08 倍）算掉手",
    ),
    "guard_back": (
        "护手肘明显退到身体后方",
        "hand",
        "护手从身前收回到身旁不算错，但手肘不能往身体背后拉扯超过约五分之一上身长（0.18 倍）",
    ),
    "guard_elbow": (
        "向上抬肘",
        "arm",
        "护手侧手肘自然下垂护肋，不能架起来抬高到快齐平肩膀（肘离肩下不足约五分之一上身长即算抬肘）",
    ),
    "guard_jaw": (
        "后手未护住下颚",
        "hand",
        "后手紧贴下巴旁做好防守；手腕离嘴边超过约三分之一上身长（0.34 倍）算没护住（夹臂贴肋单独另查）",
    ),
}

PUNCH_CHECK_META: dict[str, tuple[str, str, str]] = {
    "punch_predrop": (
        "出拳前先下沉",
        "hand",
        "从护位直接出拳，起动前拳头不能先往下沉（比站好时掉下超过约一成半上身长算错；收拳绕回不算）",
    ),
    "punch_prepull": (
        "出拳前先向后拉手",
        "hand",
        "直接出拳不能先往后拉手蓄力；起动前手腕退到肩膀后面超过约五分之一上身长（0.22 倍）算预摆拉手",
    ),
    "punch_recover": (
        "后手收回后未护住下颚",
        "hand",
        "后手打完必须收回下巴旁，不能停在胸口；收回后手腕离嘴边超过约三分之一上身长（0.34 倍）算没回位",
    ),
    "shoulder_level": (
        "出拳未与肩部齐平",
        "arm",
        f"打到最远瞬间拳头要和肩膀大致齐平；手腕高于或低于肩膀超过约一拳高（{PARAMS['shoulder_height']:.2f} 倍上身长）算偏高或偏低",
    ),
    "punch_unrecovered": (
        "出拳后未及时收回",
        "hand",
        "打完拳要立刻收手回护位；打到最远后停在外面超过 0.45 秒不收，或一直伸着不收回都算错",
    ),
    "combo_punch_pause": (
        "前后手衔接停顿过长",
        "hand",
        "前后手组合要连贯出击（边收前手边出后手最好）；前手收完后停顿超过 0.25 秒才出后手算脱节",
    ),
}

BODY_POSTURE_CHECK_META: dict[str, tuple[str, str, str]] = {
    "head_down": (
        "踢腿时低头",
        "head",
        "动作时目视前方、不要低头看地；侧面看鼻尖相对肩膀的位置比站好时明显下沉"
        f"（超过 {KICK_PARAMS['head_down_drop']:.2f} 倍上身长）算低头",
    ),
    "body_lean": (
        "踢腿时上体明显侧倾",
        "torso",
        f"踢腿时上身尽量稳住、别歪得太狠；正面看两肩连线倾斜超过 {KICK_PARAMS['body_lean_angle']:g}° 算明显侧倾",
    ),
    "trunk_lean_sagittal": (
        "出拳时上体前倾或后仰",
        "torso",
        "出拳时上身保持直立中正；侧面看肩到胯的躯干中线向前趴或向后仰超过 10° 算错",
    ),
    "hip_turn": (
        "未充分翻胯",
        "hip",
        f"发力时要明显转腰翻胯；正面看两胯宽度没比站好时收窄到 {KICK_PARAMS['hip_turn_width_ratio'] * 100:g}% 以内，算没充分转动",
    ),
    "pivot_heel": (
        "支撑脚未碾脚内旋",
        "step_foot",
        "踢腿发力时，支撑脚以前脚掌为轴充分内旋碾地，脚跟要转到脚尖前面；脚跟没转到脚尖前方算未碾脚内旋",
    ),
}

STEP_CHECK_META: dict[str, tuple[str, str, str]] = {
    "step_feet_together": (
        "上步时两脚并拢过窄",
        "feet",
        "上步移动时保持前后步距，不能并脚站成一条线；侧身状态下，侧面看两脚跟前后距离小于 1 倍侧面肩宽算并脚",
    ),
    "step_late_landing": (
        "脚步落稳晚于出拳",
        "step_foot",
        "上步和出拳要协调同步（先落脚或拳脚齐到）；脚落稳比出拳打到顶点晚超过 0.20 秒算步慢手快",
    ),
    "step_follow_missing": (
        "后手出拳时后脚未跟步",
        "step_foot",
        "前脚上步后，后脚要紧跟着向前滑步补位；后脚往前挪动不到四分之一上身长（0.25 倍）算没跟步",
    ),
    "step_overtake": (
        "跟步时后脚超过前脚",
        "step_foot",
        "跟步时后脚留在前脚后面，不能抢到前脚前面去；后脚踝跑到前脚踝前方（超过 0.02 倍上身长）算越位",
    ),
}

# 动作独有私有域元数据
STRAIGHT_PRIVATE_META: dict[str, tuple[str, str, str]] = {
    "punch_extension": (
        "直拳出拳时肘未伸直",
        "arm",
        f"直拳打到最远时手臂要充分伸直；侧面看肘角小于 {PARAMS['punch_elbow']:g}°"
        f"（距完全伸直 180° 留了 {180 - PARAMS['punch_elbow']:g}° 余量）算没伸直",
    ),
    "punch_preelbow": (
        "出拳前先抬肘",
        "arm",
        "后手直拳发力前肘部先沉着走直线，不能先架肘再出拳；起动前手肘抬到快齐平肩膀（离肩下不到一成上身长）算先抬肘",
    ),
}

HOOK_PRIVATE_META: dict[str, tuple[str, str, str]] = {
    "hook_arm_bend": (
        "摆拳大小臂弯屈角度不当",
        "arm",
        "摆拳打到瞬间大小臂大约弯成直角（约 90°）；侧面看小臂缩得太紧（小于约 60°，投影超 0.5 倍大臂长）"
        "或甩成大抡臂（大于约 135°，投影超 0.7 倍大臂长）都算错",
    ),
    "hook_forearm_level": (
        "摆拳小臂未保持水平",
        "arm",
        "摆拳打到瞬间小臂要端平（肘和拳差不多一样高）；正面看小臂上下倾斜超过 10° 算错",
    ),
    "hook_over_center": (
        "摆拳摆过身体中线过多",
        "hand",
        "摆拳打到身体正前方中线位置就要刹住；正面看拳头冲过两肩中点超过 0.3 倍肩宽（约一拳距离）算摆过头",
    ),
}

KICK_PRIVATE_META: dict[str, tuple[str, str, str]] = {
    "pad_step_missing": (
        "未做后脚垫步",
        "step_foot",
        f"起前腿踢之前，后脚必须先往前垫一大步；后脚向前移动不到约半个上身长（{KICK_PARAMS['pad_step_min_disp']:.2f} 倍）算没垫步",
    ),
    "pad_step_overtake": (
        "垫步后脚超越前脚",
        "step_foot",
        f"后脚垫上来最多和前脚并齐，不能踩到前脚前面去；起腿前后脚超过前脚五分之一上身长（{KICK_PARAMS['pad_step_overtake_max']:.2f} 倍）算越前",
    ),
    "pad_step_pause": (
        "垫步落稳后停顿过长",
        "step_foot",
        f"后脚垫稳后要立刻顺势起腿，不能先站定再踢；垫稳到起腿中间停顿超过 {KICK_PARAMS['pad_step_pause_max']:.2f} 秒算停顿过长",
    ),
    "kick_height": (
        "踢击高度过低或过高",
        "leg",
        "低鞭腿踢在对方大腿位置；腿蹬出最长时，踢脚脚尖不能高过支撑腿胯部（留约半拳余量），也不能明显低于支撑腿膝盖",
    ),
    "kick_retract": (
        "踢后未及时收腿直接落地",
        "leg",
        f"踢完要先把小腿往回勾收再落地，不能直挺挺直接放脚；落地前脚踝还伸在胯前超过 {KICK_PARAMS['kick_retract_max_x']:.1f} 倍上身长算没收腿",
    ),
    "foot_flexed": (
        "脚背未绷直（勾脚尖）",
        "foot",
        f"踢腿时脚尖往前绷直、不要勾着脚踢；膝盖—脚踝—脚尖连线夹角小于 {KICK_PARAMS['foot_flexed_min_angle']:g}° 算没绷直脚背",
    ),
    "arm_swing": (
        "未顺势向下甩手",
        "arm",
        f"腿蹬出最长那一瞬间，同侧手臂要顺势往下甩开帮着发力；正面看手腕下垂不到约半个上身长（{KICK_PARAMS['arm_swing_drop']:.2f} 倍）"
        f"或手臂蜷着没甩开（肘角小于 {KICK_PARAMS['arm_swing_min_angle']:g}°）算没甩手",
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
