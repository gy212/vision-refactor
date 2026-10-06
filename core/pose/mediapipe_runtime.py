# -*- coding: utf-8 -*-
"""MediaPipe 的延迟导入：第一次真正推理或绘制时才压低原生日志级别并导入 ``mediapipe``。

只导入 ``core.pose`` 包（如旧路径门面 ``core.pose_features``）不再加载 mediapipe（约 0.3 s），
也不在导入时改写 ``TF_CPP_MIN_LOG_LEVEL``/``GLOG_minloglevel``（P2 评审遗留第 1 项）。
"""
from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def mediapipe():
    """返回 ``mediapipe`` 模块。日志级别硬设（不用 setdefault），避免用户环境把 glog/TFLite 刷屏重新打开。"""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["GLOG_minloglevel"] = "3"
    import mediapipe as mp

    return mp
