# -*- coding: utf-8 -*-
"""Tk 层的运行标志 ``LiveFlags``：关窗、双摄录制就绪、片段级骨架与自动比对、考试锁定，由各控制器共读。

只在 Tk 主线程写；录制计划在会话线程收尾时会读 ``record_skeleton``、``auto_compare``、骨架成片的模型设置（单值读写原子）。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LiveFlags:
    closing: bool = False           # 已进入关窗流程
    recording_ready: bool = False   # 会话可录制（单摄开始即就绪；双摄等两路首帧）
    dual_active: bool = False       # 双摄会话已接管设备并登记录制
    record_skeleton: bool = False   # 本次会话开始时冻结的骨架开关（视频文件会话恒为 False：录像本身已带骨架）
    pose_variant: str = "full"      # 本次会话的模型与手部设置：录后骨架成片沿用（D-26）
    enable_hands: bool = True
    auto_compare: bool = True       # 片段级自动比对开关（主线程维护的缓存）
    exam_locked: bool = False       # 考试进行中锁定手动录制
