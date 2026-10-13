# -*- coding: utf-8 -*-
"""旧导入路径的转发门面：语音播报队列在 ``core.workflows.announcer``（P9 删除）。"""
from core.workflows.announcer import Announcer as ExamAnnouncer

__all__ = ["ExamAnnouncer"]
