"""学生练习结果页：头部、概览卡片、相位条、检查列表、证据区、诊断区与底部操作栏，只消费报告模型。"""
from apps.tk.views.practice_result.panel import FeedbackResultPanel
from apps.tk.views.practice_result.style import STYLE_A

__all__ = ["FeedbackResultPanel", "STYLE_A"]
