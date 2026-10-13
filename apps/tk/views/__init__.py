"""Tk 视图：只负责布局与控件，业务调用经控制器。"""

from apps.tk.views.compare_dialog import CompareDialog
from apps.tk.views.settings_dialog import SettingsDialog

__all__ = ["CompareDialog", "SettingsDialog"]
