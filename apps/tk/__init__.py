"""Tk 桌面（入口层 L6）：组合根、后台到 Tk 的唯一通道、关窗协调、控制器与视图。"""

from apps.tk.dispatcher import UiDispatcher, epoch_token
from apps.tk.shutdown import ShutdownCoordinator, ShutdownSteps

__all__ = ["ShutdownCoordinator", "ShutdownSteps", "UiDispatcher", "epoch_token"]
