"""练习几何门面：配置与测量在 ``core.evaluation.rules.practice_v4``，分析在 ``core.evaluation.practice``，
相位在 ``core.evaluation.phases``（P9 删除）。"""
from core.evaluation.phases import align_views, phase_windows, share_phase_timing  # noqa: F401
from core.evaluation.practice import analyze_practice as analyze_geometry  # noqa: F401
from core.evaluation.practice import create_practice_engine as _pipeline  # noqa: F401
from core.evaluation.practice import evaluate_series, extract_series  # noqa: F401
from core.evaluation.rules.practice_v4 import FeedbackConfig, load_feedback_config  # noqa: F401
from core.evaluation.rules.practice_v4 import PRIMARY_VIEW as _PRIMARY_VIEW  # noqa: F401
from core.evaluation.rules.practice_v4 import check_view_weights as _view_weights  # noqa: F401
from core.evaluation.rules.practice_v4 import default_front_weights as _default_front_weights  # noqa: F401
from core.evaluation.rules.practice_v4 import measure_check as _measure  # noqa: F401
from core.foundation.signal import contiguous_runs  # noqa: F401
