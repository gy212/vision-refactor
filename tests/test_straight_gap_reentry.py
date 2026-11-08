"""长缺点后恢复时已伸出的手不能补算出拳，实际归位后的新拳仍保留。"""

import numpy as np
import pytest

from core.evaluation import evaluate_series
from tests.test_step_straight_combo import _series


@pytest.mark.parametrize(
    "action,view",
    [
        ("front_straight", "side"),
        ("straight_combo", "side"),
        ("step_straight_combo", "side"),
        ("front_straight", "front"),
        ("straight_combo", "front"),
    ],
)
def test_missing_departure_is_not_a_punch_and_observed_return_rearms(action, view):
    series = _series(0)
    times = np.asarray(series["times"])
    # 离开护位发生在遮挡中。恢复时手已经伸出，随后下放，不能据此补算一拳。
    series["valid_mask"][(times >= 1.0) & (times <= 1.6), 15] = False
    series["landmarks"][:, 15, 0] += np.interp(
        times, [0.0, 0.99, 1.0, 1.8, 2.2, 5.99], [0.0, 0.0, 0.18, 0.18, 0.0, 0.0]
    )
    series["landmarks"][:, 15, 1] += np.interp(
        times, [0.0, 1.8, 2.2, 2.8, 3.0, 5.99], [0.0, 0.0, 0.1, 0.1, 0.0, 0.0]
    )
    # 重新回到护位后，完整可见地出拳并收回，应保留这一拳及它的检查阶段。
    series["landmarks"][:, 15, 0] += np.interp(
        times, [0.0, 3.5, 3.8, 4.1, 5.99], [0.0, 0.0, 0.2, 0.0, 0.0]
    )

    result = evaluate_series(action, "left", {view: series})

    events = result["sharedActionRecord"]["eventDetails"]
    assert len(events) == 1
    event = events[0]
    assert event["segment"] == "front_straight"
    assert 3.5 <= event["start"] < event["peak"] < event["end"] <= 4.1
    assert event["ending"] == "returned"
    phases = result["sharedActionRecord"]["phaseInstances"]
    assert {"motion", "finish"} <= {
        phase["phase"] for phase in phases if phase.get("eventId") == event["eventId"]
    }
