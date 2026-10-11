"""Process monitoring: detection of jumps and drifts, baseline handling, false alarms."""
import numpy as np
import pytest

from stat_advisor.monitor import monitor_process, western_electric

rng = np.random.default_rng(42)


def test_spike_is_detected_and_confirmed():
    x = np.r_[rng.normal(50, 2, 40), [80, 82, 79]]
    m = monitor_process(x, baseline_n=30)
    shift = [f for f in m.findings if f.kind == "shift"]
    assert m.status == "issue" and shift and shift[0].start >= 40


def test_gradual_drift_found_with_change_point_estimate():
    x = np.r_[rng.normal(50, 2, 60), 50 + np.linspace(0, 6, 40) + rng.normal(0, 2, 40)]
    m = monitor_process(x)
    drift = [f for f in m.findings if f.kind == "drift" and f.confirmed and f.direction == "upward"]
    assert drift
    assert 55 <= drift[0].start <= 85         # true start = 60


def test_limits_come_from_baseline_not_contaminated_data():
    """v1 learned limits from ALL data, so a later shift widened its own limits."""
    base = rng.normal(50, 2, 50)
    m = monitor_process(np.r_[base, rng.normal(56, 2, 50)], baseline_n=50)
    assert m.center == pytest.approx(base.mean())
    assert m.status == "issue"


@pytest.mark.slow
def test_confirmed_false_alarm_rate_is_bounded():
    local = np.random.default_rng(0)
    alarms = sum(monitor_process(local.normal(10, 1, 100)).status == "issue" for _ in range(200))
    assert alarms / 200 < 0.2


def test_flat_series_does_not_crash():
    m = monitor_process([10.0] * 20)
    assert m.status == "stable"


def test_flat_baseline_then_change_is_flagged_with_warning():
    m = monitor_process([10.0] * 15 + [10.5] * 5, baseline_n=15)
    assert m.status == "issue" and any("no variation" in w for w in m.warnings)


def test_autocorrelation_warning():
    x = np.zeros(120)
    for i in range(1, 120):
        x[i] = 0.85 * x[i - 1] + rng.normal()
    assert any("correlated" in w for w in monitor_process(x).warnings)


def test_western_electric_rule_4():
    x = np.r_[np.zeros(10), np.full(8, 0.5)]
    rules = western_electric(x, center=0.0, sigma=1.0, sensitive=True)
    assert rules["Rule 4: 8 in a row on one side"] == [17]


@pytest.mark.parametrize("bad", [[1, 2, np.nan, 4, 5, 6, 7, 8], [1, 2, 3]])
def test_invalid_series_raises(bad):
    with pytest.raises(ValueError):
        monitor_process(bad)
