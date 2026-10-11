"""Property-based tests (hypothesis): invariants that must hold for ANY input, not just hand-picked cases."""
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from stat_advisor.compare import compare_means, compare_proportions
from stat_advisor.monitor import monitor_process

SETTINGS = settings(max_examples=60, deadline=None, derandomize=True,
                    suppress_health_check=[HealthCheck.too_slow])
ints = st.lists(st.integers(-1000, 1000), min_size=3, max_size=60)
floats = st.lists(st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False), min_size=2, max_size=80)


@SETTINGS
@given(a=floats, b=floats)
def test_p_value_always_valid(a, b):
    r = compare_means({"A": a, "B": b})
    assert 0.0 <= r.p_value <= 1.0 and not np.isnan(r.p_value)


@SETTINGS
@given(groups=st.lists(floats, min_size=3, max_size=5))
def test_p_value_valid_for_many_groups(groups):
    r = compare_means({i: g for i, g in enumerate(groups)})
    assert 0.0 <= r.p_value <= 1.0


@SETTINGS
@given(a=ints, b=ints, shift=st.integers(-10_000, 10_000))
def test_location_invariance(a, b, shift):
    """Adding the same constant to every value must not change the conclusion."""
    r1 = compare_means({"A": a, "B": b})
    r2 = compare_means({"A": np.add(a, shift), "B": np.add(b, shift)})
    assert r1.test_used == r2.test_used
    assert r2.p_value == pytest.approx(r1.p_value, rel=1e-6, abs=1e-9)


@SETTINGS
@given(a=ints, b=ints)
def test_unit_change_invariance(a, b):
    """Changing units (e.g. grams -> kg) must not change the conclusion."""
    r1 = compare_means({"A": a, "B": b})
    r2 = compare_means({"A": np.multiply(a, 0.001), "B": np.multiply(b, 0.001)})
    assert r2.p_value == pytest.approx(r1.p_value, rel=1e-6, abs=1e-9)


@SETTINGS
@given(a=ints, b=ints)
def test_swapping_control_flips_sign_only(a, b):
    r1 = compare_means({"A": a, "B": b}, control="A")
    r2 = compare_means({"A": a, "B": b}, control="B")
    assert r1.p_value == pytest.approx(r2.p_value, rel=1e-9, abs=1e-12)
    assert r1.estimate == pytest.approx(-r2.estimate)


@SETTINGS
@given(a=floats, b=floats)
def test_ci_brackets_the_estimate(a, b):
    r = compare_means({"A": a, "B": b})
    if r.ci is not None and r.route == "means":
        assert r.ci[0] - 1e-9 <= r.estimate <= r.ci[1] + 1e-9


@SETTINGS
@given(data=st.lists(st.tuples(st.integers(1, 400), st.floats(0, 1)), min_size=2, max_size=4))
def test_proportions_always_valid(data):
    totals = [t for t, _ in data]
    succ = [int(round(t * f)) for t, f in data]
    r = compare_proportions(succ, totals)
    assert 0.0 <= r.p_value <= 1.0
    if r.ci is not None:
        assert -1 <= r.ci[0] <= r.ci[1] <= 1


@SETTINGS
@given(values=st.lists(st.floats(-500, 500, allow_nan=False), min_size=8, max_size=150))
def test_monitoring_signals_stay_in_range(values):
    m = monitor_process(values)
    n = len(values)
    assert m.status in {"stable", "watch", "issue"}
    for f in m.findings:
        assert 0 <= f.start <= f.detected_at < n
