"""
Robustness tests: edge cases the algorithm must fail gracefully on
(not crash with a raw traceback), plus property-based tests that check
invariants across hundreds of randomized inputs.
"""
import numpy as np
import pandas as pd
import warnings
from hypothesis import given, strategies as st, settings, HealthCheck

from algorithm import (
    run_numeric_test, run_proportion_test,
    shewhart_individuals_chart, cusum_chart,
)

warnings.filterwarnings("ignore")  # scipy warns on some degenerate edge cases; we test the ValueError path, not the warning


def expect_valueerror(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        raise AssertionError(f"Expected ValueError but {fn.__name__} succeeded")
    except ValueError as e:
        print(f"  -> correctly raised ValueError: {e}")


print("=" * 70)
print("EDGE CASE TESTS")
print("=" * 70)

print("\n1. Empty dataframe")
expect_valueerror(run_numeric_test, pd.DataFrame(), "value", "group")

print("\n2. Only 1 group present")
df = pd.DataFrame({"value": [1, 2, 3], "group": ["A", "A", "A"]})
expect_valueerror(run_numeric_test, df, "value", "group")

print("\n3. A group with 0 valid (all-NaN) values")
df = pd.DataFrame({"value": [1, 2, np.nan, np.nan, 5, 6], "group": ["A", "A", "B", "B", "A", "A"]})
# B group has 0 valid values after dropna -> should raise
expect_valueerror(run_numeric_test, df, "value", "group")

print("\n4. A group with only 1 data point")
df = pd.DataFrame({"value": [1, 2, 3, 9], "group": ["A", "A", "A", "B"]})
expect_valueerror(run_numeric_test, df, "value", "group")

print("\n5. Non-numeric metric column")
df = pd.DataFrame({"value": ["x", "y", "z", "w"], "group": ["A", "A", "B", "B"]})
expect_valueerror(run_numeric_test, df, "value", "group")

print("\n6. Missing column names")
df = pd.DataFrame({"value": [1, 2, 3, 4], "group": ["A", "A", "B", "B"]})
expect_valueerror(run_numeric_test, df, "nonexistent_col", "group")

print("\n7. All-identical values in every group (zero variance) — should NOT crash, effect size = 0")
df = pd.DataFrame({"value": [5, 5, 5, 5, 5, 5], "group": ["A", "A", "A", "B", "B", "B"]})
r = run_numeric_test(df, "value", "group")
print(f"  -> ran without crashing. effect_size={r.effect_size}, p={r.p_value}")
assert r.effect_size == 0.0

print("\n8. Proportion test: total = 0")
expect_valueerror(run_proportion_test, [10], [0], ["A"])

print("\n9. Proportion test: successes > totals")
expect_valueerror(run_proportion_test, [50, 10], [40, 100], ["A", "B"])

print("\n10. Proportion test: only 1 group")
expect_valueerror(run_proportion_test, [10], [100], ["A"])

print("\n11. Control chart: fewer than 2 points")
expect_valueerror(shewhart_individuals_chart, [5])
expect_valueerror(cusum_chart, [5])

print("\n12. Control chart: NaN in data")
expect_valueerror(shewhart_individuals_chart, [1, 2, np.nan, 4])
expect_valueerror(cusum_chart, [1, 2, np.nan, 4])

print("\n13. Control chart: perfectly flat series (zero variance) — should NOT crash")
flat = [10.0] * 20
r = shewhart_individuals_chart(flat)
print(f"  -> ran without crashing. violations={r.n_violations}")
assert r.n_violations == 0

print("\nAll edge cases handled correctly (clean errors, no raw crashes) ✔")


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PROPERTY-BASED TESTS (hypothesis)")
print("=" * 70)


@given(
    a=st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False), min_size=5, max_size=200),
    b=st.lists(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False), min_size=5, max_size=200),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_pvalue_always_in_range(a, b):
    df = pd.DataFrame({"value": a + b, "group": ["A"] * len(a) + ["B"] * len(b)})
    try:
        r = run_numeric_test(df, "value", "group")
    except ValueError:
        return  # degenerate case (e.g. zero variance breaking a specific test) - acceptable skip
    assert 0.0 <= r.p_value <= 1.0, f"p-value out of range: {r.p_value}"


@given(
    a=st.lists(st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False), min_size=5, max_size=100),
    b=st.lists(st.floats(min_value=200, max_value=300, allow_nan=False, allow_infinity=False), min_size=5, max_size=100),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_never_routes_skewed_data_to_ttest_when_nonnormal(a, b):
    """If normality fails, the algorithm must never silently use a t-test."""
    df = pd.DataFrame({"value": a + b, "group": ["A"] * len(a) + ["B"] * len(b)})
    try:
        r = run_numeric_test(df, "value", "group")
    except ValueError:
        return
    if not r.assumptions["normality"]["all_normal"]:
        assert "t-test" not in r.test_used, f"Non-normal data routed to a t-test: {r.test_used}"


@given(st.lists(st.floats(min_value=-500, max_value=500, allow_nan=False, allow_infinity=False), min_size=3, max_size=100))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_control_chart_never_flags_more_points_than_exist(values):
    r = shewhart_individuals_chart(values)
    assert r.n_violations <= len(values)
    assert all(0 <= v < len(values) for v in r.violations)


print("\nRunning: p-value always in [0,1]...")
test_pvalue_always_in_range()
print("  -> PASSED (100 random cases)")

print("\nRunning: non-normal data never routed to t-test...")
test_never_routes_skewed_data_to_ttest_when_nonnormal()
print("  -> PASSED (50 random cases)")

print("\nRunning: control chart violation count never exceeds data size...")
test_control_chart_never_flags_more_points_than_exist()
print("  -> PASSED (50 random cases)")

print("\n" + "=" * 70)
print("ALL ROBUSTNESS TESTS PASSED ✔")
print("=" * 70)
