"""Group comparison: test selection, correctness against reference implementations, edge cases."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.proportion import proportions_ztest

from stat_advisor.compare import (compare_means, compare_means_df, compare_proportions, dunn_test,
                                  games_howell)

rng = np.random.default_rng(123)


# ---------------------------------------------------------------- test selection
def test_normal_two_groups_use_welch():
    r = compare_means({"A": rng.normal(50, 5, 200), "B": rng.normal(53, 5, 200)})
    assert r.test_used == "Welch's t-test" and r.significant and r.estimate > 0


def test_small_skewed_groups_use_rank_test():
    r = compare_means({"A": rng.exponential(2, 20), "B": rng.exponential(2, 20) ** 2})
    assert r.test_used == "Brunner-Munzel test" and r.route == "ranks"
    r = compare_means({"A": rng.exponential(2, 8), "B": rng.exponential(2, 8) ** 3})
    assert r.test_used == "Mann-Whitney U test"            # Brunner-Munzel needs ~10+ per group


def test_rank_route_complete_separation_falls_back():
    r = compare_means({"A": np.arange(12) ** 3.0, "B": 10_000 + np.arange(12) ** 3.0}, focus="typical")
    assert r.test_used == "Mann-Whitney U test" and r.significant


def test_large_skewed_groups_use_welch_via_clt():
    """v1 sent these to Mann-Whitney because Shapiro-Wilk rejects any large sample."""
    r = compare_means({"A": rng.standard_t(10, 5000), "B": rng.standard_t(10, 5000)})
    assert r.test_used == "Welch's t-test"
    assert all(v["mean_test_valid"] for v in r.checks["validity"].values())


def test_three_normal_groups_use_welch_anova_and_games_howell():
    r = compare_means({"M1": rng.normal(100, 10, 80), "M2": rng.normal(100, 10, 80), "M3": rng.normal(112, 20, 80)})
    assert r.test_used == "Welch's ANOVA" and r.significant
    sig = r.posthoc[r.posthoc["significant"]]
    assert set(sig["group_2"]) == {"M3"}


def test_three_small_skewed_groups_use_kruskal_and_dunn():
    r = compare_means({"a": rng.lognormal(0, 1, 25), "b": rng.lognormal(0, 1, 25), "c": rng.lognormal(1.5, 1, 25)})
    assert r.test_used == "Kruskal-Wallis H test" and r.effect_size_name == "epsilon²"
    assert r.posthoc is not None and "p_adjusted" in r.posthoc


def test_focus_typical_forces_rank_test():
    r = compare_means({"A": rng.normal(0, 1, 100), "B": rng.normal(0, 1, 100)}, focus="typical")
    assert r.route == "ranks"


def test_no_posthoc_when_overall_not_significant():
    r = compare_means({g: rng.normal(0, 1, 30) for g in "abc"})
    if not r.significant:
        assert r.posthoc is None


# ---------------------------------------------------------------- statistical correctness
def test_welch_ci_matches_scipy_and_respects_alpha():
    a, b = rng.normal(0, 1, 15), rng.normal(0.4, 2, 12)
    r05 = compare_means({"A": a, "B": b}, alpha=0.05)
    r01 = compare_means({"A": a, "B": b}, alpha=0.01)
    ref = stats.ttest_ind(b, a, equal_var=False).confidence_interval(0.95)
    assert r05.ci == pytest.approx((ref.low, ref.high))
    assert (r01.ci[1] - r01.ci[0]) > (r05.ci[1] - r05.ci[0])


@pytest.mark.slow
def test_ci_coverage_at_small_n():
    """v1 used 1.96 instead of a t critical value: 92% coverage at n=6. Should be ~95%."""
    local = np.random.default_rng(0)
    hits = n = 0
    for _ in range(1500):
        r = compare_means({"A": local.normal(0, 1, 6), "B": local.normal(0, 1, 6)}, focus="average")
        if r.route == "means":
            n += 1
            hits += r.ci[0] <= 0 <= r.ci[1]
    assert 0.935 <= hits / n <= 0.965


def test_games_howell_two_groups_equals_welch():
    a, b = rng.normal(0, 1, 30), rng.normal(0.5, 2, 25)
    gh = games_howell({"a": a, "b": b})
    assert gh["p_adjusted"].iloc[0] == pytest.approx(stats.ttest_ind(b, a, equal_var=False).pvalue, abs=1e-6)


def test_welch_anova_two_groups_equals_welch_t_squared():
    a, b = rng.normal(0, 1, 30), rng.normal(0.5, 2, 25)
    f = anova_oneway([a, b], use_var="unequal").statistic
    assert f == pytest.approx(stats.ttest_ind(a, b, equal_var=False).statistic ** 2)


def test_dunn_two_groups_equals_mann_whitney_normal_approximation():
    a, b = np.round(rng.normal(0, 1, 30), 1), np.round(rng.normal(0.5, 1, 25), 1)   # with ties
    d = dunn_test({"a": a, "b": b})
    mw = stats.mannwhitneyu(b, a, method="asymptotic", use_continuity=False)
    assert d["p_adjusted"].iloc[0] == pytest.approx(mw.pvalue, abs=1e-6)


def test_direction_and_control():
    groups = {"A": rng.normal(10, 1, 50), "B": rng.normal(12, 1, 50)}
    r = compare_means(groups, control="B")
    assert r.groups[0] == "B" and r.estimate < 0


def test_dataframe_wrapper():
    df = pd.DataFrame({"v": np.r_[rng.normal(0, 1, 40), rng.normal(1, 1, 40)], "g": ["x"] * 40 + ["y"] * 40})
    assert compare_means_df(df, "v", "g").significant


# ---------------------------------------------------------------- proportions
def test_large_counts_use_z_test_matching_statsmodels():
    r = compare_proportions([120, 150], [1000, 1000], names=["Control", "Treatment"])
    assert r.test_used == "Two-proportion z-test"
    assert r.p_value == pytest.approx(proportions_ztest([150, 120], [1000, 1000])[1])
    assert r.estimate == pytest.approx(0.03) and r.ci[0] < 0.03 < r.ci[1]
    assert r.relative_lift == pytest.approx(0.25)


def test_small_counts_use_exact_test():
    """Expected counts < 5: the z-test's normal approximation is not guaranteed, so an exact test is used."""
    r = compare_proportions([1, 6], [20, 20])
    assert r.test_used == "Boschloo's exact test"
    assert r.p_value == pytest.approx(stats.boschloo_exact([[1, 19], [6, 14]]).pvalue)
    assert r.checks["expected_counts"]["ok"] is False


def test_k_proportions_chi_square_and_posthoc():
    r = compare_proportions([100, 120, 170], [1000, 1000, 1000], names=["a", "b", "c"])
    assert r.test_used == "Chi-square test of independence" and r.significant
    assert r.posthoc["significant"].any()


def test_k_sparse_proportions_use_monte_carlo_exact():
    r = compare_proportions([0, 1, 7], [30, 30, 30])
    assert "Monte Carlo" in r.test_used and 0 < r.p_value < 0.05


def test_zero_conversions_everywhere_is_not_nan():
    r = compare_proportions([0, 0], [100, 100])
    assert r.p_value == 1.0 and not r.significant and r.route == "degenerate"


# ---------------------------------------------------------------- degenerate + invalid input
def test_zero_variance_identical_groups():
    r = compare_means({"A": [5, 5, 5], "B": [5, 5, 5]})
    assert r.p_value == 1.0 and r.effect_size == 0.0


def test_zero_variance_different_groups():
    r = compare_means({"A": [5, 5, 5], "B": [7, 7, 7]})
    assert r.significant and r.estimate == 2


@pytest.mark.parametrize("df,value,group", [
    (pd.DataFrame(), "value", "group"),
    (pd.DataFrame({"value": [1, 2, 3], "group": ["A"] * 3}), "value", "group"),
    (pd.DataFrame({"value": [1, 2, np.nan, np.nan], "group": ["A", "A", "B", "B"]}), "value", "group"),
    (pd.DataFrame({"value": [1, 2, 3, 9], "group": ["A", "A", "A", "B"]}), "value", "group"),
    (pd.DataFrame({"value": list("xyzw"), "group": ["A", "A", "B", "B"]}), "value", "group"),
    (pd.DataFrame({"value": [1, 2, 3, 4], "group": ["A", "A", "B", "B"]}), "missing", "group"),
    (pd.DataFrame({"value": [1, 2, 3, 4], "group": ["A", "A", "B", "B"]}), "value", "value"),
])
def test_invalid_numeric_input_raises_clean_error(df, value, group):
    with pytest.raises(ValueError):
        compare_means_df(df, value, group)


@pytest.mark.parametrize("s,t,names", [([10], [0], None), ([50, 10], [40, 100], None), ([10], [100], None),
                                       ([1, 2], [10, 10], ["A", "A"])])
def test_invalid_proportion_input_raises_clean_error(s, t, names):
    with pytest.raises(ValueError):
        compare_proportions(s, t, names=names)


def test_unknown_control_raises():
    with pytest.raises(ValueError):
        compare_means({"A": [1, 2, 3], "B": [2, 3, 4]}, control="Z")
