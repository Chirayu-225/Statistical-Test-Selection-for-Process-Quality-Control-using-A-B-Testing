"""End-to-end: question in, answer + recommendation out, on the shipped sample datasets."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stat_advisor import (analyze_comparison, analyze_counts, analyze_monitoring, infer_metric_kind,
                          plan_conversion_test, plan_mean_test)

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def checkout():
    return pd.read_csv(DATA / "checkout_ab_test.csv")


def test_ab_test_conversion(checkout):
    r = analyze_comparison(checkout, "converted", "variant", control="A", randomized=True)
    assert r.verdict == "difference" and r.method == "Two-proportion z-test"
    assert "Roll out B" in r.recommendation
    assert "match the planned split" in r.reasoning[0]


def test_ab_test_revenue_uses_welch_thanks_to_clt(checkout):
    r = analyze_comparison(checkout, "revenue", "variant", control="A", randomized=True)
    assert r.method == "Welch's t-test"


def test_machine_defects_points_at_m3():
    r = analyze_comparison(pd.read_csv(DATA / "machine_defects.csv"), "defect_rate_pct", "machine",
                           higher_is_better=False)
    assert r.method == "Welch's ANOVA" and "M3" in r.headline and "Focus on M3" in r.recommendation


def test_support_times_use_rank_tests():
    r = analyze_comparison(pd.read_csv(DATA / "support_resolution_times.csv"), "resolution_hours", "team",
                           higher_is_better=False)
    assert r.method == "Kruskal-Wallis H test" and any("medians" in w for w in r.warnings)


def test_bottle_line_drift():
    r = analyze_monitoring(pd.read_csv(DATA / "bottle_fill_process.csv"), "fill_volume_ml", order_col="batch")
    assert r.verdict == "issue" and "drift" in r.headline


def test_sample_ratio_mismatch_overrides_result():
    df = pd.DataFrame({"g": ["A"] * 5000 + ["B"] * 4600, "y": [0] * 9000 + [1] * 600})
    r = analyze_comparison(df, "y", "g", randomized=True)
    assert r.verdict == "untrustworthy" and "Don't trust" in r.headline


def test_equivalence_when_ci_inside_threshold():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"g": ["A"] * 5000 + ["B"] * 5000, "y": rng.normal(100, 5, 10000)})
    r = analyze_comparison(df, "y", "g", control="A", min_meaningful=1.0)
    assert r.verdict == "no_meaningful_difference"


def test_inconclusive_reports_detectable_difference():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"g": ["A"] * 15 + ["B"] * 15, "y": rng.normal(0, 1, 30)})
    r = analyze_comparison(df, "y", "g", control="A", min_meaningful=0.2)
    assert r.verdict == "inconclusive"
    assert any("reliably detect" in k for k, _ in r.key_numbers)


@pytest.mark.parametrize("values,kind", [([0, 1, 1, 0], "binary"), (["yes", "no", "yes"], "binary"),
                                         ([True, False], "binary"), ([1.5, 2.0, 3.1], "numeric")])
def test_metric_kind_inference(values, kind):
    assert infer_metric_kind(pd.Series(values, name="m")) == kind


def test_yes_no_strings_are_analysed_as_rates():
    df = pd.DataFrame({"g": ["A"] * 200 + ["B"] * 200, "bought": ["yes"] * 20 + ["no"] * 180 + ["yes"] * 45 + ["no"] * 155})
    r = analyze_comparison(df, "bought", "g", control="A")
    assert r.result.route == "proportions" and r.result.estimate == pytest.approx(0.125)


def test_counts_and_markdown_export():
    r = analyze_counts(["A", "B"], [500, 560], [5000, 5000], control="A", randomized=True)
    md = r.to_markdown()
    for section in ("**Answer:**", "**Recommendation:**", "## Key numbers", "## Why this method"):
        assert section in md


def test_planning_reports():
    r = plan_conversion_test(0.10, 0.10, daily_traffic=2000)
    assert "14,745" in r.headline and "days" in r.headline
    assert "observations per group" in plan_mean_test(10, 2).headline
    assert plan_conversion_test(0.10, 0.10, groups=3).key_numbers[0][1] != r.key_numbers[0][1]
