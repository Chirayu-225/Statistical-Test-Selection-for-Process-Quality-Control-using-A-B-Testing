"""
LEGACY v1 ENGINE — kept unchanged, for the before/after calibration study only.
Do not use in the app. See stat_advisor/ for the current engine.
"""
"""
Statistical Test Selection and Decision Algorithm
====================================================
Given grouped data, this algorithm:
  1. Checks assumptions (normality, variance homogeneity)
  2. Selects the statistically valid test for the data's actual properties
  3. Runs it and computes effect size + confidence interval
  4. Returns a plain-language verdict

Domain-agnostic core — used identically for A/B experimentation
and process quality control (only the interpretation layer differs).
"""

from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.stats.multitest as smm
import statsmodels.api as sm
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower
from statsmodels.stats.proportion import proportions_ztest, proportion_confint


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    test_used: str
    p_value: float
    statistic: float
    effect_size: float
    effect_size_label: str
    confidence_interval: tuple
    alpha: float
    significant: bool
    assumptions: dict
    verdict: str
    group_summary: pd.DataFrame = field(default=None)


# ---------------------------------------------------------------------------
# Assumption checks
# ---------------------------------------------------------------------------

def check_normality(*groups, alpha=0.05) -> dict:
    """Shapiro-Wilk per group. Returns pass/fail + p-values."""
    results = {}
    all_normal = True
    for i, g in enumerate(groups):
        g = np.asarray(g)
        if len(g) < 3:
            results[f"group_{i}"] = {"p_value": None, "normal": None, "note": "n<3, skipped"}
            continue
        # Shapiro is unreliable for very large n; cap sample for the test
        sample = g if len(g) <= 5000 else np.random.choice(g, 5000, replace=False)
        stat, p = stats.shapiro(sample)
        normal = bool(p > alpha)
        all_normal = all_normal and normal
        results[f"group_{i}"] = {"p_value": round(float(p), 4), "normal": normal}
    results["all_normal"] = bool(all_normal)
    return results


def check_variance_homogeneity(*groups, alpha=0.05) -> dict:
    """Levene's test for equal variances across groups."""
    stat, p = stats.levene(*groups)
    equal_var = bool(p > alpha)
    return {"statistic": round(float(stat), 4), "p_value": round(float(p), 4), "equal_variance": equal_var}


# ---------------------------------------------------------------------------
# Effect size helpers
# ---------------------------------------------------------------------------

def cohens_d(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled_std


def eta_squared(groups) -> float:
    all_data = np.concatenate(groups)
    grand_mean = all_data.mean()
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = sum((all_data - grand_mean) ** 2)
    return ss_between / ss_total if ss_total > 0 else 0.0


def label_cohens_d(d):
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    return "large"


def label_eta_squared(eta):
    if eta < 0.01:
        return "negligible"
    elif eta < 0.06:
        return "small"
    elif eta < 0.14:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Core decision algorithm — numeric metrics (2 or more groups)
# ---------------------------------------------------------------------------

def run_numeric_test(df: pd.DataFrame, value_col: str, group_col: str, alpha: float = 0.05) -> TestResult:
    if df is None or df.empty:
        raise ValueError("No data provided — the dataset is empty.")
    if value_col not in df.columns or group_col not in df.columns:
        raise ValueError(f"Columns not found: expected '{value_col}' and '{group_col}' in the dataset.")
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        raise ValueError(f"'{value_col}' is not numeric — pick a numeric metric column.")

    groups_dict = {name: g[value_col].dropna().values for name, g in df.groupby(group_col)}
    group_names = list(groups_dict.keys())
    groups = list(groups_dict.values())
    k = len(groups)

    if k < 2:
        raise ValueError("Need at least 2 groups to compare — the group column only has 1 distinct value.")

    empty_groups = [name for name, g in zip(group_names, groups) if len(g) == 0]
    if empty_groups:
        raise ValueError(f"Group(s) {empty_groups} have no valid (non-missing) numeric values.")

    tiny_groups = [name for name, g in zip(group_names, groups) if len(g) < 2]
    if tiny_groups:
        raise ValueError(f"Group(s) {tiny_groups} have fewer than 2 data points — cannot compute variance. "
                          f"Statistical tests need at least 2 observations per group.")

    normality = check_normality(*groups, alpha=alpha)
    variance = check_variance_homogeneity(*groups, alpha=alpha)
    assumptions = {"normality": normality, "variance_homogeneity": variance}

    summary = df.groupby(group_col)[value_col].agg(["count", "mean", "std"]).reset_index()

    # Degenerate case: every group has zero variance (all values identical within
    # each group). The t-test/ANOVA formulas divide by pooled variance and return
    # NaN here, which is a real bug, not a hard case — handle it explicitly instead.
    if all(np.var(g) == 0 for g in groups):
        means = [np.mean(g) for g in groups]
        all_equal = len(set(np.round(means, 10))) == 1
        p = 1.0 if all_equal else 0.0
        stat = 0.0 if all_equal else float("inf")
        test_used = "Degenerate case (zero variance in all groups)"
        effect = 0.0 if all_equal else float("inf")
        effect_label = "negligible" if all_equal else "large"
        ci = (0.0, 0.0) if (all_equal and k == 2) else (None, None)
        significant = not all_equal
        verdict = ("All groups have zero variance (every value identical within each group). " +
                    ("Group means are equal, so there is no difference to detect." if all_equal else
                     "Group means differ with perfect separation (zero within-group variance) — "
                     "treat this as a certain, maximal difference, though real data this clean is unusual "
                     "and worth double-checking for a data collection error."))
        return TestResult(
            test_used=test_used, p_value=round(float(p), 6), statistic=stat,
            effect_size=effect, effect_size_label=effect_label, confidence_interval=ci,
            alpha=alpha, significant=significant, assumptions=assumptions,
            verdict=verdict, group_summary=summary,
        )

    if k == 2:
        a, b = groups
        if normality["all_normal"]:
            stat, p = stats.ttest_ind(a, b, equal_var=variance["equal_variance"])
            test_used = "Welch's t-test" if not variance["equal_variance"] else "Student's t-test"
            effect = cohens_d(a, b)
            effect_label = label_cohens_d(effect)
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            diff = a.mean() - b.mean()
            ci = (round(float(diff - 1.96 * se), 4), round(float(diff + 1.96 * se), 4))
        else:
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            test_used = "Mann-Whitney U test (non-parametric)"
            effect = cohens_d(a, b)  # reported for reference even though rank-based test used
            effect_label = label_cohens_d(effect)
            ci = (None, None)
    else:
        if normality["all_normal"]:
            stat, p = stats.f_oneway(*groups)
            test_used = "One-way ANOVA (F-test)"
        else:
            stat, p = stats.kruskal(*groups)
            test_used = "Kruskal-Wallis H-test (non-parametric)"
        effect = eta_squared(groups)
        effect_label = label_eta_squared(effect)
        ci = (None, None)

    significant = bool(p < alpha)
    verdict = _build_verdict(test_used, significant, p, effect_label, alpha, group_names, k)

    return TestResult(
        test_used=test_used,
        p_value=round(float(p), 6),
        statistic=round(float(stat), 4),
        effect_size=round(float(effect), 4),
        effect_size_label=effect_label,
        confidence_interval=ci,
        alpha=alpha,
        significant=significant,
        assumptions=assumptions,
        verdict=verdict,
        group_summary=summary,
    )


# ---------------------------------------------------------------------------
# Core decision algorithm — proportions (e.g. conversion rate)
# ---------------------------------------------------------------------------

def run_proportion_test(successes: list, totals: list, group_names: list = None, alpha: float = 0.05) -> TestResult:
    if len(successes) != len(totals):
        raise ValueError("successes and totals must have the same number of groups.")
    if len(successes) < 2:
        raise ValueError("Need at least 2 groups to compare proportions.")
    for s, t in zip(successes, totals):
        if t <= 0:
            raise ValueError("Each group's total (sample size) must be greater than 0.")
        if s < 0 or s > t:
            raise ValueError(f"Invalid data: successes ({s}) must be between 0 and totals ({t}).")

    if len(successes) != 2 or len(totals) != 2:
        # 3+ groups -> chi-square test of independence
        table = np.array([successes, [t - s for t, s in zip(totals, successes)]])
        stat, p, dof, expected = stats.chi2_contingency(table)
        test_used = "Chi-square test of independence"
        n = table.sum()
        effect = np.sqrt(stat / (n * (min(table.shape) - 1)))  # Cramer's V
        effect_label = label_eta_squared(effect ** 2)
        ci = (None, None)
    else:
        stat, p = proportions_ztest(successes, totals)
        test_used = "Two-proportion z-test"
        p1, p2 = successes[0] / totals[0], successes[1] / totals[1]
        se = np.sqrt(p1 * (1 - p1) / totals[0] + p2 * (1 - p2) / totals[1])
        diff = p1 - p2
        ci = (round(float(diff - 1.96 * se), 4), round(float(diff + 1.96 * se), 4))
        # effect size: Cohen's h
        h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
        effect = h
        effect_label = label_cohens_d(h)

    significant = bool(p < alpha)
    names = group_names or [f"group_{i}" for i in range(len(successes))]
    verdict = _build_verdict(test_used, significant, p, effect_label, alpha, names, len(successes))

    return TestResult(
        test_used=test_used,
        p_value=round(float(p), 6),
        statistic=round(float(stat), 4),
        effect_size=round(float(effect), 4),
        effect_size_label=effect_label,
        confidence_interval=ci,
        alpha=alpha,
        significant=significant,
        assumptions={"note": "Proportion tests do not require normality/variance checks"},
        verdict=verdict,
    )


def _build_verdict(test_used, significant, p, effect_label, alpha, group_names, k):
    sig_phrase = "a statistically significant difference" if significant else "no statistically significant difference"
    base = (
        f"Using {test_used} (alpha={alpha}), the data shows {sig_phrase} "
        f"across {k} group(s) [{', '.join(map(str, group_names))}] (p={p:.4f}). "
    )
    if significant:
        base += f"The effect size is {effect_label}, so treat this as a real, "
        base += "practically meaningful difference." if effect_label in ("medium", "large") else \
                 "but small, difference — confirm practical significance before acting."
    else:
        base += "Do not conclude a difference exists; either there truly is none, or the sample size is too small to detect it."
    return base


# ---------------------------------------------------------------------------
# Multiple comparison correction
# ---------------------------------------------------------------------------

def correct_multiple_tests(p_values: list, method: Literal["bonferroni", "fdr_bh"] = "fdr_bh", alpha: float = 0.05):
    reject, p_adjusted, _, _ = smm.multipletests(p_values, alpha=alpha, method=method)
    return {
        "method": method,
        "p_adjusted": [round(float(p), 6) for p in p_adjusted],
        "significant_after_correction": list(map(bool, reject)),
    }


# ---------------------------------------------------------------------------
# Sample size / power calculation (for planning experiments up front)
# ---------------------------------------------------------------------------

def required_sample_size_ttest(effect_size: float, alpha: float = 0.05, power: float = 0.8) -> int:
    analysis = TTestIndPower()
    n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=1.0)
    return int(np.ceil(n))


def required_sample_size_anova(effect_size: float, k_groups: int, alpha: float = 0.05, power: float = 0.8) -> int:
    """Returns required sample size PER GROUP.
    statsmodels' FTestAnovaPower.solve_power() returns TOTAL nobs across all
    groups, not per-group (unlike TTestIndPower, which is per-group) — divide
    by k_groups to get a comparable, per-group figure."""
    analysis = FTestAnovaPower()
    total_n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, k_groups=k_groups)
    return int(np.ceil(total_n / k_groups))


# ---------------------------------------------------------------------------
# Control charts — Mode B (process monitoring over time)
# Shewhart uses fixed sigma limits (good for large, sudden shifts).
# CUSUM accumulates small deviations (good for detecting slow, gradual drift).
# Reported together because they catch different failure modes.
# ---------------------------------------------------------------------------

@dataclass
class ControlChartResult:
    chart_type: str
    center_line: float
    ucl: float
    lcl: float
    values: list
    violations: list          # indices flagged as out-of-control
    n_violations: int
    verdict: str


def shewhart_individuals_chart(values, sigma_multiplier: float = 3.0) -> ControlChartResult:
    """
    Individuals (I-MR based) control chart: appropriate when you have one
    measurement per time point (e.g. daily defect rate, per-unit cycle time).
    Sigma is estimated from the average moving range, not the raw std dev,
    which is the standard, more robust approach for I-charts.
    """
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any():
        raise ValueError("Data contains missing (NaN) values — remove or impute them before running a control chart.")
    n = len(values)
    if n < 2:
        raise ValueError("Need at least 2 points for a control chart.")

    center = values.mean()
    moving_ranges = np.abs(np.diff(values))
    mr_bar = moving_ranges.mean()
    sigma_hat = mr_bar / 1.128  # d2 constant for n=2 subgroups (standard I-MR constant)

    ucl = center + sigma_multiplier * sigma_hat
    lcl = center - sigma_multiplier * sigma_hat

    violations = [i for i, v in enumerate(values) if v > ucl or v < lcl]

    if violations:
        verdict = (f"{len(violations)} point(s) fell outside the {sigma_multiplier}-sigma control limits "
                    f"({[int(i) for i in violations]}) — process shows signs of an out-of-control shift.")
    else:
        verdict = f"All {n} points stayed within the {sigma_multiplier}-sigma control limits — process appears stable."

    return ControlChartResult(
        chart_type="Shewhart Individuals (I-chart)",
        center_line=round(float(center), 4),
        ucl=round(float(ucl), 4),
        lcl=round(float(lcl), 4),
        values=values.tolist(),
        violations=violations,
        n_violations=len(violations),
        verdict=verdict,
    )


def _cusum_episodes(sh_pos, sh_neg, h):
    """Group consecutive violation points into distinct drift episodes,
    so a chart that flips direction (down then up) is reported accurately
    instead of only describing the final direction."""
    episodes = []
    current_dir, start = None, None
    for i in range(len(sh_pos)):
        direction = None
        if sh_pos[i] > h:
            direction = "upward"
        elif sh_neg[i] < -h:
            direction = "downward"
        if direction != current_dir:
            if current_dir is not None:
                episodes.append((current_dir, start, i - 1))
            current_dir, start = direction, i if direction else None
    if current_dir is not None:
        episodes.append((current_dir, start, len(sh_pos) - 1))
    return episodes


def cusum_chart(values, target: float = None, std_dev: float = None,
                 k: float = 0.5, h: float = 5.0) -> ControlChartResult:
    """
    Tabular CUSUM: accumulates deviations from a target in standardized (sigma) units.
    k = allowance/slack (typically 0.5 sigma) — deviations smaller than this are
    treated as noise, not drift.
    h = decision threshold (typically 4-5 sigma) — cumulative sum beyond this
    signals a real, sustained shift.
    Detects gradual drift that a Shewhart chart (single-point limits) would miss.
    """
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any():
        raise ValueError("Data contains missing (NaN) values — remove or impute them before running a control chart.")
    n = len(values)
    if n < 2:
        raise ValueError("Need at least 2 points for a CUSUM chart.")

    if target is None:
        target = values.mean()
    if std_dev is None:
        moving_ranges = np.abs(np.diff(values))
        std_dev = moving_ranges.mean() / 1.128

    if std_dev == 0:
        std_dev = 1e-9  # avoid div by zero on perfectly flat data

    z = (values - target) / std_dev  # standardize

    sh_pos, sh_neg = [0.0], [0.0]
    for zi in z:
        sh_pos.append(max(0.0, sh_pos[-1] + zi - k))
        sh_neg.append(min(0.0, sh_neg[-1] + zi + k))
    sh_pos, sh_neg = np.array(sh_pos[1:]), np.array(sh_neg[1:])

    violations = [i for i in range(n) if sh_pos[i] > h or sh_neg[i] < -h]
    episodes = _cusum_episodes(sh_pos, sh_neg, h)

    if episodes:
        if len(episodes) == 1:
            direction, start, _ = episodes[0]
            verdict = (f"CUSUM detected a sustained {direction} drift starting around point {start} "
                        f"(threshold h={h} exceeded at {len(violations)} point(s) total) — investigate a "
                        f"process change around that time, even though no single point looked extreme.")
        else:
            parts = [f"{'an' if d[0] in 'aeiou' else 'a'} {d} drift starting around point {s}" for d, s, _ in episodes]
            verdict = ("CUSUM detected multiple sustained drift episodes: " + "; then ".join(parts) +
                       f" (threshold h={h} exceeded at {len(violations)} point(s) total) — the process direction "
                       f"changed more than once, worth investigating each transition separately.")
    else:
        verdict = f"CUSUM found no sustained drift across {n} points — cumulative deviation stayed within bounds."

    return ControlChartResult(
        chart_type="CUSUM (cumulative sum)",
        center_line=round(float(target), 4),
        ucl=round(float(h), 4),
        lcl=round(float(-h), 4),
        values=list(zip(sh_pos.tolist(), sh_neg.tolist())),
        violations=violations,
        n_violations=len(violations),
        verdict=verdict,
    )