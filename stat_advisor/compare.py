"""
Group comparison: is there a real difference between groups?

Decision logic (numeric metrics)
--------------------------------
1. Can every group's mean be trusted?  (checks.mean_test_validity)
     yes -> compare MEANS with Welch's t-test (2 groups) or Welch's ANOVA (3+)
     no  -> rank-based: Brunner-Munzel / Mann-Whitney U (2) or Kruskal-Wallis (3+)
2. Welch versions are used unconditionally: they are as good as Student/classic
   ANOVA when variances are equal and stay correct when they are not, so no
   variance pre-test is needed to pick between them.
3. If the 3+ group test is significant, a post-hoc test says WHICH pairs differ
   (Games-Howell after Welch's ANOVA, Dunn + Holm after Kruskal-Wallis).

Decision logic (conversion / proportion metrics)
------------------------------------------------
2 groups:  two-proportion z-test, or an exact test (Boschloo) if expected counts < 5
3+ groups: chi-square test, or a Monte Carlo exact test if expected counts < 5
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from . import checks

# Brunner-Munzel replaces Mann-Whitney when both groups have at least this many
# points (its t-approximation is unreliable below ~10 per group).
BRUNNER_MUNZEL_MIN_N = 10
# Boschloo's exact test is uniformly more powerful than Fisher's (which is
# conservative) but slower; above this per-group size fall back to Fisher.
BOSCHLOO_MAX_N = 2000


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


@dataclass
class ComparisonResult:
    test_used: str
    route: str                    # "means", "ranks", "proportions", "degenerate"
    p_value: float
    statistic: float
    alpha: float
    significant: bool
    effect_size: float
    effect_size_name: str
    effect_size_label: str
    groups: list
    group_summary: pd.DataFrame
    estimate: float | None = None         # treatment - control, in the metric's own units
    estimate_name: str | None = None
    ci: tuple | None = None               # (1 - alpha) CI for `estimate`
    relative_lift: float | None = None    # estimate / control level
    posthoc: pd.DataFrame | None = None
    checks: dict = field(default_factory=dict)
    reasoning: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Effect-size labels (Cohen's conventions; context matters more than labels)
# ---------------------------------------------------------------------------

def _label(value, cuts):
    v = abs(value)
    for cut, name in zip(cuts, ("negligible", "small", "medium")):
        if v < cut:
            return name
    return "large"


LABEL_D = (0.2, 0.5, 0.8)          # Cohen's d, Cohen's h
LABEL_R = (0.1, 0.3, 0.5)          # rank-biserial r, Cramér's V (Cohen's w)
LABEL_VAR = (0.01, 0.06, 0.14)     # omega², epsilon²


def cohens_d(control, treatment) -> float:
    a, b = np.asarray(control, float), np.asarray(treatment, float)
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return 0.0 if pooled == 0 else float((b.mean() - a.mean()) / pooled)


def omega_squared(groups) -> float:
    allv = np.concatenate(groups)
    n, k = len(allv), len(groups)
    ss_between = sum(len(g) * (np.mean(g) - allv.mean()) ** 2 for g in groups)
    ss_within = sum(((g - np.mean(g)) ** 2).sum() for g in groups)
    ms_within = ss_within / (n - k)
    denom = ss_between + ss_within + ms_within
    return float(max(0.0, (ss_between - (k - 1) * ms_within) / denom)) if denom > 0 else 0.0


def _bootstrap_median_diff_ci(control, treatment, alpha, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(control, float), np.asarray(treatment, float)
    ia = rng.integers(0, len(a), (n_boot, len(a)))
    ib = rng.integers(0, len(b), (n_boot, len(b)))
    diffs = np.median(b[ib], axis=1) - np.median(a[ia], axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Post-hoc tests for 3+ groups
# ---------------------------------------------------------------------------

def games_howell(groups: dict, alpha: float = 0.05) -> pd.DataFrame:
    """Pairwise mean comparisons that do NOT assume equal variances or equal n.
    Uses the studentized range distribution, so it controls the family-wise
    error rate across all pairs by construction."""
    names = list(groups)
    k = len(names)
    rows = []
    for g1, g2 in combinations(names, 2):
        a, b = np.asarray(groups[g1], float), np.asarray(groups[g2], float)
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        diff = b.mean() - a.mean()
        se = np.sqrt(va + vb)
        if se == 0:
            p, lo, hi = (1.0 if diff == 0 else 0.0), diff, diff
        else:
            df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
            q = abs(diff) / se * np.sqrt(2)
            p = float(stats.studentized_range.sf(q, k, df))
            margin = stats.studentized_range.ppf(1 - alpha, k, df) / np.sqrt(2) * se
            lo, hi = diff - margin, diff + margin
        rows.append({"group_1": g1, "group_2": g2, "difference (2 − 1)": round(float(diff), 4),
                     "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
                     "p_adjusted": round(min(1.0, max(0.0, p)), 6)})
    out = pd.DataFrame(rows)
    out["significant"] = out["p_adjusted"] < alpha
    return out


def dunn_test(groups: dict, alpha: float = 0.05, method: str = "holm") -> pd.DataFrame:
    """Rank-based pairwise comparisons after Kruskal-Wallis (with tie correction),
    p-values adjusted with Holm's method."""
    names = list(groups)
    allv = np.concatenate([np.asarray(groups[g], float) for g in names])
    ranks = stats.rankdata(allv)
    n_total = len(allv)
    _, tie_counts = np.unique(allv, return_counts=True)
    tie_term = (tie_counts ** 3 - tie_counts).sum() / (12 * (n_total - 1))
    mean_rank, sizes, medians, start = {}, {}, {}, 0
    for g in names:
        n = len(groups[g])
        mean_rank[g], sizes[g], medians[g] = ranks[start:start + n].mean(), n, np.median(groups[g])
        start += n
    rows = []
    for g1, g2 in combinations(names, 2):
        var = (n_total * (n_total + 1) / 12 - tie_term) * (1 / sizes[g1] + 1 / sizes[g2])
        z = (mean_rank[g2] - mean_rank[g1]) / np.sqrt(var) if var > 0 else 0.0
        rows.append({"group_1": g1, "group_2": g2,
                     "median difference (2 − 1)": round(float(medians[g2] - medians[g1]), 4),
                     "z": round(float(z), 4), "p_raw": float(2 * stats.norm.sf(abs(z)))})
    out = pd.DataFrame(rows)
    reject, p_adj, _, _ = multipletests(out["p_raw"], alpha=alpha, method=method)
    out["p_adjusted"] = np.round(p_adj, 6)
    out["significant"] = reject
    out["p_raw"] = out["p_raw"].round(6)
    return out


# ---------------------------------------------------------------------------
# Numeric metrics
# ---------------------------------------------------------------------------

def _order_groups(groups: dict, control):
    if control is None:
        return groups
    if control not in groups:
        raise ValueError(f"Control group '{control}' not found. Groups are: {list(groups)}")
    return {control: groups[control], **{k: v for k, v in groups.items() if k != control}}


def _numeric_groups_from_df(df, value_col, group_col):
    if df is None or len(df) == 0:
        raise ValueError("No data provided — the dataset is empty.")
    for c in (value_col, group_col):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in the dataset.")
    if value_col == group_col:
        raise ValueError("The metric column and the group column must be different.")
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        raise ValueError(f"'{value_col}' is not numeric — pick a numeric metric column.")
    sub = df[[group_col, value_col]].dropna(subset=[group_col])
    return {g: s[value_col].dropna().to_numpy(dtype=float) for g, s in sub.groupby(group_col, sort=True)}


def compare_means(groups: dict, alpha: float = 0.05, control=None, focus: str = "average",
                  posthoc: bool = True) -> ComparisonResult:
    """Compare a numeric metric across 2+ groups. `groups` maps name -> values.
    For two groups, differences are reported as (other group − control).

    focus="average" (default): compare means whenever the mean is trustworthy —
        right when totals matter (total revenue = average × users).
    focus="typical": always use rank-based tests — compares typical values,
        robust to outliers and more powerful for heavy-tailed data.
    posthoc=False skips the pairwise follow-up (used by the simulation study for speed)."""
    run_posthoc = posthoc   # the name `posthoc` is reused below for the result table
    if focus not in ("average", "typical"):
        raise ValueError("focus must be 'average' or 'typical'.")
    groups = {k: np.asarray(v, dtype=float) for k, v in groups.items()}
    groups = {k: v[~np.isnan(v)] for k, v in groups.items()}
    groups = _order_groups(groups, control)
    names, values = list(groups), list(groups.values())
    k = len(names)

    if k < 2:
        raise ValueError("Need at least 2 groups to compare — the group column only has 1 distinct value.")
    empty = [n for n, v in groups.items() if len(v) == 0]
    if empty:
        raise ValueError(f"Group(s) {empty} have no valid (non-missing) numeric values.")
    tiny = [n for n, v in groups.items() if len(v) < 2]
    if tiny:
        raise ValueError(f"Group(s) {tiny} have fewer than 2 data points — need at least 2 per group.")

    summary = pd.DataFrame({
        "group": names, "n": [len(v) for v in values],
        "mean": [v.mean() for v in values], "median": [np.median(v) for v in values],
        "sd": [v.std(ddof=1) for v in values],
    }).round(4)

    validity = {n: checks.mean_test_validity(v, alpha) for n, v in groups.items()}
    variance = checks.variance_check(*values, alpha=alpha)
    chk = {"validity": validity, "variance": variance}
    reasoning = [f"{k} groups found: {', '.join(map(str, names))}."]

    # Degenerate case: every group constant -> test statistics are 0/0
    if all(np.ptp(v) == 0 for v in values):
        same = len({round(float(v[0]), 10) for v in values}) == 1
        reasoning.append("Every value within each group is identical, so no statistical test is needed.")
        est = float(values[1][0] - values[0][0]) if k == 2 else None
        return ComparisonResult(
            test_used="None needed (zero variance in every group)", route="degenerate",
            p_value=1.0 if same else 0.0, statistic=0.0 if same else float("inf"), alpha=alpha,
            significant=not same, effect_size=0.0 if same else float("inf"), effect_size_name="—",
            effect_size_label="negligible" if same else "large", groups=names, group_summary=summary,
            estimate=est, estimate_name="difference in means" if k == 2 else None,
            ci=(est, est) if k == 2 else None, checks=chk, reasoning=reasoning,
            warnings=[] if same else ["Groups never vary internally — check for a data-collection error."])

    for n, v in validity.items():
        reasoning.append(f"Group '{n}': {v['note']}.")
    use_means = focus == "average" and all(v["mean_test_valid"] for v in validity.values())
    if focus == "typical":
        reasoning.append("You asked to compare TYPICAL values, so rank-based tests are used regardless "
                         "(robust to outliers; answers 'do values in one group tend to be larger?').")
    reasoning.append(_cap(variance["note"]) + ".")

    warnings = []
    posthoc = None
    est = ci = lift = None
    est_name = None

    if k == 2:
        ctrl, trt = values
        if use_means:
            res = stats.ttest_ind(trt, ctrl, equal_var=False)
            ci = tuple(float(x) for x in res.confidence_interval(1 - alpha))
            test_used, route = "Welch's t-test", "means"
            stat, p = float(res.statistic), float(res.pvalue)
            est, est_name = float(trt.mean() - ctrl.mean()), "difference in means"
            eff, eff_name = cohens_d(ctrl, trt), "Cohen's d"
            eff_label = _label(eff, LABEL_D)
            lift = est / ctrl.mean() if ctrl.mean() != 0 else None
            reasoning.append("All groups pass the validity check, so means are compared with Welch's t-test "
                             "(no equal-variance assumption).")
        else:
            u = float(stats.mannwhitneyu(trt, ctrl, alternative="two-sided").statistic)
            test_used, stat, p = None, float("nan"), float("nan")
            if min(len(trt), len(ctrl)) >= BRUNNER_MUNZEL_MIN_N and np.ptp(np.r_[trt, ctrl]) > 0:
                # Brunner-Munzel tests the same question as Mann-Whitney (is P(other > control) = 0.5?)
                # but stays valid when the groups have different spreads or shapes.
                bm = stats.brunnermunzel(trt, ctrl, alternative="two-sided")
                test_used, stat, p = "Brunner-Munzel test", float(bm.statistic), float(bm.pvalue)
            if test_used is None or not np.isfinite(p):   # small n, or complete separation
                res = stats.mannwhitneyu(trt, ctrl, alternative="two-sided")
                test_used, stat, p = "Mann-Whitney U test", float(res.statistic), float(res.pvalue)
            route = "ranks"
            prob_sup = u / (len(trt) * len(ctrl))              # P(treatment > control) + ½ P(tie)
            eff, eff_name = 2 * prob_sup - 1, "rank-biserial r"
            eff_label = _label(eff, LABEL_R)
            est, est_name = float(np.median(trt) - np.median(ctrl)), "difference in medians"
            ci = _bootstrap_median_diff_ci(ctrl, trt, alpha)
            med_c = np.median(ctrl)
            lift = est / med_c if med_c != 0 else None
            reasoning.append(("At least one group fails the validity check, so the mean is not trustworthy here. "
                              if focus == "average" else "")
                             + f"Using the rank-based {test_used}, which asks whether values in one group "
                             f"tend to be larger (P(other > control) = {prob_sup:.2f})"
                             + (" and, unlike Mann-Whitney, stays valid when spreads differ." if
                                test_used.startswith("Brunner") else "."))
    else:
        if use_means:
            res = anova_oneway(values, use_var="unequal", welch_correction=True)
            test_used, route = "Welch's ANOVA", "means"
            stat, p = float(res.statistic), float(res.pvalue)
            eff, eff_name = omega_squared(values), "omega²"
            reasoning.append("All groups pass the validity check, so means are compared with Welch's ANOVA "
                             "(no equal-variance assumption).")
        else:
            res = stats.kruskal(*values)
            test_used, route = "Kruskal-Wallis H test", "ranks"
            stat, p = float(res.statistic), float(res.pvalue)
            n_all = sum(len(v) for v in values)
            eff, eff_name = float(stat / (n_all - 1)), "epsilon²"
            if variance.get("equal_variance") is False:
                warnings.append("Spreads differ between groups. Kruskal-Wallis can then flag a difference even when "
                                "typical values are equal — treat a borderline p-value with caution.")
            reasoning.append(("At least one group fails the validity check, so the rank-based Kruskal-Wallis "
                              "test is used instead of comparing means.") if focus == "average"
                             else "Using the rank-based Kruskal-Wallis test.")
        eff_label = _label(eff, LABEL_VAR)
        if p < alpha and run_posthoc:
            posthoc = games_howell(groups, alpha) if use_means else dunn_test(groups, alpha)
            reasoning.append(f"The overall test is significant, so a post-hoc test "
                             f"({'Games-Howell' if use_means else 'Dunn with Holm correction'}) identifies "
                             f"which pairs differ while controlling false positives across all "
                             f"{k * (k - 1) // 2} pairs.")

    if np.isnan(p):  # defensive: should be unreachable after the degenerate branch
        raise ValueError("The test returned an undefined p-value for this data.")

    return ComparisonResult(
        test_used=test_used, route=route, p_value=float(p), statistic=float(stat), alpha=alpha,
        significant=bool(p < alpha), effect_size=float(eff), effect_size_name=eff_name,
        effect_size_label=eff_label, groups=names, group_summary=summary, estimate=est,
        estimate_name=est_name, ci=ci, relative_lift=lift, posthoc=posthoc, checks=chk,
        reasoning=reasoning, warnings=warnings)


def compare_means_df(df: pd.DataFrame, value_col: str, group_col: str, alpha: float = 0.05,
                     control=None, focus: str = "average") -> ComparisonResult:
    return compare_means(_numeric_groups_from_df(df, value_col, group_col), alpha=alpha, control=control,
                         focus=focus)


# ---------------------------------------------------------------------------
# Proportions (conversion rates)
# ---------------------------------------------------------------------------

def _two_prop_p(s, t):
    exp_ok = checks.expected_counts_ok(s, t)
    if exp_ok["ok"]:
        stat, p = proportions_ztest(s, t)
        return "Two-proportion z-test", float(stat), float(p), exp_ok
    table = [[s[0], t[0] - s[0]], [s[1], t[1] - s[1]]]
    if max(t) <= BOSCHLOO_MAX_N:
        res = stats.boschloo_exact(table, alternative="two-sided")
        return "Boschloo's exact test", float(res.statistic), float(min(1.0, res.pvalue)), exp_ok
    odds, p = stats.fisher_exact(table)
    return "Fisher's exact test", float(odds), float(p), exp_ok


def _mc_chi2_pvalue(successes, totals, n_sim=20000, seed=0):
    """Exact conditional test via simulation: with the margins fixed, the
    successes are distributed across groups by a multivariate hypergeometric."""
    s, t = np.asarray(successes, dtype=np.int64), np.asarray(totals, dtype=np.int64)
    exp_s = t * s.sum() / t.sum()      # expected successes per group under H0
    exp_f = t - exp_s                  # expected failures per group under H0

    def chi2_stat(succ):               # works on one table (k,) or many (n_sim, k)
        return ((succ - exp_s) ** 2 / exp_s + ((t - succ) - exp_f) ** 2 / exp_f).sum(axis=-1)

    observed = float(chi2_stat(s))
    sims = np.random.default_rng(seed).multivariate_hypergeometric(t, int(s.sum()), size=n_sim)
    p = (1 + np.sum(chi2_stat(sims) >= observed - 1e-9)) / (n_sim + 1)
    return float(observed), float(p)


def compare_proportions(successes, totals, names=None, alpha: float = 0.05, control=None) -> ComparisonResult:
    """Compare conversion rates across 2+ groups. For two groups, differences are
    (other − control) in proportion units (0.012 = +1.2 percentage points)."""
    successes, totals = [int(x) for x in successes], [int(x) for x in totals]
    if len(successes) != len(totals):
        raise ValueError("successes and totals must have the same number of groups.")
    if len(successes) < 2:
        raise ValueError("Need at least 2 groups to compare proportions.")
    for s, t in zip(successes, totals):
        if t <= 0:
            raise ValueError("Each group's total (sample size) must be greater than 0.")
        if s < 0 or s > t:
            raise ValueError(f"Invalid data: successes ({s}) must be between 0 and totals ({t}).")
    names = list(names) if names is not None else [f"group_{i + 1}" for i in range(len(successes))]
    if len(set(names)) != len(names):
        raise ValueError("Group names must be unique.")
    if control is not None:
        if control not in names:
            raise ValueError(f"Control group '{control}' not found.")
        i = names.index(control)
        order = [i] + [j for j in range(len(names)) if j != i]
        names = [names[j] for j in order]
        successes = [successes[j] for j in order]
        totals = [totals[j] for j in order]

    s, t = np.array(successes), np.array(totals)
    rates = s / t
    k = len(s)
    summary = pd.DataFrame({"group": names, "conversions": s, "total": t, "rate": rates.round(5)})
    exp_ok = checks.expected_counts_ok(s, t)
    chk = {"expected_counts": exp_ok}
    reasoning = [f"{k} groups found: {', '.join(map(str, names))}.",
                 "The metric is a yes/no outcome, so conversion rates are compared directly "
                 "(no normality check needed)."]

    if s.sum() == 0 or s.sum() == t.sum():
        what = "zero conversions" if s.sum() == 0 else "100% conversion"
        reasoning.append(f"Every group has {what}, so there is no variation to test.")
        return ComparisonResult(
            test_used=f"None needed ({what} in every group)", route="degenerate", p_value=1.0, statistic=0.0,
            alpha=alpha, significant=False, effect_size=0.0, effect_size_name="—",
            effect_size_label="negligible", groups=names, group_summary=summary,
            estimate=0.0 if k == 2 else None, estimate_name="difference in rates" if k == 2 else None,
            ci=None, checks=chk, reasoning=reasoning,
            warnings=[f"No variation in the outcome ({what}). Collect more data before drawing conclusions."])

    reasoning.append(_cap(exp_ok["note"]) + ".")
    posthoc = None
    est = ci = lift = None
    est_name = None

    if k == 2:
        test_used, stat, p, _ = _two_prop_p(successes, totals)
        est, est_name = float(rates[1] - rates[0]), "difference in rates"
        lo, hi = confint_proportions_2indep(s[1], t[1], s[0], t[0], method="newcomb", compare="diff", alpha=alpha)
        ci = (float(lo), float(hi))
        eff = float(2 * np.arcsin(np.sqrt(rates[1])) - 2 * np.arcsin(np.sqrt(rates[0])))
        eff_name, eff_label = "Cohen's h", _label(eff, LABEL_D)
        lift = est / rates[0] if rates[0] > 0 else None
        reasoning.append(f"Using {test_used}; the confidence interval uses Newcombe's method, which stays "
                         f"accurate for small or extreme rates.")
    else:
        table = np.vstack([s, t - s])
        if exp_ok["ok"]:
            stat, p, _, _ = stats.chi2_contingency(table, correction=False)
            test_used = "Chi-square test of independence"
        else:
            stat, p = _mc_chi2_pvalue(s, t)
            test_used = "Exact chi-square test (Monte Carlo, 20,000 simulations)"
        eff = float(np.sqrt(stat / table.sum()))   # Cramér's V for a 2 x k table
        eff_name, eff_label = "Cramér's V", _label(eff, LABEL_R)
        reasoning.append(f"Using {test_used} across all {k} groups at once.")
        if p < alpha:
            rows = []
            for i, j in combinations(range(k), 2):
                name, _, pij, _ = _two_prop_p([successes[i], successes[j]], [totals[i], totals[j]])
                rows.append({"group_1": names[i], "group_2": names[j],
                             "rate difference (2 − 1)": round(float(rates[j] - rates[i]), 5),
                             "test": name, "p_raw": pij})
            posthoc = pd.DataFrame(rows)
            reject, p_adj, _, _ = multipletests(posthoc["p_raw"], alpha=alpha, method="holm")
            posthoc["p_adjusted"], posthoc["significant"] = np.round(p_adj, 6), reject
            posthoc["p_raw"] = posthoc["p_raw"].round(6)
            reasoning.append("The overall test is significant, so every pair is tested and p-values are "
                             "Holm-adjusted to keep false positives under control.")

    return ComparisonResult(
        test_used=test_used, route="proportions", p_value=float(p), statistic=float(stat), alpha=alpha,
        significant=bool(p < alpha), effect_size=eff, effect_size_name=eff_name, effect_size_label=eff_label,
        groups=names, group_summary=summary, estimate=est, estimate_name=est_name, ci=ci,
        relative_lift=lift, posthoc=posthoc, checks=chk, reasoning=reasoning)


def adjust_pvalues(p_values, method: str = "holm", alpha: float = 0.05) -> dict:
    """Multiple-comparison correction for a family of tests (holm, bonferroni, fdr_bh)."""
    reject, p_adj, _, _ = multipletests(p_values, alpha=alpha, method=method)
    return {"method": method, "p_adjusted": [float(x) for x in p_adj], "significant": [bool(x) for x in reject]}
