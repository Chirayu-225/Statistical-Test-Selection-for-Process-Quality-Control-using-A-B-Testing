"""
The end-to-end advisor: ask a business question, get an answer.

    Question                                  Function
    ----------------------------------------  ----------------------
    "Is there a real difference between       analyze_comparison()
     these groups / A/B variants?"            analyze_counts()
    "Is my process stable or drifting?"       analyze_monitoring()
    "How much data do I need?"                plan_*()

Every function returns a `Report`: a one-line answer, a recommended action,
the key numbers, the method used and WHY it was chosen, and any warnings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import checks, planning
from .compare import ComparisonResult, compare_means, compare_proportions
from .monitor import monitor_process


@dataclass
class Report:
    question: str
    verdict: str                 # machine-readable outcome, e.g. "difference", "inconclusive", "issue"
    headline: str                # the direct answer
    recommendation: str          # what to do next
    method: str
    key_numbers: list = field(default_factory=list)    # [(label, value_str)]
    reasoning: list = field(default_factory=list)      # numbered "why this method" trail
    warnings: list = field(default_factory=list)
    tables: dict = field(default_factory=dict)         # name -> DataFrame
    result: object = None                              # underlying ComparisonResult / MonitoringResult

    def to_markdown(self) -> str:
        lines = [f"# {self.question}", "", f"**Answer:** {self.headline}", "",
                 f"**Recommendation:** {self.recommendation}", "", f"**Method:** {self.method}", ""]
        if self.key_numbers:
            lines += ["## Key numbers", "", "| Measure | Value |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in self.key_numbers] + [""]
        if self.reasoning:
            lines += ["## Why this method", ""] + [f"{i}. {r}" for i, r in enumerate(self.reasoning, 1)] + [""]
        if self.warnings:
            lines += ["## Warnings", ""] + [f"- {w}" for w in self.warnings] + [""]
        for name, t in self.tables.items():
            if t is not None and len(t):
                lines += [f"## {name}", "", t.to_markdown(index=False), ""]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _num(x, digits=4):
    if x is None:
        return "—"
    if isinstance(x, float) and not np.isfinite(x):
        return "∞" if x > 0 else "−∞"
    return f"{x:,.{digits}g}"


def _p(p):
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def _p_eq(p):
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.4f}"


def _signed(x, is_rate):
    if is_rate:
        return f"{x * 100:+.2f} percentage points"
    return f"{x:+,.4g}"


def _pct(x):
    return "—" if x is None else f"{x * 100:+.1f}%"


# ---------------------------------------------------------------------------
# Comparing groups
# ---------------------------------------------------------------------------

_BINARY_TRUE = {"1", "true", "yes", "y", "converted", "success"}
_BINARY_FALSE = {"0", "false", "no", "n", "not converted", "failure"}


def infer_metric_kind(series: pd.Series) -> str:
    """'binary' for yes/no outcomes (0/1, True/False, yes/no), 'numeric' otherwise."""
    s = series.dropna()
    if s.empty:
        raise ValueError("The metric column has no values.")
    if pd.api.types.is_bool_dtype(s):
        return "binary"
    uniq = {str(v).strip().lower() for v in s.unique()}
    if pd.api.types.is_numeric_dtype(s):
        return "binary" if set(s.unique()).issubset({0, 1}) else "numeric"
    if uniq.issubset(_BINARY_TRUE | _BINARY_FALSE) and len(uniq) <= 2:
        return "binary"
    raise ValueError(f"'{series.name}' is neither numeric nor a yes/no outcome — pick another metric column.")


def _to_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    return series.map(lambda v: np.nan if pd.isna(v) else float(str(v).strip().lower() in _BINARY_TRUE))


def analyze_comparison(df: pd.DataFrame, metric: str, group: str, control=None, alpha: float = 0.05,
                       higher_is_better: bool = True, randomized: bool = False, expected_split=None,
                       min_meaningful: float | None = None, focus: str = "average") -> Report:
    """Answer: is there a real difference in `metric` between the groups in `group`?

    randomized=True marks an A/B test: group sizes are checked for Sample Ratio Mismatch.
    min_meaningful: the smallest difference that matters to the business, in metric
    units (proportion units for yes/no metrics, e.g. 0.01 = 1 percentage point).
    focus: "average" (compare means when valid) or "typical" (rank-based; numeric metrics only).
    """
    if df is None or len(df) == 0:
        raise ValueError("No data provided — the dataset is empty.")
    for c in (metric, group):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in the dataset.")
    if metric == group:
        raise ValueError("The metric column and the group column must be different.")

    kind = infer_metric_kind(df[metric])
    data = df[[group, metric]].dropna(subset=[group])
    if kind == "binary":
        data = data.assign(**{metric: _to_binary(data[metric])}).dropna()
        agg = data.groupby(group, sort=True)[metric].agg(["sum", "count"])
        if control is not None and control not in agg.index:
            raise ValueError(f"Control group '{control}' not found. Groups are: {agg.index.tolist()}")
        result = compare_proportions(agg["sum"].astype(int).tolist(), agg["count"].tolist(),
                                     names=agg.index.tolist(), alpha=alpha, control=control)
    else:
        groups = {g: s[metric].dropna().to_numpy(float) for g, s in data.groupby(group, sort=True)}
        result = compare_means(groups, alpha=alpha, control=control, focus=focus)

    sizes = result.group_summary["total" if kind == "binary" else "n"].tolist()
    srm = checks.sample_ratio_mismatch(sizes, expected_split) if randomized else None
    return _comparison_report(result, metric, kind, higher_is_better, randomized, srm, min_meaningful)


def analyze_counts(names, conversions, totals, control=None, alpha: float = 0.05,
                   higher_is_better: bool = True, randomized: bool = False, expected_split=None,
                   min_meaningful: float | None = None, metric: str = "conversion rate") -> Report:
    """Same question as analyze_comparison, from summary counts (conversions out of totals)."""
    result = compare_proportions(conversions, totals, names=names, alpha=alpha, control=control)
    srm = checks.sample_ratio_mismatch(result.group_summary["total"].tolist(), expected_split) if randomized else None
    return _comparison_report(result, metric, "binary", higher_is_better, randomized, srm, min_meaningful)


def _comparison_report(r: ComparisonResult, metric, kind, higher_is_better, randomized, srm, min_meaningful):
    is_rate = kind == "binary"
    k = len(r.groups)
    ctrl = r.groups[0]
    question = (f"Did the variant change {metric}?" if randomized and k == 2
                else f"Is there a real difference in {metric} between {', '.join(map(str, r.groups))}?")
    reasoning = list(r.reasoning)
    warnings = list(r.warnings)
    summary = r.group_summary.copy()
    key = []

    if srm is not None:
        reasoning.insert(0, "This is a randomized experiment, so the group sizes were checked first: "
                         + srm["note"] + ".")

    # ---- headline -------------------------------------------------------------------------
    verdict = "difference" if r.significant else "no_difference"
    level = "rate" if is_rate else ("median" if r.route == "ranks" else "mean")
    what = {"rate": metric if "rate" in str(metric).lower() else f"rate of '{metric}'", "mean": f"average {metric}", "median": f"typical (median) {metric}"}[level]

    if k == 2:
        other = r.groups[1]
        est, ci = r.estimate, r.ci
        if r.significant:
            up = est > 0
            headline = (f"Yes. {other}'s {what} is {'higher' if up else 'lower'} than {ctrl}'s: "
                        f"{_signed(est, is_rate)}"
                        + (f" ({_pct(r.relative_lift)} relative)" if r.relative_lift is not None else "")
                        + f". This is unlikely to be chance ({_p_eq(r.p_value)}).")
        else:
            headline = (f"No clear difference. {other}'s {what} differs from {ctrl}'s by "
                        f"{_signed(est, is_rate)}, which is within what random noise produces ({_p_eq(r.p_value)}).")
        key += [("Test used", r.test_used), ("p-value", _p(r.p_value)),
                (f"Difference in {level} ({other} − {ctrl})", _signed(est, is_rate))]
        if ci is not None:
            conf = int(round((1 - r.alpha) * 100))
            fmt = (lambda x: f"{x * 100:+.2f} pp") if is_rate else (lambda x: _num(x))
            key.append((f"{conf}% confidence interval", f"{fmt(ci[0])} to {fmt(ci[1])}"))
        if r.relative_lift is not None:
            key.append(("Relative change", _pct(r.relative_lift)))
    else:
        if r.significant:
            if r.posthoc is not None and r.posthoc["significant"].any():
                diff_col = next(c for c in r.posthoc.columns if "difference" in c)
                sig = r.posthoc[r.posthoc["significant"]]
                pairs = [f"{g2} vs {g1} ({_signed(float(d), is_rate)})"
                         for g1, g2, d in sig[["group_1", "group_2", diff_col]].values]
                headline = (f"Yes, the groups differ ({_p_eq(r.p_value)}). The pairs that really differ are: "
                            f"{'; '.join(pairs)}.")
            else:
                headline = (f"The groups differ overall ({_p_eq(r.p_value)}), but no single pair is clearly "
                            f"different after correcting for multiple comparisons — the effect is spread out.")
        else:
            headline = (f"No clear difference among the {k} groups ({_p_eq(r.p_value)}); the variation looks like "
                        f"random noise.")
        col = "rate" if is_rate else ("median" if r.route == "ranks" else "mean")
        best = summary.loc[summary[col].idxmax() if higher_is_better else summary[col].idxmin(), "group"]
        worst = summary.loc[summary[col].idxmin() if higher_is_better else summary[col].idxmax(), "group"]
        key += [("Test used", r.test_used), ("p-value", _p(r.p_value)),
                (f"Best {col}", f"{best}"), (f"Worst {col}", f"{worst}")]

    if r.route == "proportions" and k == 2:
        # Standardised effects on conversion rates are almost always "negligible" by Cohen's labels even when the
        # business impact is large, so the label is omitted; the absolute and relative differences carry the story.
        key.append((f"Standardised effect ({r.effect_size_name})", _num(r.effect_size, 3)))
    else:
        key.append((f"Effect size ({r.effect_size_name})", f"{_num(r.effect_size, 3)} ({r.effect_size_label})"))

    # ---- practical significance & power --------------------------------------------------
    if k == 2 and r.ci is not None and min_meaningful:
        thr = abs(min_meaningful)
        lo, hi = r.ci
        thr_s = _signed(thr, is_rate).lstrip("+")
        if lo >= thr or hi <= -thr:
            reasoning.append(f"The whole confidence interval is beyond the smallest difference that matters "
                             f"({thr_s}), so the effect is both real and big enough to matter.")
            verdict = "meaningful_difference"
        elif r.significant:
            reasoning.append(f"The effect is real, but the confidence interval includes values smaller than "
                             f"{thr_s}, so it may be too small to matter.")
            verdict = "small_difference"
        elif -thr < lo and hi < thr:
            reasoning.append(f"The whole confidence interval is inside ±{thr_s}, so any difference is too small "
                             f"to matter — this is evidence of NO meaningful difference, not just a lack of evidence.")
            verdict = "no_meaningful_difference"
        else:
            reasoning.append(f"The confidence interval still includes differences larger than {thr_s}, so a "
                             f"meaningful effect has not been ruled out — the test is inconclusive.")
            verdict = "inconclusive"

    if k == 2 and not r.significant and r.route != "degenerate":
        n_col = "total" if is_rate else "n"
        n1, n2 = summary[n_col].iloc[0], summary[n_col].iloc[1]
        try:
            if is_rate:
                base = float(summary["rate"].iloc[0])
                mde = planning.mde_for_proportions(base, int(n1), int(n2), alpha=r.alpha)
                need = planning.n_for_proportions(base, mde / 2, alpha=r.alpha) if 0 < base + mde / 2 < 1 else None
            else:
                pooled_sd = float(np.sqrt((summary["sd"] ** 2).mean()))
                mde = planning.mde_for_means(int(n1), int(n2), pooled_sd, alpha=r.alpha)
                need = planning.n_for_means(mde / 2, pooled_sd, alpha=r.alpha) if pooled_sd > 0 else None
            key.append(("Smallest difference this data could reliably detect (80% power)", _signed(mde, is_rate)))
            reasoning.append(f"With these sample sizes, only differences of about {_signed(mde, is_rate)} or more "
                             f"would be detected reliably. A real effect smaller than that could still exist"
                             + (f"; detecting half that size needs about {need:,} per group." if need else "."))
        except Exception:  # power solver can fail on extreme inputs; the report is still valid without it
            pass

    if r.route == "ranks" and "TYPICAL" not in " ".join(reasoning):
        warnings.append("Because the data is skewed or has outliers in a small sample, this compares typical "
                        "values (medians / rankings), not averages. If the AVERAGE matters (e.g. total revenue), "
                        "collect more data so the mean-based test becomes valid.")
    small = [g for g, n in zip(summary["group"], summary["total" if is_rate else "n"]) if n < 10]
    if small:
        warnings.append(f"Very small groups ({', '.join(map(str, small))}): results are fragile.")

    # ---- recommendation --------------------------------------------------------------------
    if srm is not None and srm["mismatch"]:
        verdict = "untrustworthy"
        warnings.insert(0, srm["note"])
        headline = ("Don't trust this result yet. The groups are not the sizes the experiment planned, which "
                    "usually means assignment or tracking is broken. (Statistical result shown below for reference.)")
        recommendation = ("Find and fix the cause of the sample ratio mismatch (bots, redirects, logging, "
                          "assignment bugs), then rerun the experiment.")
    elif verdict in ("difference", "meaningful_difference", "small_difference"):
        if k == 2:
            other = r.groups[1]
            better = (r.estimate > 0) == higher_is_better
            if randomized:
                recommendation = (f"{other} performs {'better' if better else 'worse'} than {ctrl}. "
                                  + (f"Roll out {other}" if better else f"Keep {ctrl}; do not roll out {other}")
                                  + (" — but the gain may be smaller than what matters to you; weigh it against "
                                     "the cost." if verdict == "small_difference" else "."))
            else:
                recommendation = (f"{other} is {'better' if better else 'worse'} than {ctrl}. Investigate what "
                                  f"distinguishes them" + (" and replicate it." if better else " and fix it."))
        else:
            recommendation = (f"Focus on {worst} (worst) and learn from {best} (best). Check the pairwise table to "
                              f"see which differences are real.")
    elif verdict == "no_meaningful_difference":
        recommendation = "Treat the groups as equivalent for practical purposes; choose on cost or simplicity."
    else:
        if k == 2:
            recommendation = ("Don't act on this difference yet. If an effect smaller than the 'smallest detectable "
                              "difference' above would still matter, collect more data before deciding.")
        else:
            recommendation = (f"Don't treat the groups as different yet. {worst} looks worst, but the evidence is "
                              f"not strong enough; collect more data per group if it matters.")

    tables = {"Group summary": summary}
    if r.posthoc is not None:
        tables["Which pairs differ (adjusted for multiple comparisons)"] = r.posthoc
    return Report(question=question, verdict=verdict, headline=headline, recommendation=recommendation,
                  method=r.test_used, key_numbers=key, reasoning=reasoning, warnings=warnings,
                  tables=tables, result=r)


# ---------------------------------------------------------------------------
# Monitoring a process
# ---------------------------------------------------------------------------

def analyze_monitoring(df: pd.DataFrame, metric: str, order_col: str | None = None, baseline_n: int | None = None,
                       sensitive_rules: bool = False, sigma_multiplier: float = 3.0, cusum_k: float = 0.5,
                       cusum_h: float = 5.0) -> Report:
    """Answer: is the process stable, or did it shift / drift? Points are taken in
    `order_col` order (e.g. a timestamp or batch number), else in row order."""
    if df is None or len(df) == 0:
        raise ValueError("No data provided — the dataset is empty.")
    if metric not in df.columns:
        raise ValueError(f"Column '{metric}' not found in the dataset.")
    if not pd.api.types.is_numeric_dtype(df[metric]):
        raise ValueError(f"'{metric}' is not numeric — pick a numeric measurement column.")
    data = df
    if order_col:
        if order_col not in df.columns:
            raise ValueError(f"Column '{order_col}' not found in the dataset.")
        data = df.sort_values(order_col, kind="stable")
    dropped = int(data[metric].isna().sum())
    data = data.dropna(subset=[metric])
    labels = ([f"{order_col} {v}" for v in data[order_col].astype(str)] if order_col
              else [f"row {i + 1}" for i in range(len(data))])

    m = monitor_process(data[metric].to_numpy(float), baseline_n=baseline_n, sigma_multiplier=sigma_multiplier,
                        cusum_k=cusum_k, cusum_h=cusum_h, sensitive_rules=sensitive_rules)
    warnings = list(m.warnings)
    if dropped:
        warnings.append(f"{dropped} missing value(s) were skipped.")

    def where(i):
        return labels[i] if 0 <= i < len(labels) else f"point {i}"

    confirmed = [f for f in m.findings if f.confirmed]
    weak = [f for f in m.findings if not f.confirmed]
    describe = {"shift": "a sudden {d} jump", "drift": "a gradual {d} drift",
                "variability": "an increase in variability", "pattern": "an unusual run pattern"}

    def phrase(f):
        text = describe[f.kind].format(d=f.direction)
        when = (f"starting around {where(f.start)} (flagged at {where(f.detected_at)})"
                if f.start != f.detected_at else f"at {where(f.start)}")
        return f"{text} {when}"

    if m.status == "stable":
        headline = f"Yes, the process looks stable: no signals across {len(m.values)} points."
        recommendation = "No action needed. Keep monitoring with the same limits."
    elif m.status == "watch":
        headline = (f"Probably stable, with one brief signal worth watching: {phrase(weak[0])}. Single signals "
                    f"like this also happen by chance in stable processes.")
        recommendation = f"Check what happened around {where(weak[0].start)}, and watch whether the signal repeats."
    else:
        first = confirmed[0]
        headline = f"No, the process changed. Detected {phrase(first)}."
        if len(confirmed) > 1:
            headline += " Also: " + "; ".join(phrase(f) for f in confirmed[1:3]) + "."
        recommendation = (f"Investigate what changed around {where(first.start)} (new material lot, shift change, "
                          f"maintenance, settings, measurement system). Once fixed, re-learn the limits.")

    key = [("Points analysed", f"{len(m.values)}"), ("Baseline (learning) period", f"first {m.baseline_n} points"),
           ("Center line", _num(m.center)), ("Control limits", f"{_num(m.lcl)} to {_num(m.ucl)}"),
           ("Points outside limits", str(len(next(iter(m.rule_signals.values()))))),
           ("Drift episodes (CUSUM)", str(sum(f.kind == "drift" for f in m.findings)))]
    findings = pd.DataFrame([{"what": f.kind, "direction": f.direction, "started": where(f.start),
                              "flagged": where(f.detected_at), "strength": "confirmed" if f.confirmed else "watch",
                              "detail": f.detail} for f in m.findings])
    return Report(question=f"Is the {metric} process stable over time?", verdict=m.status, headline=headline,
                  recommendation=recommendation, method="Individuals control chart + CUSUM (limits from a baseline period)",
                  key_numbers=key, reasoning=list(m.reasoning), warnings=warnings,
                  tables={"Signals": findings}, result=m)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan_conversion_test(baseline_rate: float, min_change: float, relative: bool = True, alpha: float = 0.05,
                         power: float = 0.8, groups: int = 2, daily_traffic: float | None = None) -> Report:
    """How many users per group to detect a change in a conversion rate.
    min_change is relative (0.10 = a 10% lift) unless relative=False (then 0.01 = 1 percentage point)."""
    abs_change = baseline_rate * min_change if relative else min_change
    n = planning.n_for_proportions(baseline_rate, abs_change, alpha=alpha, power=power)
    if groups > 2:  # Bonferroni across the (groups - 1) comparisons against control
        n = planning.n_for_proportions(baseline_rate, abs_change, alpha=alpha / (groups - 1), power=power)
    key = [("Users needed per group", f"{n:,}"), ("Total users", f"{n * groups:,}"),
           ("Detects", f"{baseline_rate:.2%} → {baseline_rate + abs_change:.2%}")]
    days = planning.duration_days(n, groups, daily_traffic) if daily_traffic else None
    if days:
        key.append(("Estimated duration", f"{days:.0f} days at {daily_traffic:,.0f} users/day"))
    reasoning = [f"Uses the normal approximation with Cohen's h for comparing two proportions, "
                 f"alpha = {alpha}, power = {power:.0%}."]
    if groups > 2:
        reasoning.append(f"With {groups} groups, alpha is split across the {groups - 1} comparisons with the "
                         f"control (Bonferroni) so the overall false-positive rate stays at {alpha}.")
    headline = (f"You need about {n:,} users per group ({n * groups:,} in total) to reliably detect a change from "
                f"{baseline_rate:.2%} to {baseline_rate + abs_change:.2%}" + (f" — roughly {days:.0f} days." if days else "."))
    return Report(question="How much data do I need for a conversion-rate test?", verdict="plan", headline=headline,
                  recommendation="Decide the sample size up front and do not stop early when the result looks good: "
                                 "repeatedly peeking at p-values inflates false positives.",
                  method="Power analysis (two proportions)", key_numbers=key, reasoning=reasoning)


def plan_mean_test(sd: float, min_difference: float, alpha: float = 0.05, power: float = 0.8, groups: int = 2,
                   daily_traffic: float | None = None) -> Report:
    """How many observations per group to detect a difference in an average metric."""
    d = abs(min_difference) / sd
    if groups == 2:
        n = planning.n_for_means(min_difference, sd, alpha=alpha, power=power)
        method = "Power analysis (two-sample t-test)"
        reasoning = [f"Standardised effect size d = difference / SD = {d:.3f}; t-test power analysis "
                     f"with alpha = {alpha}, power = {power:.0%}."]
    else:
        f = d * np.sqrt(groups - 1) / groups  # Cohen's f when one group differs by d from all others
        n = planning.n_for_anova(f, groups, alpha=alpha, power=power)
        method = "Power analysis (one-way ANOVA)"
        reasoning = [f"Converted the difference to Cohen's f = {f:.3f} (one group differing by d = {d:.3f}, "
                     f"a conservative pattern) for {groups} groups, alpha = {alpha}, power = {power:.0%}."]
    key = [("Observations needed per group", f"{n:,}"), ("Total", f"{n * groups:,}"),
           ("Standardised effect (Cohen's d)", f"{d:.3f}")]
    days = planning.duration_days(n, groups, daily_traffic) if daily_traffic else None
    if days:
        key.append(("Estimated duration", f"{days:.0f} days at {daily_traffic:,.0f} per day"))
    headline = (f"You need about {n:,} observations per group ({n * groups:,} in total) to reliably detect a "
                f"difference of {min_difference:,.4g}" + (f" — roughly {days:.0f} days." if days else "."))
    return Report(question="How much data do I need to compare averages?", verdict="plan", headline=headline,
                  recommendation="Estimate the SD from historical data; if unsure, plan with a larger SD to be safe.",
                  method=method, key_numbers=key, reasoning=reasoning)
