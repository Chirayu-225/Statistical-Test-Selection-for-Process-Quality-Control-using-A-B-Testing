"""
Process monitoring: is the process stable, or has it shifted / drifted?

Three complementary detectors run on the same time-ordered series:

* Individuals (I) chart  — catches SUDDEN jumps (a point beyond 3 sigma)
* Moving-range (MR) chart — catches a jump in VARIABILITY
* CUSUM                   — catches SMALL, GRADUAL drift that no single point reveals

Limits are learned from a BASELINE period (Phase I) and then applied to the
rest of the data (Phase II). Learning limits from data that already contains
the shift would widen them and hide the very problem we are looking for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from . import checks


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]

D2 = 1.128         # E[moving range] / sigma for n=2 (standard SPC constant)
D4 = 3.267         # MR chart upper-limit factor for n=2
CONFIRM_RUN = 3    # a CUSUM excursion must stay beyond h this many points to count as confirmed


@dataclass
class Finding:
    kind: str          # "shift", "drift", "variability", "pattern"
    start: int         # index where the problem appears to begin
    detected_at: int   # index where the chart first signalled
    direction: str     # "upward" / "downward" / "—"
    detail: str
    confirmed: bool = True   # False = a single, brief signal that could be chance


@dataclass
class MonitoringResult:
    values: np.ndarray
    baseline_n: int
    center: float
    sigma: float
    ucl: float
    lcl: float
    moving_range: np.ndarray
    mr_ucl: float
    rule_signals: dict             # rule name -> list of indices
    mr_signals: list
    cusum_pos: np.ndarray
    cusum_neg: np.ndarray
    cusum_h: float
    cusum_k: float
    cusum_signals: list
    findings: list
    status: str                    # "stable", "watch" (only weak signals), "issue"
    checks: dict = field(default_factory=dict)
    reasoning: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def default_baseline(n: int) -> int:
    """Default Phase I window: the first half of the data (at least 8, at most
    100 points). Short baselines give noisy sigma estimates and false alarms."""
    return int(min(100, max(min(n, 8), n // 2)))


def western_electric(values, center, sigma, limit: float = 3.0, sensitive: bool = False) -> dict:
    """Rule 1 (always): one point beyond 3 sigma.
    Sensitive rules (optional; more false alarms, faster detection):
      Rule 2: 2 of 3 consecutive points beyond 2 sigma on the same side
      Rule 3: 4 of 5 consecutive points beyond 1 sigma on the same side
      Rule 4: 8 consecutive points on the same side of the center line"""
    x = np.asarray(values, float)
    z = (x - center) / sigma if sigma > 0 else np.sign(x - center) * np.inf
    out = {f"Rule 1: point beyond {limit:g}σ": [int(i) for i in np.where(np.abs(z) > limit)[0]]}
    if not sensitive:
        return out

    def window_rule(width, need, thresh):
        hits = []
        for i in range(width - 1, len(z)):
            w = z[i - width + 1:i + 1]
            if (w > thresh).sum() >= need or (w < -thresh).sum() >= need:
                hits.append(i)
        return hits

    out["Rule 2: 2 of 3 beyond 2σ"] = window_rule(3, 2, 2)
    out["Rule 3: 4 of 5 beyond 1σ"] = window_rule(5, 4, 1)
    out["Rule 4: 8 in a row on one side"] = window_rule(8, 8, 0)
    return out


def cusum(values, target, sigma, k: float = 0.5, h: float = 5.0):
    """Tabular CUSUM in sigma units. k = slack (half the shift you want to catch,
    0.5 -> tuned for a 1-sigma shift). h = decision threshold (5 -> about one
    false alarm per ~465 in-control points per side)."""
    x = np.asarray(values, float)
    z = (x - target) / sigma if sigma > 0 else np.zeros_like(x)
    pos, neg = np.zeros(len(x)), np.zeros(len(x))
    cp, cn = 0.0, 0.0
    for i, zi in enumerate(z):
        cp, cn = max(0.0, cp + zi - k), min(0.0, cn + zi + k)
        pos[i], neg[i] = cp, cn
    return pos, neg


def _cusum_findings(pos, neg, h):
    """Turn CUSUM excursions into findings, estimating WHEN each drift began:
    the change point is the last time the cumulative sum was at zero before
    it crossed the threshold (standard CUSUM change-point estimate)."""
    findings, signals = [], []
    for series, direction, sign in ((pos, "upward", 1), (neg, "downward", -1)):
        above = sign * series > h
        signals += np.where(above)[0].tolist()
        i, n = 0, len(series)
        while i < n:
            if above[i]:
                j = i
                while j >= 0 and series[j] != 0:
                    j -= 1
                start = j + 1
                end = i
                while end + 1 < n and series[end + 1] != 0:
                    end += 1
                n_above = int(above[i:end + 1].sum())
                findings.append(Finding("drift", start, i, direction,
                                        f"cumulative {direction} deviation stayed beyond h={h:g} "
                                        f"for {n_above} point(s)", confirmed=n_above >= CONFIRM_RUN))
                i = end + 1
            else:
                i += 1
    return sorted(findings, key=lambda f: f.detected_at), sorted(set(signals))


def monitor_process(values, baseline_n: int | None = None, sigma_multiplier: float = 3.0,
                    cusum_k: float = 0.5, cusum_h: float = 5.0, sensitive_rules: bool = False,
                    target: float | None = None) -> MonitoringResult:
    x = np.asarray(values, dtype=float)
    if np.isnan(x).any():
        raise ValueError("Data contains missing (NaN) values — remove or impute them first.")
    n = len(x)
    if n < 8:
        raise ValueError(f"Need at least 8 time-ordered points to monitor a process (got {n}).")

    reasoning, warnings = [], []
    b = default_baseline(n) if baseline_n is None else int(baseline_n)
    if not 5 <= b <= n:
        raise ValueError(f"Baseline length must be between 5 and {n} points.")
    base = x[:b]
    if b == n:
        warnings.append("The baseline covers all the data, so limits are learned from the same points being "
                        "judged. Collect more data or shorten the baseline for a fair test.")
    if b < 20:
        warnings.append(f"Baseline has only {b} points; 20+ (ideally 50+) are recommended for reliable limits.")

    center = float(base.mean()) if target is None else float(target)
    mr_base = np.abs(np.diff(base))
    sigma = float(mr_base.mean() / D2) if len(mr_base) else 0.0
    if sigma == 0:
        warnings.append("The baseline has no variation at all, so any change will be flagged.")
        sigma = 1e-12
    ucl, lcl = center + sigma_multiplier * sigma, center - sigma_multiplier * sigma
    reasoning.append(f"Learned normal behaviour from the first {b} points: center {center:.4g}, "
                     f"sigma {sigma:.4g} (estimated from the average moving range, which is robust to slow drift).")

    rules = western_electric(x, center, sigma, limit=sigma_multiplier, sensitive=sensitive_rules)

    mr = np.r_[np.nan, np.abs(np.diff(x))]
    mr_ucl = D4 * sigma * D2
    mr_signals = [int(i) for i in np.where(mr > mr_ucl)[0]]

    pos, neg = cusum(x, center, sigma, cusum_k, cusum_h)
    cusum_findings, cusum_signals = _cusum_findings(pos, neg, cusum_h)

    findings = []
    r1 = next(v for k, v in rules.items() if k.startswith("Rule 1"))
    if r1:
        first = r1[0]
        direction = "upward" if x[first] > center else "downward"
        findings.append(Finding("shift", first, first, direction,
                                f"{len(r1)} point(s) outside the {sigma_multiplier:g}σ limits",
                                confirmed=len(r1) >= 2))
    findings += cusum_findings
    if mr_signals and sensitive_rules:
        findings.append(Finding("variability", mr_signals[0], mr_signals[0], "—",
                                f"{len(mr_signals)} jump(s) between consecutive points larger than normal",
                                confirmed=len(mr_signals) >= 2))
    for name, idx in rules.items():
        if not name.startswith("Rule 1") and idx:
            findings.append(Finding("pattern", idx[0], idx[0], "—", f"{name} ({len(idx)} time(s))",
                                    confirmed=len(idx) >= 2))
    findings.sort(key=lambda f: f.detected_at)

    chk = {"autocorrelation": checks.autocorrelation_check(base if b >= 10 else x)}
    if chk["autocorrelation"]["flag"]:
        warnings.append(_cap(chk["autocorrelation"]["note"]) + ".")
    if b >= 8 and np.ptp(base) > 0:
        sp = float(stats.shapiro(base).pvalue)
        chk["baseline_normality_p"] = round(sp, 4)
        if sp < 0.01:
            warnings.append(f"Baseline values are clearly non-normal (Shapiro-Wilk p={sp:.3f}); 3σ limits will "
                            f"give more false alarms than usual. Treat isolated single-point signals with caution.")

    reasoning.append("Checked every point with complementary detectors: the I-chart (sudden jumps) and CUSUM "
                     "(small persistent drift that never crosses the 3σ limits)"
                     + ("; sensitive mode also adds the moving-range chart and Western Electric rules 2-4."
                        if sensitive_rules else "."))
    reasoning.append("A signal counts as CONFIRMED when it repeats (2+ points outside the limits, or CUSUM beyond "
                     f"its threshold for {CONFIRM_RUN}+ points); a single brief signal is reported as 'worth "
                     "watching' because stable processes produce those by chance.")

    return MonitoringResult(
        values=x, baseline_n=b, center=center, sigma=sigma, ucl=ucl, lcl=lcl, moving_range=mr, mr_ucl=mr_ucl,
        rule_signals=rules, mr_signals=mr_signals, cusum_pos=pos, cusum_neg=neg, cusum_h=cusum_h,
        cusum_k=cusum_k, cusum_signals=cusum_signals, findings=findings,
        status="issue" if any(f.confirmed for f in findings) else ("watch" if findings else "stable"),
        checks=chk, reasoning=reasoning, warnings=warnings)
