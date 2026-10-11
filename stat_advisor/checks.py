"""
Assumption and data-quality checks.

Each check returns a small dict (machine-readable) plus a plain-English
`note`, so the advisor can explain *why* a method was chosen.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

# A group is "safe for a mean-based test" if it either looks normal (small
# samples) or is large enough for the Central Limit Theorem to make the
# sample mean approximately normal. "Large enough" uses Cochran's rule,
# n > 25 * skewness^2, with a floor of 30.
CLT_FLOOR = 30
SHAPIRO_MAX_N = 5000


def mean_test_validity(values, alpha: float = 0.05, rng_seed: int = 0) -> dict:
    """Decide whether a mean-based test (Welch t / Welch ANOVA) is valid for one group."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    skew = float(stats.skew(x, bias=False)) if n >= 3 and np.ptp(x) > 0 else 0.0
    clt_needed = max(CLT_FLOOR, int(np.ceil(25 * skew ** 2)))

    shapiro_p = None
    if n >= 3 and np.ptp(x) > 0:
        sample = x
        if n > SHAPIRO_MAX_N:  # Shapiro is defined up to n=5000; subsample deterministically
            sample = np.random.default_rng(rng_seed).choice(x, SHAPIRO_MAX_N, replace=False)
        shapiro_p = float(stats.shapiro(sample).pvalue)

    looks_normal = shapiro_p is not None and shapiro_p > alpha
    large_enough = n >= clt_needed
    valid = bool(looks_normal or large_enough)

    if looks_normal:
        reason = f"looks normal (Shapiro-Wilk p={shapiro_p:.3f})"
    elif large_enough:
        reason = (f"not normal, but n={n} is large enough for the Central Limit Theorem "
                  f"(skewness {skew:.2f} needs n ≥ {clt_needed})")
    else:
        sp = f"Shapiro-Wilk p={shapiro_p:.3f}" if shapiro_p is not None else "too few points to check"
        reason = (f"not normal ({sp}) and n={n} is too small for the Central Limit Theorem "
                  f"(skewness {skew:.2f} needs n ≥ {clt_needed})")

    return {"n": n, "skewness": round(skew, 3), "shapiro_p": None if shapiro_p is None else round(shapiro_p, 4),
            "clt_n_needed": clt_needed, "mean_test_valid": valid, "note": reason}


def variance_check(*groups, alpha: float = 0.05) -> dict:
    """Brown-Forsythe (median-centred Levene) test. Diagnostic only: Welch
    methods are used regardless, because pre-testing variances to choose
    between Student and Welch inflates error rates."""
    if all(np.ptp(np.asarray(g, dtype=float)) == 0 for g in groups):
        return {"p_value": None, "equal_variance": None, "sd_ratio": None,
                "note": "all groups are constant; variance test not applicable"}
    p = float(stats.levene(*groups, center="median").pvalue)
    sds = [np.std(g, ddof=1) for g in groups]
    ratio = max(sds) / min(sds) if min(sds) > 0 else float("inf")
    equal = bool(p > alpha)
    note = (f"spreads are similar (Brown-Forsythe p={p:.3f})" if equal else
            f"spreads differ (Brown-Forsythe p={p:.3f}, largest/smallest SD = {ratio:.1f}) — "
            f"Welch's method handles this")
    return {"p_value": round(p, 4), "equal_variance": equal, "sd_ratio": round(float(ratio), 3), "note": note}


def sample_ratio_mismatch(counts, expected_ratios=None, threshold: float = 0.001) -> dict:
    """Sample Ratio Mismatch check for randomized experiments.

    If traffic was meant to split 50/50 but arrived 52/48 on 10,000 users,
    randomization or logging is probably broken and the test result cannot
    be trusted. Industry convention flags SRM at p < 0.001.
    """
    counts = np.asarray(counts, dtype=float)
    k = len(counts)
    ratios = np.ones(k) / k if expected_ratios is None else np.asarray(expected_ratios, dtype=float)
    ratios = ratios / ratios.sum()
    expected = ratios * counts.sum()
    p = float(stats.chisquare(counts, expected).pvalue)
    mismatch = bool(p < threshold)
    observed_share = counts / counts.sum()
    note = (f"group sizes {counts.astype(int).tolist()} match the planned split (SRM p={p:.3f})" if not mismatch else
            f"SAMPLE RATIO MISMATCH: observed split {np.round(observed_share * 100, 1).tolist()}% vs planned "
            f"{np.round(ratios * 100, 1).tolist()}% (p={p:.2g}). Randomization or tracking is likely broken.")
    return {"p_value": p, "mismatch": mismatch, "observed_share": observed_share.round(4).tolist(),
            "expected_share": ratios.round(4).tolist(), "note": note}


def expected_counts_ok(successes, totals) -> dict:
    """Large-sample tests on proportions need expected cell counts ≥ 5."""
    s, t = np.asarray(successes, float), np.asarray(totals, float)
    table = np.vstack([s, t - s])
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / table.sum()
    min_exp = float(expected.min())
    frac_small = float((expected < 5).mean())
    ok = bool(min_exp >= 5)
    return {"min_expected": round(min_exp, 2), "fraction_below_5": round(frac_small, 3), "ok": ok,
            "note": (f"all expected counts ≥ 5 (min {min_exp:.1f}), large-sample test is valid" if ok else
                     f"some expected counts are below 5 (min {min_exp:.1f}) — using an exact test instead of the large-sample approximation")}


def autocorrelation_check(values, threshold: float = 0.3) -> dict:
    """Control charts assume consecutive points are independent. Strong
    lag-1 autocorrelation makes limits too narrow (false alarms)."""
    x = np.asarray(values, dtype=float)
    if len(x) < 10 or np.ptp(x) == 0:
        return {"lag1": None, "flag": False, "note": "too few points to check autocorrelation"}
    r1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    flag = bool(abs(r1) > threshold)
    note = (f"consecutive points look independent (lag-1 autocorrelation {r1:.2f})" if not flag else
            f"consecutive points are correlated (lag-1 autocorrelation {r1:.2f}); control limits may be "
            f"too tight and produce false alarms — consider charting residuals of a time-series model")
    return {"lag1": round(r1, 3), "flag": flag, "note": note}
