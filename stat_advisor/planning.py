"""
Sample-size planning: how much data do I need before I can trust the result?

Two directions:
* BEFORE an experiment: "to detect a difference of X, I need N per group"
* AFTER an inconclusive one: "with the N I have, the smallest difference I
  could reliably detect is X" (minimum detectable effect, MDE). This replaces
  "observed power", which is just a restatement of the p-value.
"""
from __future__ import annotations

import numpy as np
from statsmodels.stats.power import FTestAnovaPower, NormalIndPower, TTestIndPower


def _check(alpha, power):
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be between 0 and 1.")


def n_for_means(mde: float, sd: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-group n to detect a difference in means of `mde` (metric units) when the SD is `sd`."""
    _check(alpha, power)
    if sd <= 0 or mde == 0:
        raise ValueError("SD must be > 0 and the difference to detect must be non-zero.")
    return n_for_cohens_d(abs(mde) / sd, alpha, power)


def n_for_cohens_d(d: float, alpha: float = 0.05, power: float = 0.8) -> int:
    _check(alpha, power)
    if d <= 0:
        raise ValueError("Effect size must be > 0.")
    return int(np.ceil(TTestIndPower().solve_power(effect_size=d, alpha=alpha, power=power, ratio=1.0)))


def n_for_proportions(baseline: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-group n to detect a change in conversion rate from `baseline` to
    `baseline + mde_abs` (e.g. 0.10 -> 0.11 is mde_abs=0.01, one percentage point)."""
    _check(alpha, power)
    target = baseline + mde_abs
    if not (0 < baseline < 1 and 0 < target < 1) or mde_abs == 0:
        raise ValueError("Baseline and baseline + MDE must both be strictly between 0 and 1.")
    h = abs(2 * np.arcsin(np.sqrt(target)) - 2 * np.arcsin(np.sqrt(baseline)))   # Cohen's h
    return int(np.ceil(NormalIndPower().solve_power(effect_size=h, alpha=alpha, power=power, ratio=1.0)))


def n_for_anova(cohens_f: float, k_groups: int, alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-group n for a one-way comparison of k groups. statsmodels returns the
    TOTAL n across groups for ANOVA, so divide by k."""
    _check(alpha, power)
    if cohens_f <= 0 or k_groups < 2:
        raise ValueError("Cohen's f must be > 0 and k_groups ≥ 2.")
    total = FTestAnovaPower().solve_power(effect_size=cohens_f, alpha=alpha, power=power, k_groups=k_groups)
    return int(np.ceil(total / k_groups))


def mde_for_means(n1: int, n2: int, sd: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest difference in means detectable with the given group sizes."""
    d = TTestIndPower().solve_power(nobs1=n1, ratio=n2 / n1, alpha=alpha, power=power, effect_size=None)
    return float(d * sd)


def mde_for_proportions(baseline: float, n1: int, n2: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute increase over `baseline` detectable with the given group sizes."""
    h = NormalIndPower().solve_power(nobs1=n1, ratio=n2 / n1, alpha=alpha, power=power, effect_size=None)
    phi = np.arcsin(np.sqrt(np.clip(baseline, 1e-9, 1 - 1e-9))) + h / 2
    target = np.sin(min(phi, np.pi / 2)) ** 2
    return float(target - baseline)


def duration_days(n_per_group: int, k_groups: int, daily_traffic: float) -> float:
    if daily_traffic <= 0:
        raise ValueError("Daily traffic must be > 0.")
    return float(np.ceil(n_per_group * k_groups / daily_traffic))
