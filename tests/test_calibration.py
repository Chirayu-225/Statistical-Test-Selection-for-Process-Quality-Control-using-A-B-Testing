"""Small Monte Carlo checks that the promised false-positive rate (alpha = 5%) is actually delivered.
The full study lives in simulations/calibration_study.py."""
import numpy as np
import pytest

from stat_advisor.compare import compare_means, compare_proportions

pytestmark = pytest.mark.slow
REPS = 400   # standard error ~1.1 points at 5%; thresholds below leave ~3 SE of room


def false_positive_rate(gen, test):
    rng = np.random.default_rng(2024)
    return np.mean([test(gen(rng)) for _ in range(REPS)])


def test_unequal_variance_three_groups():
    """The scenario that broke v1 (23% false positives)."""
    rate = false_positive_rate(lambda r: {0: r.normal(0, 1, 60), 1: r.normal(0, 1, 15), 2: r.normal(0, 4, 15)},
                               lambda g: compare_means(g).significant)
    assert rate < 0.085


def test_unequal_variance_two_groups():
    rate = false_positive_rate(lambda r: {0: r.normal(0, 1, 60), 1: r.normal(0, 3, 15)},
                               lambda g: compare_means(g).significant)
    assert rate < 0.085


def test_sparse_proportions_do_not_exceed_alpha():
    rate = false_positive_rate(lambda r: r.binomial(40, 0.05, 2),
                               lambda s: s.sum() > 0 and compare_proportions(s.tolist(), [40, 40]).significant)
    assert rate < 0.085
