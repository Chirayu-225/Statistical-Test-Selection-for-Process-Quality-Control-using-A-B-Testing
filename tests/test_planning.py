"""Sample size / power: checked against published reference values."""
import numpy as np
import pytest
from scipy import stats

from stat_advisor import planning


def test_t_test_matches_cohen_tables():
    assert planning.n_for_cohens_d(0.5) == 64        # Cohen (1988), Table 2.4.1
    assert planning.n_for_cohens_d(0.3) == 176
    assert planning.n_for_means(mde=2, sd=10) == planning.n_for_cohens_d(0.2) == 394


def test_anova_is_per_group_not_total():
    """v1 bug: statsmodels returns TOTAL n for ANOVA. f=0.25, k=3 -> 159 total (G*Power) -> 53 per group."""
    assert planning.n_for_anova(0.25, 3) == 53


def test_proportions_close_to_textbook_formula():
    p1, p2 = 0.10, 0.12
    za, zb = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
    pbar = (p1 + p2) / 2
    textbook = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2
    assert planning.n_for_proportions(p1, 0.02) == pytest.approx(textbook, rel=0.01)


def test_mde_round_trips():
    n = planning.n_for_means(mde=1.5, sd=6)
    assert planning.mde_for_means(n, n, 6) == pytest.approx(1.5, rel=0.02)
    n = planning.n_for_proportions(0.2, 0.03)
    assert planning.mde_for_proportions(0.2, n, n) == pytest.approx(0.03, rel=0.02)


def test_duration_and_validation():
    assert planning.duration_days(1000, 2, 300) == 7
    with pytest.raises(ValueError):
        planning.n_for_proportions(0.99, 0.05)
    with pytest.raises(ValueError):
        planning.n_for_means(1, 0)
