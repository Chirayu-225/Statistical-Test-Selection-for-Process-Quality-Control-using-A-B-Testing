# Statistical Test Selection and Decision Algorithm for A/B Experimentation and Process Quality Control

Built an algorithm that checks statistical assumptions before auto-selecting the correct test (t-test, ANOVA, Mann-Whitney, etc.) or control chart method, applied to both A/B experiment analysis and manufacturing process monitoring. Validated with 150+ property-based tests and 24 edge-case scenarios, catching two real defects pre-deployment, and added multiple-comparison correction to guard against false positives.

## What it is

Most A/B testing and quality-control tools assume a t-test or ANOVA is always valid. This algorithm doesn't assume — it checks. Given grouped or time-ordered data, it:

1. Checks assumptions first: normality (Shapiro-Wilk), variance homogeneity (Levene's test)
2. Selects the statistically correct test or monitoring method based on what the data actually looks like
3. Runs it and returns effect size, confidence interval, and a plain-language verdict
4. Guards against false positives from repeated testing via multiple-comparison correction

The same core algorithm powers two views on one dataset:
- **Group Comparison** — is there a real difference between groups (A/B variants, machines, shifts)?
- **Time Monitoring** — is the process stable, or is it drifting over time?

## Decision logic

| Groups | Data type | Normal? | Test used |
|---|---|---|---|
| 2 | Numeric | Yes | Student's t-test (or Welch's if unequal variance) |
| 2 | Numeric | No | Mann-Whitney U |
| 3+ | Numeric | Yes | One-way ANOVA (F-test) |
| 3+ | Numeric | No | Kruskal-Wallis |
| 2 | Proportion | — | Two-proportion z-test |
| 3+ | Proportion | — | Chi-square test of independence |

Degenerate cases (e.g. zero variance in every group) are handled explicitly rather than silently returning `NaN`.

## Control charts (Time Monitoring)

- **Shewhart Individuals (I-chart)** — sigma-based limits from the average moving range; flags sudden shifts
- **CUSUM** — accumulates small deviations over time; flags gradual drift a Shewhart chart would miss, and reports every distinct drift episode (not just the final direction) when the process changes direction mid-series

## Also wired into the UI

- Sample size / power planning for both the t-test and ANOVA cases
- Multiple-comparison correction (Bonferroni / Benjamini-Hochberg), automatically offered when comparing 3+ groups pairwise, with a before/after table showing which "significant" results were actually false positives
- Assumption checks shown transparently, so the app explains *why* a test was chosen

## Project structure

```
.
├── algorithm.py         # Core algorithm: test selection + control charts
├── app.py                # Streamlit UI (unified Group Comparison + Time Monitoring)
├── validate.py            # Correctness tests against known-outcome scenarios
├── test_robustness.py     # Edge-case tests + property-based tests (hypothesis)
├── requirements.txt
├── .gitignore
└── README.md
```

## Run it locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
streamlit run app.py
```

## Testing

```bash
python3 validate.py           # 11 correctness scenarios with known ground truth
python3 test_robustness.py    # 13 edge cases + 150 property-based randomized cases
```

Both suites pass cleanly. Property-based testing (via [`hypothesis`](https://hypothesis.readthedocs.io/)) caught two real defects during development:
- Two constant/identical groups produced a `NaN` p-value from the raw scipy t-test formula — now handled as an explicit degenerate case
- The ANOVA sample-size planner was returning `statsmodels`' *total* sample size instead of *per-group* — fixed and verified against the standard Cohen's textbook reference value

## Tech stack

Python · scipy · statsmodels · pandas · Streamlit · Plotly

## License

MIT
