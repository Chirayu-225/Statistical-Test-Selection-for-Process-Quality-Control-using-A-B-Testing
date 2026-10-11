# Methodology

This document explains every statistical decision Stat Advisor makes and why. The goal throughout is simple: **when the advisor says "significant at 5%", it should be wrong no more than 5% of the time when there is truly nothing there**, and it should say clearly what it can and cannot conclude.

## 1. Comparing groups on a numeric metric

### Step 1: can the mean be trusted?

For each group the advisor asks whether a test based on the **mean** gives valid p-values. It says yes if either:

- **The data looks normal.** Shapiro-Wilk p > alpha. This matters mainly for small samples.
- **The sample is large enough for the Central Limit Theorem.** It uses Cochran's rule: n ≥ max(30, 25 × skewness²). Skewed data needs more observations before the sample mean behaves normally.

Why not just use Shapiro-Wilk, as v1 did? Shapiro-Wilk answers the wrong question at both ends:

- **Large samples.** With thousands of rows it rejects normality for trivial departures, even though the CLT makes the t-test valid anyway. v1 sent 10,000 rows of almost-normal data to Mann-Whitney.
- **Small samples.** It has little power, so passing it is weak evidence.

Cochran's rule targets what actually matters, which is whether the *sampling distribution of the mean* is approximately normal.

### Step 2: which test?

| Groups | Mean trustworthy? | Test | Effect size | Follow-up |
|---|---|---|---|---|
| 2 | yes | Welch's t-test | Cohen's d, difference in means with t-based CI, relative lift | — |
| 2 | no | Brunner-Munzel (Mann-Whitney U below 10 per group) | rank-biserial r, P(other > control), difference in medians with bootstrap CI | — |
| 3+ | yes | Welch's ANOVA | omega² | Games-Howell (if significant) |
| 3+ | no | Kruskal-Wallis | epsilon² | Dunn's test + Holm (if significant) |

**Why Welch's tests, always?** v1 used Levene's test to choose between Student's and Welch's t-test, and for 3+ groups it ignored unequal variances completely. That broke the false-positive rate: with group sizes 60/15/15 and SDs 1/1/4, v1 reported a "significant" difference **23% of the time when none existed** (promised: 5%). Welch's tests lose almost nothing when variances are equal and stay correct when they are not. Pre-testing variances to pick a test also adds its own errors. Current guidance (Delacre, Lakens & Leys, 2017 and 2019) is therefore to use Welch's tests by default. Levene's (Brown-Forsythe) test is still shown, but only as a diagnostic.

**Why post-hoc tests instead of "pairwise t-tests + correction"?**

- **Games-Howell** is the standard follow-up to Welch's ANOVA. It does not assume equal variances or equal group sizes, and it controls the family-wise error rate through the studentized range distribution.
- **Dunn's test** is the matching follow-up to Kruskal-Wallis. It uses the same pooled ranks, so the pairwise results are consistent with the overall test. v1 re-ran the full test selection for each pair, so different pairs could end up using different tests.

**Why Brunner-Munzel instead of Mann-Whitney?** Both ask the same question: is a random value from one group more likely to be larger than one from the other? Mann-Whitney's p-value, however, assumes the two distributions have the same shape under the null. When spreads differ it can over-reject, which is the rank-test version of the Behrens-Fisher problem. Brunner-Munzel drops that assumption. Its approximation needs about 10 points per group, so smaller samples use Mann-Whitney. Kruskal-Wallis has the same weakness, and no comparably simple fix is available in SciPy, so the advisor warns when it is used on groups with unequal spreads.

**Average vs typical value (`focus`).** When totals matter (total revenue = average revenue × users), the *mean* is the business quantity, so the advisor compares means whenever they are valid. For heavy-tailed data, rank tests have more power. In the calibration study with t(3) data, the rank-based route found the shift noticeably more often than Welch did. Users who care about the typical customer can set `focus="typical"`. The choice is explicit because the two options answer different questions.

### Confidence intervals

- **Welch's t-test:** a t-based interval using Welch-Satterthwaite degrees of freedom, at confidence level 1 − alpha. v1 used 1.96 regardless of sample size and alpha, which gave 92% actual coverage at n = 6 per group.
- **Rank route:** a percentile bootstrap interval (2,000 resamples, fixed seed) for the difference in medians.

### Degenerate data

If every group is constant, the test statistics are 0/0. The advisor handles this explicitly instead of returning NaN: identical constants give p = 1, and different constants are flagged as a certain difference with a data-quality warning.

## 2. Comparing conversion rates (yes/no outcomes)

The advisor detects a yes/no metric automatically: 0/1, True/False, or yes/no strings.

| Groups | Expected counts ≥ 5 | Test |
|---|---|---|
| 2 | yes | Two-proportion z-test |
| 2 | no | Boschloo's exact test (Fisher's above 2,000 per group) |
| 3+ | yes | Chi-square test of independence (+ Holm-adjusted pairwise tests) |
| 3+ | no | Exact conditional chi-square test by Monte Carlo (20,000 multivariate-hypergeometric tables) |

- **Boschloo vs Fisher.** Both are exact, but Fisher's test is conservative. Boschloo's test is uniformly more powerful and still controls the error rate.
- **Confidence interval** for the difference in rates: Newcombe's hybrid score method. It is accurate near 0% and 100%, where the textbook Wald interval fails.
- **Effect sizes:** Cohen's h for 2 groups and Cramér's V for 3+ groups. These are reported alongside the absolute difference (percentage points) and the relative lift, which are the numbers a business actually uses.

## 3. A/B-test safeguards

- **Sample Ratio Mismatch (SRM).** If the experiment planned a 50/50 split but the groups arrived as 5,000 vs 4,600, something upstream is broken: bots, redirects, logging or the assignment code. A chi-square goodness-of-fit test on the group sizes flags this at p < 0.001, the usual industry threshold. When SRM fires, the advisor refuses to call a winner.
- **Practical significance.** If the user gives the smallest difference that matters, the advisor compares the confidence interval to that threshold. There are four possible outcomes:
  - **Meaningful:** the whole interval is beyond the threshold.
  - **Real but possibly small:** the result is significant, but the interval reaches below the threshold.
  - **Equivalent:** the whole interval is inside ±threshold. This is positive evidence of *no meaningful difference*.
  - **Inconclusive:** none of the above.
- **Minimum detectable effect instead of "observed power".** After a non-significant result, the advisor reports the smallest difference the current sample could have detected with 80% power, and how much data would detect half of it. Post-hoc "observed power" is not used because it is just a transformation of the p-value.

## 4. Monitoring a process over time

- **Phase I / Phase II.** Limits are learned from a baseline period (default: the first half of the data, at most 100 points) and then applied to every point. v1 learned its limits from all the data, including the shift it was supposed to detect, which widened the limits. Its CUSUM target was the overall mean, which sits halfway between the "before" and "after" levels, so v1 signalled a drift in the stable period before the real change.
- **Sigma** is estimated from the average moving range (MR̄ / 1.128), the standard I-MR estimator. It is robust to slow drift.
- **Detectors.** The I-chart (3σ) catches sudden jumps. The tabular CUSUM (k = 0.5σ, h = 5σ) catches small, persistent drift. Sensitive mode adds Western Electric rules 2-4 and the moving-range chart.
- **When did it start?** The CUSUM change-point estimate is the last time the cumulative sum was zero before it crossed the threshold. This is usually several points earlier than the alarm itself.
- **Confirmed vs watch.** Even a perfectly stable process produces occasional signals: with 3σ limits, about 1 in 370 points falls outside by chance. A signal is *confirmed* when it repeats (2+ points outside the limits, or CUSUM beyond h for 3+ points). A single brief signal is reported as "worth watching". In the calibration study this roughly halved false alarms on stable 100-point series while detection of a 1σ shift stayed high (see `simulations/results.md`).
- **Assumption checks.** The advisor warns about lag-1 autocorrelation above 0.3, because correlated data makes the limits too tight, and about clearly non-normal baselines.

## 5. Planning sample size

| Situation | Method |
|---|---|
| Conversion rate | Normal approximation with Cohen's h (within 1% of the textbook pooled formula) |
| Average, 2 groups | Two-sample t-test power (statsmodels `TTestIndPower`) |
| Average, 3+ groups | One-way ANOVA power. statsmodels returns the **total** n, so it is divided by k. v1 reported the total as per-group. |
| More than one variant | Alpha is split across comparisons with control (Bonferroni) |

The advisor also reminds the user not to stop a test early when it "looks significant". Repeated peeking inflates false positives, and this tool does not implement sequential testing.

## 6. Known limitations

- **One metric at a time.** If you test many metrics, adjust for multiplicity yourself (`stat_advisor.adjust_pvalues`).
- **Independent observations only.** No paired or repeated-measures designs, no clustered data (for example, randomizing by user but analysing per session).
- **No variance reduction or sequential testing.** No CUPED, no stratification, no always-valid p-values.
- **Control charts assume independent points.** The advisor warns about autocorrelation but does not model it.

## References

- Delacre, Lakens & Leys (2017). *Why psychologists should by default use Welch's t-test instead of Student's t-test.* International Review of Social Psychology.
- Delacre et al. (2019). *Taking parametric assumptions seriously: arguments for the use of Welch's F-test instead of the classical F-test in one-way ANOVA.*
- Cochran (1952), via Boos & Hughes-Oliver (2000). *How large does n have to be for Z and t intervals?* The American Statistician.
- Games & Howell (1976); Dunn (1964); Holm (1979).
- Boschloo (1970). *Raised conditional level of significance for the 2×2 table when testing the equality of two probabilities.*
- Newcombe (1998). *Interval estimation for the difference between independent proportions.* Statistics in Medicine.
- Fabijan et al. (2019). *Diagnosing sample ratio mismatch in online controlled experiments.* KDD.
- Montgomery. *Introduction to Statistical Quality Control* (I-MR charts, CUSUM, Western Electric rules).
- Hoenig & Heisey (2001). *The abuse of power: the pervasive fallacy of power calculations for data analysis.*
