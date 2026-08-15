"""
Validation harness: throws known synthetic scenarios at the algorithm
and checks it picks the right test and reaches the right conclusion.
"""
import numpy as np
import pandas as pd
from algorithm import (
    run_numeric_test, run_proportion_test, correct_multiple_tests,
    required_sample_size_ttest, required_sample_size_anova,
    shewhart_individuals_chart, cusum_chart
)

np.random.seed(42)

def scenario(name):
    print(f"\n{'='*70}\nSCENARIO: {name}\n{'='*70}")


# ---------------------------------------------------------------------
scenario("1. Two normal groups, TRUE difference exists -> expect t-test, significant")
a = np.random.normal(50, 5, 200)
b = np.random.normal(53, 5, 200)
df = pd.DataFrame({"value": np.concatenate([a, b]), "group": ["A"]*200 + ["B"]*200})
r = run_numeric_test(df, "value", "group")
print(f"Test used: {r.test_used}")
print(f"p={r.p_value}, significant={r.significant}, effect={r.effect_size} ({r.effect_size_label})")
print(r.verdict)
assert "t-test" in r.test_used
assert r.significant

# ---------------------------------------------------------------------
scenario("2. Two normal groups, NO true difference -> expect not significant")
a = np.random.normal(50, 5, 200)
b = np.random.normal(50, 5, 200)
df = pd.DataFrame({"value": np.concatenate([a, b]), "group": ["A"]*200 + ["B"]*200})
r = run_numeric_test(df, "value", "group")
print(f"Test used: {r.test_used}")
print(f"p={r.p_value}, significant={r.significant}")
assert not r.significant

# ---------------------------------------------------------------------
scenario("3. Skewed/non-normal data -> expect Mann-Whitney, not t-test")
a = np.random.exponential(2, 150)
b = np.random.exponential(4, 150)
df = pd.DataFrame({"value": np.concatenate([a, b]), "group": ["A"]*150 + ["B"]*150})
r = run_numeric_test(df, "value", "group")
print(f"Test used: {r.test_used}")
print(f"p={r.p_value}, significant={r.significant}")
assert "Mann-Whitney" in r.test_used

# ---------------------------------------------------------------------
scenario("4. Three groups, normal -> expect ANOVA")
g1 = np.random.normal(100, 10, 100)
g2 = np.random.normal(105, 10, 100)
g3 = np.random.normal(110, 10, 100)
df = pd.DataFrame({
    "value": np.concatenate([g1, g2, g3]),
    "group": ["Machine1"]*100 + ["Machine2"]*100 + ["Machine3"]*100
})
r = run_numeric_test(df, "value", "group")
print(f"Test used: {r.test_used}")
print(f"p={r.p_value}, significant={r.significant}, eta_sq={r.effect_size} ({r.effect_size_label})")
assert "ANOVA" in r.test_used
assert r.significant

# ---------------------------------------------------------------------
scenario("5. Conversion rate A/B test (proportions) -> expect two-proportion z-test")
r = run_proportion_test(successes=[120, 150], totals=[1000, 1000], group_names=["Control", "Treatment"])
print(f"Test used: {r.test_used}")
print(f"p={r.p_value}, significant={r.significant}, CI={r.confidence_interval}")
print(r.verdict)
assert "z-test" in r.test_used

# ---------------------------------------------------------------------
scenario("6. Multiple comparison correction")
raw_ps = [0.001, 0.02, 0.04, 0.049, 0.20]
corrected = correct_multiple_tests(raw_ps, method="fdr_bh")
print(f"Raw p-values:      {raw_ps}")
print(f"Adjusted p-values: {corrected['p_adjusted']}")
print(f"Still significant: {corrected['significant_after_correction']}")

# ---------------------------------------------------------------------
scenario("7. Sample size / power calculation for planning")
n_ttest = required_sample_size_ttest(effect_size=0.3, power=0.8)
n_anova = required_sample_size_anova(effect_size=0.25, k_groups=3, power=0.8)
print(f"Required n per group for t-test (d=0.3, power=0.8): {n_ttest}")
print(f"Required n per group for ANOVA (f=0.25, k=3, power=0.8): {n_anova}")

# ---------------------------------------------------------------------
scenario("8. Shewhart chart - stable process -> expect no violations")
stable = np.random.normal(50, 2, 40)
r = shewhart_individuals_chart(stable)
print(f"Center={r.center_line}, UCL={r.ucl}, LCL={r.lcl}, violations={r.n_violations}")
print(r.verdict)
assert r.n_violations == 0

# ---------------------------------------------------------------------
scenario("9. Shewhart chart - sudden shift/spike -> expect violation flagged")
stable = np.random.normal(50, 2, 30)
spike = np.array([80, 82, 79])  # obvious jump, way outside 3-sigma
data = np.concatenate([stable, spike])
r = shewhart_individuals_chart(data)
print(f"Center={r.center_line}, UCL={r.ucl}, LCL={r.lcl}, violations={r.violations}")
print(r.verdict)
assert r.n_violations > 0
assert min(r.violations) >= 30  # violations should be in the spike region

# ---------------------------------------------------------------------
scenario("10. CUSUM - gradual drift (Shewhart would likely miss this) -> expect detection")
baseline = np.random.normal(50, 2, 30)
drift = 50 + np.linspace(0, 6, 30) + np.random.normal(0, 2, 30)  # slow creep upward
data = np.concatenate([baseline, drift])
shewhart_r = shewhart_individuals_chart(data)
cusum_r = cusum_chart(data, target=50, std_dev=2)
print(f"Shewhart violations: {shewhart_r.n_violations}")
print(f"CUSUM violations: {cusum_r.n_violations}")
print(cusum_r.verdict)
assert cusum_r.n_violations > 0  # CUSUM should catch the gradual drift

# ---------------------------------------------------------------------
scenario("11. Process factor significance (ANOVA) - machine affects defect rate")
rng = np.random.default_rng(3)
m1 = rng.normal(2.0, 0.5, 60)   # defect rate %
m2 = rng.normal(2.1, 0.5, 60)
m3 = rng.normal(3.5, 0.5, 60)   # clearly worse machine
df = pd.DataFrame({
    "defect_rate": np.concatenate([m1, m2, m3]),
    "machine": ["M1"]*60 + ["M2"]*60 + ["M3"]*60
})
r = run_numeric_test(df, "defect_rate", "machine")
print(f"Test used: {r.test_used}, p={r.p_value}, significant={r.significant}")
assert r.significant

print("\n" + "="*70)
print("ALL SCENARIOS PASSED ✔")
print("="*70)
