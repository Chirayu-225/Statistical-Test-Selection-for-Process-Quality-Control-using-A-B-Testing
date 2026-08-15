"""
Statistical Test Selection and Decision Algorithm — Streamlit UI

One algorithm, two analytical lenses applied to the SAME uploaded dataset:
  - Group Comparison (hypothesis testing): is there a real difference between groups?
  - Time Monitoring (control charts): is the process stable or drifting over time?

Previously these were two disconnected "modes" with separate upload flows.
They are unified here because they are the same underlying question asked
two different ways — and a real dataset (e.g. a machine's defect rate) often
benefits from both views without re-uploading.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from itertools import combinations

from algorithm import (
    run_numeric_test, run_proportion_test,
    required_sample_size_ttest, required_sample_size_anova,
    correct_multiple_tests,
    shewhart_individuals_chart, cusum_chart,
)

st.set_page_config(page_title="Statistical Test Selection Algorithm", layout="wide")

st.title("Statistical Test Selection and Decision Algorithm")
st.caption("One algorithm, applied two ways to the same data: **Group Comparison** asks whether groups "
           "genuinely differ; **Time Monitoring** asks whether a process is drifting. Both check assumptions "
           "first and select the statistically valid method — they never default to a t-test or a gut-feel chart read.")

# =======================================================================
# SIDEBAR — shared settings + planning tools (not tied to loaded data)
# =======================================================================
st.sidebar.header("Settings")
alpha = st.sidebar.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)
correction_method = st.sidebar.selectbox(
    "Multiple-comparison correction",
    ["Benjamini-Hochberg (FDR)", "Bonferroni"],
    help="Applied automatically whenever more than one test is run in the same session (e.g. pairwise comparisons across 3+ groups)."
)
_correction_key = "fdr_bh" if correction_method.startswith("Benjamini") else "bonferroni"

st.sidebar.markdown("---")
st.sidebar.subheader("Sample size / power planner")
st.sidebar.caption("Plan BEFORE collecting data.")
planner_kind = st.sidebar.radio("Planning for", ["2 groups (t-test)", "3+ groups (ANOVA)"], key="planner_kind")
if planner_kind.startswith("2 groups"):
    planned_effect = st.sidebar.slider("Expected effect size (Cohen's d)", 0.1, 1.0, 0.3, 0.05, key="planner_d")
    planned_power = st.sidebar.slider("Desired power", 0.7, 0.99, 0.8, 0.01, key="planner_power_t")
    if st.sidebar.button("Calculate required sample size", key="planner_btn_t"):
        n = required_sample_size_ttest(planned_effect, alpha=alpha, power=planned_power)
        st.sidebar.success(f"Need ~{n} observations per group")
else:
    planned_f = st.sidebar.slider("Expected effect size (Cohen's f)", 0.1, 1.0, 0.25, 0.05, key="planner_f")
    planned_k = st.sidebar.number_input("Number of groups", min_value=3, max_value=10, value=3, key="planner_k")
    planned_power_a = st.sidebar.slider("Desired power", 0.7, 0.99, 0.8, 0.01, key="planner_power_a")
    if st.sidebar.button("Calculate required sample size", key="planner_btn_a"):
        n = required_sample_size_anova(planned_f, int(planned_k), alpha=alpha, power=planned_power_a)
        st.sidebar.success(f"Need ~{n} observations per group")

st.markdown("---")

# =======================================================================
# SHARED DATA LOADING — one upload, used by both lenses below
# =======================================================================
st.subheader("1. Load your data")

metric_type = st.radio(
    "Metric type",
    ["Numeric (revenue, defect rate, cycle time, time on page...)", "Conversion / proportion (signup rate, click-through...)"],
    horizontal=True,
)
is_numeric_metric = metric_type.startswith("Numeric")

df = None
if is_numeric_metric:
    data_source = st.radio("Data source", ["Upload CSV", "Use demo data"], horizontal=True)
    if data_source == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload CSV — needs a group column (e.g. variant/machine/shift) and a numeric metric column. "
            "An optional time/order column enables the Time Monitoring tab.",
            type="csv",
        )
        df = pd.read_csv(uploaded) if uploaded else None
    else:
        rng = np.random.default_rng(11)
        n_per_group = 60
        g1 = rng.normal(2.0, 0.4, n_per_group)
        g2 = rng.normal(2.1, 0.4, n_per_group)
        g3 = rng.normal(3.2, 0.4, n_per_group)
        df = pd.DataFrame({
            "order": np.arange(n_per_group * 3),
            "group": ["Group A"] * n_per_group + ["Group B"] * n_per_group + ["Group C"] * n_per_group,
            "value": np.concatenate([g1, g2, g3]),
        })
        st.info("Using demo data: a metric (e.g. defect rate) across 3 groups (e.g. machines), "
                "with Group C running worse — same dataset drives both tabs below.")

    if df is not None:
        st.dataframe(df.head(), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            group_col = st.selectbox("Group column", df.columns,
                                      index=list(df.columns).index("group") if "group" in df.columns else 0)
        with col2:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            value_col = st.selectbox("Metric column", numeric_cols,
                                      index=numeric_cols.index("value") if "value" in numeric_cols else 0)
        with col3:
            order_options = ["(row order)"] + list(df.columns)
            order_col = st.selectbox("Time/order column (optional, for Time Monitoring)", order_options)

else:
    st.caption("Proportions are entered directly (typical for conversion-rate comparisons) — "
               "Time Monitoring isn't shown for this metric type since it needs a continuous series.")
    n_groups = st.number_input("Number of groups", min_value=2, max_value=6, value=2)
    names, successes, totals = [], [], []
    cols = st.columns(int(n_groups))
    for i in range(int(n_groups)):
        with cols[i]:
            name = st.text_input("Group name", value=f"Group{i+1}", key=f"p_name_{i}")
            s = st.number_input("Conversions", min_value=0, value=100 + i * 20, key=f"p_s_{i}")
            t = st.number_input("Total", min_value=1, value=1000, key=f"p_t_{i}")
            names.append(name); successes.append(int(s)); totals.append(int(t))

st.markdown("---")

# =======================================================================
# TWO LENSES on the same data
# =======================================================================
if is_numeric_metric:
    st.subheader("2. Analyze")
    tab1, tab2 = st.tabs(["Group Comparison (hypothesis testing)", "Time Monitoring (control charts)"])

    with tab1:
        if df is None:
            st.info("Load data above to run a comparison.")
        else:
            k_groups = df[group_col].nunique()
            if st.button("Run group comparison", type="primary", key="run_group_cmp"):
                result = run_numeric_test(df, value_col, group_col, alpha=alpha)

                verdict_color = "green" if result.significant else "orange"
                st.markdown("### Verdict")
                st.markdown(f":{verdict_color}[{result.verdict}]")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Test used", result.test_used)
                c2.metric("p-value", f"{result.p_value:.4f}")
                c3.metric("Effect size", f"{result.effect_size} ({result.effect_size_label})")
                c4.metric("Significant?", "Yes" if result.significant else "No")

                with st.expander("Assumption checks (why this test was chosen)"):
                    st.write("**Normality (Shapiro-Wilk)**")
                    st.json(result.assumptions["normality"])
                    st.write("**Variance homogeneity (Levene's test)**")
                    st.json(result.assumptions["variance_homogeneity"])

                st.subheader(f"{value_col} by {group_col}")
                fig = px.box(df, x=group_col, y=value_col, color=group_col, points="all")
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Group summary")
                st.dataframe(result.group_summary, use_container_width=True)

                if result.confidence_interval != (None, None):
                    st.caption(f"95% CI on difference: {result.confidence_interval}")

                st.session_state["last_group_result"] = {"k": k_groups}

            # Pairwise comparisons + multiple-comparison correction — only meaningful for 3+ groups
            if k_groups >= 3:
                st.markdown("---")
                st.markdown("**More than 2 groups detected.** Running every pair separately inflates the "
                             "chance of a false positive — this runs all pairs and corrects for it.")
                if st.button("Run all pairwise comparisons (with correction)", key="run_pairwise"):
                    group_names = sorted(df[group_col].dropna().unique().tolist())
                    rows = []
                    for g1, g2 in combinations(group_names, 2):
                        sub = df[df[group_col].isin([g1, g2])]
                        r = run_numeric_test(sub, value_col, group_col, alpha=alpha)
                        rows.append({"pair": f"{g1} vs {g2}", "test": r.test_used,
                                     "raw_p": r.p_value, "effect_size": r.effect_size})

                    pairwise_df = pd.DataFrame(rows)
                    corrected = correct_multiple_tests(pairwise_df["raw_p"].tolist(), method=_correction_key, alpha=alpha)
                    pairwise_df["adjusted_p"] = corrected["p_adjusted"]
                    pairwise_df["significant_after_correction"] = corrected["significant_after_correction"]

                    st.dataframe(pairwise_df, use_container_width=True)
                    n_sig_raw = (pairwise_df["raw_p"] < alpha).sum()
                    n_sig_corrected = pairwise_df["significant_after_correction"].sum()
                    if n_sig_raw != n_sig_corrected:
                        st.warning(f"{n_sig_raw} pair(s) looked significant before correction, but only "
                                    f"{n_sig_corrected} remain significant after {correction_method} correction — "
                                    f"the rest were likely false positives from running multiple tests.")
                    else:
                        st.success(f"{n_sig_corrected} pair(s) remain significant after correction — "
                                    f"consistent with the uncorrected result.")

    with tab2:
        if df is None:
            st.info("Load data above to run control charts.")
        else:
            sigma_mult = st.slider("Shewhart control limit width (sigma)", 2.0, 4.0, 3.0, 0.5)
            cusum_k = st.slider("CUSUM slack (k, in sigma units)", 0.25, 1.5, 0.5, 0.25)
            cusum_h = st.slider("CUSUM decision threshold (h, in sigma units)", 3.0, 8.0, 5.0, 0.5)

            chart_slice = st.selectbox("Run control chart on which slice?",
                                        ["All data combined"] + sorted(df[group_col].dropna().unique().tolist()))
            if chart_slice == "All data combined" and df[group_col].nunique() > 1:
                st.caption("Note: combining groups stacks their data back-to-back rather than a true shared "
                            "timeline — a step-change here may just reflect the group boundary, not real drift. "
                            "Select a single group above for a genuine time-ordered read.")

            if st.button("Run control chart analysis", type="primary", key="run_cc"):
                subset = df if chart_slice == "All data combined" else df[df[group_col] == chart_slice]
                if order_col != "(row order)":
                    subset = subset.sort_values(order_col)
                series = subset[value_col].dropna().values
                x_axis = list(range(len(series)))

                shew = shewhart_individuals_chart(series, sigma_multiplier=sigma_mult)
                cus = cusum_chart(series, k=cusum_k, h=cusum_h)

                st.markdown("### Shewhart Individuals chart (catches sudden shifts)")
                st.markdown(f":{'orange' if shew.n_violations else 'green'}[{shew.verdict}]")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=x_axis, y=series, mode="lines+markers", name=value_col))
                fig1.add_hline(y=shew.center_line, line_dash="dot", line_color="gray", annotation_text="center")
                fig1.add_hline(y=shew.ucl, line_dash="dash", line_color="red", annotation_text="UCL")
                fig1.add_hline(y=shew.lcl, line_dash="dash", line_color="red", annotation_text="LCL")
                if shew.violations:
                    fig1.add_trace(go.Scatter(
                        x=[x_axis[i] for i in shew.violations], y=[series[i] for i in shew.violations],
                        mode="markers", marker=dict(color="red", size=12, symbol="x"), name="out of control"
                    ))
                fig1.update_layout(title="Shewhart Individuals Chart", xaxis_title="Order", yaxis_title=value_col)
                st.plotly_chart(fig1, use_container_width=True)

                st.markdown("### CUSUM chart (catches gradual drift)")
                st.markdown(f":{'orange' if cus.n_violations else 'green'}[{cus.verdict}]")
                sh_pos = [v[0] for v in cus.values]
                sh_neg = [v[1] for v in cus.values]
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=x_axis, y=sh_pos, mode="lines", name="CUSUM+ (upward)"))
                fig2.add_trace(go.Scatter(x=x_axis, y=sh_neg, mode="lines", name="CUSUM- (downward)"))
                fig2.add_hline(y=cus.ucl, line_dash="dash", line_color="red", annotation_text=f"h=+{cusum_h}")
                fig2.add_hline(y=cus.lcl, line_dash="dash", line_color="red", annotation_text=f"h=-{cusum_h}")
                if cus.violations:
                    fig2.add_trace(go.Scatter(
                        x=[x_axis[i] for i in cus.violations],
                        y=[sh_pos[i] if abs(sh_pos[i]) >= abs(sh_neg[i]) else sh_neg[i] for i in cus.violations],
                        mode="markers", marker=dict(color="red", size=10, symbol="x"), name="drift detected"
                    ))
                fig2.update_layout(title="CUSUM Chart", xaxis_title="Order", yaxis_title="Cumulative deviation (sigma units)")
                st.plotly_chart(fig2, use_container_width=True)

else:
    st.subheader("2. Analyze")
    if st.button("Run group comparison", type="primary", key="run_prop_cmp"):
        result = run_proportion_test(successes, totals, group_names=names, alpha=alpha)

        verdict_color = "green" if result.significant else "orange"
        st.markdown("### Verdict")
        st.markdown(f":{verdict_color}[{result.verdict}]")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test used", result.test_used)
        c2.metric("p-value", f"{result.p_value:.4f}")
        c3.metric("Effect size", f"{result.effect_size} ({result.effect_size_label})")
        c4.metric("Significant?", "Yes" if result.significant else "No")

        rates = [s / t for s, t in zip(successes, totals)]
        fig = go.Figure(go.Bar(x=names, y=rates, text=[f"{r:.2%}" for r in rates], textposition="auto"))
        fig.update_layout(title="Conversion rate by group", yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

        if result.confidence_interval != (None, None):
            st.caption(f"95% CI on difference in rates: {result.confidence_interval}")

st.markdown("---")
st.caption("Core algorithm: checks normality (Shapiro-Wilk) and variance homogeneity (Levene's test) before "
           "selecting between t-test / Mann-Whitney / ANOVA / Kruskal-Wallis / two-proportion z-test / chi-square. "
           "Control charts (Shewhart + CUSUM) run on the same validated core.")
