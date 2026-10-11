"""
Stat Advisor — Streamlit app.

Ask a question about your data, get an answer you can act on:
  1. pick a question   2. give it data   3. read the answer, the recommendation and WHY.

Run:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.stats.proportion import proportion_confint

from stat_advisor import (analyze_comparison, analyze_counts, analyze_monitoring, infer_metric_kind,
                          plan_conversion_test, plan_mean_test)

DATA = Path(__file__).parent / "data"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SIGNAL = "#c62828"
MUTED = "#8a8985"

SAMPLES_COMPARE = {
    "Checkout A/B test: did the new page lift conversion and revenue?":
        ("checkout_ab_test.csv", dict(metric="converted", group="variant", control="A", randomized=True)),
    "Machine defect rates: is one machine worse?":
        ("machine_defects.csv", dict(metric="defect_rate_pct", group="machine", higher_is_better=False)),
    "Support tickets: does one team take longer? (small, skewed samples)":
        ("support_resolution_times.csv", dict(metric="resolution_hours", group="team", higher_is_better=False)),
}
SAMPLES_MONITOR = {
    "Bottle filling line: is the fill volume drifting?": ("bottle_fill_process.csv", dict(metric="fill_volume_ml", order="batch")),
    "Machine M3 defect rate over 60 days": ("machine_defects.csv", dict(metric="defect_rate_pct", order="date",
                                                                         filter=("machine", "M3"))),
}

st.set_page_config(page_title="Stat Advisor", layout="wide")


@st.cache_data
def load_csv(path_or_buffer):
    return pd.read_csv(path_or_buffer)


def pick(label, options, preferred=None, key=None, help=None):
    options = list(options)
    idx = options.index(preferred) if preferred in options else 0
    return st.selectbox(label, options, index=idx, key=key, help=help)


# ---------------------------------------------------------------------------
# Report rendering (shared by all three questions)
# ---------------------------------------------------------------------------

GOOD = {"stable", "plan", "no_meaningful_difference"}
ALERT = {"untrustworthy", "issue"}


def render_report(report, chart=None, wide_charts=()):
    st.markdown("### Answer")
    if report.verdict in ALERT:
        box = st.error
    elif report.verdict in ("no_difference", "inconclusive", "watch", "small_difference"):
        box = st.warning
    elif report.verdict in GOOD:
        box = st.success
    else:                      # a clear difference: neutral colour, since it can be good or bad news
        box = st.info
    box(report.headline)
    st.markdown(f"**What to do:** {report.recommendation}")

    for w in report.warnings:
        st.warning(w)

    tables = dict(report.tables)
    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Key numbers**")
        st.dataframe(pd.DataFrame(report.key_numbers, columns=["Measure", "Value"]), hide_index=True,
                     width="stretch")
        st.caption(f"Method: {report.method}")
    with right:
        if chart is not None:
            st.plotly_chart(chart, width="stretch")
        elif tables:                              # no chart: show the main table beside the numbers
            name, table = next(iter(tables.items()))
            tables.pop(name)
            st.markdown(f"**{name}**")
            if table is not None and len(table):
                st.dataframe(table, hide_index=True, width="stretch")
            else:
                st.caption("None.")
    for fig in wide_charts:
        st.plotly_chart(fig, width="stretch")

    with st.expander("Why this method? (the decision trail)", expanded=True):
        st.markdown("\n".join(f"{i}. {r}" for i, r in enumerate(report.reasoning, 1)))

    for name, table in tables.items():
        if table is not None and len(table):
            with st.expander(name, expanded=name.startswith("Which pairs") or name == "Signals"):
                st.dataframe(table, hide_index=True, width="stretch")

    st.download_button("Download this report (Markdown)", report.to_markdown(), file_name="stat_advisor_report.md",
                       mime="text/markdown")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def chart_numeric(df, metric, group, order):
    fig = go.Figure()
    for i, g in enumerate(order):
        vals = df.loc[df[group] == g, metric].dropna()
        fig.add_trace(go.Box(y=vals, name=str(g), boxpoints="all" if len(vals) <= 300 else "outliers",
                             jitter=0.35, pointpos=0, marker=dict(size=4, opacity=0.5),
                             line=dict(width=1.5), marker_color=SERIES[i % len(SERIES)]))
    fig.update_layout(title=f"{metric} by {group}", showlegend=False, height=380,
                      margin=dict(l=10, r=10, t=50, b=10), yaxis_title=metric)
    return fig


def chart_rates(summary):
    lo, hi = proportion_confint(summary["conversions"], summary["total"], method="wilson")
    rates = summary["rate"]
    fig = go.Figure(go.Bar(
        x=summary["group"].astype(str), y=rates, marker_color=[SERIES[i % len(SERIES)] for i in range(len(summary))],
        error_y=dict(type="data", symmetric=False, array=hi - rates, arrayminus=rates - lo, color=MUTED, thickness=1.5),
        text=[f"{r:.2%}" for r in rates], textposition="inside", insidetextanchor="start",
        textfont=dict(color="white"),
        hovertemplate="%{x}: %{y:.2%}<extra></extra>"))
    fig.update_layout(title="Rate by group (bars show 95% intervals)", yaxis_tickformat=".1%", height=380,
                      margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    return fig


def chart_monitor(m, labels, metric):
    x = labels
    fig = go.Figure()
    fig.add_vrect(x0=x[0], x1=x[m.baseline_n - 1], fillcolor=MUTED, opacity=0.10, line_width=0,
                  annotation_text="baseline", annotation_position="top left")
    fig.add_trace(go.Scatter(x=x, y=m.values, mode="lines+markers", name=metric,
                             line=dict(color=SERIES[0], width=2), marker=dict(size=5)))
    for y, name, dash in ((m.ucl, "UCL", "dash"), (m.center, "center", "dot"), (m.lcl, "LCL", "dash")):
        fig.add_hline(y=y, line_dash=dash, line_color=MUTED, annotation_text=name, annotation_position="right")
    r1 = next(iter(m.rule_signals.values()))
    if r1:
        fig.add_trace(go.Scatter(x=[x[i] for i in r1], y=m.values[r1], mode="markers", name="outside limits",
                                 marker=dict(color=SIGNAL, size=11, symbol="x")))
    fig.update_layout(title="Individuals chart (sudden jumps)", height=330, margin=dict(l=10, r=90, t=50, b=10),
                      legend=dict(orientation="h", y=-0.2))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=m.cusum_pos, mode="lines", name="upward drift", line=dict(color=SERIES[1], width=2)))
    fig2.add_trace(go.Scatter(x=x, y=m.cusum_neg, mode="lines", name="downward drift", line=dict(color=SERIES[0], width=2)))
    for y in (m.cusum_h, -m.cusum_h):
        fig2.add_hline(y=y, line_dash="dash", line_color=MUTED, annotation_text=f"threshold {y:+g}",
                       annotation_position="right")
    for f in m.findings:
        if f.kind == "drift":
            fig2.add_vline(x=x[f.start], line_color=SIGNAL, line_dash="dot",
                           annotation_text=f"{f.direction} drift starts", annotation_position="top")
    fig2.update_layout(title="CUSUM chart (slow drift, in sigma units)", height=330, margin=dict(l=10, r=90, t=50, b=10),
                       legend=dict(orientation="h", y=-0.2))
    return fig, fig2


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("Stat Advisor")
st.markdown("Ask a question about your data. Stat Advisor **checks the data first**, picks the statistically correct "
            "method, and tells you **what the result means and what to do** — with the reasoning shown.")

QUESTIONS = {
    "compare": "Is there a real difference between groups?  (A/B test variants, machines, teams, suppliers…)",
    "monitor": "Is my process stable, or has it shifted or drifted over time?",
    "plan": "How much data do I need before running a test?",
}
st.markdown("#### 1 · What do you want to know?")
q = st.radio("Question", list(QUESTIONS), format_func=QUESTIONS.get, label_visibility="collapsed")
st.divider()

try:
    # ===================================================================== COMPARE
    if q == "compare":
        st.markdown("#### 2 · Your data")
        source = st.radio("Source", ["Try a sample dataset", "Upload a CSV", "Type in summary numbers"],
                          horizontal=True, label_visibility="collapsed")
        defaults = {}
        df = None
        if source == "Try a sample dataset":
            name = st.selectbox("Sample", list(SAMPLES_COMPARE))
            file, defaults = SAMPLES_COMPARE[name]
            df = load_csv(DATA / file)
        elif source == "Upload a CSV":
            up = st.file_uploader("One row per observation: a column naming the group, and a column with the "
                                  "measurement (a number, or 0/1 / yes/no for conversions).", type="csv")
            df = load_csv(up) if up else None

        if source == "Type in summary numbers":
            n_groups = int(st.number_input("Number of groups", 2, 8, 2))
            cols = st.columns(n_groups)
            names, conv, tot = [], [], []
            for i, c in enumerate(cols):
                with c:
                    names.append(st.text_input("Group", ["A", "B", "C", "D", "E", "F", "G", "H"][i], key=f"n{i}"))
                    tot.append(int(st.number_input("Users / items", 1, value=5000, key=f"t{i}")))
                    conv.append(int(st.number_input("Converted / defective", 0, value=500 + 40 * i, key=f"s{i}")))
            st.markdown("#### 3 · About the comparison")
            c1, c2, c3 = st.columns(3)
            control = c1.selectbox("Control / baseline group", names)
            randomized = c2.checkbox("Randomized A/B test (check group sizes)", value=True)
            hib = c3.radio("Higher rate is…", ["better", "worse"], horizontal=True) == "better"
            with st.expander("Advanced"):
                alpha = st.select_slider("Significance level (alpha)", [0.01, 0.05, 0.10], 0.05)
                mm = st.number_input("Smallest difference that matters (percentage points, 0 = not set)", 0.0, 100.0, 0.0)
            st.divider()
            report = analyze_counts(names, conv, tot, control=control, alpha=alpha, higher_is_better=hib,
                                    randomized=randomized, min_meaningful=mm / 100 or None)
            render_report(report, chart_rates(report.tables["Group summary"]))

        elif df is not None:
            with st.expander("Preview data", expanded=False):
                st.dataframe(df.head(20), width="stretch")
            st.markdown("#### 3 · About the comparison")
            c1, c2 = st.columns(2)
            with c1:
                metric = pick("What are you measuring? (metric column)", df.columns, defaults.get("metric"))
            with c2:
                group = pick("What defines the groups?", [c for c in df.columns if c != metric], defaults.get("group"))
            kind = infer_metric_kind(df[metric])
            st.caption("Detected a **yes/no outcome** — rates will be compared." if kind == "binary"
                       else "Detected a **numeric measurement**.")
            groups = sorted(df[group].dropna().unique().tolist(), key=str)
            if len(groups) > 20:
                st.error(f"'{group}' has {len(groups)} distinct values — that looks like an ID, not a group column.")
                st.stop()
            c1, c2, c3 = st.columns(3)
            ctrl_opts = ["(none)"] + groups
            control = c1.selectbox("Control / baseline group", ctrl_opts,
                                   index=ctrl_opts.index(defaults["control"]) if defaults.get("control") in ctrl_opts else 0)
            randomized = c2.checkbox("Randomized A/B test (check group sizes)", value=defaults.get("randomized", False))
            hib = c3.radio("Higher values are…", ["better", "worse"], horizontal=True,
                           index=0 if defaults.get("higher_is_better", True) else 1) == "better"
            with st.expander("Advanced"):
                alpha = st.select_slider("Significance level (alpha)", [0.01, 0.05, 0.10], 0.05)
                unit = "percentage points" if kind == "binary" else f"units of {metric}"
                mm = st.number_input(f"Smallest difference that matters ({unit}, 0 = not set)", 0.0, value=0.0)
                split = st.text_input("Planned traffic split for the A/B test (e.g. 50,50)", "")
                focus = "average"
                if kind == "numeric":
                    focus = st.radio("Compare…", ["average", "typical"], horizontal=True, format_func={
                        "average": "Averages (default — right when totals matter, e.g. revenue)",
                        "typical": "Typical values (robust to outliers, rank-based)"}.get)
            st.divider()
            mm_val = (mm / 100 if kind == "binary" else mm) or None
            split_val = [float(s) for s in split.split(",")] if split.strip() else None
            report = analyze_comparison(df, metric, group, control=None if control == "(none)" else control,
                                        alpha=alpha, higher_is_better=hib, randomized=randomized,
                                        expected_split=split_val, min_meaningful=mm_val, focus=focus)
            order = report.result.groups
            chart = (chart_rates(report.tables["Group summary"]) if kind == "binary"
                     else chart_numeric(df, metric, group, order))
            render_report(report, chart)
            if kind == "binary" and source == "Try a sample dataset":
                st.info("Tip: switch the metric to **revenue** to see how the advisor handles a heavily skewed "
                        "numeric metric.")

    # ===================================================================== MONITOR
    elif q == "monitor":
        st.markdown("#### 2 · Your data")
        source = st.radio("Source", ["Try a sample dataset", "Upload a CSV"], horizontal=True,
                          label_visibility="collapsed")
        defaults, df = {}, None
        if source == "Try a sample dataset":
            name = st.selectbox("Sample", list(SAMPLES_MONITOR))
            file, defaults = SAMPLES_MONITOR[name]
            df = load_csv(DATA / file)
        else:
            up = st.file_uploader("One row per measurement, ideally with a time or sequence column.", type="csv")
            df = load_csv(up) if up else None

        if df is not None:
            st.markdown("#### 3 · About the process")
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not num_cols:
                st.error("No numeric columns found to monitor.")
                st.stop()
            c1, c2, c3 = st.columns(3)
            metric = pick("Measurement to monitor", num_cols, defaults.get("metric"))
            order_opts = ["(row order)"] + [c for c in df.columns if c != metric]
            order = pick("Time / sequence column", order_opts, defaults.get("order"))
            filt_opts = ["(all rows)"] + [c for c in df.columns if c not in (metric, order) and df[c].nunique() <= 20]
            dflt_filter = defaults.get("filter", (None, None))
            src_key = name if source == "Try a sample dataset" else "upload"
            fcol = pick("Only one group? (filter column)", filt_opts, dflt_filter[0], key=f"fcol-{src_key}",
                        help="Monitor one machine/line at a time — mixing groups hides real changes.")
            sub = df
            if fcol != "(all rows)":
                vals = sorted(df[fcol].dropna().unique().tolist(), key=str)
                fval = pick(f"Which {fcol}?", vals, dflt_filter[1], key=f"fval-{src_key}-{fcol}")
                sub = df[df[fcol] == fval]
            n = len(sub.dropna(subset=[metric]))
            with st.expander("Advanced"):
                base = st.slider("Baseline period: learn 'normal' from the first N points", 8, max(9, n),
                                 min(100, max(8, n // 2)), key=f"base-{metric}-{order}-{fcol}-{n}")
                sensitive = st.checkbox("Sensitive mode (Western Electric rules 2-4 + moving-range chart; "
                                        "faster detection, more false alarms)")
                sig = st.slider("Control limit width (sigma)", 2.0, 4.0, 3.0, 0.5)
                h = st.slider("CUSUM threshold h (higher = fewer false alarms, slower detection)", 3.0, 8.0, 5.0, 0.5)
            st.divider()
            report = analyze_monitoring(sub, metric, order_col=None if order == "(row order)" else order,
                                        baseline_n=base, sensitive_rules=sensitive, sigma_multiplier=sig, cusum_h=h)
            ordered = sub.sort_values(order, kind="stable") if order != "(row order)" else sub
            ordered = ordered.dropna(subset=[metric])
            labels = (ordered[order].tolist() if order != "(row order)" else list(range(1, len(ordered) + 1)))
            f1, f2 = chart_monitor(report.result, labels, metric)
            render_report(report, None, wide_charts=(f1, f2))

    # ===================================================================== PLAN
    else:
        st.markdown("#### 2 · What will you measure?")
        kind = st.radio("Metric", ["A conversion rate (yes/no outcome)", "An average (revenue, time, defect rate…)"],
                        horizontal=True, label_visibility="collapsed")
        c1, c2, c3, c4 = st.columns(4)
        groups = int(c4.number_input("Groups (incl. control)", 2, 8, 2))
        traffic = c4.number_input("Users/items per day (optional)", 0, value=0)
        alpha = c3.select_slider("Significance level", [0.01, 0.05, 0.10], 0.05)
        power = c3.select_slider("Power (chance of detecting a real effect)", [0.7, 0.8, 0.9, 0.95], 0.8)
        if kind.startswith("A conversion"):
            base = c1.number_input("Current conversion rate (%)", 0.1, 99.0, 10.0) / 100
            lift = c2.number_input("Smallest relative lift worth detecting (%)", 0.5, 500.0, 10.0) / 100
            report = plan_conversion_test(base, lift, alpha=alpha, power=power, groups=groups,
                                          daily_traffic=traffic or None)
        else:
            sd = c1.number_input("Typical standard deviation of the metric", 0.0001, value=10.0)
            diff = c2.number_input("Smallest difference worth detecting (same units)", 0.0001, value=2.0)
            report = plan_mean_test(sd, diff, alpha=alpha, power=power, groups=groups, daily_traffic=traffic or None)
        st.divider()
        render_report(report)

except ValueError as e:
    st.error(f"Can't analyse this data: {e}")

st.divider()
st.caption("Methods: Welch's t-test / Welch's ANOVA + Games-Howell, Brunner-Munzel / Mann-Whitney / Kruskal-Wallis + Dunn, "
           "two-proportion z / Boschloo exact / chi-square, sample-ratio-mismatch check, Individuals + CUSUM control "
           "charts with a baseline period, power analysis. See docs/METHODOLOGY.md.")
