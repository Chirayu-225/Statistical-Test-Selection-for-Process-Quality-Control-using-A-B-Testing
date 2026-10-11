"""
Calibration study: does the engine actually deliver the error rates it promises?

A test at alpha = 0.05 should produce a "significant" result about 5% of the
time when there is truly NO difference (Type I error), and should detect real
differences as often as possible (power). We simulate thousands of datasets
where we KNOW the truth and count how often each engine gets it right.

Compares the original v1 engine (simulations/legacy_v1.py) with v2 (stat_advisor).

    python -m simulations.calibration_study            # full run, writes results (~5-15 min)
    python -m simulations.calibration_study --quick    # smoke run
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from simulations import legacy_v1 as v1
from stat_advisor.compare import compare_means, compare_proportions
from stat_advisor.monitor import monitor_process

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.05


def _v1_means(groups):
    df = pd.DataFrame({"value": np.concatenate(groups),
                       "group": np.repeat(np.arange(len(groups)), [len(g) for g in groups])})
    return v1.run_numeric_test(df, "value", "group", alpha=ALPHA).significant


def _v2_means(groups):
    return compare_means({i: g for i, g in enumerate(groups)}, alpha=ALPHA, posthoc=False).significant


# (name, truth, generator(rng) -> list of arrays)
MEAN_SCENARIOS = [
    ("2 groups · normal · equal spread · n=30/30", "no difference",
     lambda r: [r.normal(0, 1, 30), r.normal(0, 1, 30)]),
    ("2 groups · normal · spread 1 vs 3 · n=60/15", "no difference",
     lambda r: [r.normal(0, 1, 60), r.normal(0, 3, 15)]),
    ("3 groups · normal · spread 1/1/4 · n=60/15/15", "no difference",
     lambda r: [r.normal(0, 1, 60), r.normal(0, 1, 15), r.normal(0, 4, 15)]),
    ("3 groups · normal · spread 1/2/3 · n=20/20/20", "no difference",
     lambda r: [r.normal(0, 1, 20), r.normal(0, 2, 20), r.normal(0, 3, 20)]),
    ("2 groups · skewed (lognormal) · n=20/20", "no difference",
     lambda r: [r.lognormal(0, 1, 20), r.lognormal(0, 1, 20)]),
    ("2 groups · skewed (lognormal) · n=500/500", "no difference",
     lambda r: [r.lognormal(0, 1, 500), r.lognormal(0, 1, 500)]),
    ("2 groups · normal · shift d=0.5 · n=50/50", "real difference",
     lambda r: [r.normal(0, 1, 50), r.normal(0.5, 1, 50)]),
    ("3 groups · normal · one mean +0.6 · n=40 each", "real difference",
     lambda r: [r.normal(0, 1, 40), r.normal(0, 1, 40), r.normal(0.6, 1, 40)]),
    ("2 groups · heavy tails (t, df=3) · shift 0.5 · n=200/200", "real difference",
     lambda r: [r.standard_t(3, 200), r.standard_t(3, 200) + 0.5]),
]


def _v1_props(s, t):
    return v1.run_proportion_test(list(s), list(t), alpha=ALPHA).significant


def _v2_props(s, t):
    return compare_proportions(list(s), list(t), alpha=ALPHA).significant


PROP_SCENARIOS = [
    ("2 rates · 5% vs 5% · n=40/40 (sparse)", "no difference", (0.05, 0.05), (40, 40)),
    ("2 rates · 5% vs 5% · n=15/60 (sparse, unbalanced)", "no difference", (0.05, 0.05), (15, 60)),
    ("2 rates · 10% vs 10% · n=2000/2000", "no difference", (0.10, 0.10), (2000, 2000)),
    ("2 rates · 10% vs 12% · n=2000/2000", "real difference", (0.10, 0.12), (2000, 2000)),
]


def _v2_typical(groups):
    return compare_means({i: g for i, g in enumerate(groups)}, alpha=ALPHA, focus="typical", posthoc=False).significant


def _mean_worker(args):
    i, reps, seed = args
    name, truth, gen = MEAN_SCENARIOS[i]
    rng = np.random.default_rng(seed)
    hits = np.zeros(3)
    for _ in range(reps):
        data = gen(rng)
        hits += [_v1_means(data), _v2_means(data), _v2_typical(data)]
    return (name, truth, *(hits / reps))


def _prop_worker(args):
    i, reps, seed = args
    name, truth, p, n = PROP_SCENARIOS[i]
    rng = np.random.default_rng(seed)
    hits = np.zeros(2)
    for _ in range(reps):
        s = rng.binomial(n, p)
        if s.sum() == 0:                 # v1 returns a NaN p-value here (counted as not significant)
            continue
        hits += [_v1_props(s, n), _v2_props(s, n)]
    return (name, truth, *(hits / reps), np.nan)


def run_comparisons(reps: int, seed: int = 7, workers: int | None = None) -> pd.DataFrame:
    from multiprocessing import Pool
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    jobs = [(_mean_worker, (i, reps, seed + i)) for i in range(len(MEAN_SCENARIOS))] + \
           [(_prop_worker, (i, reps, seed + 100 + i)) for i in range(len(PROP_SCENARIOS))]
    print(f"  running {len(jobs)} scenarios on {workers} worker process(es); each line prints as it finishes")
    t0 = time.time()
    rows = []
    with Pool(workers) as pool:
        for done, row in enumerate(pool.imap_unordered(_run_job, jobs), 1):
            rows.append(row)
            v3 = "" if pd.isna(row[4]) else f"  v2 typical={row[4]:.3f}"
            print(f"  [{done:2d}/{len(jobs)}] {row[0]:58s} v1={row[2]:.3f}  v2={row[3]:.3f}{v3}"
                  f"   ({time.time() - t0:.0f}s)", flush=True)
    order = [s[0] for s in MEAN_SCENARIOS] + [s[0] for s in PROP_SCENARIOS]
    rows.sort(key=lambda r: order.index(r[0]))
    df = pd.DataFrame(rows, columns=["scenario", "truth", "v1", "v2", "v2_typical"])
    df["metric"] = np.where(df["truth"] == "no difference", "false-positive rate (target 5%)", "power (higher is better)")
    return df


def _run_job(job):
    fn, args = job
    return fn(args)


def run_monitoring(reps: int, seed: int = 11) -> pd.DataFrame:
    """100 points per series. In control: N(50, 2). Shift: +1 sigma from point 50 on."""
    out = {"v1 (any signal)": [0, 0, []], "v2 (any signal)": [0, 0, []], "v2 (confirmed only)": [0, 0, []]}
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        stable = rng.normal(50, 2, 100)
        shifted = np.r_[rng.normal(50, 2, 50), rng.normal(52, 2, 50)]
        for series, idx in ((stable, 0), (shifted, 1)):
            a = v1.shewhart_individuals_chart(series)
            b = v1.cusum_chart(series)
            v1_sig = sorted(set(a.violations) | set(b.violations))
            m = monitor_process(series)
            v2_any = sorted({f.detected_at for f in m.findings})
            v2_conf = sorted({f.detected_at for f in m.findings if f.confirmed})
            for key, sig in (("v1 (any signal)", v1_sig), ("v2 (any signal)", v2_any), ("v2 (confirmed only)", v2_conf)):
                if idx == 0:
                    out[key][0] += bool(sig)
                else:
                    post = [s for s in sig if s >= 50]
                    pre = [s for s in sig if s < 50]
                    out[key][1] += bool(post) and not pre
                    if post:
                        out[key][2].append(post[0] - 50)
    rows = [(k, v[0] / reps, v[1] / reps, float(np.median(v[2])) if v[2] else np.nan) for k, v in out.items()]
    return pd.DataFrame(rows, columns=["detector", "false alarm (stable series)",
                                       "correctly flags the real shift (and nothing before it)",
                                       "median delay (points)"])


def plot(df: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    null = df[df["truth"] == "no difference"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(null) + 1.4))
    y = np.arange(len(null))
    ax.barh(y + 0.19, null["v1"] * 100, height=0.36, color="#eb6834", label="v1 (original)")
    ax.barh(y - 0.19, null["v2"] * 100, height=0.36, color="#2a78d6", label="v2 (this version)")
    for yi, (a, b) in enumerate(zip(null["v1"], null["v2"])):
        box = dict(facecolor="white", edgecolor="none", pad=0.6)
        ax.text(a * 100 + 0.3, yi + 0.19, f"{a:.1%}", va="center", fontsize=8, color="#52514e", bbox=box)
        ax.text(b * 100 + 0.3, yi - 0.19, f"{b:.1%}", va="center", fontsize=8, color="#52514e", bbox=box)
    ax.axvline(5, color="#0b0b0b", lw=1, ls="--", zorder=0)
    ax.set_xlim(0, max(null["v1"].max(), null["v2"].max()) * 100 + 3)
    ax.text(5.2, len(null) - 0.45, "promised 5%", fontsize=8, color="#0b0b0b")
    ax.set_yticks(y, null["scenario"], fontsize=8)
    ax.set_xlabel("False-positive rate when there is NO real difference (%)")
    ax.set_title("How often does each engine cry wolf?", loc="left", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="200 datasets per scenario, prints only (~1-2 min)")
    ap.add_argument("--reps", type=int, default=2000, help="datasets per scenario (default 2000)")
    ap.add_argument("--workers", type=int, default=None, help="parallel processes (default: CPU cores - 1)")
    ap.add_argument("--plot-only", action="store_true", help="redraw docs/calibration.png from results.csv")
    args = ap.parse_args()
    if args.plot_only:
        plot(pd.read_csv(ROOT / "simulations" / "results.csv"), ROOT / "docs" / "calibration.png")
        return
    reps = 200 if args.quick else args.reps
    t0 = time.time()
    print(f"Group comparisons ({reps} simulated datasets per scenario)...", flush=True)
    comp = run_comparisons(reps, workers=args.workers)
    print("Process monitoring...", flush=True)
    mon = run_monitoring(reps // 4)
    print(mon.to_string(index=False))
    if args.quick:
        return
    pct = lambda col: comp[col].map(lambda x: "—" if pd.isna(x) else f"{x:.1%}")
    fmt = comp.assign(v1=pct("v1"), v2=pct("v2"), v2_typical=pct("v2_typical"))
    md = ["# Calibration study results", "",
          f"{reps:,} simulated datasets per scenario, alpha = {ALPHA}. Monte Carlo standard error ≈ 0.5 percentage "
          f"points at a 5% rate. Generated by `python -m simulations.calibration_study`.", "",
          "## Group comparisons", "",
          "Rows marked *no difference*: the number is the false-positive rate and should be close to 5%. "
          "Rows marked *real difference*: the number is power — higher is better.", "",
          "`v1` = original engine · `v2` = this version (default: compare averages when valid) · "
          "`v2_typical` = this version with focus='typical' (always rank-based).", "",
          fmt[["scenario", "truth", "metric", "v1", "v2", "v2_typical"]].to_markdown(index=False), "",
          f"## Process monitoring ({reps // 4:,} simulated series of 100 points)", "",
          "Stable series: N(50, 2). Shifted series: +1 sigma from point 50 onward.", "",
          mon.round(3).to_markdown(index=False), "",
          f"_Runtime: {time.time() - t0:.0f} s_"]
    (ROOT / "simulations" / "results.md").write_text("\n".join(md), encoding="utf-8")
    comp.to_csv(ROOT / "simulations" / "results.csv", index=False)
    plot(comp, ROOT / "docs" / "calibration.png")
    print("wrote simulations/results.md and docs/calibration.png")


if __name__ == "__main__":
    main()
