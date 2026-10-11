"""Generate the realistic sample datasets shipped with the project (deterministic).

    python data/make_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent


def checkout_ab_test(rng):
    """A/B test of a new checkout page. Each row is one user."""
    rows = []
    for variant, n, rate in (("A", 6000, 0.100), ("B", 5970, 0.118)):
        converted = rng.random(n) < rate
        revenue = np.where(converted, np.round(rng.lognormal(3.7, 0.6, n), 2), 0.0)
        rows.append(pd.DataFrame({"variant": variant, "converted": converted.astype(int), "revenue": revenue}))
    df = pd.concat(rows).sample(frac=1, random_state=1).reset_index(drop=True)
    df.insert(0, "user_id", np.arange(100001, 100001 + len(df)))
    return df


def machine_defects(rng):
    """Daily defect rate (%) for three machines on the same line. M3 is worse and less consistent."""
    dates = pd.date_range("2026-03-02", periods=60, freq="D")
    parts = []
    for machine, mu, sd in (("M1", 2.0, 0.40), ("M2", 2.1, 0.45), ("M3", 2.6, 0.80)):
        parts.append(pd.DataFrame({"date": dates.date, "machine": machine,
                                   "defect_rate_pct": np.round(np.clip(rng.normal(mu, sd, 60), 0, None), 3)}))
    return pd.concat(parts, ignore_index=True)


def support_resolution_times(rng):
    """Hours to resolve a support ticket, by team. Small, heavily skewed samples."""
    parts = []
    for team, n, median_h in (("Alpha", 25, 6.0), ("Beta", 22, 6.5), ("Gamma", 28, 16.0)):
        parts.append(pd.DataFrame({"team": team, "resolution_hours": np.round(rng.lognormal(np.log(median_h), 0.9, n), 1)}))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=2).reset_index(drop=True)


def bottle_fill_process(rng):
    """Fill volume (ml) per batch. Stable for 70 batches, then a slow upward drift
    (worn valve) that never produces an extreme single reading for a long time."""
    n, start = 120, 70
    drift = np.where(np.arange(n) >= start, (np.arange(n) - start) * 0.04, 0.0)
    vol = 500 + drift + rng.normal(0, 1.5, n)
    ts = pd.date_range("2026-05-04 06:00", periods=n, freq="2h")
    return pd.DataFrame({"batch": np.arange(1, n + 1), "timestamp": ts, "fill_volume_ml": np.round(vol, 2)})


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    for name, fn in (("checkout_ab_test", checkout_ab_test), ("machine_defects", machine_defects),
                     ("support_resolution_times", support_resolution_times),
                     ("bottle_fill_process", bottle_fill_process)):
        df = fn(rng)
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"wrote data/{name}.csv  ({len(df)} rows)")
