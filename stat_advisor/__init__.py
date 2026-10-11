"""stat_advisor — ask a statistical question about your data, get a trustworthy answer.

Quick start:
    import pandas as pd
    from stat_advisor import analyze_comparison
    report = analyze_comparison(pd.read_csv("data/checkout_ab_test.csv"),
                                metric="converted", group="variant", control="A", randomized=True)
    print(report.to_markdown())
"""
from .advisor import (Report, analyze_comparison, analyze_counts, analyze_monitoring,
                      plan_conversion_test, plan_mean_test, infer_metric_kind)
from .compare import compare_means, compare_proportions, games_howell, dunn_test, adjust_pvalues
from .monitor import monitor_process

__all__ = ["Report", "analyze_comparison", "analyze_counts", "analyze_monitoring", "plan_conversion_test",
           "plan_mean_test", "infer_metric_kind", "compare_means", "compare_proportions", "games_howell",
           "dunn_test", "adjust_pvalues", "monitor_process"]
__version__ = "2.0.0"
