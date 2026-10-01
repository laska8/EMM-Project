import pandas as pd

from temporal_plots import TemporalPlots


comparison_df = pd.read_csv(
    "temporal_eda_summary_all_subreddits.csv"
)

comparison_df["extra_return_2m"] = (
    comparison_df["return_within_2_months_pct"]
    - comparison_df["return_next_month_pct"]
)

comparison_df["extra_return_3m"] = (
    comparison_df["return_within_3_months_pct"]
    - comparison_df["return_next_month_pct"]
)


TemporalPlots.return_rate_distribution(
    comparison_df
)

TemporalPlots.return_window_comparison(
    comparison_df
)

TemporalPlots.return_window_gain(
    comparison_df
)
active_month_counts = pd.Series({
    1: 12289345,
    2: 3201954,
    3: 1500007,
    4: 868478,
    5: 557094,
    6: 384289,
    7: 278898,
    8: 210449,
    9: 165105,
    10: 137000,
    11: 163645
})
TemporalPlots.active_months_distribution(
    active_month_counts
)
pattern_summary = pd.DataFrame({
    "pattern_type": [
        "one_month_only",
        "reentry_or_intermittent",
        "other",
        "new_entrant_then_persistent",
        "quitter",
        "continuous"
    ],
    "percentage": [
        62.204802,
        26.418224,
        6.250134,
        2.693581,
        1.604939,
        0.828320
    ]
})
TemporalPlots.pattern_distribution(
    pattern_summary
)