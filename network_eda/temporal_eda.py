from pathlib import Path
import pandas as pd

from temporal_analyzer import TemporalAnalyzer
from temporal_dataset_analyzer import TemporalDatasetAnalyzer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "reply_networks"


# Analyze every subreddit separately

results = []

for file_path in DATA_DIR.glob("*.json"):

    analyzer = TemporalAnalyzer(
        file_path=file_path,
        subreddit_name=file_path.stem
    )

    summary = analyzer.get_summary()
    results.append(summary)


comparison_df = pd.DataFrame(results)

#print("Number of subreddits:", len(comparison_df))


# Dataset-level analysis

dataset_analyzer = TemporalDatasetAnalyzer(
    comparison_df
)

# Add differences between return windows
comparison_df = (
    dataset_analyzer
    .add_return_window_differences()
)


# Summary statistics
summary_stats = (
    dataset_analyzer
    .get_summary_statistics()
)

print("\nSummary statistics:")
print(summary_stats)


# Filter by subreddit size

filtered_1000 = (
    dataset_analyzer
    .filter_by_size(1000)
)

filtered_5000 = (
    dataset_analyzer
    .filter_by_size(5000)
)

print(
    "\nSubreddits with at least 1000 users:",
    len(filtered_1000)
)

print(
    "Subreddits with at least 5000 users:",
    len(filtered_5000)
)


# Highest / lowest return

print("\nLowest next-month return (>= 1000 users):")
print(
    filtered_1000
    .sort_values("return_next_month_pct")
    .head(10)
)

print("\nHighest next-month return (>= 1000 users):")
print(
    filtered_1000
    .sort_values(
        "return_next_month_pct",
        ascending=False
    )
    .head(10)
)


# Save results

comparison_df.to_csv(
    "temporal_eda_summary_all_subreddits.csv",
    index=False
)

