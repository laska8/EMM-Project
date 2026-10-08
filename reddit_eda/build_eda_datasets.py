from eda_datasets import EdaDatasets


# -----------------------------------
# Build every EDA dataset and save it to eda_outputs/
# (datasets that already exist are skipped unless rebuild=True)
# -----------------------------------

datasets = EdaDatasets(rebuild=False)

# user-month and subreddit-month datasets, straight from the JSON files
datasets.build_streamed_datasets()

# summary tables
steps = [
    datasets.activity_return,
    datasets.community_return,
    datasets.monthly_summary,
    datasets.return_by_month,
    datasets.subreddit_summary
]

for step in steps:

    df = step()

    print(f"{step.__name__}: {len(df):,} rows")
