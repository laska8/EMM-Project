from pathlib import Path

import numpy as np
import pandas as pd

from beam_search import BeamSearchEMM


BASE_DIR = Path(__file__).resolve().parent

# built by reddit_eda/build_eda_datasets.py
DATA = BASE_DIR.parent / "eda_outputs" / "reciprocity_return.csv"

COLUMNS = [
    "user",
    "subreddit",
    "month",
    "replies_sent",
    "replies_received",
    "reciprocal_partners",
    "returned_next_month",
]


def prepare_data(df):

    df["recip_binary"] = (
        df["reciprocal_partners"] > 0
    ).astype(int)

    df["log_sent"] = np.log1p(
        df["replies_sent"]
    )

    df["log_received"] = np.log1p(
        df["replies_received"]
    )

    df["user"] = df["user"].astype(
        "category"
    )

    df["subreddit"] = df["subreddit"].astype(
        "category"
    )

    return df


df = pd.read_csv(
    DATA,
    usecols=COLUMNS,
    dtype={"user": "category", "subreddit": "category"},
    keep_default_na=False,
)

df = prepare_data(df)


selector_columns = [
    "replies_sent",
    "replies_received",
]

search = BeamSearchEMM(
    df=df,
    selector_columns=selector_columns,
    n_bins=8,
    beam_width=10,
    max_depth=3,
    min_support=5000,
    min_cell=10,
    alpha=0.5,
)


results = search.search()

print(
    results.head(20)
)

results.to_csv(
    BASE_DIR / "emm_beam_search_results.csv",
    index=False
)