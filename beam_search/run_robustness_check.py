import re
from pathlib import Path

import numpy as np
import pandas as pd

from emm_selectors import Selector, apply_description
from logistic_model import fit_logistic_model


BASE_DIR = Path(__file__).resolve().parent

# built by reddit_eda/build_eda_datasets.py
DATA = BASE_DIR.parent / "eda_outputs" / "reciprocity_return.csv"

RESULTS = BASE_DIR / "emm_beam_search_results.csv"

TOP_K = 10

COLUMNS = [
    "user",
    "subreddit",
    "month",
    "replies_sent",
    "replies_received",
    "unique_users_replied_to",
    "unique_users_who_replied",
    "reciprocal_partners",
    "returned_next_month",
]

# the model of the beam search plus the number of distinct partners:
# a user who talks to more people is more likely to have a reciprocal
# partner, whatever the number of replies
FORMULA = (
    "returned_next_month ~ recip_binary + log_sent + log_received"
    " + log_partners_out + log_partners_in"
    " | month + subreddit"
)


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

    df["log_partners_out"] = np.log1p(
        df["unique_users_replied_to"]
    )

    df["log_partners_in"] = np.log1p(
        df["unique_users_who_replied"]
    )

    return df


def parse_description(description):

    selectors = []

    for part in description.split(" AND "):
        column, operator, value = re.match(
            r"(\w+?)(<=|>)(.+)", part
        ).groups()
        selectors.append(Selector(column, operator, float(value)))

    return selectors


df = pd.read_csv(
    DATA,
    usecols=COLUMNS,
    dtype={"user": "category", "subreddit": "category"},
    keep_default_na=False,
)

df = prepare_data(df)

results = pd.read_csv(RESULTS)

top = results.nlargest(TOP_K, "quality")


# -----------------------------------
# Reference model with the extra controls
# -----------------------------------

reference = fit_logistic_model(df, FORMULA)

if reference is None:
    raise RuntimeError(
        "Reference model did not converge."
    )

print(
    f"Reference: {results['reference_beta'].iloc[0]:.3f} -> "
    f"{reference['beta']:.3f} (SE {reference['se']:.3f})",
    flush=True
)


# -----------------------------------
# Top-k subgroups with the extra controls
# -----------------------------------

rows = [{
    "description": "all observations",
    "n": len(df),
    "beta": results["reference_beta"].iloc[0],
    "beta_controls": reference["beta"],
    "se_controls": reference["se"],
    "ci_low_controls": reference["ci_low"],
    "ci_high_controls": reference["ci_high"],
    "delta": 0.0,
    "delta_controls": 0.0,
}]

for row in top.itertuples():

    subgroup = df[
        apply_description(df, parse_description(row.description))
    ]

    model_result = fit_logistic_model(subgroup, FORMULA)

    # reciprocity collinear with the controls, or no convergence
    if model_result is None:
        print(f"{row.description}: no estimate", flush=True)
        continue

    rows.append({
        "description": row.description,
        "n": len(subgroup),
        "beta": row.beta,
        "beta_controls": model_result["beta"],
        "se_controls": model_result["se"],
        "ci_low_controls": model_result["ci_low"],
        "ci_high_controls": model_result["ci_high"],
        "delta": row.delta,
        "delta_controls": abs(
            model_result["beta"] - reference["beta"]
        ),
    })

    print(
        f"{row.description}: {row.beta:.3f} -> "
        f"{model_result['beta']:.3f}",
        flush=True
    )

robustness = pd.DataFrame(rows)

print(
    robustness
)

robustness.to_csv(
    BASE_DIR / "emm_robustness_results.csv",
    index=False
)
