import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

from emm_selectors import Selector, apply_description


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
    "reciprocal_partners",
    "returned_next_month",
]

# The main model holds the number of users replied to (sent) and the number
# of users who replied (received) fixed. With both fixed, a reciprocal
# partner means that the two sets overlap, so the user was in contact with
# FEWER distinct people: contacts = sent + received - reciprocal partners.
# This model holds the number of distinct contacts fixed instead.
FORMULA = (
    "returned_next_month ~ recip_binary + log_contacts"
    " | month + subreddit"
)


def prepare_data(df):

    df["recip_binary"] = (
        df["reciprocal_partners"] > 0
    ).astype(int)

    df["log_contacts"] = np.log1p(
        df["replies_sent"]
        + df["replies_received"]
        - df["reciprocal_partners"]
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


def fit(df):

    model = pf.feglm(
        FORMULA,
        data=df,
        family="logit",
        vcov={"CRV1": "user"},
        copy_data=False,
        fixef_rm="none",
        iwls_maxiter=100,
    )

    if not model.convergence:
        raise RuntimeError("Model did not converge.")

    row = model.tidy().loc["recip_binary"]

    return {
        "beta_contacts": row["Estimate"],
        "se_contacts": row["Std. Error"],
        "ci_low_contacts": row["2.5%"],
        "ci_high_contacts": row["97.5%"],
    }


df = pd.read_csv(
    DATA,
    usecols=COLUMNS,
    dtype={"user": "category", "subreddit": "category"},
    keep_default_na=False,
)

df = prepare_data(df)

results = pd.read_csv(RESULTS)

top = results.nlargest(TOP_K, "quality")


# Reference model and top-k subgroups, distinct contacts held fixed

rows = [{
    "description": "all observations",
    "n": len(df),
    "beta": results["reference_beta"].iloc[0],
    **fit(df),
}]

print(
    f"all observations: {rows[0]['beta']:.3f} -> "
    f"{rows[0]['beta_contacts']:.3f}",
    flush=True
)

for row in top.itertuples():

    subgroup = df[
        apply_description(df, parse_description(row.description))
    ]

    rows.append({
        "description": row.description,
        "n": len(subgroup),
        "beta": row.beta,
        **fit(subgroup),
    })

    print(
        f"{row.description}: {row.beta:.3f} -> "
        f"{rows[-1]['beta_contacts']:.3f}",
        flush=True
    )

check = pd.DataFrame(rows)

print(
    check
)

check.to_csv(
    BASE_DIR / "emm_breadth_check_results.csv",
    index=False
)
