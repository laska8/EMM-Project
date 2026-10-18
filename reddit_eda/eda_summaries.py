import numpy as np
import pandas as pd


class EdaSummaries:

    ACTIVITY_LEVELS = ["1", "2-3", "4-10", "11+"]

    @staticmethod
    def classify_activity(x):

        if x == 1:
            return "1 reply"
        elif x <= 3:
            return "2–3 replies"
        elif x <= 10:
            return "4–10 replies"
        else:
            return "11+ replies"


    @staticmethod
    def add_activity_group(df):

        df["activity_group"] = (
            df["replies_sent"]
            .apply(EdaSummaries.classify_activity)
        )

        return df


    @staticmethod
    def add_reciprocity_group(df):

        # fixed groups instead of qcut because of the duplicate edges problem
        df["reciprocity_group"] = pd.cut(
            df["proportional_reciprocity"],
            bins=[-0.001, 0, 0.25, 0.50, 0.75, 1.0],
            labels=[
                "0",
                ">0–0.25",
                ">0.25–0.50",
                ">0.50–0.75",
                ">0.75–1"
            ]
        )

        return df


    @staticmethod
    def return_rate_by(df, by, count_column="observations"):

        summary = (
            df
            .groupby(by, observed=True)
            .agg(**{
                count_column: ("user", "count"),
                "return_rate": ("returned_next_month", "mean")
            })
            .reset_index()
        )

        summary["return_rate_percent"] = (
            summary["return_rate"] * 100
        )

        return summary


    @staticmethod
    def reciprocity_by(reciprocity_df, by):

        return (
            reciprocity_df
            .groupby(by, observed=True)
            .agg(
                observations=("user", "count"),
                mean_reciprocity=(
                    "proportional_reciprocity",
                    "mean"
                ),
                median_reciprocity=(
                    "proportional_reciprocity",
                    "median"
                )
            )
            .reset_index()
        )


    # Analysis sample used for the paper tables and figures

    @staticmethod
    def analysis_sample(reciprocity_return_df):

        keep_cols = [
            "subreddit",
            "month",
            "replies_sent",
            "replies_received",
            "unique_users_replied_to",
            "unique_users_who_replied",
            "reciprocal_partners",
            "proportional_reciprocity",
            "returned_next_month"
        ]

        df = reciprocity_return_df[keep_cols].copy()

        # binary reciprocity: 1 if the user has at least one partner who replied back, else 0
        df["recip_binary"] = (df["reciprocal_partners"] > 0).astype(int)

        # number of distinct people the user interacted with in either direction
        # (replied to + replied by, minus the ones counted twice because they are both)
        df["union_partners"] = (
            df["unique_users_replied_to"]
            + df["unique_users_who_replied"]
            - df["reciprocal_partners"]
        )

        # activity groups by number of replies sent
        df["activity_group"] = pd.cut(
            df["replies_sent"],
            bins=[0, 1, 3, 10, np.inf],
            labels=EdaSummaries.ACTIVITY_LEVELS
        )

        # reciprocity groups (0 gets its own group because so many users have exactly 0)
        df["reciprocity_group"] = pd.cut(
            df["proportional_reciprocity"],
            bins=[-0.001, 0, 0.25, 0.5, 0.75, 1.0],
            labels=["0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1]"]
        )

        return df


    @staticmethod
    def descriptive_table(df):

        # variables to describe, with readable names for the paper
        variables = {
            "replies_sent": "Out-degree (replies sent)",
            "replies_received": "In-degree (replies received)",
            "union_partners": "Distinct partners (in or out)",
            "reciprocal_partners": "Reciprocal partners",
            "proportional_reciprocity": "Proportional reciprocity"
        }

        rows = []

        for column, label in variables.items():

            x = df[column]

            rows.append({
                "Variable": label,
                "Mean": x.mean(),
                "SD": x.std(),
                "Median": x.median(),
                "P99": x.quantile(0.99),           # 99% of users are below this value
                "Max": x.max(),
                "Skewness": x.skew(),              # > 1 means a long right tail
                "% zero": (x == 0).mean() * 100    # sparsity
            })

        return pd.DataFrame(rows).set_index("Variable")


    @staticmethod
    def top_share(values, fraction):

        # share of the total that comes from the largest `fraction` of observations
        values_sorted = np.sort(values.values)[::-1]
        top = values_sorted[: int(fraction * len(values_sorted))].sum()

        return top / values_sorted.sum() * 100


    @staticmethod
    def return_by_activity_and_reciprocity(df):

        # return rate and group size for every (activity level, reciprocity level) combination
        return (
            df
            .groupby(
                ["activity_group", "reciprocity_group"],
                observed=True
            )["returned_next_month"]
            .agg(n="size", return_rate="mean")
            .reset_index()
        )


    @staticmethod
    def reciprocity_composition(by_both):

        # share of each activity level within each reciprocity group
        shares = by_both.copy()

        shares["total_n"] = (
            shares
            .groupby("reciprocity_group", observed=True)["n"]
            .transform("sum")
        )

        shares["share"] = (shares["n"] / shares["total_n"]) * 100

        # rows = reciprocity_group, cols = activity_group
        plot_matrix = shares.pivot(
            index="reciprocity_group",
            columns="activity_group",
            values="share"
        ).fillna(0)

        return plot_matrix.reindex(
            columns=EdaSummaries.ACTIVITY_LEVELS,
            fill_value=0
        )


    @staticmethod
    def subreddit_comparison(df):

        # one row per subreddit: size, return rate, average reciprocity, average activity
        subs = (
            df
            .groupby("subreddit", observed=True)
            .agg(
                n=("returned_next_month", "size"),
                return_rate=("returned_next_month", "mean"),
                mean_reciprocity=("proportional_reciprocity", "mean"),
                mean_replies_sent=("replies_sent", "mean")
            )
            .reset_index()
        )

        # only users with >= 2 replies, because single-reply users add nothing here
        multi = df[df["replies_sent"] >= 2]

        # return rate and group size, for each subreddit x activity level x (reciprocated yes/no)
        t = (
            multi
            .groupby(
                ["subreddit", "activity_group", "recip_binary"],
                observed=True
            )["returned_next_month"]
            .agg(["mean", "size"])
            .unstack("recip_binary")    # puts "not reciprocated" (0) and "reciprocated" (1) side by side
        )

        # names like mean_0, mean_1, size_0, size_1
        t.columns = [str(stat) + "_" + str(r) for stat, r in t.columns]
        t = t.dropna()

        # need enough users in both groups
        t = t[(t["size_0"] >= 30) & (t["size_1"] >= 30)].copy()

        # return(reciprocated) - return(not reciprocated)
        t["gap"] = t["mean_1"] - t["mean_0"]
        t["weight"] = t["size_0"] + t["size_1"]
        t["gap_x_weight"] = t["gap"] * t["weight"]

        # average the gap over activity levels, weighted by group size, so activity is held roughly fixed
        gap = (
            t.groupby("subreddit", observed=True)["gap_x_weight"].sum()
            / t.groupby("subreddit", observed=True)["weight"].sum()
        )
        gap = gap.rename("adjusted_gap").reset_index()

        return subs.merge(gap, on="subreddit", how="left")


    @staticmethod
    def representative_subreddits(subs):

        by_size = subs.sort_values("n").reset_index(drop=True)
        middle = len(by_size) // 2
        by_recip = subs.sort_values("mean_reciprocity")

        return pd.concat([
            by_size.head(3),                        # 3 smallest
            by_size.iloc[middle - 1: middle + 2],   # 3 medium
            by_size.tail(3),                        # 3 largest
            by_recip.head(1),                       # lowest reciprocity
            by_recip.tail(1)                        # highest reciprocity
        ]).drop_duplicates("subreddit")
