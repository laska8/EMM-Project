import json
import os
import pandas as pd
from collections import Counter

# FAST DATA CHECK, WILL BE CHANGED

data_folder = "reply_networks"

# -----------------------------------
# 1. Helper: define "present in month"
# -----------------------------------

def get_active_users(month_data):
    """
    Here, a user is considered active if they sent at least one reply,
    i.e. if they appear as a key in the JSON.
    """
    return set(month_data.keys())


# -----------------------------------
# 2. Build user-subreddit trajectories
# -----------------------------------

trajectory_rows = []

for filename in os.listdir(data_folder):
    if not filename.endswith(".json"):
        continue

    subreddit = filename.replace(".json", "")
    filepath = os.path.join(data_folder, filename)

    with open(filepath, "r") as f:
        months = json.load(f)

    # active users for each month
    active_by_month = [
        get_active_users(month)
        for month in months
    ]

    # all users that appeared at least once in this subreddit
    all_users = set().union(*active_by_month)

    for user in all_users:

        trajectory = tuple(
            int(user in active_by_month[t])
            for t in range(len(months))
        )

        trajectory_rows.append({
            "subreddit": subreddit,
            "user": user,
            "trajectory": trajectory,
            "trajectory_string": "".join(map(str, trajectory)),
            "active_months": sum(trajectory)
        })


trajectory_df = pd.DataFrame(trajectory_rows)

print(trajectory_df.head())
print("\nTotal user-subreddit trajectories:")
print(len(trajectory_df))

# -----------------------------------
# 3. Most common exact trajectories
# -----------------------------------

common_patterns = (
    trajectory_df["trajectory_string"]
    .value_counts()
    .reset_index()
)

common_patterns.columns = ["trajectory", "count"]

common_patterns["percentage"] = (
    common_patterns["count"] / len(trajectory_df) * 100
)

print("\nMost common trajectories:")
print(common_patterns.head(20))

# -----------------------------------
# 4. Number of active months
# -----------------------------------

active_month_distribution = (
    trajectory_df["active_months"]
    .value_counts()
    .sort_index()
)

print("\nDistribution of active months:")
print(active_month_distribution)


# -----------------------------------
# 5. Classify temporal patterns
# -----------------------------------

def classify_trajectory(traj):
    traj = list(traj)
    n = len(traj)

    active_months = sum(traj)

    # active only once
    if active_months == 1:
        return "one_month_only"

    # active every month
    if active_months == n:
        return "continuous"

    first = traj.index(1)
    last = n - 1 - traj[::-1].index(1)

    # active continuously from first appearance until the end
    if all(x == 1 for x in traj[first:]):
        if first > 0:
            return "new_entrant_then_persistent"

    # active continuously from beginning and then disappears forever
    if first == 0:
        if all(x == 1 for x in traj[:last + 1]) and last < n - 1:
            return "quitter"

    # check whether user returns after at least one absence
    seen_one = False
    seen_zero_after_one = False
    reentered = False

    for x in traj:
        if x == 1:
            if seen_zero_after_one:
                reentered = True
            seen_one = True

        elif x == 0 and seen_one:
            seen_zero_after_one = True

    if reentered:
        return "reentry_or_intermittent"

    return "other"


trajectory_df["pattern_type"] = (
    trajectory_df["trajectory"]
    .apply(classify_trajectory)
)

pattern_summary = (
    trajectory_df["pattern_type"]
    .value_counts()
    .reset_index()
)

pattern_summary.columns = ["pattern_type", "count"]

pattern_summary["percentage"] = (
    pattern_summary["count"] / len(trajectory_df) * 100
)

print("\nPattern summary:")
print(pattern_summary)

# -----------------------------------
# 6. Pattern distribution by subreddit
# -----------------------------------

subreddit_pattern_summary = (
    trajectory_df
    .groupby(["subreddit", "pattern_type"])
    .size()
    .reset_index(name="count")
)

subreddit_totals = (
    trajectory_df
    .groupby("subreddit")
    .size()
    .reset_index(name="total")
)

subreddit_pattern_summary = subreddit_pattern_summary.merge(
    subreddit_totals,
    on="subreddit"
)

subreddit_pattern_summary["percentage"] = (
    subreddit_pattern_summary["count"]
    / subreddit_pattern_summary["total"]
    * 100
)

print(
    subreddit_pattern_summary[
        subreddit_pattern_summary["subreddit"] == "politics"
    ]
    .sort_values("percentage", ascending=False)
)