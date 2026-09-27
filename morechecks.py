import json
import os
import pandas as pd

data_folder = "reply_networks"

results = []

for filename in os.listdir(data_folder):
    if not filename.endswith(".json"):
        continue

    subreddit = filename.replace(".json", "")
    filepath = os.path.join(data_folder, filename)

    with open(filepath, "r") as f:
        months = json.load(f)

    # Need at least 2 months to define next-month return
    if len(months) < 2:
        continue

    total_observations = 0
    total_returns = 0
    transition_rates = []

    for t in range(len(months) - 1):
        users_t = set(months[t].keys())
        users_next = set(months[t + 1].keys())

        if len(users_t) == 0:
            continue

        returned = len(users_t & users_next)
        return_rate = returned / len(users_t)

        total_observations += len(users_t)
        total_returns += returned
        transition_rates.append(return_rate)

    if total_observations == 0:
        continue

    results.append({
        "subreddit": subreddit,
        "n_months": len(months),
        "n_transitions": len(transition_rates),
        "n_user_month_observations": total_observations,
        "n_returns": total_returns,
        "overall_return_rate": total_returns / total_observations,
        "min_monthly_return_rate": min(transition_rates),
        "max_monthly_return_rate": max(transition_rates)
    })


feasibility_df = pd.DataFrame(results)

print(feasibility_df.head())

print("\nNumber of subreddits:")
print(len(feasibility_df))

print("\nSummary:")
print(
    feasibility_df[
        [
            "n_user_month_observations",
            "overall_return_rate",
            "min_monthly_return_rate",
            "max_monthly_return_rate"
        ]
    ].describe()
)

print("\nSubreddits with at least 1000 user-month observations:")
print(
    (feasibility_df["n_user_month_observations"] >= 1000).sum()
)

print("\nSubreddits with at least 5000 user-month observations:")
print(
    (feasibility_df["n_user_month_observations"] >= 5000).sum()
)


print(
    feasibility_df[
        ["subreddit", "n_user_month_observations", "overall_return_rate"]
    ]
    .sort_values("overall_return_rate")
    .head(10)
)

print(
    feasibility_df[
        ["subreddit", "n_user_month_observations", "overall_return_rate"]
    ]
    .sort_values("overall_return_rate", ascending=False)
    .head(10)
)