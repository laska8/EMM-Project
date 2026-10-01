import json
import pandas as pd


class TemporalAnalyzer:

    def __init__(self, file_path, subreddit_name):
        self.file_path = file_path
        self.subreddit_name = subreddit_name

        with open(file_path, "r") as f:
            self.months = json.load(f)

        self.active_by_month = [
            set(month.keys())
            for month in self.months
        ]

        self.trajectory_df = None
        self.return_df = None


    def build_trajectories(self):

        all_users = set().union(*self.active_by_month)

        rows = []

        for user in all_users:

            trajectory = tuple(
                int(user in self.active_by_month[t])
                for t in range(len(self.months))
            )

            rows.append({
                "user": user,
                "trajectory": trajectory,
                "trajectory_string": "".join(map(str, trajectory)),
                "active_months": sum(trajectory)
            })

        self.trajectory_df = pd.DataFrame(rows)

        return self.trajectory_df


    @staticmethod
    def classify_trajectory(traj):

        traj = list(traj)
        n = len(traj)

        active_months = sum(traj)

        if active_months == 1:
            return "one_month_only"

        if active_months == n:
            return "continuous"

        first = traj.index(1)
        last = n - 1 - traj[::-1].index(1)

        if first > 0 and all(
            x == 1 for x in traj[first:]
        ):
            return "new_entrant_then_persistent"

        if (
            first == 0
            and all(x == 1 for x in traj[:last + 1])
            and last < n - 1
        ):
            return "quitter"

        seen_active = False
        seen_absence = False

        for x in traj:

            if x == 1:

                if seen_absence:
                    return "reentry_or_intermittent"

                seen_active = True

            elif x == 0 and seen_active:

                seen_absence = True

        return "other"

    def classify_trajectories(self):

        if self.trajectory_df is None:
            self.build_trajectories()

        self.trajectory_df["pattern_type"] = (
            self.trajectory_df["trajectory"]
            .apply(self.classify_trajectory)
        )

        return self.trajectory_df

    def calculate_returns(self):

        rows = []

        # need t+1, t+2 and t+3
        for t in range(len(self.active_by_month) - 3):

            for user in self.active_by_month[t]:

                return_1 = int(
                    user in self.active_by_month[t + 1]
                )

                return_2 = int(
                    user in self.active_by_month[t + 1]
                    or user in self.active_by_month[t + 2]
                )

                return_3 = int(
                    user in self.active_by_month[t + 1]
                    or user in self.active_by_month[t + 2]
                    or user in self.active_by_month[t + 3]
                )

                rows.append({
                    "user": user,
                    "month": t + 1,
                    "return_1": return_1,
                    "return_2": return_2,
                    "return_3": return_3
                })

        self.return_df = pd.DataFrame(rows)

        return self.return_df

    def get_summary(self):

        if self.trajectory_df is None:
            self.build_trajectories()

        if "pattern_type" not in self.trajectory_df.columns:
            self.classify_trajectories()

        if self.return_df is None:
            self.calculate_returns()

        pattern_pct = (
            self.trajectory_df["pattern_type"]
            .value_counts(normalize=True)
            * 100
        )

        summary = {
            "subreddit": self.subreddit_name,

            "unique_users": len(self.trajectory_df),

            "one_month_only_pct":
                pattern_pct.get(
                    "one_month_only", 0
                ),

            "reentry_pct":
                pattern_pct.get(
                    "reentry_or_intermittent", 0
                ),

            "continuous_pct":
                pattern_pct.get(
                    "continuous", 0
                ),

            "return_next_month_pct":
                self.return_df["return_1"].mean() * 100,

            "return_within_2_months_pct":
                self.return_df["return_2"].mean() * 100,

            "return_within_3_months_pct":
                self.return_df["return_3"].mean() * 100
        }

        return summary