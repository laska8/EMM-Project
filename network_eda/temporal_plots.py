import matplotlib.pyplot as plt


class TemporalPlots:

    @staticmethod
    def active_months_distribution(active_month_counts):

        plt.figure(figsize=(8, 5))

        values_in_millions = active_month_counts / 1_000_000

        plt.bar(
            active_month_counts.index,
            values_in_millions
        )

        plt.xlabel("Number of active months")
        plt.ylabel("Number of user–subreddit trajectories (millions)")
        plt.title("Distribution of active months")

        plt.tight_layout()
        plt.savefig(
            "active_months_distribution.png",
            dpi=300
        )

        plt.show()


    @staticmethod
    def pattern_distribution(pattern_summary):

        plot_df = pattern_summary.copy()

        label_map = {
            "one_month_only": "One month only",
            "reentry_or_intermittent": "Re-entry / intermittent",
            "other": "Other",
            "new_entrant_then_persistent": "New entrant, persistent",
            "quitter": "Quitter",
            "continuous": "Continuous"
        }

        plot_df["label"] = (
            plot_df["pattern_type"]
            .map(label_map)
        )

        plt.figure(figsize=(9, 5))

        plt.bar(
            plot_df["label"],
            plot_df["percentage"]
        )

        plt.xlabel("Temporal pattern")
        plt.ylabel("Percentage of trajectories")
        plt.title("Distribution of temporal participation patterns")

        plt.xticks(rotation=30, ha="right")

        plt.tight_layout()
        plt.savefig(
            "temporal_pattern_distribution.png",
            dpi=300
        )

        plt.show()


    @staticmethod
    def return_rate_distribution(comparison_df):

        plt.figure(figsize=(8, 5))

        plt.hist(
            comparison_df["return_next_month_pct"],
            bins=30
        )

        plt.xlabel("Next-month return rate (%)")
        plt.ylabel("Number of subreddits")
        plt.title("Distribution of next-month return rates across subreddits")

        plt.tight_layout()
        plt.savefig(
            "return_rate_distribution.png",
            dpi=300
        )

        plt.show()


    @staticmethod
    def return_window_comparison(comparison_df):

        plt.figure(figsize=(7, 6))

        x = comparison_df["return_next_month_pct"]
        y = comparison_df["return_within_3_months_pct"]

        plt.scatter(
            x,
            y,
            alpha=0.5
        )

        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())

        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--"
        )

        plt.xlabel("Next-month return rate (%)")
        plt.ylabel("Return within 3 months (%)")
        plt.title("Immediate versus broader return")

        plt.tight_layout()
        plt.savefig(
            "return_window_comparison.png",
            dpi=300
        )

        plt.show()

    @staticmethod
    def return_window_gain(comparison_df):

        plt.figure(figsize=(8, 5))

        plt.hist(
            comparison_df["extra_return_3m"],
            bins=30
        )

        plt.xlabel(
            "Increase from next-month to 3-month return (percentage points)"
        )
        plt.ylabel("Number of subreddits")

        plt.title(
            "Effect of using a broader return window"
        )

        plt.tight_layout()

        plt.savefig(
            "return_window_gain.png",
            dpi=300
        )

        plt.show()