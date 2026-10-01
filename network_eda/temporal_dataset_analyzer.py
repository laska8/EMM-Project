class TemporalDatasetAnalyzer:

    def __init__(self, comparison_df):
        self.df = comparison_df


    def get_summary_statistics(self):

        cols = [
            "unique_users",
            "one_month_only_pct",
            "reentry_pct",
            "continuous_pct",
            "return_next_month_pct",
            "return_within_2_months_pct",
            "return_within_3_months_pct"
        ]

        return self.df[cols].describe()


    def add_return_window_differences(self):

        self.df["extra_return_2m"] = (
            self.df["return_within_2_months_pct"]
            - self.df["return_next_month_pct"]
        )

        self.df["extra_return_3m"] = (
            self.df["return_within_3_months_pct"]
            - self.df["return_next_month_pct"]
        )

        return self.df


    def filter_by_size(self, min_users):

        return self.df[
            self.df["unique_users"] >= min_users
        ].copy()