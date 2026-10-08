from pathlib import Path
import pandas as pd

from reply_network_analyzer import ReplyNetworkAnalyzer
from eda_summaries import EdaSummaries

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "reply_networks"
OUTPUT_DIR = BASE_DIR / "eda_outputs"


class EdaDatasets:

    KEYS = ["subreddit", "month", "user"]

    # datasets built directly from the subreddit JSON files:
    # name -> (ReplyNetworkAnalyzer method, columns)
    RAW_DATASETS = {
        "user_activity": (
            "user_activity_rows",
            ReplyNetworkAnalyzer.ACTIVITY_COLUMNS
        ),
        "user_return": (
            "user_return_rows",
            ReplyNetworkAnalyzer.RETURN_COLUMNS
        ),
        "community": (
            "community_rows",
            ReplyNetworkAnalyzer.COMMUNITY_COLUMNS
        ),
        "user_reciprocity": (
            "reciprocity_rows",
            ReplyNetworkAnalyzer.RECIPROCITY_COLUMNS
        )
    }

    # datasets with one row per user-month (or subreddit-month):
    # written subreddit by subreddit, so they never have to fit
    # in memory while they are built
    STREAMED_DATASETS = [
        "user_activity",
        "user_return",
        "community",
        "user_reciprocity",
        "user_month",
        "user_activity_reciprocity",
        "reciprocity_return"
    ]

    # compact column types, the big datasets have ~40 million rows
    DTYPES = {
        "subreddit": "category",
        "user": "string[pyarrow]"
    }

    def __init__(
        self,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        rebuild=False
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.rebuild = rebuild

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.frames = {}
        self.built = set()


    def path(self, name):

        return self.output_dir / f"{name}.csv"


    def read(self, name):

        # round_trip reads floats back exactly as they were saved,
        # so a loaded dataset is identical to a freshly built one
        df = pd.read_csv(
            self.path(name),
            dtype=self.DTYPES,
            float_precision="round_trip"
        )

        # keep groupby("subreddit") in alphabetical order,
        # as it is for plain string columns
        if "subreddit" in df.columns:
            df["subreddit"] = df["subreddit"].cat.reorder_categories(
                sorted(df["subreddit"].cat.categories)
            )

        return df


    def needs_build(self, name):

        if name in self.built:
            return False

        return self.rebuild or not self.path(name).exists()


    def release(self, *names):

        # drop loaded datasets from memory (they stay on disk)
        for name in names:
            self.frames.pop(name, None)


    # -----------------------------------
    # Datasets built from the JSON files
    # -----------------------------------

    def subreddit_frames(self, analyzer):

        frames = {
            name: pd.DataFrame(
                getattr(analyzer, method)(),
                columns=columns
            )
            for name, (method, columns) in self.RAW_DATASETS.items()
        }

        # the merge keys include the subreddit, so merging one
        # subreddit at a time gives the same rows as merging everything
        frames["user_month"] = frames["user_activity"].merge(
            frames["user_return"],
            on=self.KEYS,
            how="inner"
        )

        frames["user_activity_reciprocity"] = frames["user_activity"].merge(
            frames["user_reciprocity"],
            on=self.KEYS,
            how="left"
        )

        frames["reciprocity_return"] = (
            frames["user_activity_reciprocity"]
            .merge(
                frames["user_return"],
                on=self.KEYS,
                how="inner"
            )
        )

        return frames


    def build_streamed_datasets(self, names=None):

        if names is None:
            names = [
                name
                for name in self.STREAMED_DATASETS
                if self.needs_build(name)
            ]

        if not names:
            return

        part_paths = {
            name: self.path(name).with_suffix(".csv.part")
            for name in names
        }

        file_paths = list(self.data_dir.glob("*.json"))

        print("building:", ", ".join(names))
        print("number of subreddit files:", len(file_paths))

        for file_index, file_path in enumerate(file_paths, start=1):

            analyzer = ReplyNetworkAnalyzer(
                file_path=file_path,
                subreddit_name=file_path.stem
            )

            frames = self.subreddit_frames(analyzer)

            for name in names:

                frames[name].to_csv(
                    part_paths[name],
                    mode="w" if file_index == 1 else "a",
                    header=file_index == 1,
                    index=False
                )

            if file_index % 100 == 0 or file_index == len(file_paths):
                print(
                    f"[{file_index}/{len(file_paths)}] "
                    f"subreddits processed",
                    flush=True
                )

        for name in names:
            part_paths[name].replace(self.path(name))
            self.built.add(name)

            print("saved dataset to:", self.path(name))


    def streamed_dataset(self, name):

        if name not in self.frames:

            if self.needs_build(name):
                self.build_streamed_datasets()

            self.frames[name] = self.read(name)

        return self.frames[name]


    def user_activity(self):

        return self.streamed_dataset("user_activity")


    def user_return(self):

        return self.streamed_dataset("user_return")


    def community(self):

        return self.streamed_dataset("community")


    def user_reciprocity(self):

        return self.streamed_dataset("user_reciprocity")


    def user_month(self):

        return self.streamed_dataset("user_month")


    def user_activity_reciprocity(self):

        return self.streamed_dataset("user_activity_reciprocity")


    def reciprocity_return(self):

        return self.streamed_dataset("reciprocity_return")


    # -----------------------------------
    # Summary datasets derived from the ones above
    # -----------------------------------

    def derived_dataset(self, name, build):

        if name not in self.frames:

            if self.needs_build(name):

                df = build()
                df.to_csv(self.path(name), index=False)
                self.built.add(name)

                print("saved dataset to:", self.path(name))

            else:
                df = self.read(name)

            self.frames[name] = df

        return self.frames[name]


    def activity_return(self):

        def build():

            user_month_df = self.user_month()

            activity_group = (
                user_month_df["replies_sent"]
                .apply(EdaSummaries.classify_activity)
                .rename("activity_group")
            )

            return EdaSummaries.return_rate_by(
                user_month_df,
                activity_group,
                count_column="users"
            )

        return self.derived_dataset("activity_return", build)


    def return_by_month(self):

        return self.derived_dataset(
            "return_by_month",
            lambda: EdaSummaries.return_rate_by(
                self.user_month(),
                "month"
            )
        )


    def community_return(self):

        def build():

            community_return_df = (
                self.user_month()
                .groupby(["subreddit", "month"], observed=True)
                .agg(
                    users=("user", "count"),
                    return_rate=("returned_next_month", "mean")
                )
                .reset_index()
            )

            return community_return_df.merge(
                self.community(),
                on=["subreddit", "month"],
                how="left"
            )

        return self.derived_dataset("community_return", build)


    def monthly_summary(self):

        return self.derived_dataset(
            "monthly_summary",
            lambda: (
                self.community()
                .groupby("month")
                .agg(
                    mean_active_users=("active_users", "mean"),
                    median_active_users=("active_users", "median"),
                    mean_edges=("edges", "mean"),
                    mean_density=("density", "mean"),
                    mean_out_degree=("average_out_degree", "mean"),
                    mean_in_degree=("average_in_degree", "mean")
                )
                .reset_index()
            )
        )


    def subreddit_summary(self):

        def build():

            subreddit_activity_summary = (
                self.user_activity()
                .groupby("subreddit", observed=True)
                .agg(
                    observations=("user", "count"),
                    mean_replies_sent=("replies_sent", "mean"),
                    median_replies_sent=("replies_sent", "median"),
                    mean_replies_received=("replies_received", "mean"),
                    median_replies_received=("replies_received", "median"),
                    mean_total_interactions=("total_interactions", "mean")
                )
                .reset_index()
            )

            subreddit_network_summary = (
                self.community()
                .groupby("subreddit", observed=True)
                .agg(
                    mean_active_users=("active_users", "mean"),
                    mean_edges=("edges", "mean"),
                    mean_density=("density", "mean"),
                    mean_out_degree=("average_out_degree", "mean")
                )
                .reset_index()
            )

            subreddit_return_summary = (
                self.user_month()
                .groupby("subreddit", observed=True)
                .agg(
                    return_rate=("returned_next_month", "mean")
                )
                .reset_index()
            )

            return (
                subreddit_activity_summary
                .merge(
                    subreddit_network_summary,
                    on="subreddit",
                    how="left"
                )
                .merge(
                    subreddit_return_summary,
                    on="subreddit",
                    how="left"
                )
            )

        return self.derived_dataset("subreddit_summary", build)
