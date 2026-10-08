import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from eda_datasets import OUTPUT_DIR
from eda_summaries import EdaSummaries

FIGURE_DIR = OUTPUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


class EdaPlots:

    @staticmethod
    def save(name):

        plt.tight_layout()

        # change .png to .pdf if you want vector figures for LaTeX
        plt.savefig(
            FIGURE_DIR / (name + ".png"),
            dpi=200
        )

        plt.show()


    # -----------------------------------
    # General plots
    # -----------------------------------

    @staticmethod
    def histogram(
        values,
        xlabel,
        ylabel,
        title,
        bins=50,
        plain_axis=None,
        thousands_axes=""
    ):

        plt.figure(figsize=(8, 5))

        plt.hist(
            values,
            bins=bins
        )

        # disable scientific notation
        if plain_axis is not None:
            plt.ticklabel_format(style="plain", axis=plain_axis)

        # thousands separators (commas)
        if "y" in thousands_axes:
            plt.gca().yaxis.set_major_formatter(
                ticker.StrMethodFormatter("{x:,.0f}")
            )

        if "x" in thousands_axes:
            plt.gca().xaxis.set_major_formatter(
                ticker.StrMethodFormatter("{x:,.0f}")
            )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        plt.show()


    @staticmethod
    def line(x, y, xlabel, ylabel, title):

        plt.figure(figsize=(8, 5))

        plt.plot(
            x,
            y,
            marker="o"
        )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        plt.show()


    @staticmethod
    def bar(
        labels,
        values,
        xlabel,
        ylabel,
        title,
        figsize=(8, 5),
        rotate_labels=False
    ):

        plt.figure(figsize=figsize)

        plt.bar(
            labels,
            values
        )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        if rotate_labels:
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

        plt.show()


    @staticmethod
    def scatter(x, y, xlabel, ylabel, title, alpha=0.3):

        plt.figure(figsize=(8, 5))

        plt.scatter(
            x,
            y,
            alpha=alpha
        )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        plt.show()


    @staticmethod
    def correlation_heatmap(correlation_matrix, title):

        plt.figure(figsize=(8, 6))

        plt.imshow(
            correlation_matrix,
            aspect="auto"
        )

        plt.colorbar(label="Correlation")

        plt.xticks(
            range(len(correlation_matrix.columns)),
            correlation_matrix.columns,
            rotation=45,
            ha="right"
        )

        plt.yticks(
            range(len(correlation_matrix.index)),
            correlation_matrix.index
        )

        plt.title(title)
        plt.tight_layout()

        plt.show()


    # -----------------------------------
    # Paper figures (saved to FIGURE_DIR)
    # -----------------------------------

    @staticmethod
    def reciprocity_return(pooled, by_both, group_order):

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        # left: everyone pooled together
        axes[0].bar(range(len(pooled)), pooled.values * 100)
        axes[0].set_xticks(range(len(pooled)))
        axes[0].set_xticklabels(pooled.index)
        axes[0].set_title("Pooled (mixes very different activity levels)")
        axes[0].set_xlabel("Proportional reciprocity")
        axes[0].set_ylabel("Next-month return rate (%)")

        # right: one line per activity level
        for activity in EdaSummaries.ACTIVITY_LEVELS:

            # skip tiny groups
            part = by_both[
                (by_both["activity_group"] == activity)
                & (by_both["n"] >= 1000)
            ]

            axes[1].plot(
                [group_order.index(g) for g in part["reciprocity_group"]],
                part["return_rate"] * 100,
                marker="o",
                label=activity + " replies"
            )

        axes[1].set_xticks(range(len(group_order)))
        axes[1].set_xticklabels(group_order)
        axes[1].set_title("Within each activity level")
        axes[1].set_xlabel("Proportional reciprocity")
        axes[1].legend(frameon=False)

        EdaPlots.save("fig_reciprocity_return")


    @staticmethod
    def reciprocity_composition(plot_matrix):

        fig, ax = plt.subplots(figsize=(8, 5))

        x_pos = np.arange(len(plot_matrix.index))
        bottoms = np.zeros(len(plot_matrix.index))

        for activity in EdaSummaries.ACTIVITY_LEVELS:

            values = plot_matrix[activity].values

            bars = ax.bar(
                x_pos,
                values,
                bottom=bottoms,
                label=f"{activity} reply" if activity == "1" else f"{activity} replies",
                width=0.6,
            )

            # percentage labels inside bars if the bar section is large enough (>= 10%)
            for i, (bar, val) in enumerate(zip(bars, values)):

                if val >= 10:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottoms[i] + val / 2,
                        f"{int(round(val))}%",
                        ha="center",
                        va="center",
                        color="white" if activity in ["1", "11+"] else "black",
                        weight="bold",
                        fontsize=9,
                    )

            bottoms += values

        ax.set_title(
            "Who is inside each reciprocity group?",
            weight="bold",
            pad=15,
            fontsize=12
        )
        ax.set_xlabel("Proportional reciprocity group", labelpad=8)
        ax.set_ylabel("Share of the group's rows (%)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_matrix.index)
        ax.set_ylim(0, 105)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(
            title="Replies sent",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
        )

        # save() calls tight_layout again; with the legend outside the
        # axes the second pass still shifts the layout slightly
        plt.tight_layout()

        EdaPlots.save("fig_reciprocity_composition")


    @staticmethod
    def subreddit_spread(subs):

        fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))

        panels = [
            ("mean_reciprocity", "Mean reciprocity"),
            ("return_rate", "Next-month return rate"),
            ("mean_replies_sent", "Mean replies sent"),
            ("adjusted_gap", "Return gap: reciprocated minus not")
        ]

        for ax, (column, label) in zip(axes, panels):
            ax.hist(subs[column].dropna(), bins=40)
            ax.set_xlabel(label)
            ax.set_ylabel("Number of subreddits")

        # gap = 0 means no difference
        axes[3].axvline(0, color="black", linewidth=1)

        EdaPlots.save("fig_subreddit_spread")
