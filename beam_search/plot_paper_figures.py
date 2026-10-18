from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from subgroup_labels import readable

BASE_DIR = Path(__file__).resolve().parent
EXPERIMENTS = BASE_DIR.parent / "experiments"

BREADTH = BASE_DIR / "emm_breadth_check_results.csv"
AME = EXPERIMENTS / "A3_probability_scale" / "tables" / "top_10_ame_comparison.csv"

COLOR = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"


def style(ax):

    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)

    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)


def save(fig, name):

    fig.tight_layout()

    for extension in ["pdf", "png"]:
        fig.savefig(BASE_DIR / f"{name}.{extension}", dpi=300, bbox_inches="tight", pad_inches=0.03)

    plt.close(fig)


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 7.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": INK,
})


# A4: coefficient with activity controls and with distinct partners held fixed

breadth = pd.read_csv(BREADTH)

labels = ["all observations"] + [
    f"{rank}  {readable(d)}"
    for rank, d in enumerate(breadth["description"].iloc[1:], start=1)
]

# all observations at the top, then quality rank 1 to 10
y = list(range(len(breadth) - 1, -1, -1))

fig, ax = plt.subplots(figsize=(3.4, 3.4))

ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)

ax.hlines(
    y,
    breadth["beta"],
    breadth["beta_contacts"],
    color=GRID,
    linewidth=1.5,
    zorder=2
)

ax.scatter(
    breadth["beta"],
    y,
    s=26,
    color=INK,
    edgecolor="white",
    linewidth=0.8,
    zorder=3,
    label="sent and received held fixed (main model)"
)

ax.scatter(
    breadth["beta_contacts"],
    y,
    s=48,
    color=COLOR,
    edgecolor="white",
    linewidth=0.8,
    zorder=3,
    label="distinct partners held fixed"
)

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.get_yticklabels()[0].set_fontweight("bold")

ax.set_xlabel("Reciprocity coefficient (log odds)")
ax.set_ylim(-0.7, len(breadth) - 0.3)

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.3, 1.0),
    ncol=1,
    frameon=False,
    fontsize=6.5,
    handletextpad=0.2,
    borderaxespad=0
)

style(ax)
save(fig, "fig_breadth_check")


# A3: coefficient against AME for the top ten

ame = pd.read_csv(AME)

fig, ax = plt.subplots(figsize=(3.4, 2.6))

ax.scatter(
    ame["beta"],
    ame["ame_pp"],
    s=26,
    color=COLOR,
    edgecolor="white",
    linewidth=0.8,
    zorder=3
)

# label offsets in points; rank 3 sits next to rank 8
offsets = {3: (-9, -8)}

for row in ame.itertuples():
    ax.annotate(
        str(row.quality_rank),
        (row.beta, row.ame_pp),
        xytext=offsets.get(row.quality_rank, (4, 2)),
        textcoords="offset points",
        fontsize=6.5,
        color=INK
    )

ax.set_xlabel("Reciprocity coefficient (log odds)")
ax.set_ylabel("AME (percentage points)")

ax.margins(0.08)
ax.grid(color=GRID, linewidth=0.6)
ax.set_axisbelow(True)

for side in ["top", "right"]:
    ax.spines[side].set_visible(False)

reference = ame.iloc[0]
ax.text(
    0.98, 0.97,
    f"reference: β = {reference['reference_beta']:.3f}, "
    f"AME = {reference['reference_ame_pp']:.2f} pp".replace("-", "−"),
    transform=ax.transAxes,
    ha="right",
    va="top",
    color=MUTED,
    fontsize=6.5
)

save(fig, "fig_coefficient_vs_ame")
