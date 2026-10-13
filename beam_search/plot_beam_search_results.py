import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

RESULTS = BASE_DIR / "emm_beam_search_results.csv"

TOP_K = 10

COLOR = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"


def readable(description):

    bounds = {}

    for part in description.split(" AND "):
        column, operator, value = re.match(
            r"replies_(\w+?)(<=|>)(.+)", part
        ).groups()
        bounds.setdefault(column, {})[operator] = f"{float(value):g}"

    conditions = []

    # always "sent" before "received"
    for column in ["sent", "received"]:
        bound = bounds.get(column, {})

        # a lower and an upper bound on one attribute form an interval
        if ">" in bound and "<=" in bound:
            conditions.append(f"{bound['>']} < {column} ≤ {bound['<=']}")
        elif ">" in bound:
            conditions.append(f"{column} > {bound['>']}")
        elif "<=" in bound:
            conditions.append(f"{column} ≤ {bound['<=']}")

    return ", ".join(conditions)


# -----------------------------------
# Forest plot of the top-k subgroups by quality
# -----------------------------------

results = pd.read_csv(RESULTS)

top = results.nlargest(TOP_K, "quality").reset_index(drop=True)
reference = results["reference_beta"].iloc[0]

# rank 1 at the top
y = range(len(top) - 1, -1, -1)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 7.5,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": INK,
})

fig, ax = plt.subplots(figsize=(3.4, 3.2))

ax.axvline(
    reference,
    color=MUTED,
    linestyle="--",
    linewidth=1,
    zorder=1
)

ax.text(
    reference,
    len(top) - 0.35,
    f"reference\n{reference:.3f}".replace("-", "−"),
    color=MUTED,
    ha="center",
    va="bottom",
    fontsize=6.5
)

ax.hlines(
    list(y),
    top["ci_low"],
    top["ci_high"],
    color=COLOR,
    linewidth=2,
    zorder=2
)

ax.scatter(
    top["beta"],
    list(y),
    s=24,
    color=COLOR,
    edgecolor="white",
    linewidth=0.8,
    zorder=3
)

ax.set_yticks(list(y))
ax.set_yticklabels([readable(d) for d in top["description"]])

# subgroup size next to each row
ax_n = ax.secondary_yaxis("right")
ax_n.set_yticks(list(y))
ax_n.set_yticklabels(
    [f"{n / 1e6:.1f}M" for n in top["n"]],
    color=MUTED
)
ax_n.tick_params(length=0)
ax_n.spines["right"].set_visible(False)

ax.set_xlabel("Reciprocity coefficient $\\beta_{1,\\phi}$ (95% CI)")
ax.set_ylim(-0.7, len(top) + 0.6)
ax.set_xlim(-0.6, -0.2)

ax.grid(axis="x", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)

for side in ["top", "right", "left"]:
    ax.spines[side].set_visible(False)

fig.tight_layout()

for extension in ["pdf", "png"]:
    fig.savefig(
        BASE_DIR / f"fig_beam_search_forest.{extension}",
        dpi=300
    )

plt.show()
