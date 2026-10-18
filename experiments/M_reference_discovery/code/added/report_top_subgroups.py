"""Report the saved beam-search results without refitting any models.

Run from the project directory:
    python experiments/M_reference_discovery/code/added/report_top_subgroups.py
"""

from pathlib import Path
import sys
import re

# Import the shared paths from this project's experiments directory.
EXPERIMENTS_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "experiments")
sys.path.insert(0, str(EXPERIMENTS_DIR / "code"))
from experiment_paths import ORIGINAL_SEARCH, M, A3, output_path

import pandas as pd


BASE_DIR = ORIGINAL_SEARCH
SOURCE = BASE_DIR / "emm_beam_search_results.csv"
OUTPUT_DIR = M

TOP_K = 10  # Reporting cutoff, independent of the search's beam width.
TOTAL_OBSERVATIONS = 36_031_393  # Full-data run; see pooled_logistic.ipynb.


def readable_description(description):
    description = description.replace("replies_sent", "Sent")
    description = description.replace("replies_received", "Received")
    description = description.replace(" AND ", " and ")
    description = re.sub(r"(<=|>|==)", r" \1 ", description)
    return re.sub(r"(\d+)\.0\b", r"\1", description)


def main():
    results = pd.read_csv(SOURCE)
    if len(results) < TOP_K:
        raise ValueError(f"Requested {TOP_K} subgroups, but found {len(results)}.")
    if results["reference_beta"].nunique() != 1:
        raise ValueError("Expected one common reference coefficient.")

    # Rank before rounding. Stable sorting preserves source order for ties.
    top = results.sort_values("quality", ascending=False, kind="stable").head(TOP_K).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    top["coverage_pct"] = 100 * top["n"] / TOTAL_OBSERVATIONS
    top["return_rate_pct"] = 100 * top["return_rate"]
    top["signed_delta"] = top["beta"] - top["reference_beta"]
    # Preserve the original delta column, which is an ABSOLUTE difference.

    ame_path = output_path(A3, "top_10_ame_comparison.csv")
    has_ames = ame_path.exists()
    if has_ames:
        ames = pd.read_csv(ame_path).set_index("description")
        matched = ames.reindex(top["description"])
        if matched["ame_pp"].isna().any():
            raise ValueError("AME results do not cover the selected subgroups. Rerun run_ame_analysis.py.")
        for column in ("n", "beta", "reference_beta", "quality"):
            if not (abs(matched[column].to_numpy() - top[column].to_numpy()) < 1e-10).all():
                raise ValueError("AME results do not match this search. Rerun run_ame_analysis.py.")
        for column in ("ame_pp", "reference_ame_pp", "delta_ame_pp"):
            top[column] = matched[column].to_numpy()

    # Rounded strings belong only in the presentation table.
    table = pd.DataFrame({
        "Rank": top["rank"],
        "Subgroup": top["description"].map(readable_description),
        "n": top["n"].map(lambda n: f"{n:,}"),
        "Coverage (%)": top["coverage_pct"].map(lambda x: f"{x:.1f}"),
        "Return (%)": top["return_rate_pct"].map(lambda x: f"{x:.1f}"),
        "Beta": top["beta"].map(lambda x: f"{x:.3f}"),
        "95% CI": [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(top["ci_low"], top["ci_high"])],
        "Signed delta": top["signed_delta"].map(lambda x: f"{x:+.3f}"),
        "Quality": top["quality"].map(lambda x: f"{x:.1f}"),
    })
    if has_ames:
        table["AME (pp)"] = top["ame_pp"].map(lambda x: f"{x:+.2f}")

    reference = float(top["reference_beta"].iloc[0])
    notes = (
        f"Top {TOP_K} of {len(results)} valid subgroup fits, ranked by quality.\n"
        f"Reference beta: {reference:.4f}; full sample: {TOTAL_OBSERVATIONS:,} observations.\n"
        "n counts user-subreddit-month observations, not distinct users.\n"
        "Sent and Received are reply counts during the current month.\n"
        "Return is the observed next-month return rate. Coverage uses the full sample.\n"
        "Beta is the adjusted binary-reciprocity coefficient on the log-odds scale.\n"
        "95% CIs use user-clustered standard errors.\n"
        "Signed delta = subgroup beta - reference beta.\n"
        "Quality = sqrt(n) * abs(signed delta). Subgroups overlap.\n"
        "Source: beam_search/emm_beam_search_results.csv."
    )
    if has_ames:
        notes += (
            f"\nReference AME: {top['reference_ame_pp'].iloc[0]:.4f} percentage points."
            "\nAME is the average discrete change from reciprocity 0 to 1 within each fitted population."
            "\nAMEs are point estimates; see experiments/A3_probability_scale/reports/a3_report.txt for scale comparisons."
        )
    else:
        notes += " AMEs are not yet included."

    OUTPUT_DIR.mkdir(exist_ok=True)
    top.to_csv(output_path(OUTPUT_DIR, f"top_{TOP_K}_subgroups.csv"), index=False)
    report = table.to_string(index=False) + "\n\n" + notes + "\n"
    output_path(OUTPUT_DIR, f"top_{TOP_K}_subgroups.txt").write_text(report, encoding="utf-8")

    # This table spans both columns in a two-column paper (requires booktabs).
    latex = (
        "\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n"
        f"\\caption{{Top {TOP_K} activity-defined subgroups ranked by quality.}}\n"
        "\\label{tab:top-subgroups}\n"
        + table.to_latex(index=False, escape=True, column_format="rlrrrrlrr" + ("r" if has_ames else ""))
        + "\\begin{minipage}{\\textwidth}\n\\footnotesize\n"
        + f"Reference coefficient: {reference:.4f}. "
        + r"$n$ counts user--subreddit--month observations. Coverage and return are percentages. "
        + r"Beta and confidence intervals are on the log-odds scale; standard errors are clustered by user. "
        + r"Signed delta is $\beta_{1,\phi}-\beta_{1,\mathrm{ref}}$; "
        + r"quality is $\sqrt{n}\,|\Delta|$. Subgroups overlap."
        + (f" Reference AME: {top['reference_ame_pp'].iloc[0]:.2f} percentage points. "
           "AMEs are average binary discrete changes within each fitted population; AME uncertainty is not estimated."
           if has_ames else "")
        + "\n\\end{minipage}\n\\end{table*}\n"
    )
    output_path(OUTPUT_DIR, f"top_{TOP_K}_subgroups.tex").write_text(latex, encoding="utf-8")
    print(report)
    print(f"Saved CSV, text, and LaTeX reports in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
