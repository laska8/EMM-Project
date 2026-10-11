"""Rebuild experiment reports from canonical files without copying or fitting models.

Run with the project's analysis Python. Then run render_report.py with a Python
environment providing reportlab. All writes are confined to experiments/.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
MANIFEST = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path = BASE / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def save_json(path, value):
    write(path, json.dumps(value, indent=2, allow_nan=False))


def inventory():
    plan = json.loads((BASE / "reproducibility/consolidation_plan.json").read_text())
    for item in plan["files"]:
        path = Path(item["keep"])
        if not path.exists():
            raise FileNotFoundError(f"Missing canonical file: {path}")
        MANIFEST.append({"canonical": path.relative_to(ROOT).as_posix(),
                         "sha256": digest(path), "bytes": path.stat().st_size,
                         "role": item["role"].replace("snapshot", "reference"),
                         "previous_source": item["source"], "previous_archive": item["archive"]})
    for path in sorted((BASE / "code").glob("*.py")):
        MANIFEST.append({"canonical": path.relative_to(ROOT).as_posix(), "sha256": digest(path),
                         "bytes": path.stat().st_size, "role": "shared experiment/report tooling"})


def describe(text):
    return text.replace("replies_sent", "Sent").replace("replies_received", "Received").replace(".0", "").replace(" AND ", " and ")


def paragraph(text):
    return {"type": "paragraph", "text": text}


def small(text):
    return {"type": "small", "text": text}


def heading(text):
    return {"type": "heading", "text": text}


def table(rows, widths):
    assert sum(widths) <= 511
    return {"type": "table", "rows": rows, "widths": widths}


def generate_report():
    m = pd.read_csv(BASE / "M_reference_discovery/tables/top_10_subgroups.csv")
    all_m = pd.read_csv(BASE / "../beam_search/emm_beam_search_results.csv")
    a1 = json.loads((BASE / "A1_reciprocity_measure/reports/summary.json").read_text())
    a2 = pd.read_csv(BASE / "A2_user_dependence/tables/a2_comparison.csv")
    a3 = pd.read_csv(BASE / "A3_probability_scale/tables/top_10_ame_comparison.csv")
    s3 = json.loads((BASE / "A3_probability_scale/reports/summary.json").read_text())
    union = pd.read_csv(BASE / "A1_reciprocity_measure/tables/top_10_comparison.csv")
    weighted = json.loads((BASE / "A1_reciprocity_measure/audits/weighted_feasibility.json").read_text())
    temporal = pd.read_csv(BASE / "../network_eda/temporal_eda_summary_all_subreddits.csv")
    communities = pd.read_csv(BASE / "../logistic_resgression/results/subreddit_logistic_results.csv")
    ref = a2.iloc[0]
    # Recover the raw 2x2 counts directly from the saved notebook output.
    notebook = json.loads((BASE / "../logistic_resgression/pooled_logistic.ipynb").read_text(encoding="utf-8"))
    saved_text = "\n".join("".join(o.get("data", {}).get("text/plain", [])) for c in notebook["cells"] for o in c.get("outputs", []))
    for count in (12447185, 4758717, 10473095, 8352396):
        assert str(count) in saved_text
    raw_no = 4758717 / (12447185 + 4758717) * 100
    raw_yes = 8352396 / (10473095 + 8352396) * 100
    assert m.description.tolist() == a3.description.tolist() == a2.iloc[1:].description.tolist()
    assert len(all_m) == 29 and len(m) == 10 and len(communities) == 2046
    assert a1["shared_top_k_count"] == 9 and weighted["repeated_target_entries"] == 0
    assert all_m.depth.value_counts().to_dict() == {2: 20, 1: 9}
    assert communities.status.eq("ok").all()
    assert int((communities.coefficient < 0).sum()) == 2021
    assert int((communities.ci_high < 0).sum()) == 1782
    assert int((communities.ci_low > 0).sum()) == 3
    assert a1["all_shared_difference_direction_agreement_count"] == 21
    assert a1["shared_top_k_difference_direction_agreement_count"] == 9
    assert s3["difference_direction_agreement_count"] == 10
    assert (a2.clustered_ci_high < 0).all() and (a2.ordinary_ci_high < 0).all()
    np.testing.assert_allclose(m.ame_pp, a3.ame_pp)
    np.testing.assert_allclose(m.beta, a2.iloc[1:].beta, rtol=0, atol=1e-7)
    negative = int((all_m.beta < 0).sum())
    ci_excludes = int(((all_m.ci_high < ref.beta) | (all_m.ci_low > ref.beta)).sum())
    rank1 = m.iloc[0]
    strongest_ame = a3.loc[a3.ame_pp.idxmin()]
    smallest_beta = all_m.loc[all_m.beta.idxmax()]
    statuses = [
        ["Analysis", "Status", "Completed scope"],
        ["Supporting analyses", "Complete", "Temporal and reciprocity EDA; pooled model; 2,046 community models"],
        ["M: reference discovery", "Complete", "Activity selectors; 29 valid groups; top 10 reported"],
        ["A1: measure", "Complete", "Binary versus proportional; overlap, signs and reference differences"],
        ["A1: weighted variant", "Not applicable", "Full raw-data audit found no repeated-reply intensity"],
        ["A2: dependence", "Complete", "Ordinary versus user-clustered inference; reference plus fixed top 10"],
        ["A3: effect scale", "Complete", "Binary AMEs; reference plus fixed top 10; ranking and direction checks"],
    ]
    intro = (
        "This report consolidates the completed analyses of reciprocity and next-month return in Reddit reply networks. "
        "The main search and robustness comparisons are complete for the implemented activity-based specification. "
        "The results describe associations and are not estimates of causal effects."
    )
    m_text = (
        f"The binary search returned {len(all_m)} valid subgroups: nine with one condition and twenty with two. "
        f"All {negative} reciprocity coefficients were negative, ranging from {all_m.beta.min():.3f} to {all_m.beta.max():.3f}, "
        f"against a global reference of {ref.beta:.4f}. The highest-quality subgroup was Sent > 3 and Received > 3 "
        f"(n = {int(rank1.n):,}; return rate {rank1.return_rate_pct:.2f}%; beta = {rank1.beta:.3f}; "
        f"95% CI [{rank1.ci_low:.3f}, {rank1.ci_high:.3f}]). Its AME was {rank1.ame_pp:.2f} percentage points."
    )
    a1_text = (
        f"Binary and proportional reciprocity retained nine of the ten highest-quality subgroup descriptions "
        f"(Jaccard overlap 9/11 = {a1['jaccard_top_k']:.3f}). All nine shared top-ten subgroups had negative coefficients "
        "and more negative coefficients than their own measure-specific references. Across all 29 shared groups, "
        "coefficient signs agreed in 29/29 cases, while the direction of the subgroup-reference difference agreed in 21/29. "
        "Thus the main selected contexts and association signs were broadly stable, but ranking and some reference-relative "
        "conclusions depended on the definition of reciprocity."
    )
    a2_text = (
        f"User clustering increased the reference standard error by {ref.se_change_pct:.1f}% and top-ten subgroup standard "
        f"errors by {a2.iloc[1:].se_change_pct.min():.1f}%-{a2.iloc[1:].se_change_pct.max():.1f}% "
        f"(median {a2.iloc[1:].se_change_pct.median():.1f}%). All ten subgroup confidence intervals remained below zero under "
        "both approaches. Coefficients and the quality ranking were unchanged. M already used user-clustered inference; "
        "A2 supplies its explicit comparison with ordinary model-based uncertainty."
    )
    a3_text = (
        f"The reference AME was {s3['reference_ame_pp']:.2f} percentage points. AMEs for the fixed top ten ranged from "
        f"{s3['ame_pp_min']:.2f} to {s3['ame_pp_max']:.2f} percentage points. All ten subgroup-reference differences had "
        "the same direction on the coefficient and probability scales. However, the Spearman correlation between rankings "
        f"by absolute coefficient difference and absolute AME difference was only {s3['spearman_abs_difference_ranks']:.3f}. "
        "Directional conclusions were consistent, while the ranking of substantive effect magnitudes depended on the scale."
    )
    limitation = (
        "A subgroup confidence interval excluding the global reference coefficient is a descriptive diagnostic, not a formal "
        "test of the difference between two estimated coefficients. The reference includes the subgroup, so a formal test "
        "must account for their joint uncertainty. Intervals are pointwise and have no multiple-testing or post-selection adjustment."
    )
    sections = []
    sections.append({"title": "Reciprocity and return\nCompleted experiments", "blocks": [
        small("FINAL EXPERIMENT REPORT  |  11 OCTOBER 2026  |  EMM PROJECT"), paragraph(intro),
        table(statuses, [119, 73, 319]), heading("What the experiments establish"),
        paragraph("1. Adjusted reciprocity-return associations are negative in the reference and all 29 activity subgroups evaluated in M."),
        paragraph("2. Nine of the top ten contexts survive the change from binary to proportional reciprocity. Some reference-relative conclusions remain measure-sensitive."),
        paragraph("3. User clustering modestly widens uncertainty. Probability-scale analysis changes the ordering of the strongest subgroup differences."),
        small("Scope: months 1-10; activity selectors only; no lagged controls or community-context selector search. The weighted variant is unavailable in the supplied data. See pages 8-9 for limits and reproduction."),
    ]})
    sections.append({"title": "1. Data, model and supporting analyses", "blocks": [
        paragraph(f"The analysis contains {int(ref.n):,} user-subreddit-month observations from {int(ref.user_clusters):,} users across 2,046 subreddits. The next-month return rate is {100*ref.return_rate:.2f}%. Eleven monthly networks provide ten observed transitions; month 11 supplies the last return label."),
        heading("Variables and fitted model"),
        paragraph("Return means appearing as a reply author in the same subreddit in the following month. Binary reciprocity is 1 when reciprocal_partners > 0. Proportional reciprocity is reciprocal_partners / outgoing_partners; this definition was checked on every analysis row."),
        paragraph("The logistic model includes reciprocity, ln(1 + replies_sent), ln(1 + replies_received), and month and subreddit fixed effects. Subgroup models use the same specification within each subgroup. The global reference uses the full analysis population, including the subgroup."),
        small("Sent and Received throughout this report refer to the existing replies_sent and replies_received fields. The raw adjacency lists contain no repeated targets, so these supplied network counts do not retain repeated-message intensity."),
        heading("Completed foundation"),
        paragraph(f"Temporal EDA found predominantly short-lived or intermittent participation. Across subreddits, extending the return window from one to three months increased return rates by an unweighted mean of {temporal.extra_return_3m.mean():.2f} percentage points. This EDA does not change the one-month outcome used in M and A1-A3."),
        paragraph(f"Unadjusted return rates were {raw_yes:.2f}% with binary reciprocity and {raw_no:.2f}% without it, a {raw_yes-raw_no:.2f}-percentage-point difference. After adjustment, the pooled coefficient was {ref.beta:.4f}, with clustered 95% CI [{ref.clustered_ci_low:.4f}, {ref.clustered_ci_high:.4f}]; the odds ratio was {np.exp(ref.beta):.3f} and the binary AME was {s3['reference_ame_pp']:.2f} percentage points. Raw group differences and adjusted model estimates represent different comparisons."),
        paragraph(f"Separate community models converged for all 2,046 subreddits. Coefficients were negative in 2,021 and positive in 25; nominal 95% intervals lay below zero in 1,782 and above zero in three. The remaining 261 included zero. The median AME was {communities.ame_pp.median():.2f} percentage points (IQR {communities.ame_pp.quantile(.25):.2f} to {communities.ame_pp.quantile(.75):.2f}). These comparisons were exploratory and unadjusted for multiple testing."),
        small("Sources: ../network_eda/; ../eda_outputs/figures/; ../logistic_resgression/pooled_logistic.ipynb; ../logistic_resgression/results/subreddit_logistic_results.csv; A2 and A3 reference checkpoints."),
    ]})
    m_rows = [["M rank", "Subgroup", "n", "Return %", "Beta [95% CI]", "Signed delta", "Quality"]]
    for r in m.itertuples():
        m_rows.append([r.rank, describe(r.description), f"{r.n:,}", f"{r.return_rate_pct:.2f}", f"{r.beta:.3f}\n[{r.ci_low:.3f}, {r.ci_high:.3f}]", f"{r.signed_delta:.3f}", f"{r.quality:.1f}"])
    sections.append({"title": "2. M: reference subgroup discovery", "blocks": [
        paragraph("The search uses Sent thresholds 1 and 3 and Received thresholds 0, 1 and 3, with <= and > conditions. Beam width is 10, maximum depth is 2, and minimum support is 5,000. Reporting k = 10 is distinct from beam width. The quality score is sqrt(n) x |beta_subgroup - beta_reference|."),
        paragraph(m_text), table(m_rows, [35, 140, 65, 50, 115, 55, 51]),
        small("Subgroups overlap; their sizes must not be added. CIs use user-clustered standard errors. Signed delta = subgroup beta minus reference beta. The original CSV field delta is the absolute difference used in quality. More negative signed delta means a stronger negative conditional association."),
        paragraph(f"The weakest negative coefficient among all 29 valid groups was {smallest_beta.beta:.3f} for {describe(smallest_beta.description)} (n = {int(smallest_beta.n):,}). {ci_excludes}/29 intervals exclude the reference point. Magnitudes and probability-scale effects are more informative than this diagnostic alone."),
        small("Sources: ../beam_search/emm_beam_search_results.csv and M_reference_discovery/tables/top_10_subgroups.csv. AME values are supplied by A3. The subgroup definitions use the same activity variables included as regression controls."),
    ]})
    a1_rows = [["Context", "Binary rank", "Prop. rank", "Binary beta", "Prop. beta"]]
    for r in union.sort_values("quality_rank_binary").itertuples():
        a1_rows.append([describe(r.description_binary), int(r.quality_rank_binary), int(r.quality_rank_proportional), f"{r.beta_binary:.3f}", f"{r.beta_proportional:.3f}"])
    sections.append({"title": "3. A1: reciprocity measure", "blocks": [
        paragraph(a1_text),
        small(f"Proportional reference: beta {a1['proportional_reference_beta']:.4f}, clustered SE {a1['proportional_reference_se']:.6f}, 95% CI [{a1['proportional_reference_ci'][0]:.4f}, {a1['proportional_reference_ci'][1]:.4f}]. The same search returned 29 valid groups from 34 attempted descriptions; five lacked reciprocity variation."),
        table(a1_rows, [203, 64, 64, 90, 90]),
        small("The table is the union of both top-ten lists; ranks above 10 identify membership changes. Sent > 3 and Received <= 3 enters (20 to 9); Sent <= 3 and Received <= 3 leaves (10 to 23). Betas represent different reciprocity contrasts and are not directly comparable probability changes."),
        heading("Weighted variant: not applicable to the supplied data"),
        paragraph(f"The audit examined all {weighted['files_checked']:,} JSON files, {weighted['months_checked']:,} monthly networks and {weighted['reply_entries_checked']:,} target entries across {weighted['outgoing_lists_checked']:,} outgoing user-month lists. No list contained repeated targets. A distinct repeated-reply-weighted experiment cannot be recovered from these inputs. This does not imply that users never replied repeatedly in the underlying Reddit activity."),
        small("Six proportional subgroups had separation exclusions; binary refits verified the same excluded rows and reproduced saved coefficients and SEs. One of these groups enters the proportional top ten. Input n remains the quality-function support; fitted_rows records actual estimation size. Sources: A1_reciprocity_measure/tables/ and audits/."),
    ]})
    a2_rows = [["M rank", "Ordinary SE", "Clustered SE", "Change %", "Ordinary 95% CI", "Clustered 95% CI"]]
    for r in a2.itertuples():
        a2_rows.append(["Ref" if pd.isna(r.quality_rank) else int(r.quality_rank), f"{r.ordinary_se:.6f}", f"{r.clustered_se:.6f}", f"{r.se_change_pct:.1f}", f"[{r.ordinary_ci_low:.4f}, {r.ordinary_ci_high:.4f}]", f"[{r.clustered_ci_low:.4f}, {r.clustered_ci_high:.4f}]"])
    sections.append({"title": "4. A2: dependence between observations", "blocks": [
        paragraph("The global reference and fixed binary top ten were refitted once each using the original likelihood and controls. User-clustered CRV1 uncertainty was reproduced, then ordinary model-based (iid) uncertainty was calculated on the same fitted model. All eleven models converged and retained their complete input samples."),
        paragraph(a2_text), table(a2_rows, [33, 73, 74, 54, 138, 139]),
        small("M ranks refer to the subgroup descriptions on page 3. SE change = 100 x (clustered SE / ordinary SE - 1). Both sets of intervals are pointwise 95% intervals. All subgroup intervals remain below the reference coefficient under both methods."),
        heading("Interpretation"),
        paragraph("Allowing dependence within users modestly widens uncertainty without overturning the negative-association evidence in these selected groups. It is expected that coefficient signs, AME point estimates and quality rankings do not change when only the covariance calculation changes. This comparison does not isolate dependence from every possible source of model misspecification."),
        paragraph(limitation),
        small("Sources: A2_user_dependence/tables/a2_comparison.csv; reports/summary.json; audits/validation.json. All ten clustered subgroup fits reproduce M; the reference agrees with the saved pooled notebook. Clustered results remain the basis for interpretation."),
    ]})
    a3_rows = [["M rank", "Signed delta beta", "AME (pp)", "Delta AME (pp)", "|Delta beta| rank", "|Delta AME| rank"]]
    for r in a3.itertuples():
        a3_rows.append([r.quality_rank, f"{r.signed_delta:.3f}", f"{r.ame_pp:.2f}", f"{r.delta_ame_pp:.2f}", int(r.coefficient_difference_rank), int(r.ame_difference_rank)])
    sections.append({"title": "5. A3: effects on the probability scale", "blocks": [
        paragraph("For binary reciprocity, the AME is the average discrete change in predicted return probability when reciprocity changes from 0 to 1, holding the other model inputs fixed for each observation. An AME of -7.55 percentage points means an average model-predicted reduction of 7.55 points for that contrast; it is not a 7.55% relative reduction and is not a causal estimate."),
        paragraph(a3_text), table(a3_rows, [38, 89, 88, 99, 98, 99]),
        small("M rank is the original size-weighted quality rank. The last two columns rank absolute subgroup-reference differences without size weighting. Reference AME: -4.44 percentage points. Subgroups are identified on page 3."),
        heading("Which context has the largest probability difference?"),
        paragraph(f"The most negative AME is {strongest_ame.ame_pp:.2f} percentage points for {describe(strongest_ame.description)} (M rank {int(strongest_ame.quality_rank)}), a difference of {strongest_ame.delta_ame_pp:.2f} points from the reference. The strongest coefficient context, Sent > 3 and Received > 3, has AME {rank1.ame_pp:.2f} points and ranks fifth by absolute AME difference."),
        paragraph("Each AME is averaged over its own fitted population. Differences therefore reflect both model estimates and covariate composition, rather than a common standardized population. A3 provides point estimates only: no AME confidence intervals or post-selection uncertainty were estimated. Coefficient/AME sign agreement is expected in this binary main-effects specification and is not an independent robustness test."),
        small("Sources: A3_probability_scale/tables/top_10_ame_comparison.csv, reports/summary.json and model_checkpoints/. Eleven refits reproduced M's coefficients to numerical precision. Ordinary covariance was used for those point-estimate refits; original clustered coefficient CIs were retained."),
    ]})
    make_figures(m, a2, a3)
    sections.append({"title": "6. Visual comparison of the results", "blocks": [
        {"type":"image", "path":"figures/coefficient_and_ame.png", "width":505, "height":360},
        small("Figure 1. The same ten subgroups, indexed by M rank. Left: coefficients and user-clustered 95% intervals; the dashed line is the global coefficient. Right: binary AME point estimates; the dashed line is the global AME. AME uncertainty was not estimated. Separate horizontal scales are intentional."),
        {"type":"image", "path":"figures/clustered_se_change.png", "width":505, "height":220},
        small("Figure 2. Percentage increase in SE after user clustering. Ref denotes the global model; 1-10 are M ranks. Both figures use the canonical results tables and can be regenerated without model refits."),
    ]})
    sections.append({"title": "7. Interpretation and remaining scope", "blocks": [
        heading("Overall interpretation"),
        paragraph("The positive raw reciprocity-return difference does not persist after adjustment for activity and fixed effects. Negative adjusted associations appear across the activity contexts evaluated. The main selected contexts are broadly robust to binary versus proportional measurement and to clustered uncertainty, while the order of effect magnitudes is sensitive to whether coefficients or probability differences are used."),
        paragraph("Substantive interpretation should report both the coefficient contrast and the AME. A more negative logistic coefficient does not necessarily imply the largest average probability difference. The AME analysis supports directional heterogeneity across the selected activity contexts, but the modest rank correlation argues against treating the coefficient-based ordering as the sole substantive result."),
        heading("Limits of the completed experiments"),
        paragraph("Activity context: subgroup selectors are Sent and Received, which are also regression controls. These results describe variation across activity levels; the search did not include separate community-context variables. Community-specific regressions are supporting analyses, not a community-selector beam search."),
        paragraph("Model scope: fitted transitions are 1-10, with contemporaneous activity controls and month/subreddit fixed effects. Lagged controls and the corresponding transitions 2-10 specification have not been run."),
        paragraph(limitation),
        paragraph("Effect scale and data: AMEs use each group's own covariate distribution and have no uncertainty intervals. Weighted repeated-reply intensity is unavailable. All results remain descriptive; none establishes that reciprocity causes lower return."),
        heading("What remains for the paper"),
        paragraph("The experiment report, tables and paste-ready results text are complete. Align the methods with the implemented activity-only, non-lagged specification; if broader selectors or lagged controls remain a requirement, those are additional experiments. Describe weighted A1 as not applicable, retain clustered CIs, explain the two effect rankings, and distinguish interval exclusion of the reference point from a formal difference test."),
        small("manuscript_results.txt and manuscript_results.tex contain results prose for M and A1-A3. No manuscript source was edited, and no new experiments were run during packaging. The existing LaTeX tables are included as exported; their layout has not been compiled in the target manuscript."),
    ]})
    sections.append({"title": "8. Files and reproduction", "blocks": [
        paragraph("Each code or result file has one canonical location. Added experiment code and outputs live under experiments/. Original search, EDA and regression code remains in its existing project folders. Duplicate copies have been removed; reproducibility/manifest.json records the current paths and SHA-256 hashes."),
        table([
            ["Folder / file", "Purpose"],
            ["README.txt", "Start here: folder map, commands and dependencies"],
            ["M_reference_discovery/", "Added reporter and top-ten tables; original search stays in ../beam_search/"],
            ["A1_reciprocity_measure/", "Measure comparison, weighted audit, results, checkpoints and single data cache"],
            ["A2_user_dependence/", "Dependence runner, SE/CI results and checkpoints"],
            ["A3_probability_scale/", "AME runner, effect/ranking results and checkpoints"],
            ["supporting_analyses/", "Guide to existing EDA and model files; no duplicate source copies"],
            ["figures/", "Report figures in PNG and vector SVG"],
            ["code/", "Report builders, shared path routing and experiment launcher"],
            ["reproducibility/", "Current file manifest, environment, source checks and relocation audit"],
        ], [184, 327]),
        heading("Rerunning"),
        paragraph("From the project root, python experiments/code/run_experiment.py A2 prints the command. Add --execute to run it. Choices: M, M-report, A1, A1-weighted-audit, A2 and A3. Added scripts can also run directly at their documented paths. The launcher preserves the original M entry point. Use the recorded analysis environment, including pyfixest 0.60.0."),
        paragraph("Generated outputs go directly to tables/, reports/, latex/, audits/ and model_checkpoints/ inside each experiment. A1/A2 share the single parquet cache under A1_reciprocity_measure/data_cache/. The analysis CSV and raw networks remain in their original locations. Completed checkpoint signatures were migrated after verifying that only paths and output routing changed; numerical estimates were preserved."),
        small("To rebuild the report without refitting, run code/build_bundle.py using the analysis Python, then code/render_report.py using a Python with reportlab. The builder reads canonical files and does not copy them. Reruns of experiments may update their generated results; they do not edit original source code."),
        small("Original evidence: ../beam_search/emm_beam_search_results.csv, ../network_eda/, ../eda_outputs/figures/ and ../logistic_resgression/. Source references in this report are relative to experiments/. Historical audits retain paths as they existed at the time; consolidation_plan.json maps those paths to current locations."),
    ]})
    save_json("reproducibility/report_content.json", {"date": "2026-10-11", "sections": sections})
    text_parts = []
    for section in sections:
        text_parts.extend([section["title"], "=" * 75])
        for block in section["blocks"]:
            if "text" in block:
                text_parts.append(block["text"])
            elif block["type"] == "table":
                text_parts.append(pd.DataFrame(block["rows"][1:], columns=block["rows"][0]).to_string(index=False))
            elif block["type"] == "image":
                text_parts.append("Figure: " + block["path"])
        text_parts.append("")
    write("FINAL_REPORT.txt", "\n\n".join(text_parts) + "\n")
    manuscript = [
        ("M. Activity-based reference subgroup discovery", m_text + " " + f"The quality score was sqrt(n) times the absolute coefficient difference from the reference. {ci_excludes}/29 subgroup intervals excluded the reference point, which is reported as a descriptive diagnostic rather than a formal test of the subgroup-reference difference. Subgroup definitions used activity variables also included as controls."),
        ("A1. Reciprocity operationalization", a1_text + " " + "The weighted variant was not applicable: an exhaustive audit of the supplied raw networks found no repeated targets in any outgoing user-month list, so repeated-reply intensity was unavailable. Coefficients under the two measures refer to different contrasts and should not be interpreted as directly comparable probability changes."),
        ("A2. Dependence between observations", a2_text + " " + limitation),
        ("A3. Probability-scale comparison", a3_text + " " + f"The most negative AME was {strongest_ame.ame_pp:.2f} percentage points for {describe(strongest_ame.description)}, rather than for the subgroup with the most negative coefficient. AMEs were evaluated over each model's own observations, and no AME confidence intervals were estimated."),
        ("Scope and interpretation", "These completed analyses use months 1-10, activity selectors and the two contemporaneous activity controls, with month and subreddit fixed effects. They do not include a community-context selector search or lagged controls. Results are descriptive rather than causal; pointwise intervals have no post-selection adjustment."),
    ]
    write("manuscript_results.txt", "RESULTS TEXT FOR THE COMPLETED EXPERIMENTS\n\n" + "\n\n".join(t + "\n" + b for t, b in manuscript) + "\n")
    def tex_escape(value):
        return value.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("<", "$<$").replace(">", "$>$")
    write("manuscript_results.tex", "% Results fragment; insert in the existing manuscript. No bibliography keys are invented.\n" + "\n\n".join("\\paragraph{" + tex_escape(t) + "}\n" + tex_escape(b) for t, b in manuscript) + "\n")
    notes = {
        "M_reference_discovery": ("M: activity-based reference subgroup discovery", m_text),
        "A1_reciprocity_measure": ("A1: binary versus proportional reciprocity", a1_text + " Weighted reciprocity is not applicable to the supplied adjacency data; see audits/weighted_feasibility.json."),
        "A2_user_dependence": ("A2: ordinary versus user-clustered uncertainty", a2_text),
        "A3_probability_scale": ("A3: coefficient versus probability scale", a3_text),
    }
    for folder, (title, interpretation) in notes.items():
        write(f"{folder}/README.txt", title + "\n\n" + interpretation + "\n\ncode/: runnable added scripts. tables/: full numeric CSV outputs. reports/: readable results and JSON summaries.\nmodel_checkpoints/: individual saved fits where available. audits/: validation and provenance where available.\nlatex/: existing exported tables, not compiled in the target manuscript.\n\nUse ../README.txt for rerun commands. Original M implementation and full results remain in ../../beam_search/.\n")
    save_json("reproducibility/numerical_checks.json", {
        "main_valid_groups": len(all_m), "main_top_k": len(m), "top_k_descriptions_match_M_A2_A3": True,
        "ame_exports_match": True, "clustered_coefficients_reproduce_M": True,
        "raw_rates_reconstructed_from_notebook_counts": {"without_reciprocity_pct": raw_no, "with_reciprocity_pct": raw_yes},
        "temporal_mean_3_month_gain_pp": float(temporal.extra_return_3m.mean()),
        "community_models": len(communities), "all_community_fits_ok": bool(communities.status.eq("ok").all()),
        "no_models_rerun_during_packaging": True,
    })


def make_figures(m, a2, a3):
    directory = BASE / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11, "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 7.2), sharey=True)
    y = m["rank"].to_numpy()
    axes[0].errorbar(m.beta, y, xerr=np.vstack([m.beta-m.ci_low, m.ci_high-m.beta]), fmt="o", color="#087e8b", capsize=3)
    axes[0].axvline(m.reference_beta.iloc[0], linestyle="--", color="#aa6034", linewidth=1.4)
    axes[0].set(xlabel="Reciprocity coefficient (log odds)", ylabel="Original M quality rank", title="Coefficient and clustered 95% CI", yticks=y)
    axes[1].scatter(a3.ame_pp, y, color="#19374b", s=48)
    axes[1].axvline(a3.reference_ame_pp.iloc[0], linestyle="--", color="#aa6034", linewidth=1.4)
    axes[1].set(xlabel="Binary AME (percentage points)", title="Probability-scale point estimate")
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="y", color="#e2e7eb", zorder=0)
    fig.tight_layout(pad=2.1)
    for ext in ("png", "svg"):
        fig.savefig(directory / f"coefficient_and_ame.{ext}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10.1, 4.4))
    bars = ax.bar(["Ref"]+[str(i) for i in range(1,11)], a2.se_change_pct, color=["#aa6034"]+["#087e8b"]*10)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in a2.se_change_pct], padding=4, fontsize=10)
    ax.set(xlabel="Reference model and original M quality rank", ylabel="Increase in standard error (%)", ylim=(0,16))
    ax.grid(axis="y", color="#e2e7eb")
    ax.set_axisbelow(True)
    fig.tight_layout(pad=1.8)
    for ext in ("png", "svg"):
        fig.savefig(directory / f"clustered_se_change.{ext}", dpi=180)
    plt.close(fig)


def readme():
    write("README.txt", """EXPERIMENTS - START HERE
========================
Open FINAL_REPORT.pdf or FINAL_REPORT.txt. manuscript_results.txt and .tex
contain results prose for M and A1-A3.

SINGLE CANONICAL LOCATIONS
- M_reference_discovery/: added top-ten reporter and its results.
- A1_reciprocity_measure/: comparison/audit code, results and model checkpoints.
- A2_user_dependence/: dependence-analysis code, results and checkpoints.
- A3_probability_scale/: AME-analysis code, results and checkpoints.
- code/: launcher, shared paths and report builders.
- figures/: final-report PNG/SVG figures (different output formats).
- reproducibility/: canonical paths, environment, checks and relocation history.
- supporting_analyses/README.txt: pointers to existing foundation analyses.

Your ORIGINAL code stays unchanged in beam_search/, network_eda/, reddit_eda/
and logistic_resgression/. The original full M results stay at
beam_search/emm_beam_search_results.csv. Duplicate copies of those files have
been removed from experiments. Added experiment scripts were removed from
beam_search after their runnable versions were established here.

HOW TO RUN (from project root, with the analysis Python)
python experiments/code/run_experiment.py M
python experiments/code/run_experiment.py M-report
python experiments/code/run_experiment.py A1
python experiments/code/run_experiment.py A1-weighted-audit
python experiments/code/run_experiment.py A2
python experiments/code/run_experiment.py A3

Commands preview their target by default. Add --execute to run.
You can also run the added scripts directly from their experiment code folders.
The M reporter is M_reference_discovery/code/added/report_top_subgroups.py.
Reruns may replace generated outputs. They do not edit original source code.
A1/A2/A3 resume verified checkpoints when inputs, code and versions match.
The relocation changed paths/imports only; model estimates were not recomputed.

DEPENDENCIES AND INPUTS
- Analysis environment: reproducibility/environment.json.
- Data: ../eda_outputs/reciprocity_return.csv.
- Raw networks for weighted audit: ../reply_networks/*.json.
- A1/A2 cache: A1_reciprocity_measure/data_cache/prepared.parquet (one copy).
- A1 can rebuild that cache from the analysis CSV; A2 requires its audit.
- A1 imports the unchanged original beam search from ../beam_search/.
- A1/A3 share the M reporter; all added scripts share code/experiment_paths.py.

REBUILD REPORTS WITHOUT MODEL FITS
python experiments/code/build_bundle.py
python experiments/code/render_report.py
The first uses pandas/numpy/matplotlib; the second needs reportlab. The builder
reads canonical results directly and does not recreate duplicate archives.
Inspect rendered PDF pages after changing report content.

COMPLETED SCOPE
M: activity-only binary search, 29 valid groups and top-ten reporting.
A1: binary/proportional comparison; weighted variant unavailable after full audit.
A2: ordinary versus user-clustered uncertainty for reference plus fixed top ten.
A3: binary AMEs and coefficient/probability difference rankings.
Prior temporal/reciprocity EDA and baseline/community models are also documented.

LIMITS
No community-context selector search or lagged-control experiment. Pointwise
coefficient CIs have no selection adjustment. CI exclusion of the estimated
reference is descriptive, not a formal difference test. AMEs are evaluated
within each population and have no uncertainty intervals. Associations are
not causal. Existing LaTeX table exports have not been compiled in the target
manuscript. No original manuscript was edited.

Historical audits may mention their former beam_search/results paths. Current
locations are in manifest.json and consolidation_plan.json. PDF/text, CSV/text/
LaTeX, and PNG/SVG files are intentional alternative formats, not duplicate code.
""")
    write("supporting_analyses/README.txt", """SUPPORTING ANALYSES - ORIGINAL FILE LOCATIONS
============================================
All paths below are relative to the project root. These files were not moved
or edited; their duplicate archive copies have been removed.

Temporal EDA: network_eda/ (source, summary CSV and figures).
Reciprocity EDA: reddit_eda/ (source/notebook); eda_outputs/figures/ (tables/plots).
Pooled baseline: logistic_resgression/pooled_logistic.ipynb (saved outputs).
Community models: logistic_resgression/subreddit_logistic.ipynb and
logistic_resgression/results/subreddit_logistic_results.csv.
Original M: beam_search/beam_search.py, emm_selectors.py, logistic_model.py,
run_beam_search.py and emm_beam_search_results.csv.

The final report reads the original evidence directly. See
../reproducibility/manifest.json for exact current paths and hashes.
""")
    write("M_reference_discovery/code/ORIGINAL_CODE_LOCATION.txt", "Original beam-search/model code remains unchanged in the project's beam_search/ directory. This folder contains only the added reporting code. See experiments/README.txt for entry points.\n")


def main():
    BASE.mkdir(exist_ok=True)
    before = {p.relative_to(ROOT).as_posix(): digest(p)
              for folder in ("beam_search", "logistic_resgression", "network_eda", "reddit_eda")
              for p in (ROOT / folder).rglob("*")
              if p.is_file() and p.suffix in (".py", ".ipynb") and "__pycache__" not in p.parts}
    inventory()
    generate_report()
    readme()
    save_json("reproducibility/manifest.json", {"updated_date":"2026-10-11", "layout":"single canonical location per file", "files": MANIFEST})
    environment = {"python": sys.version, "platform": platform.platform(), "analysis_executable": sys.executable,
                   "packages": {p: importlib.metadata.version(p) for p in ("pyfixest", "pandas", "numpy", "scipy", "pyarrow", "matplotlib")}}
    save_json("reproducibility/environment.json", environment)
    inputs = []
    for name in ["eda_outputs/reciprocity_return.csv", "experiments/A1_reciprocity_measure/data_cache/prepared.parquet"]:
        path = ROOT / name
        inputs.append({"path": name, "bytes":path.stat().st_size, "mtime_ns":path.stat().st_mtime_ns, "copied":False})
    save_json("reproducibility/external_inputs.json", {"files": inputs, "raw_network_directory":"reply_networks", "raw_file_count":2046,
              "note":"Large inputs are referenced, not duplicated; size/mtime are provenance metadata, not content hashes. See the weighted audit for the raw metadata-manifest hash."})
    changed = [name for name, sha in before.items() if digest(ROOT / name) != sha]
    copy_changes = [r["canonical"] for r in MANIFEST if digest(ROOT / r["canonical"]) != r["sha256"]]
    save_json("reproducibility/source_integrity.json", {"original_code_files_checked":len(before), "canonical_files_checked":len(MANIFEST),
              "all_original_code_unchanged":not changed, "all_canonical_files_unchanged_during_report_build":not copy_changes,
              "changed_code":changed, "changed_sources":copy_changes, "original_code_sha256":before})
    if changed or copy_changes:
        raise ValueError("A source file changed during packaging")
    print(f"Indexed {len(MANIFEST)} canonical files; checked {len(before)} original code files; no copies or model fits.")


if __name__ == "__main__":
    main()
