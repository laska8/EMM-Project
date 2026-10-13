"""Rebuild experiment reports from the saved results without fitting models.

Run with the project's analysis Python. Then run render_report.py with a Python
environment providing reportlab. All writes are confined to experiments/.
"""
from datetime import date
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


def write(path, value):
    path = BASE / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def save_json(path, value):
    write(path, json.dumps(value, indent=2, allow_nan=False))


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
    assert len(m) == 10 and len(communities) == 2046
    assert a1.get("discretization") == "local", "A1 results are from the earlier search; rerun A1"
    assert weighted["repeated_target_entries"] == 0
    assert communities.status.eq("ok").all()
    assert int((communities.coefficient < 0).sum()) == 2021
    assert int((communities.ci_high < 0).sum()) == 1782
    assert int((communities.ci_low > 0).sum()) == 3
    assert (a2.clustered_ci_high < 0).all() and (a2.ordinary_ci_high < 0).all()
    np.testing.assert_allclose(m.ame_pp, a3.ame_pp)
    np.testing.assert_allclose(m.beta, a2.iloc[1:].beta, rtol=0, atol=1e-7)
    negative = int((all_m.beta < 0).sum())
    ci_excludes = int(((all_m.ci_high < ref.beta) | (all_m.ci_low > ref.beta)).sum())
    rank1 = m.iloc[0]
    words = {1: "one condition", 2: "two conditions", 3: "three conditions"}
    conditions_text = ", ".join(f"{count} with {words.get(d, str(d) + ' conditions')}"
                                for d, count in all_m.depth.value_counts().sort_index().items())
    groups = a2.iloc[1:]
    below_reference = int(((groups.clustered_ci_high < ref.beta) & (groups.ordinary_ci_high < ref.beta)).sum())
    most_negative_beta = a3.loc[a3.beta.idxmin()]
    today = date.today()
    strongest_ame = a3.loc[a3.ame_pp.idxmin()]
    smallest_beta = all_m.loc[all_m.beta.idxmax()]
    statuses = [
        ["Analysis", "Status", "Completed scope"],
        ["Supporting analyses", "Complete", "Temporal and reciprocity EDA; pooled model; 2,046 community models"],
        ["M: reference discovery", "Complete", f"Activity selectors, local discretization; {len(all_m)} valid groups; top 10 reported"],
        ["A1: measure", "Complete", "Binary versus proportional; separate searches, top-10 overlap and reference differences"],
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
        f"The binary search returned {len(all_m)} valid subgroups: {conditions_text}. "
        f"{negative} of the {len(all_m)} reciprocity coefficients were negative; coefficients ranged from {all_m.beta.min():.3f} to {all_m.beta.max():.3f}, "
        f"against a global reference of {ref.beta:.4f}. The highest-quality subgroup was {describe(rank1.description)} "
        f"(n = {int(rank1.n):,}; return rate {rank1.return_rate_pct:.2f}%; beta = {rank1.beta:.3f}; "
        f"95% CI [{rank1.ci_low:.3f}, {rank1.ci_high:.3f}]). Its AME was {rank1.ame_pp:.2f} percentage points."
    )
    a1_text = (
        f"The beam search was repeated with proportional reciprocity as the target, with the same data, model, local "
        f"discretization and settings. It returned {a1['proportional_valid_subgroups']} valid subgroups "
        f"({a1['proportional_negative_coefficients']} with a negative coefficient), against {a1['binary_valid_subgroups']} "
        f"for binary reciprocity; {a1['visited_by_both_searches']} subgroups were visited by both searches. "
        f"The two top-ten lists share {a1['shared_top_k_count']} of {a1['union_top_k_count']} distinct descriptions "
        f"(Jaccard overlap {a1['jaccard_top_k']:.3f}). Because thresholds are recomputed in every refined subgroup, the two "
        "searches follow different paths, so identical descriptions are a strict criterion. By observations, "
        f"{100 * a1['proportional_top_k_observations_inside_binary_top_k']:.0f}% of those selected by the proportional top ten "
        f"are also selected by the binary top ten (Jaccard overlap of the two selections {a1['top_k_observation_jaccard']:.3f}). "
        "Each subgroup in either "
        f"top-ten list was also estimated under the other measure: of the {a1['union_estimated_under_both_measures']} "
        f"that could be estimated under both, the coefficient sign agreed in {a1['union_coefficient_sign_agreement_count']} "
        f"and the direction of the subgroup-reference difference in {a1['union_difference_direction_agreement_count']}. "
        f"Among all {a1['visited_by_both_searches']} subgroups visited by both searches, the direction of the difference "
        f"agreed in {a1['visited_by_both_difference_direction_agreement_count']}."
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
        f"{s3['ame_pp_min']:.2f} to {s3['ame_pp_max']:.2f} percentage points. "
        f"{s3['difference_direction_agreement_count']} of the {len(a3)} subgroup-reference differences had "
        "the same direction on the coefficient and probability scales. The Spearman correlation between rankings "
        f"by absolute coefficient difference and absolute AME difference was {s3['spearman_abs_difference_ranks']:.3f}, "
        "which shows how far the ordering of effect magnitudes depends on the scale."
    )
    limitation = (
        "A subgroup confidence interval excluding the global reference coefficient is a descriptive diagnostic, not a formal "
        "test of the difference between two estimated coefficients. The reference includes the subgroup, so a formal test "
        "must account for their joint uncertainty. Intervals are pointwise and have no multiple-testing or post-selection adjustment."
    )
    sections = []
    sections.append({"title": "Reciprocity and return\nCompleted experiments", "blocks": [
        small(f"FINAL EXPERIMENT REPORT  |  {today.day} {today:%B %Y}  |  EMM PROJECT".upper()), paragraph(intro),
        table(statuses, [119, 73, 319]), heading("What the experiments establish"),
        paragraph(f"1. Adjusted reciprocity-return associations are negative in the reference and in {negative} of the {len(all_m)} activity subgroups evaluated in M."),
        paragraph(f"2. A separate search with proportional reciprocity shares {a1['shared_top_k_count']} of {a1['union_top_k_count']} distinct top-ten descriptions with the binary search; the direction of the subgroup-reference difference agrees in {a1['union_difference_direction_agreement_count']} of the {a1['union_estimated_under_both_measures']} top-ten subgroups estimated under both measures."),
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
        paragraph("Numeric selectors are discretized locally, following the 3-lbca strategy of Meeng and Knobbe (2021): each time a subgroup is refined, Sent and Received are split into 8 equal-frequency bins of that subgroup, and every cut point gives a <= and a > condition. Beam width is 10, the search has 3 levels, and minimum support is 5,000. A subgroup is skipped when any combination of reciprocity (yes/no) and return (yes/no) has fewer than 10 observations. Reporting k = 10 is distinct from beam width. The quality score is sqrt(n) x |beta_subgroup - beta_reference|."),
        paragraph(m_text), table(m_rows, [35, 140, 65, 50, 115, 55, 51]),
        small("Subgroups overlap; their sizes must not be added. CIs use user-clustered standard errors. Signed delta = subgroup beta minus reference beta. The original CSV field delta is the absolute difference used in quality. More negative signed delta means a stronger negative conditional association."),
        paragraph(f"The weakest negative coefficient among all {len(all_m)} valid groups was {smallest_beta.beta:.3f} for {describe(smallest_beta.description)} (n = {int(smallest_beta.n):,}). {ci_excludes}/{len(all_m)} intervals exclude the reference point. Magnitudes and probability-scale effects are more informative than this diagnostic alone."),
        small("Sources: ../beam_search/emm_beam_search_results.csv and M_reference_discovery/tables/top_10_subgroups.csv. AME values are supplied by A3. The subgroup definitions use the same activity variables included as regression controls."),
    ]})
    def a1_rank(value, searched):
        return "-" if pd.isna(value) else f"{int(value)}" + ("" if searched is True else "*")

    def a1_number(value):
        return "-" if pd.isna(value) else f"{value:.3f}"

    a1_rows = [["Context", "Binary rank", "Prop. rank", "Binary beta", "Prop. beta"]]
    for r in union.itertuples():
        a1_rows.append([r.subgroup, a1_rank(r.quality_rank_binary, r.found_by_search_binary), a1_rank(r.quality_rank_proportional, r.found_by_search_proportional), a1_number(r.beta_binary), a1_number(r.beta_proportional)])
    sections.append({"title": "3. A1: reciprocity measure", "blocks": [
        paragraph(a1_text),
        small(f"Proportional reference: beta {a1['proportional_reference_beta']:.4f}, clustered SE {a1['proportional_reference_se']:.6f}, 95% CI [{a1['proportional_reference_ci'][0]:.4f}, {a1['proportional_reference_ci'][1]:.4f}]."),
        table(a1_rows, [203, 64, 64, 90, 90]),
        small("The table is the union of both top-ten lists, binary top ten first. A rank marked * belongs to a subgroup that the search of that measure did not visit: it was fitted separately, and the rank is the place its quality would take in that search. Betas represent different reciprocity contrasts and are not directly comparable probability changes."),
        heading("Weighted variant: not applicable to the supplied data"),
        paragraph(f"The audit examined all {weighted['files_checked']:,} JSON files, {weighted['months_checked']:,} monthly networks and {weighted['reply_entries_checked']:,} target entries across {weighted['outgoing_lists_checked']:,} outgoing user-month lists. No list contained repeated targets. A distinct repeated-reply-weighted experiment cannot be recovered from these inputs. This does not imply that users never replied repeatedly in the underlying Reddit activity."),
        small("Sources: A1_reciprocity_measure/tables/, reports/ and audits/."),
    ]})
    a2_rows = [["M rank", "Ordinary SE", "Clustered SE", "Change %", "Ordinary 95% CI", "Clustered 95% CI"]]
    for r in a2.itertuples():
        a2_rows.append(["Ref" if pd.isna(r.quality_rank) else int(r.quality_rank), f"{r.ordinary_se:.6f}", f"{r.clustered_se:.6f}", f"{r.se_change_pct:.1f}", f"[{r.ordinary_ci_low:.4f}, {r.ordinary_ci_high:.4f}]", f"[{r.clustered_ci_low:.4f}, {r.clustered_ci_high:.4f}]"])
    sections.append({"title": "4. A2: dependence between observations", "blocks": [
        paragraph("The global reference and fixed binary top ten were refitted once each using the original likelihood and controls. User-clustered CRV1 uncertainty was reproduced, then ordinary model-based (iid) uncertainty was calculated on the same fitted model. All eleven models converged and retained their complete input samples."),
        paragraph(a2_text), table(a2_rows, [33, 73, 74, 54, 138, 139]),
        small(f"M ranks refer to the subgroup descriptions on page 3. SE change = 100 x (clustered SE / ordinary SE - 1). Both sets of intervals are pointwise 95% intervals. {below_reference} of the {len(groups)} subgroup intervals lie below the reference coefficient under both methods."),
        heading("Interpretation"),
        paragraph("Allowing dependence within users modestly widens uncertainty without overturning the negative-association evidence in these selected groups. It is expected that coefficient signs, AME point estimates and quality rankings do not change when only the covariance calculation changes. This comparison does not isolate dependence from every possible source of model misspecification."),
        paragraph(limitation),
        small("Sources: A2_user_dependence/tables/a2_comparison.csv; reports/summary.json. All ten clustered subgroup fits reproduce M; the reference agrees with the saved pooled notebook. Clustered results remain the basis for interpretation."),
    ]})
    a3_rows = [["M rank", "Signed delta beta", "AME (pp)", "Delta AME (pp)", "|Delta beta| rank", "|Delta AME| rank"]]
    for r in a3.itertuples():
        a3_rows.append([r.quality_rank, f"{r.signed_delta:.3f}", f"{r.ame_pp:.2f}", f"{r.delta_ame_pp:.2f}", int(r.coefficient_difference_rank), int(r.ame_difference_rank)])
    sections.append({"title": "5. A3: effects on the probability scale", "blocks": [
        paragraph(f"For binary reciprocity, the AME is the average discrete change in predicted return probability when reciprocity changes from 0 to 1, holding the other model inputs fixed for each observation. An AME of {rank1.ame_pp:.2f} percentage points means an average model-predicted change of {rank1.ame_pp:.2f} points for that contrast; it is not a {abs(rank1.ame_pp):.2f}% relative change and is not a causal estimate."),
        paragraph(a3_text), table(a3_rows, [38, 89, 88, 99, 98, 99]),
        small(f"M rank is the original size-weighted quality rank. The last two columns rank absolute subgroup-reference differences without size weighting. Reference AME: {s3['reference_ame_pp']:.2f} percentage points. Subgroups are identified on page 3."),
        heading("Which context has the largest probability difference?"),
        paragraph(f"The most negative AME is {strongest_ame.ame_pp:.2f} percentage points for {describe(strongest_ame.description)} (M rank {int(strongest_ame.quality_rank)}), a difference of {strongest_ame.delta_ame_pp:.2f} points from the reference. The highest-quality context, {describe(rank1.description)}, has AME {rank1.ame_pp:.2f} points and rank {a3.iloc[0].ame_difference_rank:g} of {len(a3)} by absolute AME difference."),
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
        paragraph("The positive raw reciprocity-return difference does not persist after adjustment for activity and fixed effects. Negative adjusted associations appear across the activity contexts evaluated. The selected contexts are robust to clustered uncertainty, while the order of effect magnitudes depends on whether coefficients or probability differences are used. Under proportional reciprocity a separate search selects different descriptions, while the direction of the subgroup-reference differences is compared in A1."),
        paragraph(f"Substantive interpretation should report both the coefficient contrast and the AME. A more negative logistic coefficient does not necessarily imply the largest average probability difference. The AME analysis supports directional heterogeneity across the selected activity contexts, but the rank correlation between the two orderings ({s3['spearman_abs_difference_ranks']:.2f}) argues against treating the coefficient-based ordering as the sole substantive result."),
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
        paragraph("Added experiment code and outputs live under experiments/. The search, EDA and regression code is in the project folders beam_search/, reddit_eda/, network_eda/ and logistic_resgression/."),
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
            ["reproducibility/", "Analysis environment, data dictionary and the report content used for the PDF"],
        ], [184, 327]),
        heading("Rerunning"),
        paragraph("From the project root, python experiments/code/run_experiment.py A2 prints the command. Add --execute to run it. Choices: M, M-report, A1, A1-weighted-audit, A2 and A3. Added scripts can also run directly at their documented paths. The launcher preserves the original M entry point. Use the recorded analysis environment, including pyfixest 0.60.0."),
        paragraph("Generated outputs go directly to tables/, reports/, latex/, audits/ and model_checkpoints/ inside each experiment. A1/A2 share the single parquet cache under A1_reciprocity_measure/data_cache/. The analysis CSV and raw networks remain in their original locations."),
        small("To rebuild the report without refitting, run code/build_bundle.py using the analysis Python, then code/render_report.py using a Python with reportlab. The builder reads canonical files and does not copy them. Reruns of experiments may update their generated results; they do not edit original source code."),
        small("Original evidence: ../beam_search/emm_beam_search_results.csv, ../network_eda/, ../eda_outputs/figures/ and ../logistic_resgression/. Source references in this report are relative to experiments/."),
    ]})
    save_json("reproducibility/report_content.json", {"date": today.isoformat(), "sections": sections})
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
        ("M. Activity-based reference subgroup discovery", m_text + " " + f"The quality score was sqrt(n) times the absolute coefficient difference from the reference. {ci_excludes}/{len(all_m)} subgroup intervals excluded the reference point, which is reported as a descriptive diagnostic rather than a formal test of the subgroup-reference difference. Subgroup definitions used activity variables also included as controls."),
        ("A1. Reciprocity operationalization", a1_text + " " + "The weighted variant was not applicable: an exhaustive audit of the supplied raw networks found no repeated targets in any outgoing user-month list, so repeated-reply intensity was unavailable. Coefficients under the two measures refer to different contrasts and should not be interpreted as directly comparable probability changes."),
        ("A2. Dependence between observations", a2_text + " " + limitation),
        ("A3. Probability-scale comparison", a3_text + " " + f"The most negative AME was {strongest_ame.ame_pp:.2f} percentage points for {describe(strongest_ame.description)}, {'which is also' if strongest_ame.description == most_negative_beta.description else 'rather than for'} the top-ten subgroup with the most negative coefficient. AMEs were evaluated over each model's own observations, and no AME confidence intervals were estimated."),
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
    ax.set(xlabel="Reference model and original M quality rank", ylabel="Increase in standard error (%)", ylim=(0, max(16, 1.2 * a2.se_change_pct.max())))
    ax.grid(axis="y", color="#e2e7eb")
    ax.set_axisbelow(True)
    fig.tight_layout(pad=1.8)
    for ext in ("png", "svg"):
        fig.savefig(directory / f"clustered_se_change.{ext}", dpi=180)
    plt.close(fig)


def readme():
    write("README.txt", """EXPERIMENTS - START HERE
========================
manuscript_results.txt and .tex contain results prose for M and A1-A3.
FINAL_REPORT.pdf and FINAL_REPORT.txt are not stored in the repository; build
them with the two commands under REBUILD REPORTS below.

SINGLE CANONICAL LOCATIONS
- M_reference_discovery/: added top-ten reporter and its results.
- A1_reciprocity_measure/: comparison/audit code, results and model checkpoints.
- A2_user_dependence/: dependence-analysis code, results and checkpoints.
- A3_probability_scale/: AME-analysis code, results and checkpoints.
- code/: launcher, shared paths and report builders.
- figures/: final-report PNG/SVG figures (different output formats).
- reproducibility/: analysis environment, data dictionary, report content.
- supporting_analyses/README.txt: pointers to existing foundation analyses.

The M search code lives in beam_search/ and now discretizes locally (see
../readme.md); network_eda/, reddit_eda/ and logistic_resgression/ are
unchanged. The full M results are at
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

DEPENDENCIES AND INPUTS
- Analysis environment: reproducibility/environment.json.
- Data: ../eda_outputs/reciprocity_return.csv.
- Raw networks for weighted audit: ../reply_networks/*.json.
- A1/A2 cache: A1_reciprocity_measure/data_cache/prepared.parquet (one copy).
- A1 can rebuild that cache from the analysis CSV; A2 requires its audit.
- A1 imports the beam search from ../beam_search/ and checkpoints every fit.
- A1/A3 share the M reporter; all added scripts share code/experiment_paths.py.

REBUILD REPORTS WITHOUT MODEL FITS
python experiments/code/build_bundle.py
python experiments/code/render_report.py
The first uses pandas/numpy/matplotlib; the second needs reportlab. The builder
reads canonical results directly and does not recreate duplicate archives.
Inspect rendered PDF pages after changing report content.

COMPLETED SCOPE
M: activity-only binary search with local discretization; top-ten reporting.
A1: separate proportional search compared with the binary one; weighted
    variant unavailable after full audit.
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

PDF/text, CSV/text/LaTeX, and PNG/SVG files are intentional alternative formats.
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

The final report reads the original evidence directly.
""")


def main():
    BASE.mkdir(exist_ok=True)
    generate_report()
    readme()
    environment = {"python": sys.version, "platform": platform.platform(), "analysis_executable": sys.executable,
                   "packages": {p: importlib.metadata.version(p) for p in ("pyfixest", "pandas", "numpy", "scipy", "pyarrow", "matplotlib")}}
    save_json("reproducibility/environment.json", environment)
    print("Report content, READMEs and manuscript text rebuilt; no model fits.")


if __name__ == "__main__":
    main()
