# EMM Project

Project on Exceptional Model Mining for Reddit interaction networks.

## Dataset

We use the SNAP Reddit Interaction Networks dataset.

Download the reply-based interaction networks and extract them into:

reply_networks/

The dataset folder is ignored by Git and is not stored in this repository.

## Current files

- `data_parsing.py` - initial data loading and parsing
- `datachecks.py` - basic checks of the dataset structure
- `eda.py` - initial exploratory data analysis

## Setup

Install required packages:

pip install pandas numpy matplotlib networkx pyarrow jinja2 pyfixest

# Current work

temporal_eda.py
Runs temporal EDA on all subreddit JSON files and saves subreddit-level summaries.

plot_temporal_results.py
Loads precomputed summaries and generates temporal EDA figures.

temporal_eda_summary_all_subreddits.csv
Final subreddit-level temporal EDA results.

## reddit_eda

reddit_eda.ipynb
EDA notebook (user activity, return, community networks, reciprocity). Only loads datasets, prints summaries and plots.

build_eda_datasets.py
Builds all EDA datasets from the subreddit JSON files and saves them to eda_outputs/.

reply_network_analyzer.py
Computes activity, return, community and reciprocity features for one subreddit JSON file.

eda_datasets.py
Builds each dataset or loads it from eda_outputs/ if it already exists.

eda_summaries.py
Grouping and summary tables used in the notebook.

eda_plots.py
Plots used in the notebook. Paper figures are saved to eda_outputs/figures/.

## beam_search

run_beam_search.py
Runs the beam search on eda_outputs/reciprocity_return.csv and saves the subgroups to emm_beam_search_results.csv.

beam_search.py
Beam search for Exceptional Model Mining (BeamSearchEMM).

emm_selectors.py
Subgroup conditions on one column (e.g. replies_sent > 3).

logistic_model.py
The model fitted in every subgroup: the same logit as pooled_logistic.ipynb (month and subreddit fixed effects, standard errors clustered by user).

emm_beam_search_results.csv
All subgroups from the full-data run, sorted by quality.

report_top_subgroups.py
Generates a top-10 report from the saved results without rerunning the models:

```bash
python experiments/M_reference_discovery/code/added/report_top_subgroups.py
```

Outputs are saved in `experiments/M_reference_discovery/`: a CSV retaining numeric precision,
a readable `.txt` table, and a `.tex` table for the paper (requires `booktabs`).
The report includes coverage and signed coefficient differences. The original
results' `delta` column contains absolute differences. Change `TOP_K` in the
script to change the reporting cutoff; `TOTAL_OBSERVATIONS` describes the
current full-data run and must match the source run if that changes.

run_ame_analysis.py
Completes the A3 effect-scale comparison for the fixed top 10 from M:

```bash
python experiments/A3_probability_scale/code/run_ame_analysis.py
python experiments/M_reference_discovery/code/added/report_top_subgroups.py
```

The first command refits each selected subgroup and the global reference,
checks the fitted coefficients against the saved search, and calculates the
average discrete change in return probability when binary reciprocity is set
from 0 to 1. Other inputs, including fixed effects, stay fixed for each observation.
Each AME is averaged over that model's own population and reported in percentage
points. These are adjusted associations, not causal effects.

Fits run sequentially in separate processes and save checkpoints under
`experiments/A3_probability_scale/`; rerunning resumes matching completed fits. The refits
use the original likelihood and model specification. User clustering is omitted
for these point-estimate calculations to reduce memory use; it does not affect
coefficients or fitted probabilities. The original clustered coefficient CIs
remain in the reports. AME confidence intervals are not estimated.

`a3_report.txt` and `a3_table.tex` report the probability-scale comparisons;
`top_10_ame_comparison.csv` retains numeric results and `summary.json` records
rank correlation and direction agreement. Rankings compare absolute subgroup
differences from the global reference on each scale, without size weighting.
They differ from M's original quality ranking, which includes subgroup size.
Once A3 is complete, the second command adds AMEs to the main top-10 report.

### Main results

Setup of the run:

- 36,031,393 user-month observations from all 2,046 subreddits
- target: the coefficient of recip_binary (at least one reciprocal partner) on next-month return, controlling for log replies sent and log replies received
- subgroups: conditions on replies_sent (thresholds 1 and 3) and replies_received (thresholds 0, 1 and 3), up to 2 conditions, beam width 10, at least 5,000 observations
- quality = sqrt(n) * |beta_subgroup - beta_reference|

Reference model (all observations): beta = -0.241 (SE 0.001). Holding activity fixed, users with a reciprocal partner are less likely to return.

Top subgroups by quality:

| Subgroup | n | Return rate | beta | SE | Difference from reference | Quality |
|---|---|---|---|---|---|---|
| replies_sent > 3 AND replies_received > 3 | 5,056,348 | 75.8% | -0.522 | 0.008 | 0.281 | 632 |
| replies_sent > 1 AND replies_received > 1 | 11,289,290 | 59.6% | -0.429 | 0.002 | 0.187 | 629 |
| replies_sent > 3 AND replies_received > 1 | 6,448,956 | 72.4% | -0.487 | 0.005 | 0.246 | 625 |
| replies_sent > 1 AND replies_received > 0 | 14,380,085 | 55.4% | -0.400 | 0.002 | 0.159 | 602 |
| replies_sent > 3 AND replies_received > 0 | 6,871,536 | 71.4% | -0.442 | 0.004 | 0.201 | 526 |
| replies_sent > 1 | 16,170,924 | 53.7% | -0.362 | 0.002 | 0.120 | 484 |

What the search shows:

- 29 subgroups were evaluated (9 with one condition, 20 with two). The reciprocity coefficient is negative in all of them, between -0.177 and -0.522.
- The negative effect is strongest for active users. It roughly doubles for users who sent and received more than 3 replies (-0.522 against -0.241).
- For users who sent a single reply (19.9 million observations, 55% of the data) the coefficient is the same as in the reference model (-0.239).
- The effect is weakest for users who sent one reply but received more than 3 (-0.177, 642,328 observations).

### A3: Effect-scale comparison (completed for the top 10)

The refitted reference AME is -4.4366 percentage points, reproducing the pooled
notebook's rounded -4.44 result. Across the ten selected subgroups, AMEs range
from -9.0109 to -5.5867 percentage points. All ten subgroup AMEs are more
negative than the reference: the signed subgroup-reference differences agree
in direction on the coefficient and probability scales in 10/10 cases.

The rankings differ substantially. Spearman correlation between rankings by
absolute coefficient difference and absolute AME difference is 0.3697. These
comparisons use unrounded values and do not include M's size weighting.
The subgroup with more than three replies sent and received ranks first by
coefficient difference but fifth by AME difference (AME -7.5527 points).
The largest absolute AME difference belongs to observations with more than one
reply sent and at most three received (AME -9.0109 points; difference from
reference -4.5743 points).

Thus, the direction of the differences is consistent across scales, but the
coefficient ranking should not be treated as a ranking of probability-scale
differences. Use AMEs when discussing the size of differences in predicted
return probability. Each AME averages over its own fitted population, so the
comparison reflects both model estimates and population composition. No AME
confidence intervals or post-selection uncertainty are estimated.

All eleven fits (ten subgroups and the reference) converged, retained the
expected observation counts, and reproduced the saved reciprocity coefficients
to within 4e-12. See `experiments/A3_probability_scale/reports/a3_report.txt` for the full table.

### A1: Reciprocity-measure comparison

Run the proportional-reciprocity search and compare it with the saved binary M:

```bash
python experiments/A1_reciprocity_measure/code/run_reciprocity_comparison.py
```

This uses the original beam-search expansion and pruning code, the same
36,031,393 observations, fixed activity thresholds, controls, fixed effects,
user-clustered standard errors, and quality function. It fits a separate global
reference using proportional reciprocity (reciprocal partners / outgoing
partners). Binary results are not overwritten. Fits run in separate processes
and are checkpointed, so rerunning resumes matching completed fits.

Outputs are under `experiments/A1_reciprocity_measure/`:
`a1_report.txt` is the readable report; `proportional_beam_search_results.csv`
contains all valid proportional subgroups; `top_10_comparison.csv` compares
subgroups in either top-ten list; and `summary.json` contains Jaccard overlap
and sign/difference-direction agreement. LaTeX tables are also exported.
The generated `data_cache/` is ignored by Git and can be rebuilt from the CSV.

Subgroup conditions are normalized before comparison, so reversed condition
order does not create false nonmatches. Signed differences use each measure's
own reference coefficient. Coefficients represent different contrasts and are
not directly comparable probability effects.

Some small subgroups contain fixed-effect levels that the estimator excludes
because of separation. For each affected subgroup, the runner refits the binary
model, verifies identical excluded row positions, and checks its coefficient
and clustered SE against M. `n` remains the input subgroup size used by M's
quality function; `fitted_rows` and `excluded_for_separation` record the actual
estimation sample. Details are saved in `search_audit.json` and fit checkpoints.

Check whether repeated replies are available for the optional weighted variant:

```bash
python experiments/A1_reciprocity_measure/code/audit_repeated_replies.py
```

This scans every supplied raw network and saves `weighted_feasibility.json` in
the same results directory. It checks whether an outgoing user-month list
contains the same target more than once.

The completed audit checked all 2,046 files (22,506 monthly networks,
40,050,308 outgoing user-month lists, and 136,829,959 target entries). It found
zero repeated targets within an outgoing list. Thus the supplied networks do
not retain repeated-reply intensity for a distinct weighted-reciprocity test.
The conditional weighted variant of A1 is not applicable to these supplied
data; this does not imply that users never exchanged repeated replies in the
underlying Reddit activity.

#### Completed A1 results

The proportional reference coefficient is -0.372847 (user-clustered SE
0.001347; 95% CI [-0.375487, -0.370207]). The search returned the same 29 valid
subgroup descriptions as M, with negative coefficients in all 29. Thirty
proportional models converged: the reference plus those 29 subgroups. Five
additional candidate descriptions were skipped because reciprocity did not vary.

The binary and proportional top-ten lists share nine descriptions, giving a
Jaccard overlap of 9/11 = 0.8182. All nine shared top-ten subgroups have negative
coefficients and more negative coefficients than their respective global
references. Thus coefficient signs and subgroup-reference difference directions
agree in all nine shared top-ten cases.

Their order changes. `replies_sent > 3 AND replies_received > 0` moves from
binary rank 5 to proportional rank 1. The binary winner, `replies_sent > 3 AND
replies_received > 3`, ranks fourth under proportional reciprocity.

| Top-ten membership change | Binary rank | Proportional rank |
|---|---:|---:|
| Enters: replies_sent > 3 AND replies_received <= 3 | 20 | 9 |
| Leaves: replies_sent <= 3 AND replies_received <= 3 | 10 | 23 |

Across all 29 common subgroups, coefficient signs agree in 29/29 cases, but
directions of differences from each measure's reference agree in 21/29. The
main top-ten contexts and association signs are therefore largely stable, while
rankings and some reference-relative conclusions depend on the reciprocity
definition. Coefficient magnitudes represent different contrasts and should
not be interpreted as directly comparable changes in return probability.

Sample verification found six subgroups with separation exclusions. Each was
checked against a binary refit: the excluded row positions matched exactly,
and the original binary coefficient and clustered SE were reproduced. The
original binary results were not modified. All six one-reply subgroup models,
where binary and proportional reciprocity coincide, also reproduced their
binary coefficients, SEs and confidence intervals.

## Consolidated experiments

Added experiment code and generated results now live only in `experiments/`.
Original search, EDA and regression code stays in its existing location.
See [the experiment guide](experiments/README.txt) for all commands and paths.
The A1/A2 data cache has moved to `experiments/A1_reciprocity_measure/data_cache/`.
The report builder reads canonical files directly and does not recreate duplicate copies.
