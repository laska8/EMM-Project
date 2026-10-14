# Subgroup Discovery for Exceptional Behavior in Networked Data

Code and results for our Exceptional Model Mining (EMM) study of Reddit reply
networks. We ask **in which groups of users the link between reciprocity and
coming back to a community differs from the overall pattern**.

For every user who replies in a subreddit in a given month, we model whether
they reply in that subreddit again next month. The predictor of interest is
*reciprocity*: did at least one person they replied to also reply to them? We
fit one logistic regression on all data (the reference model), then use beam
search to find activity-defined subgroups where the reciprocity coefficient is
most different from the reference.

---

## Contents

- [Repository layout](#repository-layout)
- [Data](#data)
- [Setup](#setup)
- [Running the pipeline](#running-the-pipeline)
- [Where each paper result comes from](#where-each-paper-result-comes-from)
- [Model and search settings](#model-and-search-settings)
- [Robustness experiments (A1 to A3)](#robustness-experiments-a1-to-a3)
- [Authors](#authors)

---

## Repository layout

```
EMM-Project/
├── reply_networks/          raw data, one JSON file per subreddit (not in Git)
├── network_eda/             temporal EDA: activity months, return rates, return windows
├── reddit_eda/              builds the analysis datasets; reciprocity EDA notebook
├── eda_outputs/             generated datasets (CSV, not in Git) and paper figures
│   └── figures/             descriptive table, reciprocity figures (in Git)
├── logistic_resgression/    pooled baseline and per-subreddit logistic regressions
├── beam_search/             the EMM beam search (main experiment, "M")
└── experiments/             robustness experiments A1–A3, launcher, report builder
```

## Data

We use the monthly directed **Reddit reply networks** released with
Hamilton et al., *Loyalty in Online Communities*, ICWSM 2017
([doi:10.1609/icwsm.v11i1.14972](https://doi.org/10.1609/icwsm.v11i1.14972)),
distributed through the Stanford SNAP data collection.

- 2,046 subreddits, 11 consecutive monthly networks from 2014.
- One JSON file per subreddit: a list of months, each a dict mapping a user to
  the list of users they replied to.
- Download the reply networks and extract them so the files sit directly in
  `reply_networks/*.json` at the project root. The folder is git-ignored.

**Unit of analysis.** One observation is a user–subreddit–month in which the
user wrote at least one reply. Next-month return is defined for months 1–10,
giving **36,031,393 observations**.

| Variable | Meaning |
|---|---|
| `replies_sent` / `replies_received` | Out-degree / in-degree in that month's reply network (users replied to / replied by) |
| `reciprocal_partners` | Users who were both replied to and replied by |
| `recip_binary` | 1 if `reciprocal_partners > 0` (main measure) |
| `proportional_reciprocity` | `reciprocal_partners / outgoing_partners` (used in A1) |
| `returned_next_month` | 1 if the user replies in the same subreddit next month |

Column definitions for every result file are in
[`experiments/reproducibility/DATA_DICTIONARY.txt`](experiments/reproducibility/DATA_DICTIONARY.txt).

---

## Setup

Python 3.12. The analysis was run with the versions in
[`experiments/reproducibility/environment.json`](experiments/reproducibility/environment.json).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Hardware.** The full dataset is 36M rows. Building `eda_outputs/` produces
several GB of CSV, and each subgroup model is a fixed-effects logit on millions
of rows, so the beam search and A1–A3 need plenty of RAM and take 1.5 to 2 hours.
A1–A3 checkpoint every fit and resume where they stopped.

---

## Running the pipeline

Run everything from the project root unless noted. Steps 1–2 need the raw
data; every later step only needs `eda_outputs/reciprocity_return.csv`.

**1. Build the analysis datasets** (reads `reply_networks/`, writes `eda_outputs/*.csv`)

```bash
python reddit_eda/build_eda_datasets.py
```

Datasets that already exist are skipped. If the build is interrupted and the
merged files are incomplete, `logistic_resgression/repair.ipynb` rebuilds
`reciprocity_return.csv` from `user_activity.csv`, `user_reciprocity.csv` and
`user_return.csv` (set `FOLDER` to your `eda_outputs/` path first).

**2. Temporal EDA** (these scripts read and write in their own folder)

```bash
cd network_eda
python temporal_eda.py            # → temporal_eda_summary_all_subreddits.csv
python plot_temporal_results.py   # → *.png figures
cd ..
```

**3. Reciprocity EDA and descriptive table:** open `reddit_eda/reddit_eda.ipynb`
and run all cells. Paper figures and tables go to `eda_outputs/figures/`.

**4. Baseline regressions:** run `logistic_resgression/pooled_logistic.ipynb`
and `logistic_resgression/subreddit_logistic.ipynb`. Both notebooks have a
hard-coded `DATA` path near the top; point it at your
`eda_outputs/reciprocity_return.csv`.

**5. EMM beam search (main experiment)**

```bash
python beam_search/run_beam_search.py           # → beam_search/emm_beam_search_results.csv
python beam_search/plot_beam_search_results.py  # → forest plot
python beam_search/run_breadth_check.py         # A4: contact-breadth check
python experiments/code/run_experiment.py M-report --execute   # top-10 txt/csv/tex report
```

**6. Robustness experiments**

```bash
python experiments/code/run_experiment.py A1 --execute   # proportional reciprocity
python experiments/code/run_experiment.py A2 --execute   # ordinary vs clustered SEs
python experiments/code/run_experiment.py A3 --execute   # probability scale (AMEs)
```

Without `--execute` the launcher only prints the command. Re-running replaces
generated results but never edits source code.

**7. Rebuild the summary report without refitting anything**

```bash
python experiments/code/build_bundle.py
python experiments/code/render_report.py   # needs reportlab
```

---

## Where each paper result comes from

| Paper item | Produced by | Output in this repo |
|---|---|---|
| Fig. *Temporal participation and return* (3 panels) | `network_eda/temporal_eda.py`, `plot_temporal_results.py` | `network_eda/temporal_pattern_distribution.png`, `return_rate_distribution.png`, `return_window_gain.png` |
| Table *Descriptive statistics* | `reddit_eda/reddit_eda.ipynb` (`EdaSummaries.descriptive_table`) | `eda_outputs/figures/descriptive_table.csv` |
| Fig. *Next-month return by proportional reciprocity* | `reddit_eda/reddit_eda.ipynb` → `eda_plots.py` | `eda_outputs/figures/fig_reciprocity_return.png` |
| Fig. *Distribution across subreddits* | same | `eda_outputs/figures/fig_subreddit_spread.png` |
| Fig. *Activity composition of each reciprocity group* | same | `eda_outputs/figures/fig_reciprocity_composition.png` |
| Table *Pooled baseline logistic regression* | `logistic_resgression/pooled_logistic.ipynb` | saved notebook outputs |
| Table *Selected subreddit-level regression results* | `logistic_resgression/subreddit_logistic.ipynb` | `logistic_resgression/results/subreddit_logistic_results.csv` |
| Results § *Exceptional model mining* | `beam_search/` | `beam_search/emm_beam_search_results.csv`, top-10 in `experiments/M_reference_discovery/` (see [Known issues](#known-issues)) |
| Protocol § *Family A: Model Robustness* | `experiments/A1_*`, `A2_*`, `A3_*` | `reports/*.txt`, `tables/*.csv`, `latex/*.tex` in each folder; prose in `experiments/manuscript_results.txt` |

---

## Model and search settings

**Model fitted in the full data and in every subgroup** (`beam_search/logistic_model.py`):

```
returned_next_month ~ recip_binary + log(1 + replies_sent) + log(1 + replies_received)
                      | month + subreddit fixed effects,   SEs clustered by user
```

estimated with [`pyfixest`](https://github.com/py-econometrics/pyfixest).

**Quality of a subgroup S**: `q(S) = sqrt(|S|) · |β_S − β_reference|`, with
selectors on `replies_sent` and `replies_received` only (reciprocity itself is
never a selector).

Settings in the current `run_beam_search.py`: beam width 10, maximum depth 3,
at least 5,000 observations per subgroup, at least 10 observations in each
reciprocity × return cell (avoids separated logits), and cut points
re-computed inside each subgroup being refined (8 bins).

---

## Robustness experiments (A1 to A3)

| | Experiment | Result (top 10 subgroups) |
|---|---|---|
| **M** | Main search | 347 valid subgroups, all with a negative reciprocity coefficient (−1.150 to −0.216, reference −0.241). Top subgroup: sent > 1, received > 1, sent ≤ 14 (n = 10.0M, β = −0.449, q = 657.1). |
| **A1** | Does the result depend on how reciprocity is measured? | Proportional reciprocity gives the same signs and difference directions for all 20 top-ten subgroups; the exact descriptions differ because thresholds are re-computed along different search paths. |
| **A2** | Do repeated observations of the same user matter? | User clustering raises SEs by 13% for the reference and 2–5% for subgroups; every interval stays below zero. |
| **A3** | Do differences look the same on the probability scale? | Reference AME −4.44 pp; subgroup AMEs −10.14 to −7.25 pp. Directions agree 10/10, but the ranking by size changes. |

Full numbers: [`experiments/manuscript_results.txt`](experiments/manuscript_results.txt).
Detailed commands, inputs and limits: [`experiments/README.txt`](experiments/README.txt).

---

---

## Authors

Asmita Chakraborty, Lamya Gandhi, Mick van Gelder, Taha Koç,
Valerija Krumplevskaja.
