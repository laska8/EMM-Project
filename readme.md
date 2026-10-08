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
