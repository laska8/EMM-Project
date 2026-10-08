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
