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

pip install pandas networkx