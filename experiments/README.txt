EXPERIMENTS - START HERE
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
