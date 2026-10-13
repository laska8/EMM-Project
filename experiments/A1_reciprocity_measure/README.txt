A1: binary versus proportional reciprocity

The beam search was repeated with proportional reciprocity as the target, with the same data, model, local discretization and settings. It returned 409 valid subgroups (409 with a negative coefficient), against 347 for binary reciprocity; 147 subgroups were visited by both searches. The two top-ten lists share 0 of 20 distinct descriptions (Jaccard overlap 0.000). Because thresholds are recomputed in every refined subgroup, the two searches follow different paths, so identical descriptions are a strict criterion. By observations, 95% of those selected by the proportional top ten are also selected by the binary top ten (Jaccard overlap of the two selections 0.442). Each subgroup in either top-ten list was also estimated under the other measure: of the 20 that could be estimated under both, the coefficient sign agreed in 20 and the direction of the subgroup-reference difference in 20. Among all 147 subgroups visited by both searches, the direction of the difference agreed in 136. Weighted reciprocity is not applicable to the supplied adjacency data; see audits/weighted_feasibility.json.

code/: runnable added scripts. tables/: full numeric CSV outputs. reports/: readable results and JSON summaries.
model_checkpoints/: individual saved fits where available. audits/: validation and provenance where available.
latex/: existing exported tables, not compiled in the target manuscript.

Use ../README.txt for rerun commands. Original M implementation and full results remain in ../../beam_search/.
