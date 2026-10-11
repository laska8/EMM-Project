"""A2: ordinary versus user-clustered inference for M's fixed top ten.

Run: python experiments/A2_user_dependence/code/run_dependence_analysis.py
Reads existing inputs; writes only experiments/A2_user_dependence/. Original code is unchanged.
Uses the validated A1 parquet cache. Each model runs in a fresh process.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

for variable in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

# Import the shared paths from this project's experiments directory.
EXPERIMENTS_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "experiments")
sys.path.insert(0, str(EXPERIMENTS_DIR / "code"))
from experiment_paths import ROOT, ORIGINAL_SEARCH, M, A1, A2, A3, output_path

import numpy as np
import pandas as pd

BASE = ORIGINAL_SEARCH
ROOT = BASE.parent
SOURCE = BASE / "emm_beam_search_results.csv"
DATA = ROOT / "eda_outputs" / "reciprocity_return.csv"
CACHE = A1 / "data_cache" / "prepared.parquet"
OUT = A2
FORMULA = "returned_next_month ~ recip_binary + log_sent + log_received | month + subreddit"
TOP_K = 10
N = 36_031_393


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def selected():
    return pd.read_csv(SOURCE).sort_values("quality", ascending=False, kind="stable").head(TOP_K)


def validate_cache():
    expected = {"data_size": DATA.stat().st_size, "data_mtime_ns": DATA.stat().st_mtime_ns,
                "total_observations": N, "cache_format_version": 1}
    if not CACHE.exists() or json.loads(CACHE.with_name("metadata.json").read_text()) != expected:
        raise ValueError("Missing or stale A1 cache; run run_reciprocity_comparison.py first")
    audit = json.loads(output_path(A1, "data_audit.json").read_text())
    if not audit["definition_verified_all_rows"]:
        raise ValueError("Reciprocity definition has not been verified")
    return expected


def signature():
    record = {**validate_cache(), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "path_helper_sha256": hashlib.sha256((EXPERIMENTS_DIR / "code" / "experiment_paths.py").read_bytes()).hexdigest(),
              "cache_size": CACHE.stat().st_size, "cache_mtime_ns": CACHE.stat().st_mtime_ns,
              "pyfixest": importlib.metadata.version("pyfixest"), "formula": FORMULA, "top_k": TOP_K}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def filters(description):
    if description == "reference":
        return None
    result = []
    for condition in description.split(" AND "):
        match = re.fullmatch(r"(replies_sent|replies_received)(<=|>|==)([0-9.]+)", condition)
        if match is None:
            raise ValueError(f"Unsupported selector: {condition}")
        column, operator, threshold = match.groups()
        result.append((column, operator, float(threshold)))
    return result


def extract(model):
    row = model.tidy().loc["recip_binary"]
    result = {name: float(row[column]) for name, column in {
        "beta": "Estimate", "se": "Std. Error", "ci_low": "2.5%",
        "ci_high": "97.5%", "p_value_beta_zero": "Pr(>|t|)"}.items()}
    if not np.isfinite(list(result.values())).all() or result["se"] <= 0:
        raise ValueError("Invalid coefficient or uncertainty estimate")
    return result


def fit_one(key, run_signature):
    import pyfixest as pf

    started = time.monotonic()
    top = selected()
    row = None if key == "reference" else top.iloc[int(key) - 1]
    description = "reference" if row is None else row["description"]
    expected_n = N if row is None else int(row["n"])
    expected_beta = float(top.iloc[0]["reference_beta"] if row is None else row["beta"])
    print(f"[{key}] Loading {description}", flush=True)
    columns = ["user", "subreddit", "month", "returned_next_month",
               "proportional_reciprocity", "log_sent", "log_received"]
    df = pd.read_parquet(CACHE, columns=columns, filters=filters(description))
    for column in ("user", "subreddit"):
        df[column] = df[column].cat.remove_unused_categories()
    # A1 verified proportional = reciprocal_partners / positive outgoing_partners
    # for every row, so >0 exactly recovers M's binary reciprocity.
    df["recip_binary"] = (df.pop("proportional_reciprocity") > 0).astype("int8")
    if len(df) != expected_n or df.isna().any().any() or not df["month"].between(1, 10).all():
        raise ValueError("Sample differs from M")
    return_rate = float(df["returned_next_month"].mean())
    if row is not None and not np.isclose(return_rate, row["return_rate"], rtol=0, atol=1e-12):
        raise ValueError("Return rate differs from M")
    user_clusters = int(df["user"].nunique())
    gc.collect()
    print(f"[{key}] Fitting {len(df):,} observations, {user_clusters:,} users", flush=True)
    model = pf.feglm(FORMULA, data=df, family="logit", vcov={"CRV1": "user"},
                     copy_data=False, fixef_rm="none", iwls_maxiter=100)
    if not model.convergence or len(model.na_index):
        raise ValueError("Model failed to converge or excluded observations")
    clustered = extract(model)
    if not np.isclose(clustered["beta"], expected_beta, rtol=0, atol=1e-7):
        raise ValueError("Coefficient does not reproduce M")
    if row is not None:
        for field in ("se", "ci_low", "ci_high"):
            if not np.isclose(clustered[field], row[field], rtol=0, atol=1e-8):
                raise ValueError(f"Clustered {field} does not reproduce M")
    else:
        # Pooled notebook displays six decimal places; validate at that precision.
        for field, saved in {"se": 0.001146, "ci_low": -0.243514, "ci_high": -0.239020}.items():
            if abs(clustered[field] - saved) > 0.00000051:
                raise ValueError(f"Reference clustered {field} differs from the pooled notebook")
    # Reuse the same likelihood fit and fitted sample, changing only covariance.
    # pyfixest.vcov recomputes SEs, confidence intervals and p-values in place.
    model.vcov("iid")
    ordinary = extract(model)
    if ordinary["beta"] != clustered["beta"]:
        raise ValueError("Covariance change unexpectedly altered the coefficient")
    record = {"signature": run_signature, "key": key, "description": description,
              "quality_rank": None if row is None else int(key), "n": len(df),
              "user_clusters": user_clusters, "return_rate": return_rate,
              "converged": True, "excluded_observations": 0,
              "saved_beta": expected_beta, "beta_difference_from_saved": clustered["beta"] - expected_beta,
              "saved_quality": None if row is None else float(row["quality"]),
              "ordinary": ordinary, "clustered": clustered,
              "elapsed_seconds": time.monotonic() - started}
    write_json(output_path(OUT, f"fit_{key}.json"), record)
    print(f"[{key}] Complete: SE ordinary={ordinary['se']:.6f}, clustered={clustered['se']:.6f}, "
          f"ratio={clustered['se']/ordinary['se']:.3f}; {record['elapsed_seconds']:.0f}s", flush=True)


def build_report(run_signature):
    records = [json.loads((output_path(OUT, f"fit_{key}.json")).read_text())
               for key in ["reference"] + [str(i) for i in range(1, TOP_K + 1)]]
    if any(r["signature"] != run_signature for r in records):
        raise ValueError("Checkpoint signature mismatch")
    reference_beta = records[0]["clustered"]["beta"]
    rows = []
    for record in records:
        row = {k: record[k] for k in ("description", "quality_rank", "n", "user_clusters", "return_rate")}
        row["reference_beta"] = reference_beta
        row["beta"] = record["clustered"]["beta"]
        row["signed_delta"] = row["beta"] - reference_beta
        for kind in ("ordinary", "clustered"):
            estimate = record[kind]
            row.update({f"{kind}_{k}": v for k, v in estimate.items()})
            low, high = estimate["ci_low"], estimate["ci_high"]
            row[f"{kind}_ci_width"] = high - low
            row[f"{kind}_ci_excludes_zero"] = bool(low > 0 or high < 0)
            row[f"{kind}_ci_excludes_reference_point"] = bool(low > reference_beta or high < reference_beta)
        row["se_ratio_clustered_over_ordinary"] = row["clustered_se"] / row["ordinary_se"]
        row["se_change_pct"] = 100 * (row["se_ratio_clustered_over_ordinary"] - 1)
        row["ci_width_ratio"] = row["clustered_ci_width"] / row["ordinary_ci_width"]
        row["coefficient_unchanged"] = row["ordinary_beta"] == row["clustered_beta"]
        row["saved_quality"] = record["saved_quality"]
        row["recomputed_quality"] = None if record["key"] == "reference" else np.sqrt(row["n"]) * abs(row["signed_delta"])
        if record["key"] != "reference" and not np.isclose(row["recomputed_quality"], row["saved_quality"], rtol=0, atol=1e-4):
            raise ValueError("Quality no longer reproduces M")
        rows.append(row)
    comparison = pd.DataFrame(rows)
    groups = comparison.iloc[1:]
    summary = {
        "top_k": TOP_K, "models": len(records), "all_converged": all(r["converged"] for r in records),
        "all_coefficients_unchanged": bool(comparison["coefficient_unchanged"].all()),
        "all_coefficients_negative": bool((comparison["beta"] < 0).all()),
        "reference_se_ratio": float(comparison.iloc[0]["se_ratio_clustered_over_ordinary"]),
        "subgroup_se_ratio_min": float(groups["se_ratio_clustered_over_ordinary"].min()),
        "subgroup_se_ratio_median": float(groups["se_ratio_clustered_over_ordinary"].median()),
        "subgroup_se_ratio_max": float(groups["se_ratio_clustered_over_ordinary"].max()),
        "subgroups_with_larger_clustered_se": int((groups["se_change_pct"] > 0).sum()),
        "ordinary_subgroup_cis_exclude_zero": int(groups["ordinary_ci_excludes_zero"].sum()),
        "clustered_subgroup_cis_exclude_zero": int(groups["clustered_ci_excludes_zero"].sum()),
        "ordinary_subgroup_cis_exclude_reference_point": int(groups["ordinary_ci_excludes_reference_point"].sum()),
        "clustered_subgroup_cis_exclude_reference_point": int(groups["clustered_ci_excludes_reference_point"].sum()),
        "reference_point_diagnostic_changed_count": int((groups["ordinary_ci_excludes_reference_point"] != groups["clustered_ci_excludes_reference_point"]).sum()),
        "max_abs_beta_difference_from_saved": max(abs(r["beta_difference_from_saved"]) for r in records),
        "quality_ranking_unchanged": bool(groups["recomputed_quality"].is_monotonic_decreasing),
        "scope": "Fixed binary top ten and global reference, same controls and months 1-10; no new search.",
        "inference": "Ordinary model-based iid versus user-clustered CRV1, with pyfixest default small-sample corrections.",
        "limitation": "CI exclusion of the reference point is descriptive, not a formal test of subgroup-reference difference. The reference is estimated and overlaps each subgroup; joint covariance and post-selection adjustment are not estimated.",
    }
    comparison.to_csv(output_path(OUT, "a2_comparison.csv"), index=False)
    write_json(output_path(OUT, "summary.json"), summary)
    table = pd.DataFrame({
        "M rank": ["Ref"] + list(range(1, TOP_K + 1)),
        "Subgroup": comparison["description"].str.replace("replies_sent", "Sent").str.replace("replies_received", "Received").str.replace(".0", "", regex=False),
        "Beta": comparison["beta"].map(lambda x: f"{x:.4f}"),
        "Ordinary SE": comparison["ordinary_se"].map(lambda x: f"{x:.6f}"),
        "Clustered SE": comparison["clustered_se"].map(lambda x: f"{x:.6f}"),
        "SE ratio": comparison["se_ratio_clustered_over_ordinary"].map(lambda x: f"{x:.2f}"),
        "Ordinary 95% CI": [f"[{r.ordinary_ci_low:.4f}, {r.ordinary_ci_high:.4f}]" for r in comparison.itertuples()],
        "Clustered 95% CI": [f"[{r.clustered_ci_low:.4f}, {r.clustered_ci_high:.4f}]" for r in comparison.itertuples()],
    })
    notes = (
        "A2: DEPENDENCE BETWEEN OBSERVATIONS\n\n" + table.to_string(index=False) + "\n\n"
        f"Reference SE ratio (clustered/ordinary): {summary['reference_se_ratio']:.3f}.\n"
        f"Subgroup SE ratios: {summary['subgroup_se_ratio_min']:.3f} to {summary['subgroup_se_ratio_max']:.3f}; "
        f"median {summary['subgroup_se_ratio_median']:.3f}.\n"
        f"Clustered SE is larger in {summary['subgroups_with_larger_clustered_se']}/{TOP_K} subgroups.\n"
        f"Subgroup CIs exclude zero: ordinary {summary['ordinary_subgroup_cis_exclude_zero']}/{TOP_K}; "
        f"clustered {summary['clustered_subgroup_cis_exclude_zero']}/{TOP_K}.\n"
        f"Subgroup CIs exclude reference point: ordinary {summary['ordinary_subgroup_cis_exclude_reference_point']}/{TOP_K}; "
        f"clustered {summary['clustered_subgroup_cis_exclude_reference_point']}/{TOP_K}.\n"
        "Coefficients, directions, AME point estimates and the size-weighted quality ranking are unaffected by covariance choice.\n"
        "M already used user-clustered standard errors; A2 adds the ordinary-error comparison.\n"
        "Ordinary SEs assume independent observations and the model-based variance. CRV1 allows within-user dependence.\n"
        "The comparison does not isolate dependence from all possible model misspecification.\n"
        "These are pointwise 95% coefficient CIs; p-values test beta=0, not beta=reference_beta.\n"
        + summary["limitation"] + "\n"
        "Use clustered uncertainty for interpretation. These associations are descriptive, not causal.\n\n"
        "Run: python experiments/A2_user_dependence/code/run_dependence_analysis.py\n"
        "Completed fits are checkpointed. Inputs are read only; output is confined to experiments/A2_user_dependence/.\n"
        "The validated A1 parquet cache is required. Binary reciprocity is recovered as proportional reciprocity > 0.\n"
        "Each likelihood is fitted once with clustered inference; pyfixest.vcov('iid') recalculates ordinary inference on the same fit.\n"
    )
    (output_path(OUT, "a2_report.txt")).write_text(notes, encoding="utf-8")
    print(notes, flush=True)


def protected_files():
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in tracked if Path(name).suffix in (".py", ".ipynb", ".csv", ".md")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-key", help=argparse.SUPPRESS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run_signature = signature()
    if args.fit_key:
        fit_one(args.fit_key, run_signature)
        return
    before = protected_files()
    write_json(output_path(OUT, "protected_files_before.json"), before)
    try:
        for key in [str(i) for i in range(1, TOP_K + 1)] + ["reference"]:
            checkpoint = output_path(OUT, f"fit_{key}.json")
            if checkpoint.exists() and json.loads(checkpoint.read_text()).get("signature") == run_signature:
                print(f"[{key}] Using completed checkpoint", flush=True)
                continue
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--fit-key", key], check=True)
        build_report(run_signature)
    finally:
        after = protected_files()
        changed = [name for name in set(before) | set(after) if before.get(name) != after.get(name)]
        write_json(output_path(OUT, "source_integrity.json"), {"unchanged": not changed, "changed_files": changed, "before": before, "after": after})
        if changed:
            raise RuntimeError(f"Protected files changed during A2: {changed}")


if __name__ == "__main__":
    main()
