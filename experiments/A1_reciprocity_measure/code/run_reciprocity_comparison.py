"""A1: repeat M with proportional reciprocity and compare the top-k lists.

Run: python experiments/A1_reciprocity_measure/code/run_reciprocity_comparison.py
Uses the original beam-search expansion/pruning code and fixed M thresholds.
The original binary results are read only. Fits are checkpointed for resumption.
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
sys.path.insert(0, str(M / "code" / "added"))
sys.path.insert(0, str(ORIGINAL_SEARCH))

import numpy as np
import pandas as pd

from beam_search import BeamSearchEMM
from emm_selectors import Selector
from report_top_subgroups import readable_description

BASE = ORIGINAL_SEARCH
DATA = BASE.parent / "eda_outputs" / "reciprocity_return.csv"
BINARY = BASE / "emm_beam_search_results.csv"
OUT = A1
CACHE = output_path(OUT, "data_cache")
PARQUET = CACHE / "prepared.parquet"
TARGET = "proportional_reciprocity"
FORMULA = f"returned_next_month ~ {TARGET} + log_sent + log_received | month + subreddit"
TOTAL_OBSERVATIONS = 36_031_393
TOP_K = 10
SETTINGS = {"beam_width": 10, "max_depth": 2, "min_support": 5000, "alpha": 0.5}
THRESHOLDS = {"replies_sent": [1.0, 3.0], "replies_received": [0.0, 1.0, 3.0]}


def conditions(description):
    if description == "reference":
        return []
    parsed = []
    for condition in description.split(" AND "):
        match = re.fullmatch(r"(replies_sent|replies_received)(<=|>|==)([0-9.]+)", condition)
        if match is None:
            raise ValueError(f"Unsupported condition: {condition}")
        column, operator, value = match.groups()
        parsed.append((column, operator, float(value)))
    return sorted(parsed)


def canonical(description):
    return " AND ".join(f"{column}{operator}{value:g}" for column, operator, value in conditions(description)) or "reference"


def cache_metadata():
    return {"data_size": DATA.stat().st_size, "data_mtime_ns": DATA.stat().st_mtime_ns,
            "total_observations": TOTAL_OBSERVATIONS, "cache_format_version": 1}


def run_signature():
    record = {**cache_metadata(), "binary_sha256": hashlib.sha256(BINARY.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "path_helper_sha256": hashlib.sha256((EXPERIMENTS_DIR / "code" / "experiment_paths.py").read_bytes()).hexdigest(),
              "search_sha256": hashlib.sha256((BASE / "beam_search.py").read_bytes()).hexdigest(),
              "pyfixest": importlib.metadata.version("pyfixest"),
              "formula": FORMULA, "settings": SETTINGS, "thresholds": THRESHOLDS}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def write_json(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare_cache():
    print("Preparing compact model data from the full CSV", flush=True)
    dtypes = {"user": "category", "subreddit": "category", "month": "int8",
              "replies_sent": "int32", "replies_received": "int32",
              "reciprocal_partners": "int32", "outgoing_partners": "int32",
              TARGET: "float64", "returned_next_month": "int8"}
    df = pd.read_csv(DATA, usecols=list(dtypes), dtype=dtypes, keep_default_na=False)
    if len(df) != TOTAL_OBSERVATIONS or df.isna().any().any():
        raise ValueError("Unexpected observation count or missing values")
    if set(df["month"].unique()) != set(range(1, 11)):
        raise ValueError("Expected transitions 1-10")
    if not df[TARGET].between(0, 1).all() or not (df["outgoing_partners"] > 0).all():
        raise ValueError("Invalid proportional reciprocity inputs")
    if not np.allclose(df[TARGET], df["reciprocal_partners"] / df["outgoing_partners"], rtol=0, atol=1e-12):
        raise ValueError("Proportional reciprocity does not match its documented definition")
    for column, thresholds in THRESHOLDS.items():
        actual = df[column].quantile([0.25, 0.5, 0.75]).unique().tolist()
        if actual != thresholds:
            raise ValueError(f"M thresholds changed for {column}: {actual}")
    audit = {**cache_metadata(), "users": int(df["user"].nunique()),
             "subreddits": int(df["subreddit"].nunique()),
             "return_rate": float(df["returned_next_month"].mean()),
             "definition": "reciprocal_partners / outgoing_partners",
             "definition_verified_all_rows": True, "thresholds": THRESHOLDS}
    df = df.drop(columns=["reciprocal_partners", "outgoing_partners"])
    df["log_sent"] = np.log1p(df["replies_sent"].astype("float64"))
    df["log_received"] = np.log1p(df["replies_received"].astype("float64"))
    df.to_parquet(PARQUET, index=False, compression="snappy")
    write_json(CACHE / "metadata.json", cache_metadata())
    write_json(output_path(OUT, "data_audit.json"), audit)
    print(f"Prepared {len(df):,} observations; all definitions and thresholds verified", flush=True)


def fit_path(description):
    key = hashlib.sha256(canonical(description).encode()).hexdigest()[:16]
    return output_path(OUT, "fits") / f"{key}.json"


def fit_worker(description):
    import pyfixest as pf

    started = time.monotonic()
    filters = conditions(description) or None
    df = pd.read_parquet(PARQUET, filters=filters)
    for column in ("user", "subreddit"):
        df[column] = df[column].cat.remove_unused_categories()
    record = {"signature": run_signature(), "description": canonical(description), "n": len(df),
              "return_rate": float(df["returned_next_month"].mean()), "status": "pending"}
    if description == "reference" and len(df) != TOTAL_OBSERVATIONS:
        raise ValueError("Reference sample count differs from M")
    binary = pd.read_csv(BINARY)
    matched = binary.loc[binary["description"].map(canonical).eq(canonical(description))]
    if not matched.empty:
        original = matched.iloc[0]
        if len(df) != int(original["n"]) or not np.isclose(record["return_rate"], original["return_rate"], rtol=0, atol=1e-12):
            raise ValueError("Subgroup observations differ from M")
    if len(df) < SETTINGS["min_support"]:
        record.update(status="skipped", reason="Below minimum support")
    elif df["returned_next_month"].nunique() < 2:
        record.update(status="skipped", reason="No outcome variation")
    elif df[TARGET].nunique() < 2:
        record.update(status="skipped", reason="No reciprocity variation")
    else:
        print(f"Fitting {canonical(description)}: {len(df):,} observations", flush=True)
        df.drop(columns=["replies_sent", "replies_received"], inplace=True)
        gc.collect()
        model = pf.feglm(FORMULA, data=df, family="logit", vcov={"CRV1": "user"},
                         copy_data=False, fixef_rm="none", iwls_maxiter=100)
        if not model.convergence:
            raise RuntimeError(f"Fit did not converge: {description}")
        estimates = model.tidy()
        if TARGET not in estimates.index:
            record.update(status="skipped", reason="Reciprocity dropped as collinear")
        else:
            estimate = estimates.loc[TARGET]
            predictions = np.asarray(model.predict(type="link")).ravel()
            excluded = np.sort(np.asarray(model.na_index, dtype="int64"))
            if len(predictions) != record["n"] - len(excluded) or not np.isfinite(predictions).all():
                raise ValueError("Missing predictions or unexplained sample exclusions")
            if len(excluded) != getattr(model, "n_separation_na", 0):
                raise ValueError("Unexpected exclusions other than fixed-effect separation")
            record.update(status="ok", converged=True, fitted_rows=len(predictions),
                          excluded_for_separation=len(excluded),
                          beta=float(estimate["Estimate"]), se=float(estimate["Std. Error"]),
                          ci_low=float(estimate["2.5%"]), ci_high=float(estimate["97.5%"]),
                          p_value=float(estimate["Pr(>|t|)"]), user_clusters=int(df["user"].nunique()))
            if not np.isfinite([record[k] for k in ("beta", "se", "ci_low", "ci_high")]).all() or record["se"] <= 0:
                raise ValueError("Invalid coefficient or uncertainty estimate")
            if len(excluded):
                # M used the same default separation handling, but saved only
                # the subgroup's input size. Verify the actual binary fit sample.
                del model, predictions, df
                gc.collect()
                audit_df = pd.read_parquet(PARQUET, filters=filters)
                for column in ("user", "subreddit"):
                    audit_df[column] = audit_df[column].cat.remove_unused_categories()
                audit_df["recip_binary"] = (audit_df[TARGET] > 0).astype("int8")
                binary_model = pf.feglm(FORMULA.replace(TARGET, "recip_binary"), data=audit_df,
                                       family="logit", vcov={"CRV1": "user"}, copy_data=False,
                                       fixef_rm="none", iwls_maxiter=100)
                binary_excluded = np.sort(np.asarray(binary_model.na_index, dtype="int64"))
                if not binary_model.convergence or not np.array_equal(excluded, binary_excluded):
                    raise ValueError("Binary and proportional models use different fitted observations")
                if matched.empty:
                    raise ValueError("No saved binary result available to verify the affected subgroup")
                binary_estimate = binary_model.tidy().loc["recip_binary"]
                if not np.isclose(binary_estimate["Estimate"], original["beta"], rtol=0, atol=1e-7):
                    raise ValueError("Binary separation-audit coefficient differs from saved M")
                if not np.isclose(binary_estimate["Std. Error"], original["se"], rtol=0, atol=1e-8):
                    raise ValueError("Binary separation-audit SE differs from saved M")
                record["separation_audit"] = {
                    "same_excluded_rows_as_binary": True,
                    "excluded_row_positions": excluded.tolist(),
                    "binary_refit_beta": float(binary_estimate["Estimate"]),
                    "binary_refit_se": float(binary_estimate["Std. Error"]),
                    "binary_fitted_rows": len(binary_model.predict(type="link")),
                }
                print(f"Verified identical exclusions in binary and proportional models: {len(excluded)} rows", flush=True)
    record["elapsed_seconds"] = time.monotonic() - started
    write_json(fit_path(description), record)
    print(f"Completed {canonical(description)}: {record['status']} ({record['elapsed_seconds']:.0f}s)", flush=True)


class CheckpointedSearch(BeamSearchEMM):
    """Reuse M's search traversal, running each model in a fresh worker process."""

    def __init__(self):
        selectors = [Selector(column, operator, value) for column, values in THRESHOLDS.items()
                     for value in values for operator in ("<=", ">")]
        super().__init__(df=None, selectors=selectors, **SETTINGS)
        self.signature = run_signature()
        self.records = {}

    def fitted(self, description):
        path = fit_path(description)
        result = json.loads(path.read_text()) if path.exists() else None
        if result is None or result.get("signature") != self.signature:
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--worker", description], check=True)
            result = json.loads(path.read_text())
        else:
            print(f"Using checkpoint: {canonical(description)}", flush=True)
        if result["signature"] != self.signature:
            raise ValueError("Fit settings changed while search was running")
        self.records[canonical(description)] = result
        return result

    def fit_reference_model(self):
        self.reference_result = self.fitted("reference")
        if self.reference_result["status"] != "ok":
            raise RuntimeError("Reference fit was not successful")
        return self.reference_result

    def evaluate_subgroup(self, description):
        text = " AND ".join(str(selector) for selector in description)
        fit = self.fitted(text)
        if fit["status"] != "ok":
            return None
        beta_ref = self.reference_result["beta"]
        delta = abs(fit["beta"] - beta_ref)
        return {"description": text, "depth": len(description), "n": fit["n"],
                "fitted_rows": fit["fitted_rows"], "excluded_for_separation": fit.get("excluded_for_separation", 0),
                "return_rate": fit["return_rate"], **{k: fit[k] for k in ("beta", "se", "ci_low", "ci_high", "p_value")},
                "reference_beta": beta_ref, "delta": delta,
                "quality": fit["n"] ** self.alpha * delta, "_selectors": description}


def compare_results(proportional, reference):
    binary = pd.read_csv(BINARY)
    frames = {}
    for name, frame in (("binary", binary), ("proportional", proportional)):
        frame = frame.sort_values("quality", ascending=False, kind="stable").reset_index(drop=True).copy()
        frame["key"] = frame["description"].map(canonical)
        if frame["key"].duplicated().any():
            raise ValueError(f"Duplicate subgroup descriptions in {name}")
        frame.insert(0, "quality_rank", range(1, len(frame) + 1))
        frame["signed_delta"] = frame["beta"] - frame["reference_beta"]
        frames[name] = frame
    left, right = frames["binary"], frames["proportional"]
    left_keys, right_keys = set(left.head(TOP_K)["key"]), set(right.head(TOP_K)["key"])
    shared, union = left_keys & right_keys, left_keys | right_keys
    shared_all = left.merge(right, on="key", suffixes=("_binary", "_proportional"), validate="one_to_one")
    if not (shared_all["n_binary"] == shared_all["n_proportional"]).all():
        raise ValueError("Matched subgroup sizes differ between measures")
    shared_all["coefficient_sign_agrees"] = np.sign(shared_all["beta_binary"]) == np.sign(shared_all["beta_proportional"])
    shared_all["difference_direction_agrees"] = np.sign(shared_all["signed_delta_binary"]) == np.sign(shared_all["signed_delta_proportional"])
    shared_all["in_both_top_10"] = shared_all["key"].isin(shared)
    shared_top = shared_all.loc[shared_all["in_both_top_10"]].copy()
    comparison = shared_all.loc[shared_all["key"].isin(union)].copy()
    comparison["in_binary_top_10"] = comparison["key"].isin(left_keys)
    comparison["in_proportional_top_10"] = comparison["key"].isin(right_keys)
    left.head(TOP_K).to_csv(output_path(OUT, "binary_top_10.csv"), index=False)
    right.head(TOP_K).to_csv(output_path(OUT, "proportional_top_10.csv"), index=False)
    shared_all.to_csv(output_path(OUT, "all_shared_subgroups.csv"), index=False)
    comparison.to_csv(output_path(OUT, "top_10_comparison.csv"), index=False)
    summary = {
        "comparison": "binary versus proportional reciprocity", "top_k": TOP_K,
        "binary_valid_subgroups": len(left), "proportional_valid_subgroups": len(right),
        "shared_top_k_count": len(shared), "union_top_k_count": len(union),
        "jaccard_top_k": len(shared) / len(union),
        "binary_reference_beta": float(left.iloc[0]["reference_beta"]),
        "proportional_reference_beta": reference["beta"],
        "proportional_reference_se": reference["se"],
        "proportional_reference_ci": [reference["ci_low"], reference["ci_high"]],
        "shared_top_k_coefficient_sign_agreement_count": int(shared_top["coefficient_sign_agrees"].sum()),
        "shared_top_k_difference_direction_agreement_count": int(shared_top["difference_direction_agrees"].sum()),
        "all_shared_subgroup_count": len(shared_all),
        "all_shared_coefficient_sign_agreement_count": int(shared_all["coefficient_sign_agrees"].sum()),
        "all_shared_difference_direction_agreement_count": int(shared_all["difference_direction_agrees"].sum()),
        "proportional_negative_coefficients": int((right["beta"] < 0).sum()),
        "proportional_positive_coefficients": int((right["beta"] > 0).sum()),
        "same_sample_and_search_settings": True,
        "proportional_subgroups_with_separation_exclusions": int((proportional["excluded_for_separation"] > 0).sum()),
    }
    write_json(output_path(OUT, "summary.json"), summary)
    top_table = pd.DataFrame({
        "Rank": right.head(TOP_K)["quality_rank"],
        "Subgroup": right.head(TOP_K)["description"].map(readable_description),
        "n": right.head(TOP_K)["n"].map(lambda x: f"{x:,}"),
        "Beta": right.head(TOP_K)["beta"].map(lambda x: f"{x:+.3f}"),
        "95% CI": [f"[{r.ci_low:.3f}, {r.ci_high:.3f}]" for r in right.head(TOP_K).itertuples()],
        "Signed delta": right.head(TOP_K)["signed_delta"].map(lambda x: f"{x:+.3f}"),
        "Quality": right.head(TOP_K)["quality"].map(lambda x: f"{x:.1f}"),
    })
    comparison_table = pd.DataFrame({
        "Subgroup": comparison["key"].map(readable_description),
        "Binary rank": comparison["quality_rank_binary"],
        "Prop. rank": comparison["quality_rank_proportional"],
        "Binary beta": comparison["beta_binary"].map(lambda x: f"{x:+.3f}"),
        "Prop. beta": comparison["beta_proportional"].map(lambda x: f"{x:+.3f}"),
        "Binary delta": comparison["signed_delta_binary"].map(lambda x: f"{x:+.3f}"),
        "Prop. delta": comparison["signed_delta_proportional"].map(lambda x: f"{x:+.3f}"),
        "Both top 10": comparison["in_both_top_10"].map({True: "Yes", False: "No"}),
    })
    report = (
        "A1: BINARY VERSUS PROPORTIONAL RECIPROCITY\n\n"
        f"Top-{TOP_K} overlap: {len(shared)} shared descriptions / {len(union)} distinct descriptions.\n"
        f"Jaccard overlap: {summary['jaccard_top_k']:.4f}.\n"
        f"Coefficient-sign agreement among shared top-{TOP_K}: {summary['shared_top_k_coefficient_sign_agreement_count']}/{len(shared)}.\n"
        f"Difference-direction agreement among shared top-{TOP_K}: {summary['shared_top_k_difference_direction_agreement_count']}/{len(shared)}.\n"
        f"Binary reference beta: {summary['binary_reference_beta']:.6f}.\n"
        f"Proportional reference beta: {reference['beta']:.6f} (95% CI [{reference['ci_low']:.6f}, {reference['ci_high']:.6f}]).\n\n"
        "PROPORTIONAL TOP 10\n" + top_table.to_string(index=False)
        + "\n\nCOMPARISON OF SUBGROUPS APPEARING IN EITHER TOP 10\n" + comparison_table.to_string(index=False)
        + "\n\nNotes: All models use the same observations, activity controls, month/subreddit fixed effects, and user-clustered SEs.\n"
        "n is the input subgroup size used for M's quality function. Fitted row counts and separation exclusions are recorded separately.\n"
        "For affected subgroups, binary refits verify identical excluded row positions and reproduce the original coefficients and SEs.\n"
        "Signed delta compares each subgroup with the global reference fitted using the SAME reciprocity measure.\n"
        "Ranks use sqrt(n) * abs(signed delta). Conditions are canonicalized before list comparisons.\n"
        "Binary reciprocity indicates any mutual partner; proportional reciprocity is reciprocal/outgoing partners.\n"
        "Coefficients refer to different contrasts and should not be read as directly comparable probability effects.\n"
        "Subgroups overlap. CIs are pointwise; no post-selection correction is applied. Associations are descriptive.\n"
        "Source: saved binary M results and the full reciprocity_return.csv analysis dataset.\n"
        "The weighted-reciprocity search is not part of this binary/proportional comparison.\n"
    )
    (output_path(OUT, "a1_report.txt")).write_text(report, encoding="utf-8")
    for name, table, caption in (("proportional_top_10", top_table, "Top ten subgroups using proportional reciprocity."),
                                 ("a1_comparison", comparison_table, "Binary and proportional reciprocity: subgroups in either top-ten list.")):
        latex = "\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n"
        latex += f"\\caption{{{caption}}}\n" + table.to_latex(index=False, escape=True)
        latex += "\\begin{minipage}{\\textwidth}\n\\footnotesize\n"
        latex += "Ranks use the original size-weighted quality. Signed differences use each measure's own global reference. "
        latex += "Coefficients are on the log-odds scale and represent different reciprocity contrasts. "
        latex += "Standard errors are clustered by user. Subgroups overlap.\n\\end{minipage}\n\\end{table*}\n"
        (output_path(OUT, f"{name}.tex")).write_text(latex, encoding="utf-8")
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    (output_path(OUT, "fits")).mkdir(exist_ok=True)
    if args.prepare:
        prepare_cache()
        return
    if args.worker:
        fit_worker(args.worker)
        return
    metadata_path = CACHE / "metadata.json"
    if not PARQUET.exists() or not metadata_path.exists() or json.loads(metadata_path.read_text()) != cache_metadata():
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--prepare"], check=True)
    search = CheckpointedSearch()
    proportional = search.search()
    proportional.to_csv(output_path(OUT, "proportional_beam_search_results.csv"), index=False)
    write_json(output_path(OUT, "search_audit.json"), {
        "signature": search.signature, "settings": SETTINGS, "thresholds": THRESHOLDS,
        "reference": search.reference_result,
        "attempted_subgroups": len(search.records) - 1, "valid_subgroups": len(proportional),
        "skipped": [r for r in search.records.values() if r["status"] != "ok"],
        "separation_audits": [r for r in search.records.values() if r.get("excluded_for_separation", 0) > 0],
    })
    compare_results(proportional, search.reference_result)


if __name__ == "__main__":
    main()
