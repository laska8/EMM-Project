"""Refit M's fixed top-k subgroups and compare coefficient and AME differences.

Run: python experiments/A3_probability_scale/code/run_ame_analysis.py
Completed fits are checkpointed; rerunning resumes the same analysis.
For binary reciprocity, AME means the average discrete change from 0 to 1.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

# Match the stable, single-threaded setup in subreddit_logistic.ipynb.
for variable in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

# Import the shared paths from this project's experiments directory.
EXPERIMENTS_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "experiments")
sys.path.insert(0, str(EXPERIMENTS_DIR / "code"))
from experiment_paths import ROOT, ORIGINAL_SEARCH, M, A1, A2, A3, output_path
sys.path.insert(0, str(M / "code" / "added"))

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr

from report_top_subgroups import SOURCE, TOTAL_OBSERVATIONS, TOP_K, readable_description

BASE_DIR = ORIGINAL_SEARCH
DATA = BASE_DIR.parent / "eda_outputs" / "reciprocity_return.csv"
OUT = A3
FORMULA = "returned_next_month ~ recip_binary + log_sent + log_received | month + subreddit"


def selected_results():
    return pd.read_csv(SOURCE).sort_values("quality", ascending=False, kind="stable").head(TOP_K).copy()


def signature():
    import importlib.metadata

    settings = {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "data_size": DATA.stat().st_size,
        "data_mtime_ns": DATA.stat().st_mtime_ns,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "path_helper_sha256": hashlib.sha256((EXPERIMENTS_DIR / "code" / "experiment_paths.py").read_bytes()).hexdigest(),
        "pyfixest": importlib.metadata.version("pyfixest"),
        "top_k": TOP_K,
        "total_observations": TOTAL_OBSERVATIONS,
    }
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def load_sample(description):
    dtypes = {"subreddit": "category", "month": "int8", "replies_sent": "int32",
              "replies_received": "int32", "reciprocal_partners": "int32", "returned_next_month": "int8"}
    parts = []
    for chunk in pd.read_csv(DATA, usecols=list(dtypes), dtype=dtypes, keep_default_na=False, chunksize=1_000_000):
        mask = np.ones(len(chunk), dtype=bool)
        if description != "reference":
            for condition in description.split(" AND "):
                match = re.fullmatch(r"(replies_sent|replies_received)(<=|>|==)([0-9.]+)", condition)
                if match is None:
                    raise ValueError(f"Unsupported selector: {condition}")
                column, operator, threshold = match.groups()
                values = chunk[column].to_numpy()
                threshold = float(threshold)
                mask &= {"<=": np.less_equal, ">": np.greater, "==": np.equal}[operator](values, threshold)
        selected = chunk.loc[mask].copy()
        selected["recip_binary"] = (selected.pop("reciprocal_partners") > 0).astype("int8")
        selected["log_sent"] = np.log1p(selected.pop("replies_sent").astype("float64"))
        selected["log_received"] = np.log1p(selected.pop("replies_received").astype("float64"))
        parts.append(selected)
    df = pd.concat(parts, ignore_index=True)
    del parts
    df["subreddit"] = df["subreddit"].astype("category")
    gc.collect()
    return df


def average_discrete_change(eta, reciprocity, beta, chunk_size=1_000_000):
    """Average [p(R=1, X) - p(R=0, X)], preserving each row's X and FEs."""
    total = 0.0
    for start in range(0, len(eta), chunk_size):
        stop = start + chunk_size
        eta_zero = eta[start:stop] - beta * reciprocity[start:stop]
        total += np.sum(expit(eta_zero + beta) - expit(eta_zero), dtype="float64")
    return float(total / len(eta))


def fit_one(key, run_signature):
    import pyfixest as pf

    selected = selected_results()
    row = None if key == "reference" else selected.iloc[int(key) - 1]
    description = "reference" if row is None else row["description"]
    expected_n = TOTAL_OBSERVATIONS if row is None else int(row["n"])
    expected_beta = float(selected.iloc[0]["reference_beta"] if row is None else row["beta"])
    started = time.monotonic()
    print(f"[{key}] Loading {description}", flush=True)
    df = load_sample(description)
    if len(df) != expected_n:
        raise ValueError(f"Sample mismatch: {len(df)} != {expected_n}")
    if not df["month"].between(1, 10).all() or df.isna().any().any():
        raise ValueError("Unexpected analysis months or missing data")
    return_rate = float(df["returned_next_month"].mean())
    if row is not None and not np.isclose(return_rate, row["return_rate"], atol=1e-12, rtol=0):
        raise ValueError("Return rate differs from saved M results")
    print(f"[{key}] Fitting {len(df):,} observations", flush=True)
    # Covariance choice does not change point estimates or fitted probabilities.
    # Original clustered CIs remain in M; A3 computes AME point estimates only.
    # Omitting user IDs and clustering reduces the memory needed for these refits.
    model = pf.feglm(FORMULA, data=df, family="logit", vcov="iid",
                     copy_data=False, fixef_rm="none", iwls_maxiter=100)
    if not model.convergence:
        raise RuntimeError("Model did not converge")
    beta = float(model.coef().loc["recip_binary"])
    if abs(beta - expected_beta) > 1e-5:
        raise ValueError(f"Coefficient mismatch: refit={beta}, saved={expected_beta}")
    eta = np.asarray(model.predict(type="link")).ravel()
    if len(eta) != len(df) or not np.isfinite(eta).all():
        raise ValueError("Predictions are missing or observations were excluded")
    ame = average_discrete_change(eta, df["recip_binary"].to_numpy(), beta)
    if np.sign(ame) != np.sign(beta) or not -1 <= ame <= 1:
        raise ValueError("Invalid AME")
    result = {
        "signature": run_signature, "description": description, "n": len(df),
        "return_rate": return_rate, "beta": beta, "saved_beta": expected_beta,
        "beta_difference_from_saved": beta - expected_beta, "ame": ame, "ame_pp": 100 * ame,
        "converged": bool(model.convergence), "elapsed_seconds": time.monotonic() - started,
    }
    path = output_path(OUT, f"fit_{key}.json")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(f"[{key}] Complete: beta={beta:.6f}, AME={100 * ame:.4f} pp; {result['elapsed_seconds']:.0f}s", flush=True)


def build_report():
    reference = json.loads((output_path(OUT, "fit_reference.json")).read_text())
    result = selected_results().reset_index(drop=True)
    result.insert(0, "quality_rank", range(1, len(result) + 1))
    fits = [json.loads((output_path(OUT, f"fit_{i}.json")).read_text()) for i in result["quality_rank"]]
    result["refit_beta"] = [r["beta"] for r in fits]
    result["ame_pp"] = [r["ame_pp"] for r in fits]
    result["reference_ame_pp"] = reference["ame_pp"]
    result["signed_delta"] = result["beta"] - result["reference_beta"]
    result["delta_ame_pp"] = result["ame_pp"] - result["reference_ame_pp"]
    result["abs_delta"] = result["signed_delta"].abs()
    result["abs_delta_ame_pp"] = result["delta_ame_pp"].abs()
    result["coefficient_difference_rank"] = result["abs_delta"].rank(ascending=False, method="average")
    result["ame_difference_rank"] = result["abs_delta_ame_pp"].rank(ascending=False, method="average")
    result["difference_direction_agrees"] = np.sign(result["signed_delta"]) == np.sign(result["delta_ame_pp"])
    result["effect_sign_agrees"] = np.sign(result["beta"]) == np.sign(result["ame_pp"])
    rho = float(spearmanr(result["abs_delta"], result["abs_delta_ame_pp"]).statistic)
    summary = {
        "top_k": len(result), "reference_beta": reference["beta"], "reference_ame_pp": reference["ame_pp"],
        "spearman_abs_difference_ranks": rho,
        "difference_direction_agreement_count": int(result["difference_direction_agrees"].sum()),
        "difference_direction_agreement_pct": 100 * float(result["difference_direction_agrees"].mean()),
        "effect_sign_agreement_count": int(result["effect_sign_agrees"].sum()),
        "ame_pp_min": float(result["ame_pp"].min()), "ame_pp_max": float(result["ame_pp"].max()),
        "max_abs_beta_refit_difference": max(abs(r["beta_difference_from_saved"]) for r in [reference] + fits),
        "definition": "Mean discrete change p(R=1,X)-p(R=0,X), averaged within each fitted population.",
        "scope": "Fixed top-k from M, not a new search. AME point estimates only; no AME confidence intervals.",
    }
    result.to_csv(output_path(OUT, "top_10_ame_comparison.csv"), index=False)
    (output_path(OUT, "summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table = pd.DataFrame({
        "M rank": result["quality_rank"], "Subgroup": result["description"].map(readable_description),
        "Beta": result["beta"].map(lambda x: f"{x:.3f}"),
        "Delta beta": result["signed_delta"].map(lambda x: f"{x:+.3f}"),
        "AME (pp)": result["ame_pp"].map(lambda x: f"{x:+.2f}"),
        "Delta AME (pp)": result["delta_ame_pp"].map(lambda x: f"{x:+.2f}"),
        "Beta rank": result["coefficient_difference_rank"].map(lambda x: f"{x:g}"),
        "AME rank": result["ame_difference_rank"].map(lambda x: f"{x:g}"),
        "Direction agrees": result["difference_direction_agrees"].map({True: "Yes", False: "No"}),
    })
    notes = (
        f"Reference AME: {reference['ame_pp']:.4f} percentage points.\n"
        f"Spearman correlation between absolute-difference rankings: {rho:.4f}.\n"
        f"Difference-direction agreement: {summary['difference_direction_agreement_count']}/{len(result)}.\n"
        f"Coefficient/AME sign agreement: {summary['effect_sign_agreement_count']}/{len(result)}.\n"
        "Beta rank uses abs(subgroup beta - reference beta); AME rank uses abs(subgroup AME - reference AME).\n"
        "M rank is the original size-weighted quality rank, not the coefficient-difference rank.\n"
        "AME is the average binary 0-to-1 discrete change, evaluated over each model's own observations.\n"
        "Differences therefore reflect both fitted models and the populations over which they are averaged.\n"
        "All AME values and AME differences are in percentage points. These are descriptive associations.\n"
        "AME uncertainty and post-selection inference are not estimated. Subgroups overlap.\n"
        "Refits use the same likelihood and specification as M; covariance choice affects uncertainty, not AME point estimates.\n"
        "Original user-clustered coefficient CIs are retained in the source/report CSV.\n"
        "Source: beam_search/emm_beam_search_results.csv and eda_outputs/reciprocity_return.csv."
    )
    report = table.to_string(index=False) + "\n\n" + notes + "\n"
    (output_path(OUT, "a3_report.txt")).write_text(report, encoding="utf-8")
    latex = "\\begin{table*}[t]\n\\centering\n\\small\n"
    latex += "\\caption{Coefficient and probability-scale differences for the top ten subgroups from M.}\n"
    latex += "\\label{tab:ame-comparison}\n" + table.to_latex(index=False, escape=True)
    latex += "\\begin{minipage}{\\textwidth}\n\\footnotesize\n"
    latex += f"Reference AME: {reference['ame_pp']:.2f} percentage points. "
    latex += "AMEs are average binary discrete changes within each fitted population. "
    latex += "Beta and AME ranks use absolute differences from the reference, without size weighting. "
    latex += "Direction agreement compares signed differences. Subgroups overlap. AME uncertainty is not estimated.\n"
    latex += "\\end{minipage}\n\\end{table*}\n"
    (output_path(OUT, "a3_table.tex")).write_text(latex, encoding="utf-8")
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-key", help=argparse.SUPPRESS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run_signature = signature()
    if args.fit_key:
        fit_one(args.fit_key, run_signature)
        return
    # Separate processes release each model's memory before the next fit.
    for key in [str(i) for i in range(1, TOP_K + 1)] + ["reference"]:
        checkpoint = output_path(OUT, f"fit_{key}.json")
        if checkpoint.exists() and json.loads(checkpoint.read_text()).get("signature") == run_signature:
            print(f"[{key}] Using completed checkpoint", flush=True)
            continue
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--fit-key", key], check=True)
    build_report()


if __name__ == "__main__":
    main()
