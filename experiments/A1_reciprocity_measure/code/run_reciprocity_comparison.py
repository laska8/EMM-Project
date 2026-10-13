"""A1: repeat the M beam search with proportional reciprocity and compare the top-k lists.

Run: python experiments/A1_reciprocity_measure/code/run_reciprocity_comparison.py
The search is the unchanged beam search of beam_search/ (local discretization),
with proportional_reciprocity as the model target instead of recip_binary.
The saved binary results are read only. Every fit is checkpointed, so an
interrupted run continues where it stopped when the command is repeated.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import time

# Import the shared paths from this project's experiments directory.
EXPERIMENTS_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "experiments")
sys.path.insert(0, str(EXPERIMENTS_DIR / "code"))
from experiment_paths import ROOT, ORIGINAL_SEARCH, M, A1, A2, A3, output_path
sys.path.insert(0, str(M / "code" / "added"))
sys.path.insert(0, str(ORIGINAL_SEARCH))

import numpy as np
import pandas as pd

BASE = ORIGINAL_SEARCH
DATA = BASE.parent / "eda_outputs" / "reciprocity_return.csv"
BINARY = BASE / "emm_beam_search_results.csv"
OUT = A1
CACHE = output_path(OUT, "data_cache")
PARQUET = CACHE / "prepared.parquet"
CHECKPOINTS = CACHE / "fit_checkpoints.jsonl"
TARGET = "proportional_reciprocity"
BINARY_TARGET = "recip_binary"
TOTAL_OBSERVATIONS = 36_031_393
TOP_K = 10
SELECTOR_COLUMNS = ["replies_sent", "replies_received"]
# same settings as beam_search/run_beam_search.py
SETTINGS = {"n_bins": 8, "beam_width": 10, "max_depth": 3, "min_support": 5000, "min_cell": 10, "alpha": 0.5}
ESTIMATES = ("beta", "se", "ci_low", "ci_high", "p_value")


def conditions(description):
    parsed = []
    for condition in description.split(" AND "):
        match = re.fullmatch(r"(replies_sent|replies_received)(<=|>|==)([0-9.]+)", condition)
        if match is None:
            raise ValueError(f"Unsupported condition: {condition}")
        column, operator, value = match.groups()
        parsed.append((column, operator, float(value)))
    return sorted(parsed)


def canonical(description):
    """Order-independent key: 'A AND B' and 'B AND A' are the same subgroup."""
    return " AND ".join(f"{column}{operator}{value:g}" for column, operator, value in conditions(description))


def cache_metadata():
    return {"data_size": DATA.stat().st_size, "data_mtime_ns": DATA.stat().st_mtime_ns,
            "total_observations": TOTAL_OBSERVATIONS, "cache_format_version": 1}


def run_signature():
    record = {**cache_metadata(), "binary_sha256": hashlib.sha256(BINARY.read_bytes()).hexdigest(),
              "search_sha256": [hashlib.sha256((BASE / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                for name in ("beam_search.py", "emm_selectors.py", "logistic_model.py")],
              "pyfixest": importlib.metadata.version("pyfixest"),
              "target": TARGET, "settings": SETTINGS, "selector_columns": SELECTOR_COLUMNS}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def write_json(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare_cache():
    """Compact copy of the analysis data; also used by A2."""
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
    audit = {**cache_metadata(), "users": int(df["user"].nunique()),
             "subreddits": int(df["subreddit"].nunique()),
             "return_rate": float(df["returned_next_month"].mean()),
             "definition": "reciprocal_partners / outgoing_partners",
             "definition_verified_all_rows": True}
    df = df.drop(columns=["reciprocal_partners", "outgoing_partners"])
    df["log_sent"] = np.log1p(df["replies_sent"].astype("float64"))
    df["log_received"] = np.log1p(df["replies_received"].astype("float64"))
    df.to_parquet(PARQUET, index=False, compression="snappy")
    write_json(CACHE / "metadata.json", cache_metadata())
    write_json(output_path(OUT, "data_audit.json"), audit)
    print(f"Prepared {len(df):,} observations; reciprocity definition verified on every row", flush=True)


class Checkpoints:
    """One JSON line per finished fit; lines of another signature are ignored."""

    def __init__(self, signature):
        self.signature = signature
        self.records = {}
        if CHECKPOINTS.exists():
            for line in CHECKPOINTS.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if record["signature"] == signature:
                    self.records[record["key"]] = record["result"]
        print(f"{len(self.records)} checkpointed fits can be reused", flush=True)

    def get(self, key, compute):
        if key not in self.records:
            result = compute()
            self.records[key] = result
            with CHECKPOINTS.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"signature": self.signature, "key": key, "result": result}) + "\n")
        return self.records[key]


def make_search(df, target, checkpoints):
    from beam_search import BeamSearchEMM

    class CheckpointedSearch(BeamSearchEMM):
        """M's search, storing every fit so that the run can be resumed."""

        def fit_reference_model(self):
            self.reference_result = checkpoints.get(
                f"{target}|reference",
                lambda: {k: float(v) for k, v in BeamSearchEMM.fit_reference_model(self).items()})
            return self.reference_result

        def evaluate_subgroup(self, description, subgroup):
            text = " AND ".join(str(selector) for selector in description)

            def compute():
                started = time.monotonic()
                result = BeamSearchEMM.evaluate_subgroup(self, description, subgroup)
                if result is None:
                    return None
                result = {k: (v.item() if hasattr(v, "item") else v) for k, v in result.items() if k != "_selectors"}
                print(f"  {text}: n={result['n']:,} beta={result['beta']:.4f} ({time.monotonic() - started:.0f}s)", flush=True)
                return result

            result = checkpoints.get(f"{target}|{canonical(text)}", compute)
            return None if result is None else {**result, "_selectors": description}

    return CheckpointedSearch(df=df, selector_columns=SELECTOR_COLUMNS, target=target, **SETTINGS)


def cross_fit(search, df, description):
    """Fit one given subgroup description with the search's own model and guards."""
    from emm_selectors import Selector, apply_description

    selectors = tuple(Selector(column, operator, value) for column, operator, value in conditions(description))
    result = search.evaluate_subgroup(selectors, df[apply_description(df, selectors)])
    if result is None:
        return None
    return {k: v for k, v in result.items() if k != "_selectors"}


def ranked(frame):
    frame = frame.sort_values("quality", ascending=False, kind="stable").reset_index(drop=True).copy()
    frame["key"] = frame["description"].map(canonical)
    if frame["key"].duplicated().any():
        raise ValueError("Duplicate subgroup descriptions")
    frame.insert(0, "quality_rank", range(1, len(frame) + 1))
    frame["signed_delta"] = frame["beta"] - frame["reference_beta"]
    return frame


def side(row, frame, searched):
    """Estimates of one subgroup under one measure, with the rank its quality has in that search."""
    if row is None:
        return {"available": False}
    return {"available": True, "found_by_search": searched,
            "quality_rank": int((frame["quality"] > row["quality"]).sum()) + 1,
            "n": int(row["n"]), "beta": float(row["beta"]), "se": float(row["se"]),
            "ci_low": float(row["ci_low"]), "ci_high": float(row["ci_high"]),
            "reference_beta": float(row["reference_beta"]),
            "signed_delta": float(row["beta"] - row["reference_beta"]), "quality": float(row["quality"])}


def compare_results(df, proportional, proportional_search, checkpoints):
    from report_top_subgroups import readable_description

    binary = ranked(pd.read_csv(BINARY))
    proportional = ranked(proportional)
    binary_keys = list(binary.head(TOP_K)["key"])
    proportional_keys = list(proportional.head(TOP_K)["key"])
    shared = [key for key in binary_keys if key in proportional_keys]
    union = binary_keys + [key for key in proportional_keys if key not in binary_keys]

    # A subgroup of one top-k list may never have been visited by the other
    # search. It is then fitted separately so that both measures can be compared.
    binary_search = make_search(df, BINARY_TARGET, checkpoints)
    binary_search.reference_result = {"beta": float(binary["reference_beta"].iloc[0])}
    by_key = {"binary": binary.set_index("key"), "proportional": proportional.set_index("key")}
    rows = []
    for key in union:
        row = {"key": key, "subgroup": readable_description(key),
               "in_binary_top_10": key in binary_keys, "in_proportional_top_10": key in proportional_keys}
        for name, frame, search in (("binary", binary, binary_search), ("proportional", proportional, proportional_search)):
            searched = key in by_key[name].index
            estimate = by_key[name].loc[key] if searched else cross_fit(search, df, key)
            row.update({f"{field}_{name}": value for field, value in side(estimate, frame, searched).items()})
        both = row["available_binary"] and row["available_proportional"]
        row["coefficient_sign_agrees"] = bool(both and np.sign(row["beta_binary"]) == np.sign(row["beta_proportional"]))
        row["difference_direction_agrees"] = bool(both and np.sign(row["signed_delta_binary"]) == np.sign(row["signed_delta_proportional"]))
        rows.append(row)
    comparison = pd.DataFrame(rows)

    # Identical descriptions are a strict criterion when thresholds are local,
    # so the two lists are also compared by the observations they select.
    from emm_selectors import Selector, apply_description

    def members(key):
        selectors = [Selector(column, operator, value) for column, operator, value in conditions(key)]
        return apply_description(df, selectors).to_numpy()

    masks = {key: members(key) for key in union}

    def jaccard(left, right):
        return float((left & right).sum() / (left | right).sum())

    def closest(key, others):
        return max(jaccard(masks[key], masks[other]) for other in others)

    comparison["closest_in_other_top_10_jaccard"] = [
        closest(key, proportional_keys if key in binary_keys else binary_keys) for key in union]
    covered_binary = np.logical_or.reduce([masks[key] for key in binary_keys])
    covered_proportional = np.logical_or.reduce([masks[key] for key in proportional_keys])
    covered_both = int((covered_binary & covered_proportional).sum())
    del masks
    both = comparison.loc[comparison["available_binary"] & comparison["available_proportional"]]

    # subgroups that both searches happened to visit
    visited = binary.merge(proportional, on="key", suffixes=("_binary", "_proportional"), validate="one_to_one")
    if not (visited["n_binary"] == visited["n_proportional"]).all():
        raise ValueError("Matched subgroup sizes differ between measures")

    reference = proportional_search.reference_result
    summary = {
        "comparison": "binary versus proportional reciprocity", "top_k": TOP_K,
        "discretization": "local", "settings": SETTINGS,
        "binary_valid_subgroups": len(binary), "proportional_valid_subgroups": len(proportional),
        "shared_top_k_count": len(shared), "union_top_k_count": len(union),
        "jaccard_top_k": len(shared) / len(union),
        "top_k_observations_binary": int(covered_binary.sum()),
        "top_k_observations_proportional": int(covered_proportional.sum()),
        "top_k_observations_in_both": covered_both,
        "top_k_observation_jaccard": covered_both / int((covered_binary | covered_proportional).sum()),
        "proportional_top_k_observations_inside_binary_top_k": covered_both / int(covered_proportional.sum()),
        "binary_top_k_observations_inside_proportional_top_k": covered_both / int(covered_binary.sum()),
        "closest_match_jaccard_binary_top_k": [float(v) for v in comparison.loc[comparison["in_binary_top_10"], "closest_in_other_top_10_jaccard"]],
        "closest_match_jaccard_proportional_top_k": [float(v) for v in comparison.loc[comparison["in_proportional_top_10"], "closest_in_other_top_10_jaccard"]],
        "binary_reference_beta": float(binary["reference_beta"].iloc[0]),
        "proportional_reference_beta": reference["beta"],
        "proportional_reference_se": reference["se"],
        "proportional_reference_ci": [reference["ci_low"], reference["ci_high"]],
        "union_estimated_under_both_measures": len(both),
        "union_coefficient_sign_agreement_count": int(both["coefficient_sign_agrees"].sum()),
        "union_difference_direction_agreement_count": int(both["difference_direction_agrees"].sum()),
        "union_fitted_outside_binary_search": int((comparison["available_binary"] & ~comparison["found_by_search_binary"].fillna(False).astype(bool)).sum()),
        "union_fitted_outside_proportional_search": int((comparison["available_proportional"] & ~comparison["found_by_search_proportional"].fillna(False).astype(bool)).sum()),
        "visited_by_both_searches": len(visited),
        "visited_by_both_coefficient_sign_agreement_count": int((np.sign(visited["beta_binary"]) == np.sign(visited["beta_proportional"])).sum()),
        "visited_by_both_difference_direction_agreement_count": int((np.sign(visited["signed_delta_binary"]) == np.sign(visited["signed_delta_proportional"])).sum()),
        "proportional_negative_coefficients": int((proportional["beta"] < 0).sum()),
        "proportional_positive_coefficients": int((proportional["beta"] > 0).sum()),
    }
    proportional.drop(columns=["key"]).head(TOP_K).to_csv(output_path(OUT, "proportional_top_10.csv"), index=False)
    comparison.to_csv(output_path(OUT, "top_10_comparison.csv"), index=False)
    write_json(output_path(OUT, "summary.json"), summary)

    def rank(value, searched):
        return "-" if pd.isna(value) else f"{int(value)}" + ("" if searched else "*")

    def number(value):
        return "-" if pd.isna(value) else f"{value:+.3f}"

    table = pd.DataFrame({
        "Subgroup": comparison["subgroup"],
        "Binary rank": [rank(r.quality_rank_binary, r.found_by_search_binary) for r in comparison.itertuples()],
        "Prop. rank": [rank(r.quality_rank_proportional, r.found_by_search_proportional) for r in comparison.itertuples()],
        "Binary beta": comparison["beta_binary"].map(number),
        "Prop. beta": comparison["beta_proportional"].map(number),
        "Binary delta": comparison["signed_delta_binary"].map(number),
        "Prop. delta": comparison["signed_delta_proportional"].map(number),
        "Closest match": comparison["closest_in_other_top_10_jaccard"].map(lambda value: f"{value:.2f}"),
    })
    report = (
        "A1: BINARY VERSUS PROPORTIONAL RECIPROCITY\n\n" + table.to_string(index=False) + "\n\n"
        f"Proportional reference: beta {reference['beta']:.4f}, SE {reference['se']:.6f}.\n"
        f"Valid subgroups: binary {len(binary)}, proportional {len(proportional)}; {len(visited)} were visited by both searches.\n"
        f"Top-{TOP_K} overlap: {len(shared)} shared descriptions / {len(union)} distinct descriptions "
        f"(Jaccard {summary['jaccard_top_k']:.3f}).\n"
        f"Observations selected by at least one top-{TOP_K} subgroup: binary {summary['top_k_observations_binary']:,}, "
        f"proportional {summary['top_k_observations_proportional']:,}; {covered_both:,} are selected by both lists "
        f"(Jaccard {summary['top_k_observation_jaccard']:.3f}; {100 * summary['proportional_top_k_observations_inside_binary_top_k']:.1f}% "
        f"of the proportional and {100 * summary['binary_top_k_observations_inside_proportional_top_k']:.1f}% of the binary selection).\n"
        "Closest match is the largest Jaccard overlap of observations with any subgroup of the other top list.\n"
        f"Of the {len(union)} subgroups in either top-{TOP_K} list, {len(both)} could be estimated under both measures; "
        f"the direction of the subgroup-reference difference agrees in {summary['union_difference_direction_agreement_count']}.\n"
        "Both searches use the same data, model, local discretization and settings; only the reciprocity measure differs.\n"
        "The thresholds are recomputed in every refined subgroup, so the two searches follow different paths and\n"
        "need not visit the same subgroups. A rank marked * belongs to a subgroup that the search of that measure\n"
        "did not visit: it was fitted separately and the rank is the place its quality would take in that search.\n"
        "Signed delta compares each subgroup with the global reference fitted using the SAME reciprocity measure.\n"
        "Binary reciprocity indicates any mutual partner; proportional reciprocity is reciprocal/outgoing partners.\n"
        "Coefficients refer to different contrasts and should not be read as directly comparable probability effects.\n"
        "Subgroups overlap. CIs are pointwise; no post-selection correction is applied. Associations are descriptive.\n"
    )
    output_path(OUT, "a1_report.txt").write_text(report, encoding="utf-8")
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="only build the data cache (needed by A2)")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    metadata_path = CACHE / "metadata.json"
    if not PARQUET.exists() or not metadata_path.exists() or json.loads(metadata_path.read_text()) != cache_metadata():
        prepare_cache()
    if args.prepare:
        return
    df = pd.read_parquet(PARQUET)
    # M's binary measure; the beam search also uses it for its minimum cell count
    df[BINARY_TARGET] = (df[TARGET] > 0).astype(int)
    checkpoints = Checkpoints(run_signature())
    search = make_search(df, TARGET, checkpoints)
    started = time.monotonic()
    proportional = search.search()
    print(f"Proportional search finished in {(time.monotonic() - started) / 60:.0f} min", flush=True)
    proportional.to_csv(output_path(OUT, "proportional_beam_search_results.csv"), index=False)
    compare_results(df, proportional, search, checkpoints)


if __name__ == "__main__":
    main()
