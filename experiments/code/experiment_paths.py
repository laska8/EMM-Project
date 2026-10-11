"""Shared locations for the consolidated experiments; no duplicated source code."""
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENTS.parent
ORIGINAL_SEARCH = ROOT / "beam_search"
M = EXPERIMENTS / "M_reference_discovery"
A1 = EXPERIMENTS / "A1_reciprocity_measure"
A2 = EXPERIMENTS / "A2_user_dependence"
A3 = EXPERIMENTS / "A3_probability_scale"


def output_path(experiment, name):
    """Route an output to its single canonical location in the experiment folder."""
    if name == "fits":
        path = experiment / "model_checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path
    if name == "data_cache":
        path = experiment / name
        path.mkdir(parents=True, exist_ok=True)
        return path
    suffix = Path(name).suffix
    if name.startswith("fit_") and suffix == ".json":
        folder = "model_checkpoints"
    elif suffix == ".csv":
        folder = "tables"
    elif suffix == ".tex":
        folder = "latex"
    elif suffix == ".txt" or name == "summary.json":
        folder = "reports"
    elif suffix == ".json":
        folder = "audits"
    else:
        raise ValueError(f"Unrecognized experiment output: {name}")
    path = experiment / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
