"""Check all raw networks for repeated directed replies needed by weighted A1."""

import hashlib
import json
from pathlib import Path
import sys
import time

# Import the shared paths from this project's experiments directory.
EXPERIMENTS_DIR = next(p for p in Path(__file__).resolve().parents if p.name == "experiments")
sys.path.insert(0, str(EXPERIMENTS_DIR / "code"))
from experiment_paths import ROOT, ORIGINAL_SEARCH, M, A1, A2, A3, output_path

BASE = ORIGINAL_SEARCH
RAW = BASE.parent / "reply_networks"
OUT = output_path(A1, "weighted_feasibility.json")


def main():
    started = time.monotonic()
    files = sorted(RAW.glob("*.json"))
    if not files:
        raise ValueError("No raw network files found")
    months = outgoing_lists = reply_entries = repeated_lists = repeated_entries = 0
    manifest = hashlib.sha256()
    for index, path in enumerate(files, 1):
        stat = path.stat()
        manifest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
        networks = json.loads(path.read_text(encoding="utf-8"))
        months += len(networks)
        for month in networks:
            for targets in month.values():
                outgoing_lists += 1
                reply_entries += len(targets)
                repeats = len(targets) - len(set(targets))
                repeated_lists += int(repeats > 0)
                repeated_entries += repeats
        del networks
        if index % 250 == 0 or index == len(files):
            print(f"Raw-network audit: {index}/{len(files)} files; {repeated_entries:,} repeated target entries", flush=True)
    result = {
        "scope": "All supplied raw subreddit JSON files, all available monthly networks",
        "files_checked": len(files), "months_checked": months,
        "outgoing_lists_checked": outgoing_lists, "reply_entries_checked": reply_entries,
        "lists_with_repeated_targets": repeated_lists, "repeated_target_entries": repeated_entries,
        "repeated_replies_detected": repeated_entries > 0,
        "weighted_measure_implemented": False,
        "conclusion": (
            "Repeated directed replies are available; a weighted measure can be defined and evaluated."
            if repeated_entries else
            "No repeated targets occur within any user-month outgoing list. The supplied adjacency lists do not provide repeated-reply intensity for a distinct weighted-reciprocity experiment. This does not establish that real users never replied repeatedly."
        ),
        "raw_file_manifest_sha256": manifest.hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
