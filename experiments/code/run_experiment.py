"""Print or run a project experiment without editing any source code.

Default: print the command. Add --execute to run it. Run from any directory.
Added analyses live in experiments/. The original M entry point remains in
beam_search/ so the original implementation does not need to be edited.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
COMMANDS = {
    "M": "beam_search/run_beam_search.py",
    "M-report": "experiments/M_reference_discovery/code/added/report_top_subgroups.py",
    "A1": "experiments/A1_reciprocity_measure/code/run_reciprocity_comparison.py",
    "A1-weighted-audit": "experiments/A1_reciprocity_measure/code/audit_repeated_replies.py",
    "A2": "experiments/A2_user_dependence/code/run_dependence_analysis.py",
    "A3": "experiments/A3_probability_scale/code/run_ame_analysis.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=COMMANDS)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    command = [sys.executable, "-u", str(ROOT / COMMANDS[args.experiment])]
    print(subprocess.list2cmdline(command), flush=True)
    if args.execute:
        print("This reruns the analysis and may replace its generated result files. Source code is not modified.", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        print("Command only. Add --execute to run. See experiments/README.txt for dependencies and output locations.")


if __name__ == "__main__":
    main()
