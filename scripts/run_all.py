#!/usr/bin/env python3
"""End-to-end reproduction: audit -> 5 training seeds -> 150-run evaluation -> tables -> figures."""
import subprocess, sys
def run(cmd): print("+", " ".join(cmd)); subprocess.check_call(cmd)
def main() -> None:
    py = sys.executable
    run([py, "scripts/audit_data.py"])
    for seed in ("17", "29", "43", "71", "101"):
        run([py, "scripts/train.py", "--seed", seed])
    run([py, "scripts/evaluate.py"])
    run([py, "scripts/build_tables.py"])
    run([py, "scripts/build_figures.py"])
if __name__ == "__main__":
    main()
