#!/usr/bin/env python3
"""Run the frozen 150-run PEMS-BAY primary evaluation (3 models x 5 seeds x 10 conditions)."""
import argparse, os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from _bootstrap import project_root  # noqa: E402
from fargfrc.evaluation.pems_bay_final_multi_model_primary_evaluator_v4 import run_final_evaluation  # noqa: E402
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    run_final_evaluation(project_root=project_root(a.project_root), device_name=a.device)
if __name__ == "__main__":
    main()
