#!/usr/bin/env python3
"""Train one frozen FAR-GF-RC seed (30 epochs, frozen curriculum, selection-only checkpointing)."""
import argparse, os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from _bootstrap import project_root  # noqa: E402
from fargfrc.training.pems_bay_far_gf_rc_runner_v2 import train_fargfrc_seed  # noqa: E402
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--seed", type=int, required=True, choices=[17, 29, 43, 71, 101])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    summary = train_fargfrc_seed(project_root=project_root(a.project_root), model_seed=a.seed, device=a.device)
    print(summary.get("best_epoch"), summary.get("best_selection_raw_speed_mae"))
if __name__ == "__main__":
    main()
