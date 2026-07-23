# Reproduction Guide

Environment: `conda env create -f environment.yml` (frozen run: Python 3.11.1,
PyTorch 2.5.1+cu121, cuDNN 9.1, RTX 3060, Windows 10; Linux works identically -
determinism requires `CUBLAS_WORKSPACE_CONFIG=:4096:8`, which the runner enforces,
plus `torch.use_deterministic_algorithms(True)` and deterministic cuDNN, which the
runner sets).

1. `python scripts/verify_release.py` - all shipped artifact hashes + 150/150 records.
2. `pytest -q` - data-free checks: model shapes and log-scale clipping; the
   uniform-reliability no-op identity; elapsed-gap recurrence vs brute force and the
   native-carry continuation; training-mask determinism/containment/rates/structure;
   metric accumulators vs direct computation.
3. `python scripts/download_data.py` then `python scripts/audit_data.py` - raw hashes,
   split boundaries, native mask, nine frozen test masks.
4. `python scripts/train.py --seed {17,29,43,71,101}` - 30 frozen epochs each;
   expected best selection MAE ~= 1.747-1.754 (per-seed values recorded in
   `results/raw/*training_history*.json` and checkpoint metadata).
5. `python scripts/evaluate.py` - regenerates the 150-run primary evaluation; outputs
   are expected to match `results/raw/pems_bay_final_primary_evaluation_v1/` exactly
   under the pinned environment (per-seed CSV values reproduce to float64).
6. `python scripts/build_tables.py && python scripts/build_figures.py` - every LaTeX
   table and figure from result files; no hand-typed numbers.

Known non-reproduced items (by design, disclosed): five-seed PEMS-BAY ablations,
sensitivity sweeps, external baselines (see README and the manuscript Limitations).
