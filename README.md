# FAR-GF-RC: Failure-Aware Reliability-Gated Graph Forecasting with Robust Curriculum

Repository: https://github.com/aalbata/FAR-GF-RC

Reference implementation, frozen experimental protocol, and archived results for
"FAR-GF-RC: Robust Dual-Graph Traffic-Speed Forecasting Under Structured Sensor Dropout".

FAR-GF-RC is a reliability-gated dual-graph (geographic + adaptive) multi-horizon
traffic-speed forecaster trained with a progressive sensor-failure curriculum. It is
evaluated on **PEMS-BAY** (325 sensors, 52,116 five-minute steps) under a frozen,
leakage-safe protocol with **10 controlled history-dropout conditions × 3 models ×
5 seeds = 150 verified runs**, against equally failure-trained MS-GRU and MS-TCN-v2
baselines. A supporting METR-LA line (5-seed comparatives; seed-17 ablations) is archived.

## Headline result (PEMS-BAY primary test, overall MAE, 5-seed mean)

| Condition | FAR-GF-RC | MS-GRU | MS-TCN-v2 |
|---|---|---|---|
| Clean native history | **1.627** | 1.735 | 1.743 |
| IID 50% dropout | **1.733** | 1.841 | 1.858 |
| Temporal tail 75% / 12 steps | **2.290** | 2.554 | 2.549 |
| Spatial cluster 32 (full history) | **1.709** | 1.841 | 1.847 |

Every one of the 100 paired seed-level comparisons favors FAR-GF-RC (minimum
improvement +4.87%); all 20 condition×baseline paired t-tests survive Holm
correction (max adjusted p = 8.5e-5); FAR-GF-RC ranks first in all 50
(condition, seed) blocks (Friedman χ² = 78.2, p ≈ 1e-17).

## Repository layout

```
configs/frozen/        Byte-exact frozen protocol JSONs (Windows paths, original hashes)
configs/experiments/   Portable copies: project-root-relative paths; one documented hash correction
configs/{datasets,models,dropout,ablations}/  Category views of the portable configs
data/processed/        Frozen PEMS-BAY artifacts: split, native mask, normalization, graph,
                       spatial topology, 9 frozen test dropout masks + selection composite (hash-pinned)
data/README.md         Raw-data download + SHA-256 verification (raw HDF5 not redistributed)
src/fargfrc/           Verbatim frozen sources (hash-preserving) organized as a package
scripts/               Thin CLIs: download/audit/train/evaluate/build tables & figures/verify
results/raw/           Frozen final evaluation outputs + training histories (the paper's evidence)
results/processed/     Derived statistics (descriptives, paired tests + Holm, ranks, Friedman)
results/{tables,figures}/  Regenerated LaTeX tables and the archived publication figures
checkpoints/README.md  SHA-256 manifest for the 15 released checkpoints
docs/                  data_protocol.md, model_formulation.md, reproduction.md
tests/                 Data-free unit tests (model shapes, elapsed-gap recurrence, masks, metrics)
```

## Quick start

```bash
# 1) Environment (CPU is sufficient for tests/tables; training used CUDA 12.1)
conda env create -f environment.yml && conda activate fargfrc   # or: pip install -r requirements-lock.txt

# 2) Integrity of shipped artifacts
python scripts/verify_release.py

# 3) Data-free unit tests
pytest -q

# 4) Raw data (required only for training / re-evaluation)
python scripts/download_data.py --dest data/raw/PEMSBAY
python scripts/audit_data.py --project-root .

# 5) Reproduce training (one seed shown; seeds: 17 29 43 71 101)
python scripts/train.py --project-root . --seed 17 --device cuda

# 6) Reproduce the frozen 150-run primary evaluation
python scripts/evaluate.py --project-root . --device cuda

# 7) Regenerate every manuscript table and figure from result files
python scripts/build_tables.py --project-root .
python scripts/build_figures.py --project-root .
```

`run_all.py` chains steps 4–7. Training determinism requires
`CUBLAS_WORKSPACE_CONFIG=:4096:8` (the runner sets and enforces this).

## What is and is not reproduced here

Archived and regenerable from this repository: the 150-run PEMS-BAY primary
evaluation, all statistics, all tables, all figures, and strict checkpoint
restoration (tested). **Not yet run** (protocols included, honestly scoped in the
manuscript's Limitations): five-seed ablations on PEMS-BAY, sensitivity sweeps,
and external graph baselines (DCRNN / Graph WaveNet / AGCRN). `run_ablations.py`
and `run_sensitivity.py` document the planned frozen protocols and exit with a
clear message rather than pretending to results.

## Data attribution

PEMS-BAY and METR-LA originate from Li et al., *Diffusion Convolutional Recurrent
Neural Network* (ICLR 2018); files are obtained from the public DL-Traff-Graph
mirror (github.com/deepkashiwa20/DL-Traff-Graph, commit `ccc038a`) and verified
against SHA-256 `65d69fb0…153c8f` (pems-bay.h5) and `276ee010…b7a6c`
(graph_sensor_locations_bay.csv). See `data/README.md`.

## Citation

See `CITATION.cff`. License: MIT (code and configs); dataset licenses remain with
their original distributors.
