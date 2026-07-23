# Frozen Data Protocol (PEMS-BAY)

Raw source: `pems-bay.h5` (52,116 x 325 five-minute speeds) and
`graph_sensor_locations_bay.csv`, obtained from the DL-Traff-Graph mirror
(commit `ccc038aeef05ffd43fab42e0752c8f94b90163a7`) and pinned to
SHA-256 `65d69fb0a2323dba9867179eb7af47c8b814186bc459ff0a4937d21614153c8f` /
`276ee01059610774d4e59572507f7e32eaac21f1f5882fcd9e3d7d426a4b7a6c`.
Timestamps are the file's naive local axis.

Native observation mask: a cell is observed iff its value is finite and > 0
(PEMS-BAY availability 99.997%). Chronological split (raw indices, [start, end)):
train [0, 36481), selection [36481, 39087), calibration [39087, 41692) - reserved
and never accessed - test [41692, 52116). Valid target windows: 36,458 / 2,595 /
2,594 / 10,413. Normalization: per-sensor z-score fit on native-observed TRAIN
values only; unobserved history cells carry a 0.0 placeholder after masking.

Features per history step (7): normalized value, effective mask, causal elapsed gap
(recurrence gap<-0 if observed else min(gap+1, 288), seeded from the native state at
t-13; normalized by 288), daily sin/cos, weekly sin/cos. Future calendar (4):
daily/weekly sin-cos for each of the 12 target steps.

Controlled dropout: history-only; targets remain native. Nine frozen test
conditions + clean: IID {10,30,50}%, temporal tails {25%/3, 50%/6, 75%/12},
geographic clusters {8,16,32} = center + (k-1) nearest by Haversine distance
(protocol seed 20260702; one frozen mask file per condition shared by every
model and seed; every removed cell verified native-observed). The training
curriculum regenerates the same scenario families dynamically and
deterministically per (seed, epoch, window); its spatial clusters use the
neighbour-order array, which EXCLUDES the center sensor - a documented
train/eval composition difference (both are k-sensor geographic blobs applied
identically to all models).

Leakage controls: no primary-test target is used for selection, tuning,
calibration, early stopping, or design; checkpoints are chosen on the frozen
selection composite only; the only calibration-interval access is rows
[41680, 41692) as causal INPUT history for the first 12 test windows, per the
boundary-history addendum. Elapsed-gap carry for test windows reads
availability bits only. All artifacts above are SHA-256-pinned and re-checked by
`scripts/verify_release.py` and `scripts/audit_data.py`.
