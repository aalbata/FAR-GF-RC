#!/usr/bin/env python3
"""Five-seed PEMS-BAY ablations: protocol is frozen; runs are NOT yet executed.

Archived evidence: results/raw/metr_la_far_gf_rc_ablation_v1_seed_17_..._v2_....npz
(single-seed METR-LA; overall-MAE deltas vs full model: removing the progressive
curriculum costs +0.074..+0.147; removing failure-awareness inputs or the
reconstruction objective changes -0.02..+0.001). The manuscript scopes these as
single-seed evidence. This script intentionally refuses to fabricate the
five-seed PEMS-BAY ablation results."""
import sys
sys.exit("Ablation training for PEMS-BAY (5 seeds x 3 variants) is defined in "
         "configs/ablations/metr_la_far_gf_rc_ablation_protocol_v1.json (variant semantics) "
         "but has not been executed. Run scripts/train.py-style runs per variant, then evaluate.py.")
