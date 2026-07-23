#!/usr/bin/env python3
"""Verify every hash-pinned shipped artifact and the internal provenance chain."""
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def sha(p): 
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
def main() -> None:
    failures = []
    def check(label, path, expected):
        got = sha(ROOT / path)
        ok = got == expected
        print(f"{'OK ' if ok else 'FAIL'} {label}")
        if not ok:
            failures.append((label, expected, got))
    # frozen model source must match the hash recorded in checkpoints & protocol
    check("model source (far_gf_rc.py)", "src/fargfrc/models/far_gf_rc.py",
          "6dd6cc74f27cffd70cfef6e4dda89d31e5db16a31d66517a2da1ba2004070d25")
    check("training runner v2", "src/fargfrc/training/pems_bay_far_gf_rc_runner_v2.py",
          "f0c4934c2954e4f528c391ec03aba2c55a9956d930646003049076d29451a433")
    proto = json.load(open(ROOT / "configs/frozen/pems_bay_far_gf_rc_training_protocol_v2.json"))
    dc = proto["data_contract"]
    check("physical graph", "data/processed/pems_bay_geographic_knn4_self_tuning_gaussian_physical_graph_v1.npz",
          dc["physical_graph_sha256"])
    check("spatial topology", "data/processed/pems_bay_geographic_knn4_spatial_failure_topology_v1.npz",
          dc["spatial_topology_sha256"])
    check("selection composite mask",
          "data/processed/controlled_dropout/pems_bay_v1/pems_bay_controlled_sensor_dropout_selection_composite_mask_v1.npz",
          dc["selection_mask_sha256"])
    check("test mask manifest",
          "data/processed/controlled_dropout/pems_bay_v1/pems_bay_controlled_sensor_dropout_test_primary_manifest_v1.json",
          dc["test_mask_manifest_sha256"])
    man = json.load(open(ROOT / "data/processed/controlled_dropout/pems_bay_v1/"
                                "pems_bay_controlled_sensor_dropout_test_primary_manifest_v1.json"))
    for r in man["primary_test"]["scenario_mask_records"]:
        name = Path(r["path"].replace("\\", "/")).name
        check(f"test mask {r['scenario_id']}", f"data/processed/controlled_dropout/pems_bay_v1/{name}", r["sha256"])
    # 150-run integrity of the archived final evaluation
    import csv
    rows = list(csv.DictReader(open(ROOT / "results/raw/pems_bay_final_primary_evaluation_v1/per_seed_metrics.csv")))
    runs = {(r["model_identifier"], r["model_seed"], r["condition_identifier"]) for r in rows}
    print(f"{'OK ' if len(runs) == 150 else 'FAIL'} archived run records: {len(runs)}/150")
    if len(runs) != 150:
        failures.append(("run count", 150, len(runs)))
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" ", f)
        sys.exit(1)
    print("\nAll release integrity checks passed.")
if __name__ == "__main__":
    main()
