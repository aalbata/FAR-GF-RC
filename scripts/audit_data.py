#!/usr/bin/env python3
"""Audit raw + frozen processed artifacts against the frozen protocol (shape, hashes, split, masks)."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--project-root", type=Path, default=Path("."))
    root = ap.parse_args().project_root.resolve()
    proto = json.load(open(root / "configs/experiments/pems_bay_far_gf_rc_data_protocol_v1.json"))
    failures = []
    raw = root / "data/raw/PEMSBAY/pems-bay.h5"
    if raw.exists():
        exp = proto["source_data"]["traffic_hdf5_sha256"]
        got = sha256(raw)
        print(f"raw pems-bay.h5: {'OK' if got == exp else 'MISMATCH'}")
        if got != exp: failures.append("raw h5 hash")
    else:
        print("raw pems-bay.h5: absent (training/evaluation need it; frozen artifacts do not)")
    split = np.load(root / "data/processed/pems_bay_temporal_split_v1.npz")
    b = split["split_boundaries"]
    assert (b == np.array([[0, 36481], [36481, 39087], [39087, 41692], [41692, 52116]])).all(), "split boundaries changed"
    print("temporal split boundaries: OK", b.tolist())
    native = np.load(root / "data/processed/pems_bay_native_observation_mask_v1.npz")["native_observation_mask"]
    assert native.shape == (52116, 325)
    print(f"native mask: OK shape={native.shape} availability={native.mean():.6f}")
    man = json.load(open(root / "data/processed/controlled_dropout/pems_bay_v1/pems_bay_controlled_sensor_dropout_test_primary_manifest_v1.json"))
    for r in man["primary_test"]["scenario_mask_records"]:
        p = root / "data/processed/controlled_dropout/pems_bay_v1" / Path(r["path"].replace("\\", "/")).name
        ok = sha256(p) == r["sha256"]
        print(f"test mask {r['scenario_id']}: {'OK' if ok else 'MISMATCH'}")
        if not ok: failures.append(r["scenario_id"])
    sys.exit(1 if failures else 0)
if __name__ == "__main__":
    main()
