#!/usr/bin/env python3
"""Download the raw PEMS-BAY files from the public DL-Traff-Graph mirror and verify SHA-256."""
import argparse, hashlib, sys, urllib.request
from pathlib import Path
FILES = {
    "pems-bay.h5": (
        "https://github.com/deepkashiwa20/DL-Traff-Graph/raw/ccc038aeef05ffd43fab42e0752c8f94b90163a7/PEMSBAY/pems-bay.h5",
        "65d69fb0a2323dba9867179eb7af47c8b814186bc459ff0a4937d21614153c8f",
    ),
    "graph_sensor_locations_bay.csv": (
        "https://github.com/deepkashiwa20/DL-Traff-Graph/raw/ccc038aeef05ffd43fab42e0752c8f94b90163a7/PEMSBAY/graph_sensor_locations_bay.csv",
        "276ee01059610774d4e59572507f7e32eaac21f1f5882fcd9e3d7d426a4b7a6c",
    ),
}
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=Path("data/raw/PEMSBAY"))
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, (url, expected) in FILES.items():
        target = args.dest / name
        if not target.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, target)
        got = sha256(target)
        status = "OK" if got == expected else "HASH MISMATCH"
        ok &= got == expected
        print(f"{name}: {status}\n  expected {expected}\n  got      {got}")
    sys.exit(0 if ok else 1)
if __name__ == "__main__":
    main()
