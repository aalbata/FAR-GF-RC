# Data

`data/processed/` ships every frozen derived artifact (split, native mask,
train-only normalization, physical graph, spatial topology, selection composite,
nine test dropout masks) - all SHA-256-pinned; verify with
`python scripts/verify_release.py`.

Raw files are NOT redistributed. Fetch + verify:

    python scripts/download_data.py --dest data/raw/PEMSBAY

Sources: PEMS-BAY / METR-LA benchmarks introduced by Li et al. (ICLR 2018,
DCRNN); files mirrored at github.com/deepkashiwa20/DL-Traff-Graph
(commit ccc038a). Expected hashes:

    pems-bay.h5                       65d69fb0a2323dba9867179eb7af47c8b814186bc459ff0a4937d21614153c8f
    graph_sensor_locations_bay.csv    276ee01059610774d4e59572507f7e32eaac21f1f5882fcd9e3d7d426a4b7a6c
