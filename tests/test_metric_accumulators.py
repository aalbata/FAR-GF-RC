import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from fargfrc.evaluation.pems_bay_final_multi_model_primary_evaluator_v4 import (
    empty_metric_accumulator, update_metric_accumulator, finalize_metric_accumulator)

def test_accumulator_matches_direct_computation():
    rng = np.random.default_rng(0)
    pred = torch.tensor(rng.normal(60, 5, (7, 12, 11)), dtype=torch.float64)
    targ = torch.tensor(rng.uniform(30, 70, (7, 12, 11)), dtype=torch.float64)
    mask = torch.tensor(rng.random((7, 12, 11)) > 0.2)
    acc = empty_metric_accumulator()
    for b in range(7):  # streaming in chunks
        update_metric_accumulator(acc, pred[b:b + 1], targ[b:b + 1], mask[b:b + 1])
    out = finalize_metric_accumulator(acc)
    p, t = pred[mask].numpy(), targ[mask].numpy()
    assert np.isclose(out["mae"], np.abs(p - t).mean())
    assert np.isclose(out["rmse"], np.sqrt(((p - t) ** 2).mean()))
    assert np.isclose(out["mape_percent"], 100 * (np.abs(p - t) / np.abs(t)).mean())
    assert np.isclose(out["wmape_percent"], 100 * np.abs(p - t).sum() / np.abs(t).sum())
    assert out["scored_cells"] == int(mask.sum())
