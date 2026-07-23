import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import fargfrc.training.pems_bay_far_gf_rc_runner_v2 as R

def _native(rng):
    return rng.random((R.EXPECTED_INPUT_STEPS, R.EXPECTED_SENSOR_COUNT)) > 0.05

def test_masks_deterministic_and_contained():
    rng = np.random.default_rng(2)
    native = _native(rng)
    order = np.argsort(np.random.default_rng(3).random((R.EXPECTED_SENSOR_COUNT,
                                                        R.EXPECTED_SENSOR_COUNT)), axis=1)
    for sid in ["iid_random_30pct", "temporal_tail_50pct_sensors_6steps",
                "spatial_geographic_knn4_cluster_16_full_history"]:
        m1 = R.make_training_artificial_mask(native, sid, order, 17, 3, 12345)
        m2 = R.make_training_artificial_mask(native, sid, order, 17, 3, 12345)
        assert (m1 == m2).all(), "not deterministic"
        assert not (m1 & ~native).any(), "removed a native-missing cell"

def test_iid_rate_and_tail_structure():
    rng = np.random.default_rng(4)
    native = _native(rng)
    order = np.argsort(np.random.default_rng(5).random((R.EXPECTED_SENSOR_COUNT,
                                                        R.EXPECTED_SENSOR_COUNT)), axis=1)
    iid = R.make_training_artificial_mask(native, "iid_random_50pct", order, 17, 1, 99)
    rate = iid.sum() / native.sum()
    assert abs(rate - 0.5) < 0.03
    tail = R.make_training_artificial_mask(native, "temporal_tail_25pct_sensors_3steps",
                                           order, 17, 1, 99)
    assert not tail[:-3].any()

def test_spatial_cluster_size():
    rng = np.random.default_rng(6)
    native = np.ones((R.EXPECTED_INPUT_STEPS, R.EXPECTED_SENSOR_COUNT), dtype=bool)
    order = np.argsort(np.random.default_rng(7).random((R.EXPECTED_SENSOR_COUNT,
                                                        R.EXPECTED_SENSOR_COUNT)), axis=1)
    m = R.make_training_artificial_mask(native, "spatial_geographic_knn4_cluster_8_full_history",
                                        order, 29, 2, 500)
    assert m.any(axis=0).sum() == 8
    assert m.all(axis=0).sum() == 8  # full history for the chosen cluster
