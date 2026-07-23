import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from fargfrc.training.pems_bay_far_gf_rc_runner_v2 import (
    build_native_elapsed_steps, build_window_elapsed_feature)

def test_native_elapsed_matches_bruteforce():
    rng = np.random.default_rng(0)
    mask = rng.random((200, 5)) > 0.3
    cap = 24
    fast = build_native_elapsed_steps(mask, cap)
    slow = np.empty_like(fast)
    for s in range(5):
        gap = cap
        for t in range(200):
            gap = 0 if mask[t, s] else min(gap + 1, cap)
            slow[t, s] = gap
    assert (fast == slow).all()

def test_window_recurrence_continues_native_carry():
    rng = np.random.default_rng(1)
    mask = rng.random((100, 4)) > 0.4
    cap = 288
    native = build_native_elapsed_steps(mask, cap)
    t0 = 40
    win = build_window_elapsed_feature(mask[t0:t0 + 12], native[t0 - 1], cap)
    assert win.shape == (12, 4)
    assert (native[t0:t0 + 12].astype(np.float32) / cap == win).all()
