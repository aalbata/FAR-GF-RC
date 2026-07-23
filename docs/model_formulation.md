# FAR-GF-RC - Verified Model Formulation

Every module below exists in the executable frozen source
`src/fargfrc/models/far_gf_rc.py`
(SHA-256 `6dd6cc74f27cffd70cfef6e4dda89d31e5db16a31d66517a2da1ba2004070d25`, byte-identical to
the source recorded in every checkpoint's provenance). Shapes use B = batch, T = 12 input steps,
K = 12 output steps, N = 325 sensors, D = 64 latent dim. Notation: value x, effective mask m ∈ {0,1},
normalized elapsed gap g ∈ [0,1] (cap 288 steps), calendar c ∈ R⁴ (daily/weekly sin–cos).

| # | Module | Input → Output | Equation | Params | Train/Infer | Loss role | Code |
|---|--------|----------------|----------|-------:|-------------|-----------|------|
| 1 | Reliability estimator | (m,g): B×T×N → r: B×T×N | ā_i = meanₜ m_{t,i}; r = clip( σ(MLP₃→₃₂→₁([m,g,ā])) · (m + (1−m)e^{−3g}), 0, 1) | 161 | both | BCE(r, m), weight 0.05 | `ReliabilityEstimator` |
| 2 | History projection + gate | B×T×N×(7+1) → H⁰: B×T×N×D | h = Drop(GELU(W[x‖r])); h ← h·(0.25 + 0.75 r) | 576 | both | - | `history_projection` |
| 3 | Temporal GRU (sensor-shared) | B·N×T×D → B·N×T×D | standard 1-layer GRU over time, weights shared across sensors | 24,960 | both | - | `temporal_encoder` |
| 4 | Adaptive functional graph | - → Ã: N×N | Ã = softmax(E_s E_tᵀ / √16), row-stochastic, input-independent | 10,400 | both | - | `AdaptiveFunctionalGraph` |
| 5 | Reliability-gated dual-graph layers ×2 | H: B×N×D, r_T: B×N → B×N×D | For S ∈ {Â, Ã}: M_S = rownorm(S ⊙ r_T)H; H ← LN(H + MLP([H‖M_Â‖M_Ã‖r_T])). Â = rownorm(clip(A_phys,0)+I) is a fixed buffer. Uniform r cancels exactly (unit-tested). | 33,408 | both | - | `ReliabilityGatedGraphLayer` |
| 6 | Future-calendar encoder | B×K×4 → B×K×D | FC + GELU + dropout | 320 | both | - | `future_calendar_encoder` |
| 7 | Horizon embedding | K → K×D | learned lookup | 768 | both | - | `horizon_embedding` |
| 8 | Forecast decoder | B×K×N×D → (ŷ, log s): B×K×N each | z_{k,i} = H_i + FC(c_k) + e_k; [ŷ, log s] = MLP_{D→64→2}(z); log s clipped to [−6, 3] | 4,290 | both | ŷ: masked L1 (w=1.0); log s: Gaussian NLL ½[(e/s)² + 2 log s] (w=0.05) | `forecast_decoder` |
| 9 | Reconstruction head | [GRU states ‖ broadcast H]: B×T×N×2D → x̂: B×T×N | MLP_{2D→64→1} | 8,321 | training-only output | masked L1 on **artificially removed** native cells (w=0.25) | `reconstruction_head` |

Total: 83,204 trainable parameters (MS-GRU 21,609; MS-TCN-v2 106,921 - the proposed model
is smaller than the stronger baseline).

Training objective (frozen weights from `configs/frozen/pems_bay_far_gf_rc_training_protocol_v2.json`):

L = 1.0·L1_masked(ŷ, y; m_target) + 0.05·NLL_masked + 0.25·L1_recon(x̂, x_native; m_artificial) + 0.05·BCE(r, m_history)

with AdamW (lr 1e-3, wd 1e-4), grad-clip 5.0, batch 32, 30 epochs, four-stage progressive failure
curriculum (clean → IID → temporal → mixed), checkpoint selected by minimum raw-speed MAE on the
frozen balanced nine-scenario selection composite.

## Honest empirical characterization (informs the manuscript's framing)

Verified from frozen artifacts, not asserted:

1. The reliability BCE targets the mask, which is also an input to the head; the loss collapses
   from 0.0746 to 0.0012 in one epoch and the trained head saturates to a binary mask copy
   (inference: r = 1.0000 observed / 0.0002 dropped). The e^{−3g} recency prior is numerically
   inert within a 12-step window (g ≤ 12/288 ⇒ prior ∈ [0.88, 1]).
2. Single-seed METR-LA ablations: removing the progressive curriculum costs +0.074…+0.147
   overall MAE across conditions (≈12–23× the seed-level SD); removing the failure-awareness
   inputs or the reconstruction objective changes −0.020…+0.001. The paper therefore credits
   robustness primarily to the curriculum and level accuracy to the dual-graph architecture, and
   reports the reliability/reconstruction components as design choices whose measured marginal
   contribution is ≈ 0 (single-seed evidence; five-seed ablations are open work).
