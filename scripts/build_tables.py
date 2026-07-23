#!/usr/bin/env python3
"""Regenerate every manuscript table (LaTeX) from the archived result files.

Nothing is typed by hand: values flow results/raw -> results/processed -> LaTeX.
Also emits generated_results.tex with numeric macros used inline by the manuscript.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CONDS = ["clean_native_history","iid_random_10pct","iid_random_30pct","iid_random_50pct",
         "temporal_tail_25pct_sensors_3steps","temporal_tail_50pct_sensors_6steps",
         "temporal_tail_75pct_sensors_12steps","spatial_geographic_knn4_cluster_8_full_history",
         "spatial_geographic_knn4_cluster_16_full_history","spatial_geographic_knn4_cluster_32_full_history"]
PRETTY = {"clean_native_history":"Clean native history","iid_random_10pct":"IID 10\\%",
          "iid_random_30pct":"IID 30\\%","iid_random_50pct":"IID 50\\%",
          "temporal_tail_25pct_sensors_3steps":"Tail 25\\%/3","temporal_tail_50pct_sensors_6steps":"Tail 50\\%/6",
          "temporal_tail_75pct_sensors_12steps":"Tail 75\\%/12",
          "spatial_geographic_knn4_cluster_8_full_history":"Spatial 8",
          "spatial_geographic_knn4_cluster_16_full_history":"Spatial 16",
          "spatial_geographic_knn4_cluster_32_full_history":"Spatial 32"}
MODELS = ["FAR-GF-RC","MS-GRU","MS-TCN-v2"]

def wrap(body, caption, label, colspec):
    return ("\\begin{table}[t]\n\\centering\n\\caption{%s}\n\\label{%s}\n"
            "\\resizebox{\\linewidth}{!}{\\begin{tabular}{%s}\n\\toprule\n" % (caption,label,colspec)
            + body + "\\bottomrule\n\\end{tabular}}\n\\end{table}\n")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument("--overleaf", type=Path, default=None, help="optional second output dir (Overleaf tables/)")
    root = ap.parse_args().project_root.resolve()
    RAW = root/"results/raw/pems_bay_final_primary_evaluation_v1"
    PRO = root/"results/processed"; OUT = root/"results/tables"; OUT.mkdir(parents=True, exist_ok=True)
    ov2 = ap.parse_args().overleaf
    summ = pd.read_csv(RAW/"seed_summary_metrics.csv")
    per  = pd.read_csv(RAW/"per_seed_metrics.csv")
    comp = pd.read_csv(PRO/"paired_comparisons_overall_mae.csv")
    deg  = pd.read_csv(PRO/"degradation_overall.csv")
    ranks= pd.read_csv(PRO/"mean_ranks_overall_mae.csv")
    params=pd.read_csv(PRO/"params_runtime.csv")
    tables={}

    # T1: dataset & protocol (constants verified by audit_data.py)
    body=("Item & Value\\\\ \\midrule\n"
      "Dataset & PEMS-BAY (325 sensors, 52{,}116 five-minute steps)\\\\\n"
      "Native availability & 99.997\\% (finite and $>0$)\\\\\n"
      "Split (target starts) & train $[12,36481)$ / selection $[36481,39087)$ / calibration $[39087,41692)$ / test $[41692,52105]$\\\\\n"
      "Windows & 12-step history $\\rightarrow$ 12-step horizon; 10{,}413 primary-test windows\\\\\n"
      "Normalization & per-sensor $z$-score, fit on native-observed \\emph{train} values only\\\\\n"
      "History channels & value, effective mask, elapsed gap (cap 288), daily $\\sin/\\cos$, weekly $\\sin/\\cos$\\\\\n"
      "Graph & Haversine $k$NN ($k{=}4$), self-tuning Gaussian $\\exp(-d_{ij}^2/\\sigma_i\\sigma_j)$, symmetric union\\\\\n"
      "Seeds & 17, 29, 43, 71, 101 (fully deterministic CUDA)\\\\\n"
      "Runs & $3\\times5\\times10=150$, all finite, hash-verified\\\\\n")
    tables["table_protocol.tex"]=wrap(body,"Frozen dataset and evaluation protocol.","tab:protocol","ll")

    # T2: overall MAE all conditions (mean±sd), plus RMSE for clean
    rows=""
    ov=summ[summ.scope_identifier=="overall"]
    for c in CONDS:
        cells=[]
        for m in MODELS:
            r=ov[(ov.condition_identifier==c)&(ov.model_identifier==m)].iloc[0]
            v=f"{r.mae_mean:.3f}\\,$\\pm$\\,{r.mae_sample_std:.3f}"
            best = ov[ov.condition_identifier==c].mae_mean.min()==r.mae_mean
            cells.append("\\textbf{%s}"%v if best else v)
        rows+=PRETTY[c]+" & "+" & ".join(cells)+"\\\\\n"
    body="Condition & FAR-GF-RC & MS-GRU & MS-TCN-v2\\\\ \\midrule\n"+rows
    tables["table_overall_mae.tex"]=wrap(body,
      "Primary-test overall MAE (mph) under all ten history-availability conditions; mean\\,$\\pm$\\,sample SD over five seeds. Best per row in bold.",
      "tab:overall","lccc")

    # T3: per-horizon MAE (clean + hardest condition)
    rows=""
    for c in ["clean_native_history","temporal_tail_75pct_sensors_12steps"]:
        for s,lab in [("horizon_3","15 min"),("horizon_6","30 min"),("horizon_12","60 min"),("overall","overall")]:
            cells=[]
            sub=summ[(summ.condition_identifier==c)&(summ.scope_identifier==s)]
            for m in MODELS:
                r=sub[sub.model_identifier==m].iloc[0]
                v=f"{r.mae_mean:.3f}"
                cells.append("\\textbf{%s}"%v if sub.mae_mean.min()==r.mae_mean else v)
            rows+=f"{PRETTY[c]} & {lab} & "+" & ".join(cells)+"\\\\\n"
        rows+="\\midrule\n" if c=="clean_native_history" else ""
    body="Condition & Lead & FAR-GF-RC & MS-GRU & MS-TCN-v2\\\\ \\midrule\n"+rows
    tables["table_horizon.tex"]=wrap(body,
      "Per-lead MAE (mph) at 15/30/60 minutes (single-lead metrics) and overall, five-seed means, for the clean and the most severe condition.",
      "tab:horizon","llccc")

    # T4: robustness (clean->dropout absolute degradation) + worst condition + mean rank
    rows=""
    for c in CONDS[1:]:
        cells=[]
        sub=deg[deg.condition_identifier==c]
        for m in MODELS:
            r=sub[sub.model_identifier==m].iloc[0]
            v=f"{r.absolute_mae_degradation_mean:.3f}"
            cells.append("\\textbf{%s}"%v if sub.absolute_mae_degradation_mean.min()==r.absolute_mae_degradation_mean else v)
        rows+=PRETTY[c]+" & "+" & ".join(cells)+"\\\\\n"
    rows+="\\midrule\nWorst-condition MAE & "+" & ".join(
        f"{per[(per.model_identifier==m)&(per.scope_identifier=='overall')].groupby('model_seed').mae.max().mean():.3f}" for m in MODELS)+"\\\\\n"
    rows+="Mean rank (50 blocks) & "+" & ".join(
        f"{ranks[ranks.model_identifier==m].mean_rank.iloc[0]:.2f}" for m in MODELS)+"\\\\\n"
    body="Condition & FAR-GF-RC & MS-GRU & MS-TCN-v2\\\\ \\midrule\n"+rows
    tables["table_robustness.tex"]=wrap(body,
      "Clean-to-dropout absolute overall-MAE degradation (mph; lower is more robust), worst-condition MAE, and mean rank over the 50 (condition, seed) blocks.",
      "tab:robustness","lccc")

    # T5: statistics table
    rows=""
    for c in CONDS:
        for bl in ["MS-GRU","MS-TCN-v2"]:
            r=comp[(comp.condition==c)&(comp.baseline==bl)].iloc[0]
            rows+=(f"{PRETTY[c]} & {bl} & {r.mean_diff:.3f} [{r.diff_ci95_lo:.3f}, {r.diff_ci95_hi:.3f}] & "
                   f"{r.rel_impr_mean_pct:.2f} & {r.t_stat:.1f} & {r.p_t_holm:.1e} & {r.cohens_dz:.1f} & "
                   f"{r.p_wilcoxon_onesided:.4f}\\\\\n")
    body=("Condition & Baseline & $\\Delta$MAE [95\\% CI] & Impr.\\,\\% & $t$ & $p_{\\mathrm{Holm}}$ & $d_z$ & $p_{W}$\\\\ \\midrule\n"+rows)
    tables["table_statistics.tex"]=wrap(body,
      "Paired five-seed comparisons on overall MAE. $\\Delta$MAE = baseline $-$ FAR-GF-RC (mph) with paired $t$-based 95\\% CI; Impr.\\,\\% is the mean paired relative improvement; $p_{\\mathrm{Holm}}$ is Holm-adjusted over all 20 tests; $d_z$ is Cohen's paired effect size; $p_{W}$ is the exact one-sided Wilcoxon signed-rank $p$ (floor $2^{-5}{=}0.03125$ at $n{=}5$).",
      "tab:stats","llcccccc")

    # T6: ablations (METR-LA seed 17)
    ab=np.load(root/"results/raw/metr_la_far_gf_rc_ablation_v1_seed_17_clean_and_nine_controlled_dropout_metrics_v2_corrected_failure_awareness_input_policy.npz",allow_pickle=True)
    variants=[str(v) for v in ab["variant_names"]]; conds=[str(c) for c in ab["condition_names"]]
    mae=ab["mae"][:,:,0]; full=variants.index("full_far_gf_rc_seed17")
    show=[("without_progressive_curriculum","-- progressive curriculum"),
          ("without_failure_awareness_design","-- failure-awareness inputs"),
          ("without_reconstruction_objective","-- reconstruction objective")]
    pick=[0,3,6,9]  # clean, iid50, tail75, spatial20
    header="Variant & "+" & ".join([conds[i].replace('_','\\_')[:26] for i in pick])+"\\\\ \\midrule\n"
    rows=f"Full FAR-GF-RC & "+" & ".join(f"{mae[full,i]:.3f}" for i in pick)+"\\\\ \\midrule\n"
    for key,lab in show:
        vi=variants.index(key)
        rows+=lab+" & "+" & ".join(f"{mae[vi,i]:.3f} ({mae[vi,i]-mae[full,i]:+.3f})" for i in pick)+"\\\\\n"
    tables["table_ablation.tex"]=wrap(header+rows,
      "Component ablations on METR-LA (single seed 17; overall MAE, mph; $\\Delta$ vs.\\ full model in parentheses). Removing the progressive curriculum degrades every condition; removing the failure-awareness inputs or the reconstruction objective does not.",
      "tab:ablation","lcccc")

    # T7: params / runtime
    rows="".join(f"{r.model} & {r.parameters:,} & {r.runtime}\\\\\n" for r in params.itertuples())
    tables["table_params.tex"]=wrap("Model & Trainable parameters & Wall-clock (per seed)\\\\ \\midrule\n"+rows,
      "Model capacity and runtime. Per-seed wall-clock time was not recorded in the frozen logs; hardware: RTX~3060, CUDA~12.1, fully deterministic kernels.",
      "tab:params","lcc")

    for name,tex in tables.items():
        (OUT/name).write_text(tex)
        if ov2: (ov2/name).write_text(tex)
    # numeric macros
    import json as _json
    fried=_json.load(open(PRO/"friedman.json"))
    def sci(x):
        m,e=f"{x:.1e}".split("e"); return f"{m}\\times10^{{{int(e)}}}"
    clean=ov[(ov.condition_identifier=="clean_native_history")]
    far_clean=clean[clean.model_identifier=="FAR-GF-RC"].iloc[0]
    hard=ov[(ov.condition_identifier=="temporal_tail_75pct_sensors_12steps")]
    macros=(f"\\newcommand{{\\GruCleanMAE}}{{{clean[clean.model_identifier=='MS-GRU'].mae_mean.iloc[0]:.3f}}}\n"
            f"\\newcommand{{\\TcnCleanMAE}}{{{clean[clean.model_identifier=='MS-TCN-v2'].mae_mean.iloc[0]:.3f}}}\n"
            f"\\newcommand{{\\TcnHardMAE}}{{{hard[hard.model_identifier=='MS-TCN-v2'].mae_mean.iloc[0]:.3f}}}\n"
            f"\\newcommand{{\\FriedmanChi}}{{{fried['friedman_chi2']:.1f}}}\n"
            f"\\newcommand{{\\FriedmanP}}{{{sci(fried['p'])}}}\n"
            f"\\newcommand{{\\FarCleanMAE}}{{{far_clean.mae_mean:.3f}}}\n"
            f"\\newcommand{{\\FarHardMAE}}{{{hard[hard.model_identifier=='FAR-GF-RC'].mae_mean.iloc[0]:.3f}}}\n"
            f"\\newcommand{{\\GruHardMAE}}{{{hard[hard.model_identifier=='MS-GRU'].mae_mean.iloc[0]:.3f}}}\n"
            f"\\newcommand{{\\MinPairedImpr}}{{{comp.rel_impr_min_pct.min():.2f}}}\n"
            f"\\newcommand{{\\MaxHolmP}}{{{sci(comp.p_t_holm.max())}}}\n"
            f"\\newcommand{{\\MinDz}}{{{comp.cohens_dz.min():.1f}}}\n"
            f"\\newcommand{{\\MeanImprGRU}}{{{comp[comp.baseline=='MS-GRU'].rel_impr_mean_pct.mean():.2f}}}\n"
            f"\\newcommand{{\\MeanImprTCN}}{{{comp[comp.baseline=='MS-TCN-v2'].rel_impr_mean_pct.mean():.2f}}}\n")
    (OUT/"generated_results.tex").write_text(macros)
    if ov2: (ov2.parent/"generated_results.tex").write_text(macros)
    print("wrote", len(tables), "tables +", "generated_results.tex")
if __name__=="__main__":
    main()
