#!/usr/bin/env python3
"""Regenerate submission figures from the archived result tables.

The hash-pinned publication figures remain untouched. New figures are written
alongside them with a ``submission_`` prefix and use the manuscript's GC and TT
condition abbreviations.
"""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


CONDITIONS = [
    "clean_native_history",
    "iid_random_10pct",
    "iid_random_30pct",
    "iid_random_50pct",
    "temporal_tail_25pct_sensors_3steps",
    "temporal_tail_50pct_sensors_6steps",
    "temporal_tail_75pct_sensors_12steps",
    "spatial_geographic_knn4_cluster_8_full_history",
    "spatial_geographic_knn4_cluster_16_full_history",
    "spatial_geographic_knn4_cluster_32_full_history",
]
TICK_LABELS = [
    "Clean",
    "IID-10%",
    "IID-30%",
    "IID-50%",
    "TT-25/3",
    "TT-50/6",
    "TT-75/12",
    "GC-8",
    "GC-16",
    "GC-32",
]
MODELS = ["FAR-GF-RC", "MS-GRU", "MS-TCN-v2"]
COLORS = {
    "FAR-GF-RC": "#1f77b4",
    "MS-GRU": "#ff7f0e",
    "MS-TCN-v2": "#2ca02c",
}
MARKERS = {"FAR-GF-RC": "o", "MS-GRU": "s", "MS-TCN-v2": "^"}
LINESTYLES = {"FAR-GF-RC": "-", "MS-GRU": "--", "MS-TCN-v2": ":"}


def _row(frame: pd.DataFrame, **filters: str) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def _format_axis(
    ax: plt.Axes,
    ylabel: str,
    xlabel: str,
    tick_fontsize: float = 9,
    label_fontsize: float = 10,
    tick_rotation: float = 0,
) -> None:
    ax.set_xticks(np.arange(len(TICK_LABELS)))
    ax.set_xticklabels(
        TICK_LABELS,
        fontsize=tick_fontsize,
        rotation=tick_rotation,
        ha="right" if tick_rotation else "center",
    )
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.grid(axis="y", alpha=0.28, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=tick_fontsize)


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _line_figure(
    summary: pd.DataFrame,
    scope: str,
    title: str,
    ylabel: str,
    stem: str,
    output_dir: Path,
    compact: bool = False,
) -> None:
    if compact:
        fig, ax = plt.subplots(figsize=(5.0, 2.6))
        title_fontsize = 11
        tick_fontsize = 8
        label_fontsize = 9
        legend_fontsize = 7.5
        tick_rotation = 20
    else:
        fig, ax = plt.subplots(figsize=(12.62, 5.92))
        title_fontsize = 13
        tick_fontsize = 9
        label_fontsize = 10
        legend_fontsize = 9
        tick_rotation = 0
    x = np.arange(len(CONDITIONS))
    for model in MODELS:
        rows = [
            _row(
                summary,
                condition_identifier=condition,
                model_identifier=model,
                scope_identifier=scope,
            )
            for condition in CONDITIONS
        ]
        ax.errorbar(
            x,
            [row.mae_mean for row in rows],
            yerr=[row.mae_sample_std for row in rows],
            color=COLORS[model],
            marker=MARKERS[model],
            linestyle=LINESTYLES[model],
            linewidth=1.8,
            markersize=5.5,
            capsize=3,
            label=model,
        )
    ax.set_title(title, fontsize=title_fontsize)
    _format_axis(
        ax,
        ylabel,
        "History availability condition",
        tick_fontsize=tick_fontsize,
        label_fontsize=label_fontsize,
        tick_rotation=tick_rotation,
    )
    ax.legend(
        title="Model",
        frameon=False,
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
    )
    _save(fig, output_dir, stem)


def _degradation_figure(
    degradation: pd.DataFrame, output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(12.62, 5.92))
    conditions = CONDITIONS[1:]
    labels = TICK_LABELS[1:]
    x = np.arange(len(conditions))
    width = 0.25
    for index, model in enumerate(MODELS):
        rows = [
            _row(
                degradation,
                condition_identifier=condition,
                model_identifier=model,
                scope_identifier="overall",
            )
            for condition in conditions
        ]
        ax.bar(
            x + (index - 1) * width,
            [row.absolute_mae_degradation_mean for row in rows],
            width=width,
            yerr=[row.absolute_mae_degradation_sample_std for row in rows],
            capsize=3,
            color=COLORS[model],
            label=model,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Absolute MAE increase from clean history (mph)", fontsize=11)
    ax.set_xlabel("Controlled dropout condition", fontsize=10)
    ax.set_title(
        "Forecasting-error degradation under sensor dropout", fontsize=13
    )
    ax.grid(axis="y", alpha=0.28, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(title="Model", frameon=False, fontsize=9, title_fontsize=9)
    _save(
        fig,
        output_dir,
        "submission_Figure_2_MAE_degradation_across_dropout_conditions",
    )


def _improvement_figure(
    improvement: pd.DataFrame, output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(12.62, 5.92))
    x = np.arange(len(CONDITIONS))
    baselines = ["MS-GRU", "MS-TCN-v2"]
    width = 0.34
    for index, baseline in enumerate(baselines):
        rows = [
            _row(
                improvement,
                condition_identifier=condition,
                baseline_model=baseline,
                scope_identifier="overall",
            )
            for condition in CONDITIONS
        ]
        ax.bar(
            x + (index - 0.5) * width,
            [row.paired_mae_improvement_percent_mean for row in rows],
            width=width,
            yerr=[
                row.paired_mae_improvement_percent_sample_std for row in rows
            ],
            capsize=3,
            color=COLORS[baseline],
            label=baseline,
        )
    ax.set_title(
        "Paired MAE improvement of FAR-GF-RC over baselines", fontsize=13
    )
    _format_axis(
        ax,
        "Paired FAR-GF-RC MAE improvement (%)",
        "History availability condition",
    )
    ax.legend(title="Baseline", frameon=False, fontsize=9, title_fontsize=9)
    _save(
        fig,
        output_dir,
        "submission_Figure_3_paired_FAR_MAE_improvement",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.project_root.resolve()
    raw = root / "results/raw/pems_bay_final_primary_evaluation_v1"
    output_dir = root / "results/figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(raw / "seed_summary_metrics.csv")
    degradation = pd.read_csv(
        raw / "clean_to_dropout_degradation_summary.csv"
    )
    improvement = pd.read_csv(
        raw / "paired_far_mae_improvement_summary.csv"
    )

    _line_figure(
        summary,
        "overall",
        "Overall forecasting error under controlled sensor dropout",
        "Overall MAE (mph)",
        "submission_Figure_1_overall_MAE_across_conditions",
        output_dir,
    )
    _degradation_figure(degradation, output_dir)
    _improvement_figure(improvement, output_dir)

    for horizon, minutes in ((3, 15), (6, 30), (12, 60)):
        _line_figure(
            summary,
            f"horizon_{horizon}",
            f"Horizon {horizon} ({minutes}-minute) forecasting MAE",
            "MAE (mph)",
            f"submission_Figure_4_horizon_{horizon}_MAE_across_conditions",
            output_dir,
            compact=True,
        )

    print("Wrote six vector-PDF submission figures.")


if __name__ == "__main__":
    main()
