import numpy as np


STATISTIC_KEYS = [
    "valid_count",
    "absolute_error_sum",
    "squared_error_sum",
    "target_absolute_sum",
    "mape_error_sum",
    "mape_count",
]

METRIC_NAMES = [
    "MAE",
    "RMSE",
    "MAPE",
    "WMAPE",
]


def compute_metrics_from_components(
    components,
):
    """
    Compute global metrics from raw sufficient statistics.

    Components may contain scalar values or one-element arrays.
    """
    valid_count = float(
        np.asarray(
            components["valid_count"],
            dtype=np.float64,
        ).sum()
    )

    absolute_error_sum = float(
        np.asarray(
            components["absolute_error_sum"],
            dtype=np.float64,
        ).sum()
    )

    squared_error_sum = float(
        np.asarray(
            components["squared_error_sum"],
            dtype=np.float64,
        ).sum()
    )

    target_absolute_sum = float(
        np.asarray(
            components["target_absolute_sum"],
            dtype=np.float64,
        ).sum()
    )

    mape_error_sum = float(
        np.asarray(
            components["mape_error_sum"],
            dtype=np.float64,
        ).sum()
    )

    mape_count = float(
        np.asarray(
            components["mape_count"],
            dtype=np.float64,
        ).sum()
    )

    if valid_count <= 0.0:
        raise ValueError(
            "valid_count must be positive."
        )

    if target_absolute_sum <= 0.0:
        raise ValueError(
            "target_absolute_sum must be positive."
        )

    if mape_count <= 0.0:
        raise ValueError(
            "mape_count must be positive."
        )

    metrics = {
        "MAE": absolute_error_sum / valid_count,
        "RMSE": float(
            np.sqrt(
                squared_error_sum / valid_count
            )
        ),
        "MAPE": float(
            100.0
            * mape_error_sum
            / mape_count
        ),
        "WMAPE": float(
            100.0
            * absolute_error_sum
            / target_absolute_sum
        ),
    }

    for metric_name, metric_value in metrics.items():
        if not np.isfinite(metric_value):
            raise ValueError(
                f"{metric_name} is NaN or infinite."
            )

    return metrics


def aggregate_daily_statistics(
    daily_statistics,
    temporal_indices,
    horizon_indices,
):
    """
    Aggregate a seed's daily sufficient statistics over resampled
    forecast-origin days and selected horizons.
    """
    temporal_indices = np.asarray(
        temporal_indices,
        dtype=np.int64,
    )

    horizon_indices = np.asarray(
        horizon_indices,
        dtype=np.int64,
    )

    if temporal_indices.ndim != 1:
        raise ValueError(
            "temporal_indices must be one-dimensional."
        )

    if horizon_indices.ndim != 1:
        raise ValueError(
            "horizon_indices must be one-dimensional."
        )

    if temporal_indices.size == 0:
        raise ValueError(
            "temporal_indices cannot be empty."
        )

    if horizon_indices.size == 0:
        raise ValueError(
            "horizon_indices cannot be empty."
        )

    components = {}

    for statistic_key in STATISTIC_KEYS:
        statistic_array = np.asarray(
            daily_statistics[statistic_key],
            dtype=np.float64,
        )

        if statistic_array.ndim != 2:
            raise ValueError(
                f"{statistic_key} must have shape "
                "[days, horizons]."
            )

        components[statistic_key] = float(
            statistic_array[
                temporal_indices[:, None],
                horizon_indices[None, :],
            ].sum()
        )

    return components


def generate_noncircular_temporal_plan(
    group_count,
    block_length,
    random_generator,
):
    """
    Sample a non-circular moving-block bootstrap plan.

    Blocks are contiguous and never wrap from the final day back
    to the first day.
    """
    group_count = int(group_count)
    block_length = int(block_length)

    if group_count < 1:
        raise ValueError(
            "group_count must be positive."
        )

    if block_length < 1:
        raise ValueError(
            "block_length must be positive."
        )

    if block_length > group_count:
        raise ValueError(
            "block_length cannot exceed group_count."
        )

    block_count = int(
        np.ceil(group_count / block_length)
    )

    maximum_start = group_count - block_length

    block_starts = random_generator.integers(
        low=0,
        high=maximum_start + 1,
        size=block_count,
        dtype=np.int64,
    )

    temporal_indices = np.concatenate(
        [
            np.arange(
                block_start,
                block_start + block_length,
                dtype=np.int64,
            )
            for block_start in block_starts
        ]
    )[:group_count]

    if temporal_indices.size != group_count:
        raise RuntimeError(
            "Temporal bootstrap sample has incorrect length."
        )

    if temporal_indices.min() < 0 or temporal_indices.max() >= group_count:
        raise RuntimeError(
            "Temporal bootstrap plan wraps or exceeds day bounds."
        )

    for block_start in block_starts:
        block_end_exclusive = int(
            block_start + block_length
        )

        if block_end_exclusive > group_count:
            raise RuntimeError(
                "A temporal bootstrap block wraps beyond final day."
            )

    return temporal_indices, block_starts


def generate_hierarchical_bootstrap_plan(
    seed_count,
    group_count,
    block_length,
    resample_count,
    random_seed,
):
    """
    Create a frozen hierarchical bootstrap plan.

    Each replicate:
    - samples `seed_count` seed slots with replacement;
    - samples non-circular temporal moving blocks;
    - shares that temporal sample across selected seed slots.

    Sharing temporal draws represents common uncertainty from the same
    chronological test period while seed resampling captures variation
    from model initialization and stochastic training.
    """
    seed_count = int(seed_count)
    group_count = int(group_count)
    resample_count = int(resample_count)

    if seed_count < 1:
        raise ValueError(
            "seed_count must be positive."
        )

    if resample_count < 1:
        raise ValueError(
            "resample_count must be positive."
        )

    random_generator = np.random.default_rng(
        int(random_seed)
    )

    seed_indices = random_generator.integers(
        low=0,
        high=seed_count,
        size=(resample_count, seed_count),
        dtype=np.int64,
    )

    blocks_per_replicate = int(
        np.ceil(group_count / int(block_length))
    )

    temporal_indices = np.empty(
        shape=(resample_count, group_count),
        dtype=np.int64,
    )

    temporal_block_starts = np.empty(
        shape=(resample_count, blocks_per_replicate),
        dtype=np.int64,
    )

    for replicate_index in range(resample_count):
        (
            sampled_day_indices,
            sampled_block_starts,
        ) = generate_noncircular_temporal_plan(
            group_count=group_count,
            block_length=block_length,
            random_generator=random_generator,
        )

        temporal_indices[
            replicate_index
        ] = sampled_day_indices

        temporal_block_starts[
            replicate_index
        ] = sampled_block_starts

    if seed_indices.min() < 0 or seed_indices.max() >= seed_count:
        raise RuntimeError(
            "Seed bootstrap indices are out of bounds."
        )

    if temporal_indices.min() < 0 or temporal_indices.max() >= group_count:
        raise RuntimeError(
            "Temporal bootstrap indices are out of bounds."
        )

    return {
        "seed_indices": seed_indices,
        "temporal_indices": temporal_indices,
        "temporal_block_starts": temporal_block_starts,
        "seed_count": seed_count,
        "group_count": group_count,
        "block_length": int(block_length),
        "resample_count": resample_count,
    }


def hierarchical_metric_confidence_intervals(
    seed_daily_statistics,
    horizon_indices,
    bootstrap_plan,
    lower_percentile=2.5,
    upper_percentile=97.5,
):
    """
    Compute mean-across-seeds point metrics and hierarchical CIs.

    The point estimate is the arithmetic mean of each global,
    seed-level metric. It is not calculated by pooling raw values
    across seeds.
    """
    if not seed_daily_statistics:
        raise ValueError(
            "seed_daily_statistics cannot be empty."
        )

    seed_count = len(seed_daily_statistics)

    if int(bootstrap_plan["seed_count"]) != seed_count:
        raise ValueError(
            "Bootstrap plan seed count does not match statistics."
        )

    group_count = int(
        np.asarray(
            seed_daily_statistics[0]["valid_count"]
        ).shape[0]
    )

    if int(bootstrap_plan["group_count"]) != group_count:
        raise ValueError(
            "Bootstrap plan day count does not match statistics."
        )

    full_day_indices = np.arange(
        group_count,
        dtype=np.int64,
    )

    seed_point_metrics = []

    for daily_statistics in seed_daily_statistics:
        seed_components = aggregate_daily_statistics(
            daily_statistics=daily_statistics,
            temporal_indices=full_day_indices,
            horizon_indices=horizon_indices,
        )

        seed_point_metrics.append(
            compute_metrics_from_components(
                seed_components
            )
        )

    point_estimates = {
        metric_name: float(
            np.mean(
                [
                    seed_metrics[metric_name]
                    for seed_metrics in seed_point_metrics
                ]
            )
        )
        for metric_name in METRIC_NAMES
    }

    resample_count = int(
        bootstrap_plan["resample_count"]
    )

    replicate_metrics = {
        metric_name: np.empty(
            resample_count,
            dtype=np.float64,
        )
        for metric_name in METRIC_NAMES
    }

    for replicate_index in range(resample_count):
        selected_seed_indices = bootstrap_plan[
            "seed_indices"
        ][replicate_index]

        selected_temporal_indices = bootstrap_plan[
            "temporal_indices"
        ][replicate_index]

        slot_metrics = {
            metric_name: []
            for metric_name in METRIC_NAMES
        }

        for selected_seed_index in selected_seed_indices:
            sampled_components = aggregate_daily_statistics(
                daily_statistics=seed_daily_statistics[
                    int(selected_seed_index)
                ],
                temporal_indices=selected_temporal_indices,
                horizon_indices=horizon_indices,
            )

            sampled_metrics = compute_metrics_from_components(
                sampled_components
            )

            for metric_name in METRIC_NAMES:
                slot_metrics[metric_name].append(
                    sampled_metrics[metric_name]
                )

        for metric_name in METRIC_NAMES:
            replicate_value = float(
                np.mean(slot_metrics[metric_name])
            )

            if not np.isfinite(replicate_value):
                raise RuntimeError(
                    f"Replicate {replicate_index}, {metric_name} "
                    "is NaN or infinite."
                )

            replicate_metrics[metric_name][
                replicate_index
            ] = replicate_value

    confidence_intervals = {}

    for metric_name in METRIC_NAMES:
        ci_lower = float(
            np.percentile(
                replicate_metrics[metric_name],
                lower_percentile,
            )
        )

        ci_upper = float(
            np.percentile(
                replicate_metrics[metric_name],
                upper_percentile,
            )
        )

        point_estimate = point_estimates[metric_name]

        if not (
            np.isfinite(ci_lower)
            and np.isfinite(ci_upper)
        ):
            raise RuntimeError(
                f"{metric_name} confidence interval is invalid."
            )

        confidence_intervals[metric_name] = {
            "point_estimate": point_estimate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "effective_resamples": int(
                np.isfinite(
                    replicate_metrics[metric_name]
                ).sum()
            ),
        }

    return {
        "point_estimates": point_estimates,
        "seed_point_metrics": seed_point_metrics,
        "confidence_intervals": confidence_intervals,
        "replicate_metrics": replicate_metrics,
    }


def run_hierarchical_ci_self_tests():
    """
    Run exact and structural validation checks.
    """
    random_generator = np.random.default_rng(20260629)

    temporal_indices, block_starts = (
        generate_noncircular_temporal_plan(
            group_count=5,
            block_length=3,
            random_generator=random_generator,
        )
    )

    if temporal_indices.size != 5:
        raise RuntimeError(
            "Toy temporal-plan length check failed."
        )

    if temporal_indices.min() < 0 or temporal_indices.max() >= 5:
        raise RuntimeError(
            "Toy temporal-plan bound check failed."
        )

    if np.any(block_starts < 0) or np.any(
        block_starts + 3 > 5
    ):
        raise RuntimeError(
            "Toy non-circular block check failed."
        )

    constant_daily_statistics = {
        "valid_count": np.full(
            (6, 1),
            10.0,
            dtype=np.float64,
        ),
        "absolute_error_sum": np.full(
            (6, 1),
            20.0,
            dtype=np.float64,
        ),
        "squared_error_sum": np.full(
            (6, 1),
            50.0,
            dtype=np.float64,
        ),
        "target_absolute_sum": np.full(
            (6, 1),
            100.0,
            dtype=np.float64,
        ),
        "mape_error_sum": np.full(
            (6, 1),
            1.0,
            dtype=np.float64,
        ),
        "mape_count": np.full(
            (6, 1),
            10.0,
            dtype=np.float64,
        ),
    }

    identical_seed_statistics = [
        constant_daily_statistics,
        {
            statistic_key: statistic_value.copy()
            for statistic_key, statistic_value
            in constant_daily_statistics.items()
        },
    ]

    toy_plan = generate_hierarchical_bootstrap_plan(
        seed_count=2,
        group_count=6,
        block_length=3,
        resample_count=100,
        random_seed=11,
    )

    toy_result = hierarchical_metric_confidence_intervals(
        seed_daily_statistics=identical_seed_statistics,
        horizon_indices=np.asarray([0], dtype=np.int64),
        bootstrap_plan=toy_plan,
    )

    expected_metrics = {
        "MAE": 2.0,
        "RMSE": float(np.sqrt(5.0)),
        "MAPE": 10.0,
        "WMAPE": 20.0,
    }

    for metric_name, expected_value in expected_metrics.items():
        observed_point = toy_result["point_estimates"][
            metric_name
        ]

        if not np.isclose(
            observed_point,
            expected_value,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Toy point estimate failed for {metric_name}."
            )

        if not np.allclose(
            toy_result["replicate_metrics"][metric_name],
            expected_value,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Toy bootstrap invariance failed for {metric_name}."
            )

    return {
        "noncircular_temporal_plan_check_passed": True,
        "constant_statistic_bootstrap_invariance_passed": True,
        "metric_formula_toy_check_passed": True,
    }
