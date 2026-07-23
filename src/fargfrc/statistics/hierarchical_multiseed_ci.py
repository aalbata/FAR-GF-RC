import numpy as np


METRIC_NAMES = ("MAE", "RMSE", "MAPE", "WMAPE")

STATISTIC_KEYS = (
    "valid_count",
    "absolute_error_sum",
    "squared_error_sum",
    "target_absolute_sum",
    "mape_error_sum",
    "mape_count",
)


def _metrics_from_sufficient_statistics(
    statistics,
    strict=True,
):
    """
    Compute MAE, RMSE, MAPE, and WMAPE from globally aggregated
    sufficient statistics.
    """
    missing_keys = set(STATISTIC_KEYS) - set(statistics.keys())

    if missing_keys:
        raise KeyError(
            "Missing sufficient-statistics keys: "
            + ", ".join(sorted(missing_keys))
        )

    valid_count = np.asarray(
        statistics["valid_count"],
        dtype=np.float64,
    )

    absolute_error_sum = np.asarray(
        statistics["absolute_error_sum"],
        dtype=np.float64,
    )

    squared_error_sum = np.asarray(
        statistics["squared_error_sum"],
        dtype=np.float64,
    )

    target_absolute_sum = np.asarray(
        statistics["target_absolute_sum"],
        dtype=np.float64,
    )

    mape_error_sum = np.asarray(
        statistics["mape_error_sum"],
        dtype=np.float64,
    )

    mape_count = np.asarray(
        statistics["mape_count"],
        dtype=np.float64,
    )

    if strict:
        if np.any(valid_count <= 0):
            raise ValueError(
                "At least one metric entry has zero valid targets."
            )

        if np.any(target_absolute_sum <= 0):
            raise ValueError(
                "At least one WMAPE denominator is zero."
            )

        if np.any(mape_count <= 0):
            raise ValueError(
                "At least one MAPE denominator is zero."
            )

    with np.errstate(divide="ignore", invalid="ignore"):
        mae = absolute_error_sum / valid_count
        rmse = np.sqrt(squared_error_sum / valid_count)
        wmape = absolute_error_sum / target_absolute_sum * 100.0
        mape = mape_error_sum / mape_count * 100.0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "WMAPE": wmape,
    }


def _validate_seeded_grouped_statistics(
    seed_grouped_statistics,
):
    """
    Validate a list of grouped sufficient-statistics dictionaries.

    Each list element represents one independently trained model seed.
    Every seed must use identical chronological day groups and have
    identically shaped horizon-wise statistic arrays.
    """
    if not isinstance(seed_grouped_statistics, (list, tuple)):
        raise TypeError(
            "seed_grouped_statistics must be a list or tuple."
        )

    if len(seed_grouped_statistics) == 0:
        raise ValueError(
            "At least one seed-level grouped-statistics object is required."
        )

    canonical_groups = None
    canonical_horizon_shape = None

    for seed_index, grouped_statistics in enumerate(
        seed_grouped_statistics
    ):
        required_keys = {"groups"} | set(STATISTIC_KEYS)

        missing_keys = required_keys - set(grouped_statistics.keys())

        if missing_keys:
            raise KeyError(
                f"Seed {seed_index} is missing grouped-statistics keys: "
                + ", ".join(sorted(missing_keys))
            )

        groups = np.asarray(
            grouped_statistics["groups"]
        ).astype(str)

        if groups.ndim != 1 or groups.size == 0:
            raise ValueError(
                f"Seed {seed_index} does not contain valid groups."
            )

        if canonical_groups is None:
            canonical_groups = groups

        elif not np.array_equal(groups, canonical_groups):
            raise ValueError(
                "All seed-level statistics must use identical "
                "chronological day groups."
            )

        for key in STATISTIC_KEYS:
            values = np.asarray(
                grouped_statistics[key],
                dtype=np.float64,
            )

            if values.shape[0] != len(groups):
                raise ValueError(
                    f"Seed {seed_index}, statistic '{key}' has an "
                    "inconsistent number of day groups."
                )

            if canonical_horizon_shape is None:
                canonical_horizon_shape = values.shape[1:]

            elif values.shape[1:] != canonical_horizon_shape:
                raise ValueError(
                    "All seed-level statistics must have identical "
                    "forecast-horizon shapes."
                )

    return canonical_groups, canonical_horizon_shape


def _sum_one_seed_statistics(
    grouped_statistics,
    sampled_group_indices,
):
    """
    Aggregate selected temporal groups for one model seed.
    """
    sampled_group_indices = np.asarray(
        sampled_group_indices,
        dtype=np.int64,
    )

    return {
        key: np.asarray(
            grouped_statistics[key],
            dtype=np.float64,
        )[sampled_group_indices].sum(axis=0)
        for key in STATISTIC_KEYS
    }


def _noncircular_block_bootstrap_plan(
    number_of_groups,
    block_length,
    number_of_resamples,
    random_generator,
):
    """
    Generate non-circular moving-block bootstrap indices.

    The final chronological group never wraps to the first group.
    """
    if number_of_groups < 1:
        raise ValueError(
            "number_of_groups must be positive."
        )

    if block_length < 1:
        raise ValueError(
            "block_length must be positive."
        )

    if block_length > number_of_groups:
        raise ValueError(
            "block_length cannot exceed number_of_groups."
        )

    if number_of_resamples < 1:
        raise ValueError(
            "number_of_resamples must be positive."
        )

    blocks_needed = int(
        np.ceil(number_of_groups / block_length)
    )

    maximum_valid_start = number_of_groups - block_length

    sampled_block_starts = random_generator.integers(
        low=0,
        high=maximum_valid_start + 1,
        size=(number_of_resamples, blocks_needed),
    )

    block_offsets = np.arange(
        block_length,
        dtype=np.int64,
    )

    bootstrap_indices = np.empty(
        (number_of_resamples, number_of_groups),
        dtype=np.int64,
    )

    for resample_index in range(number_of_resamples):
        sampled_indices = np.concatenate(
            [
                block_start + block_offsets
                for block_start in sampled_block_starts[resample_index]
            ]
        )[:number_of_groups]

        bootstrap_indices[resample_index] = sampled_indices

    return {
        "bootstrap_indices": bootstrap_indices,
        "sampled_block_starts": sampled_block_starts,
    }


def _point_metrics_across_seeds(
    seed_grouped_statistics,
):
    """
    Point estimate = arithmetic mean of seed-level globally aggregated
    metrics. This is intentionally not metric computation after pooling
    predictions across seeds.
    """
    number_of_groups = len(
        np.asarray(seed_grouped_statistics[0]["groups"])
    )

    all_group_indices = np.arange(
        number_of_groups,
        dtype=np.int64,
    )

    seed_metric_values = {
        metric_name: []
        for metric_name in METRIC_NAMES
    }

    for grouped_statistics in seed_grouped_statistics:
        seed_statistics = _sum_one_seed_statistics(
            grouped_statistics=grouped_statistics,
            sampled_group_indices=all_group_indices,
        )

        seed_metrics = _metrics_from_sufficient_statistics(
            seed_statistics,
            strict=True,
        )

        for metric_name in METRIC_NAMES:
            seed_metric_values[metric_name].append(
                seed_metrics[metric_name]
            )

    point_metrics = {
        metric_name: np.mean(
            np.stack(metric_values, axis=0),
            axis=0,
        )
        for metric_name, metric_values in seed_metric_values.items()
    }

    seed_metric_arrays = {
        metric_name: np.stack(
            metric_values,
            axis=0,
        )
        for metric_name, metric_values in seed_metric_values.items()
    }

    return point_metrics, seed_metric_arrays


def hierarchical_multiseed_metric_confidence_intervals(
    seed_grouped_statistics,
    number_of_resamples=2000,
    block_length=3,
    random_seed=20260629,
    lower_percentile=2.5,
    upper_percentile=97.5,
):
    """
    Hierarchical 95% CI for learned models.

    Each bootstrap replicate:
    1. Resamples contiguous non-circular temporal day blocks.
    2. Resamples neural-model seeds with replacement.
    3. Computes the metric within each selected seed.
    4. Averages the selected seed metrics.

    This represents uncertainty from both the observed temporal period
    and stochastic neural-model training.
    """
    groups, horizon_shape = _validate_seeded_grouped_statistics(
        seed_grouped_statistics
    )

    if not 0.0 < lower_percentile < upper_percentile < 100.0:
        raise ValueError(
            "Require 0 < lower_percentile < upper_percentile < 100."
        )

    number_of_groups = len(groups)
    number_of_seeds = len(seed_grouped_statistics)

    random_generator = np.random.default_rng(random_seed)

    bootstrap_plan = _noncircular_block_bootstrap_plan(
        number_of_groups=number_of_groups,
        block_length=block_length,
        number_of_resamples=number_of_resamples,
        random_generator=random_generator,
    )

    bootstrap_indices = bootstrap_plan["bootstrap_indices"]

    sampled_seed_indices = random_generator.integers(
        low=0,
        high=number_of_seeds,
        size=(number_of_resamples, number_of_seeds),
    )

    bootstrap_metric_values = {
        metric_name: np.full(
            (number_of_resamples,) + horizon_shape,
            np.nan,
            dtype=np.float64,
        )
        for metric_name in METRIC_NAMES
    }

    for resample_index in range(number_of_resamples):
        selected_seed_metrics = {
            metric_name: []
            for metric_name in METRIC_NAMES
        }

        for seed_index in sampled_seed_indices[resample_index]:
            sampled_statistics = _sum_one_seed_statistics(
                grouped_statistics=seed_grouped_statistics[int(seed_index)],
                sampled_group_indices=bootstrap_indices[resample_index],
            )

            sampled_metrics = _metrics_from_sufficient_statistics(
                sampled_statistics,
                strict=False,
            )

            for metric_name in METRIC_NAMES:
                selected_seed_metrics[metric_name].append(
                    sampled_metrics[metric_name]
                )

        for metric_name in METRIC_NAMES:
            bootstrap_metric_values[metric_name][resample_index] = (
                np.mean(
                    np.stack(
                        selected_seed_metrics[metric_name],
                        axis=0,
                    ),
                    axis=0,
                )
            )

    point_metrics, seed_metric_arrays = _point_metrics_across_seeds(
        seed_grouped_statistics
    )

    confidence_intervals = {}

    for metric_name in METRIC_NAMES:
        metric_bootstrap_values = bootstrap_metric_values[metric_name]

        confidence_intervals[metric_name] = {
            "point_estimate": point_metrics[metric_name],
            "ci_lower": np.nanpercentile(
                metric_bootstrap_values,
                lower_percentile,
                axis=0,
            ),
            "ci_upper": np.nanpercentile(
                metric_bootstrap_values,
                upper_percentile,
                axis=0,
            ),
            "effective_resamples": np.sum(
                np.isfinite(metric_bootstrap_values),
                axis=0,
            ),
            "seed_level_metric_values": seed_metric_arrays[metric_name],
        }

    return {
        "groups": groups,
        "number_of_seeds": int(number_of_seeds),
        "number_of_resamples": int(number_of_resamples),
        "block_length": int(block_length),
        "random_seed": int(random_seed),
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "bootstrap_indices": bootstrap_indices,
        "sampled_block_starts": bootstrap_plan["sampled_block_starts"],
        "sampled_seed_indices": sampled_seed_indices,
        "confidence_intervals": confidence_intervals,
    }


def _resolve_paired_seed_slots(
    seed_statistics_a,
    seed_statistics_b,
):
    """
    Resolve compatible seed slots for paired model comparison.

    Allowed cases:
    - Equal numbers of seed runs: seed i is paired with seed i.
    - One deterministic model (one run) vs a multi-seed learned model:
      the deterministic model is repeated across learned-model seed slots.
    """
    number_of_seeds_a = len(seed_statistics_a)
    number_of_seeds_b = len(seed_statistics_b)

    if number_of_seeds_a == number_of_seeds_b:
        return {
            "number_of_seed_slots": number_of_seeds_a,
            "a_seed_map": np.arange(
                number_of_seeds_a,
                dtype=np.int64,
            ),
            "b_seed_map": np.arange(
                number_of_seeds_b,
                dtype=np.int64,
            ),
            "pairing_type": "matched_seed_indices",
        }

    if number_of_seeds_a == 1 and number_of_seeds_b > 1:
        return {
            "number_of_seed_slots": number_of_seeds_b,
            "a_seed_map": np.zeros(
                number_of_seeds_b,
                dtype=np.int64,
            ),
            "b_seed_map": np.arange(
                number_of_seeds_b,
                dtype=np.int64,
            ),
            "pairing_type": "deterministic_a_repeated_against_b_seeds",
        }

    if number_of_seeds_b == 1 and number_of_seeds_a > 1:
        return {
            "number_of_seed_slots": number_of_seeds_a,
            "a_seed_map": np.arange(
                number_of_seeds_a,
                dtype=np.int64,
            ),
            "b_seed_map": np.zeros(
                number_of_seeds_a,
                dtype=np.int64,
            ),
            "pairing_type": "deterministic_b_repeated_against_a_seeds",
        }

    raise ValueError(
        "Paired comparison requires equal seed counts or one deterministic "
        "model with exactly one prediction set."
    )


def paired_hierarchical_multiseed_difference_confidence_intervals(
    seed_grouped_statistics_a,
    seed_grouped_statistics_b,
    number_of_resamples=2000,
    block_length=3,
    random_seed=20260629,
    lower_percentile=2.5,
    upper_percentile=97.5,
):
    """
    Paired hierarchical CI for:

        Delta = Metric(Model A) - Metric(Model B)

    The same non-circular temporal blocks are used for both models.
    Neural seeds are sampled through matched seed slots. This preserves
    paired temporal uncertainty and includes seed-level variation.
    """
    groups_a, horizon_shape_a = _validate_seeded_grouped_statistics(
        seed_grouped_statistics_a
    )

    groups_b, horizon_shape_b = _validate_seeded_grouped_statistics(
        seed_grouped_statistics_b
    )

    if not np.array_equal(groups_a, groups_b):
        raise ValueError(
            "Paired comparison requires identical chronological groups."
        )

    if horizon_shape_a != horizon_shape_b:
        raise ValueError(
            "Paired comparison requires identical horizon shapes."
        )

    if not 0.0 < lower_percentile < upper_percentile < 100.0:
        raise ValueError(
            "Require 0 < lower_percentile < upper_percentile < 100."
        )

    seed_slot_info = _resolve_paired_seed_slots(
        seed_grouped_statistics_a,
        seed_grouped_statistics_b,
    )

    number_of_groups = len(groups_a)
    number_of_seed_slots = seed_slot_info["number_of_seed_slots"]

    random_generator = np.random.default_rng(random_seed)

    bootstrap_plan = _noncircular_block_bootstrap_plan(
        number_of_groups=number_of_groups,
        block_length=block_length,
        number_of_resamples=number_of_resamples,
        random_generator=random_generator,
    )

    bootstrap_indices = bootstrap_plan["bootstrap_indices"]

    sampled_seed_slots = random_generator.integers(
        low=0,
        high=number_of_seed_slots,
        size=(number_of_resamples, number_of_seed_slots),
    )

    delta_bootstrap_values = {
        metric_name: np.full(
            (number_of_resamples,) + horizon_shape_a,
            np.nan,
            dtype=np.float64,
        )
        for metric_name in METRIC_NAMES
    }

    for resample_index in range(number_of_resamples):
        selected_metrics_a = {
            metric_name: []
            for metric_name in METRIC_NAMES
        }

        selected_metrics_b = {
            metric_name: []
            for metric_name in METRIC_NAMES
        }

        for seed_slot in sampled_seed_slots[resample_index]:
            model_a_seed_index = seed_slot_info["a_seed_map"][
                seed_slot
            ]

            model_b_seed_index = seed_slot_info["b_seed_map"][
                seed_slot
            ]

            sampled_statistics_a = _sum_one_seed_statistics(
                grouped_statistics=seed_grouped_statistics_a[
                    int(model_a_seed_index)
                ],
                sampled_group_indices=bootstrap_indices[resample_index],
            )

            sampled_statistics_b = _sum_one_seed_statistics(
                grouped_statistics=seed_grouped_statistics_b[
                    int(model_b_seed_index)
                ],
                sampled_group_indices=bootstrap_indices[resample_index],
            )

            sampled_metrics_a = _metrics_from_sufficient_statistics(
                sampled_statistics_a,
                strict=False,
            )

            sampled_metrics_b = _metrics_from_sufficient_statistics(
                sampled_statistics_b,
                strict=False,
            )

            for metric_name in METRIC_NAMES:
                selected_metrics_a[metric_name].append(
                    sampled_metrics_a[metric_name]
                )

                selected_metrics_b[metric_name].append(
                    sampled_metrics_b[metric_name]
                )

        for metric_name in METRIC_NAMES:
            average_metric_a = np.mean(
                np.stack(
                    selected_metrics_a[metric_name],
                    axis=0,
                ),
                axis=0,
            )

            average_metric_b = np.mean(
                np.stack(
                    selected_metrics_b[metric_name],
                    axis=0,
                ),
                axis=0,
            )

            delta_bootstrap_values[metric_name][resample_index] = (
                average_metric_a - average_metric_b
            )

    point_metrics_a, seed_metrics_a = _point_metrics_across_seeds(
        seed_grouped_statistics_a
    )

    point_metrics_b, seed_metrics_b = _point_metrics_across_seeds(
        seed_grouped_statistics_b
    )

    confidence_intervals = {}

    for metric_name in METRIC_NAMES:
        metric_delta_values = delta_bootstrap_values[metric_name]

        confidence_intervals[metric_name] = {
            "point_difference_a_minus_b": (
                point_metrics_a[metric_name]
                - point_metrics_b[metric_name]
            ),
            "ci_lower": np.nanpercentile(
                metric_delta_values,
                lower_percentile,
                axis=0,
            ),
            "ci_upper": np.nanpercentile(
                metric_delta_values,
                upper_percentile,
                axis=0,
            ),
            "effective_resamples": np.sum(
                np.isfinite(metric_delta_values),
                axis=0,
            ),
            "model_a_seed_level_metric_values": (
                seed_metrics_a[metric_name]
            ),
            "model_b_seed_level_metric_values": (
                seed_metrics_b[metric_name]
            ),
        }

    return {
        "groups": groups_a,
        "number_of_resamples": int(number_of_resamples),
        "block_length": int(block_length),
        "random_seed": int(random_seed),
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "seed_pairing_type": seed_slot_info["pairing_type"],
        "number_of_seed_slots": int(number_of_seed_slots),
        "bootstrap_indices": bootstrap_indices,
        "sampled_block_starts": bootstrap_plan["sampled_block_starts"],
        "sampled_seed_slots": sampled_seed_slots,
        "confidence_intervals": confidence_intervals,
    }
