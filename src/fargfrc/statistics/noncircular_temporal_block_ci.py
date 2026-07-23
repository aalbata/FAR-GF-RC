import numpy as np


def _validate_grouped_statistics(grouped_statistics):
    required_keys = {
        "groups",
        "valid_count",
        "absolute_error_sum",
        "squared_error_sum",
        "target_absolute_sum",
        "mape_error_sum",
        "mape_count",
    }

    missing_keys = required_keys - set(grouped_statistics.keys())

    if missing_keys:
        raise KeyError(
            "Missing grouped-statistics keys: "
            + ", ".join(sorted(missing_keys))
        )

    groups = np.asarray(grouped_statistics["groups"]).astype(str)

    if groups.ndim != 1 or groups.size == 0:
        raise ValueError(
            "Grouped statistics must contain at least one chronological group."
        )

    number_of_groups = groups.shape[0]
    expected_horizon_shape = None

    for key in required_keys - {"groups"}:
        values = np.asarray(
            grouped_statistics[key],
            dtype=np.float64,
        )

        if values.shape[0] != number_of_groups:
            raise ValueError(
                f"Grouped statistic '{key}' has inconsistent group count."
            )

        if expected_horizon_shape is None:
            expected_horizon_shape = values.shape[1:]

        elif values.shape[1:] != expected_horizon_shape:
            raise ValueError(
                "Grouped-statistics arrays have inconsistent horizon shapes."
            )

    return groups


def _sum_grouped_statistics(
    grouped_statistics,
    sampled_group_indices,
):
    statistic_keys = [
        "valid_count",
        "absolute_error_sum",
        "squared_error_sum",
        "target_absolute_sum",
        "mape_error_sum",
        "mape_count",
    ]

    sampled_group_indices = np.asarray(
        sampled_group_indices,
        dtype=np.int64,
    )

    return {
        key: np.asarray(
            grouped_statistics[key],
            dtype=np.float64,
        )[sampled_group_indices].sum(axis=0)
        for key in statistic_keys
    }


def metrics_from_sufficient_statistics(
    statistics,
    strict=True,
):
    required_keys = {
        "valid_count",
        "absolute_error_sum",
        "squared_error_sum",
        "target_absolute_sum",
        "mape_error_sum",
        "mape_count",
    }

    missing_keys = required_keys - set(statistics.keys())

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


def noncircular_moving_block_indices(
    number_of_groups,
    block_length,
    number_of_resamples,
    random_seed,
):
    """
    Generate chronological moving-block bootstrap samples without
    wrapping the final day back to the first day.
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

    rng = np.random.default_rng(random_seed)

    blocks_needed = int(
        np.ceil(number_of_groups / block_length)
    )

    maximum_valid_start = number_of_groups - block_length

    bootstrap_indices = np.empty(
        (number_of_resamples, number_of_groups),
        dtype=np.int64,
    )

    sampled_block_starts = np.empty(
        (number_of_resamples, blocks_needed),
        dtype=np.int64,
    )

    block_offsets = np.arange(
        block_length,
        dtype=np.int64,
    )

    for resample_index in range(number_of_resamples):
        block_starts = rng.integers(
            low=0,
            high=maximum_valid_start + 1,
            size=blocks_needed,
        )

        sampled_block_starts[resample_index] = block_starts

        sampled_indices = np.concatenate(
            [
                block_start + block_offsets
                for block_start in block_starts
            ]
        )[:number_of_groups]

        bootstrap_indices[resample_index] = sampled_indices

    return {
        "indices": bootstrap_indices,
        "block_starts": sampled_block_starts,
    }


def bootstrap_metric_confidence_intervals(
    grouped_statistics,
    number_of_resamples=2000,
    block_length=3,
    random_seed=20260629,
    lower_percentile=2.5,
    upper_percentile=97.5,
):
    """
    Non-circular moving-block bootstrap percentile CIs for
    MAE, RMSE, MAPE, and WMAPE.
    """
    groups = _validate_grouped_statistics(
        grouped_statistics
    )

    if not 0.0 < lower_percentile < upper_percentile < 100.0:
        raise ValueError(
            "Require 0 < lower_percentile < upper_percentile < 100."
        )

    number_of_groups = len(groups)

    bootstrap_plan = noncircular_moving_block_indices(
        number_of_groups=number_of_groups,
        block_length=block_length,
        number_of_resamples=number_of_resamples,
        random_seed=random_seed,
    )

    bootstrap_indices = bootstrap_plan["indices"]

    horizon_shape = np.asarray(
        grouped_statistics["valid_count"]
    ).shape[1:]

    metric_names = ["MAE", "RMSE", "MAPE", "WMAPE"]

    bootstrap_metric_values = {
        metric_name: np.full(
            (number_of_resamples,) + horizon_shape,
            np.nan,
            dtype=np.float64,
        )
        for metric_name in metric_names
    }

    for resample_index, sampled_indices in enumerate(
        bootstrap_indices
    ):
        sampled_statistics = _sum_grouped_statistics(
            grouped_statistics,
            sampled_indices,
        )

        sampled_metrics = metrics_from_sufficient_statistics(
            sampled_statistics,
            strict=False,
        )

        for metric_name in metric_names:
            bootstrap_metric_values[metric_name][resample_index] = (
                sampled_metrics[metric_name]
            )

    point_statistics = _sum_grouped_statistics(
        grouped_statistics,
        np.arange(number_of_groups, dtype=np.int64),
    )

    point_metrics = metrics_from_sufficient_statistics(
        point_statistics,
        strict=True,
    )

    confidence_intervals = {}

    for metric_name in metric_names:
        bootstrap_values = bootstrap_metric_values[metric_name]

        confidence_intervals[metric_name] = {
            "point_estimate": point_metrics[metric_name],
            "ci_lower": np.nanpercentile(
                bootstrap_values,
                lower_percentile,
                axis=0,
            ),
            "ci_upper": np.nanpercentile(
                bootstrap_values,
                upper_percentile,
                axis=0,
            ),
            "effective_resamples": np.sum(
                np.isfinite(bootstrap_values),
                axis=0,
            ),
        }

    return {
        "groups": groups,
        "bootstrap_indices": bootstrap_indices,
        "sampled_block_starts": bootstrap_plan["block_starts"],
        "number_of_resamples": int(number_of_resamples),
        "block_length": int(block_length),
        "random_seed": int(random_seed),
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "confidence_intervals": confidence_intervals,
    }


def paired_bootstrap_difference_confidence_intervals(
    grouped_statistics_a,
    grouped_statistics_b,
    number_of_resamples=2000,
    block_length=3,
    random_seed=20260629,
    lower_percentile=2.5,
    upper_percentile=97.5,
):
    """
    Paired non-circular block-bootstrap CI for:

        Delta = Metric(Model A) - Metric(Model B)

    Identical chronological blocks are sampled for both models.
    """
    groups_a = _validate_grouped_statistics(
        grouped_statistics_a
    )

    groups_b = _validate_grouped_statistics(
        grouped_statistics_b
    )

    if not np.array_equal(groups_a, groups_b):
        raise ValueError(
            "Paired bootstrap requires identical chronological groups."
        )

    if not 0.0 < lower_percentile < upper_percentile < 100.0:
        raise ValueError(
            "Require 0 < lower_percentile < upper_percentile < 100."
        )

    number_of_groups = len(groups_a)

    bootstrap_plan = noncircular_moving_block_indices(
        number_of_groups=number_of_groups,
        block_length=block_length,
        number_of_resamples=number_of_resamples,
        random_seed=random_seed,
    )

    bootstrap_indices = bootstrap_plan["indices"]

    horizon_shape = np.asarray(
        grouped_statistics_a["valid_count"]
    ).shape[1:]

    metric_names = ["MAE", "RMSE", "MAPE", "WMAPE"]

    delta_bootstrap_values = {
        metric_name: np.full(
            (number_of_resamples,) + horizon_shape,
            np.nan,
            dtype=np.float64,
        )
        for metric_name in metric_names
    }

    for resample_index, sampled_indices in enumerate(
        bootstrap_indices
    ):
        sampled_statistics_a = _sum_grouped_statistics(
            grouped_statistics_a,
            sampled_indices,
        )

        sampled_statistics_b = _sum_grouped_statistics(
            grouped_statistics_b,
            sampled_indices,
        )

        sampled_metrics_a = metrics_from_sufficient_statistics(
            sampled_statistics_a,
            strict=False,
        )

        sampled_metrics_b = metrics_from_sufficient_statistics(
            sampled_statistics_b,
            strict=False,
        )

        for metric_name in metric_names:
            delta_bootstrap_values[metric_name][resample_index] = (
                sampled_metrics_a[metric_name]
                - sampled_metrics_b[metric_name]
            )

    all_group_indices = np.arange(
        number_of_groups,
        dtype=np.int64,
    )

    point_metrics_a = metrics_from_sufficient_statistics(
        _sum_grouped_statistics(
            grouped_statistics_a,
            all_group_indices,
        ),
        strict=True,
    )

    point_metrics_b = metrics_from_sufficient_statistics(
        _sum_grouped_statistics(
            grouped_statistics_b,
            all_group_indices,
        ),
        strict=True,
    )

    confidence_intervals = {}

    for metric_name in metric_names:
        delta_values = delta_bootstrap_values[metric_name]

        confidence_intervals[metric_name] = {
            "point_difference_a_minus_b": (
                point_metrics_a[metric_name]
                - point_metrics_b[metric_name]
            ),
            "ci_lower": np.nanpercentile(
                delta_values,
                lower_percentile,
                axis=0,
            ),
            "ci_upper": np.nanpercentile(
                delta_values,
                upper_percentile,
                axis=0,
            ),
            "effective_resamples": np.sum(
                np.isfinite(delta_values),
                axis=0,
            ),
        }

    return {
        "groups": groups_a,
        "bootstrap_indices": bootstrap_indices,
        "sampled_block_starts": bootstrap_plan["block_starts"],
        "number_of_resamples": int(number_of_resamples),
        "block_length": int(block_length),
        "random_seed": int(random_seed),
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "confidence_intervals": confidence_intervals,
    }
