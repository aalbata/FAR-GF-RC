import numpy as np
import torch


def inverse_scale_numpy(
    normalized_values,
    sensor_means,
    sensor_stds,
):
    """
    Convert normalized traffic values back to original traffic units.

    Parameters
    ----------
    normalized_values : np.ndarray
        Array whose final dimension equals the number of sensors.
    sensor_means : np.ndarray
        Per-sensor training-only mean values.
    sensor_stds : np.ndarray
        Per-sensor training-only standard deviations.
    """
    normalized_values = np.asarray(
        normalized_values,
        dtype=np.float64,
    )

    sensor_means = np.asarray(
        sensor_means,
        dtype=np.float64,
    )

    sensor_stds = np.asarray(
        sensor_stds,
        dtype=np.float64,
    )

    if normalized_values.shape[-1] != sensor_means.shape[0]:
        raise ValueError(
            "Final prediction dimension does not match sensor means."
        )

    if sensor_means.shape != sensor_stds.shape:
        raise ValueError(
            "sensor_means and sensor_stds must have identical shapes."
        )

    if not np.all(sensor_stds > 0):
        raise ValueError(
            "Every sensor standard deviation must be positive."
        )

    return (
        normalized_values * sensor_stds
        + sensor_means
    )


def inverse_scale_torch(
    normalized_values,
    sensor_means,
    sensor_stds,
):
    """
    Torch equivalent of inverse_scale_numpy().
    """
    if not isinstance(normalized_values, torch.Tensor):
        raise TypeError(
            "normalized_values must be a torch.Tensor."
        )

    means = torch.as_tensor(
        sensor_means,
        dtype=normalized_values.dtype,
        device=normalized_values.device,
    )

    stds = torch.as_tensor(
        sensor_stds,
        dtype=normalized_values.dtype,
        device=normalized_values.device,
    )

    if normalized_values.shape[-1] != means.shape[0]:
        raise ValueError(
            "Final prediction dimension does not match sensor means."
        )

    if means.shape != stds.shape:
        raise ValueError(
            "sensor_means and sensor_stds must have identical shapes."
        )

    if not torch.all(stds > 0):
        raise ValueError(
            "Every sensor standard deviation must be positive."
        )

    return normalized_values * stds + means


def _validate_prediction_target_mask(
    predictions,
    targets,
    valid_mask,
):
    """
    Validate arrays for masked metric computation.

    Expected shape:
        [samples, horizons, sensors]
    """
    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    targets = np.asarray(
        targets,
        dtype=np.float64,
    )

    valid_mask = np.asarray(
        valid_mask,
        dtype=bool,
    )

    if predictions.shape != targets.shape:
        raise ValueError(
            "predictions and targets must have identical shapes."
        )

    if valid_mask.shape != predictions.shape:
        raise ValueError(
            "valid_mask shape must match predictions."
        )

    if predictions.ndim != 3:
        raise ValueError(
            "Expected [samples, horizons, sensors] arrays."
        )

    if not np.isfinite(predictions[valid_mask]).all():
        raise ValueError(
            "Predictions contain NaN or Inf at valid target positions."
        )

    if not np.isfinite(targets[valid_mask]).all():
        raise ValueError(
            "Targets contain NaN or Inf at valid target positions."
        )

    return predictions, targets, valid_mask


def masked_sufficient_statistics(
    predictions,
    targets,
    valid_mask,
    mape_denominator_epsilon=1e-6,
):
    """
    Compute metric components by forecast horizon.

    The returned components allow correct global metric aggregation:
      MAE   = total absolute error / total valid entries
      RMSE  = sqrt(total squared error / total valid entries)
      WMAPE = total absolute error / total absolute target magnitude
      MAPE  = mean absolute percentage error over eligible entries

    Outputs contain one value per forecast horizon.
    """
    predictions, targets, valid_mask = _validate_prediction_target_mask(
        predictions,
        targets,
        valid_mask,
    )

    if mape_denominator_epsilon <= 0:
        raise ValueError(
            "mape_denominator_epsilon must be positive."
        )

    absolute_error = np.abs(predictions - targets)
    squared_error = np.square(predictions - targets)
    absolute_target = np.abs(targets)

    mape_eligible_mask = (
        valid_mask
        & (absolute_target > mape_denominator_epsilon)
    )

    safe_mape_denominator = np.maximum(
        absolute_target,
        mape_denominator_epsilon,
    )

    absolute_percentage_error = (
        absolute_error / safe_mape_denominator
    )

    statistics = {
        "valid_count": valid_mask.sum(axis=(0, 2)).astype(np.float64),
        "absolute_error_sum": (
            absolute_error * valid_mask
        ).sum(axis=(0, 2)),
        "squared_error_sum": (
            squared_error * valid_mask
        ).sum(axis=(0, 2)),
        "target_absolute_sum": (
            absolute_target * valid_mask
        ).sum(axis=(0, 2)),
        "mape_error_sum": (
            absolute_percentage_error * mape_eligible_mask
        ).sum(axis=(0, 2)),
        "mape_count": (
            mape_eligible_mask.sum(axis=(0, 2)).astype(np.float64)
        ),
    }

    return statistics


def collapse_horizon_statistics(
    horizon_statistics,
):
    """
    Aggregate all forecast horizons into one global set of components.
    """
    return {
        key: np.asarray(value, dtype=np.float64).sum()
        for key, value in horizon_statistics.items()
    }


def metrics_from_sufficient_statistics(
    statistics,
    strict=True,
):
    """
    Compute MAE, RMSE, MAPE, and WMAPE from accumulated components.

    Parameters
    ----------
    statistics : dict
        Metric components, either scalar or horizon-wise arrays.
    strict : bool
        If True, raises an error whenever a required denominator is zero.
        If False, returns NaN for undefined metric entries.
    """
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

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
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


def compute_masked_metrics(
    predictions,
    targets,
    valid_mask,
    mape_denominator_epsilon=1e-6,
):
    """
    Return both horizon-wise metrics and all-horizon global metrics.
    """
    horizon_statistics = masked_sufficient_statistics(
        predictions=predictions,
        targets=targets,
        valid_mask=valid_mask,
        mape_denominator_epsilon=mape_denominator_epsilon,
    )

    global_statistics = collapse_horizon_statistics(
        horizon_statistics
    )

    return {
        "horizon_statistics": horizon_statistics,
        "global_statistics": global_statistics,
        "horizon_metrics": metrics_from_sufficient_statistics(
            horizon_statistics,
            strict=True,
        ),
        "global_metrics": metrics_from_sufficient_statistics(
            global_statistics,
            strict=True,
        ),
    }


def grouped_sufficient_statistics(
    predictions,
    targets,
    valid_mask,
    group_labels,
    mape_denominator_epsilon=1e-6,
):
    """
    Aggregate sufficient statistics separately for chronological groups.

    group_labels should usually be forecast-origin calendar dates.
    One label is required for every sample/window.

    Returns
    -------
    dict with:
      groups : sorted unique group labels
      arrays : [groups, horizons] sufficient-statistic arrays
    """
    predictions, targets, valid_mask = _validate_prediction_target_mask(
        predictions,
        targets,
        valid_mask,
    )

    group_labels = np.asarray(group_labels).astype(str)

    if group_labels.ndim != 1:
        raise ValueError(
            "group_labels must be a one-dimensional array."
        )

    if group_labels.shape[0] != predictions.shape[0]:
        raise ValueError(
            "group_labels must have one value per sample."
        )

    unique_groups = np.asarray(
        sorted(np.unique(group_labels).tolist()),
        dtype=str,
    )

    grouped_arrays = {
        "valid_count": [],
        "absolute_error_sum": [],
        "squared_error_sum": [],
        "target_absolute_sum": [],
        "mape_error_sum": [],
        "mape_count": [],
    }

    for group in unique_groups:
        group_mask = group_labels == group

        group_statistics = masked_sufficient_statistics(
            predictions=predictions[group_mask],
            targets=targets[group_mask],
            valid_mask=valid_mask[group_mask],
            mape_denominator_epsilon=mape_denominator_epsilon,
        )

        for key in grouped_arrays:
            grouped_arrays[key].append(group_statistics[key])

    output = {
        "groups": unique_groups,
    }

    for key, values in grouped_arrays.items():
        output[key] = np.stack(values, axis=0).astype(np.float64)

    return output


def _validate_grouped_statistics(
    grouped_statistics,
):
    """
    Check expected grouped sufficient-statistics structure.
    """
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

    groups = np.asarray(
        grouped_statistics["groups"]
    ).astype(str)

    if groups.ndim != 1 or groups.size == 0:
        raise ValueError(
            "Grouped statistics must include at least one group."
        )

    number_of_groups = groups.shape[0]
    horizon_shape = None

    for key in required_keys - {"groups"}:
        values = np.asarray(
            grouped_statistics[key],
            dtype=np.float64,
        )

        if values.shape[0] != number_of_groups:
            raise ValueError(
                f"Grouped statistic '{key}' has inconsistent group count."
            )

        if horizon_shape is None:
            horizon_shape = values.shape[1:]
        elif values.shape[1:] != horizon_shape:
            raise ValueError(
                "Grouped statistics have inconsistent horizon shapes."
            )

    return groups


def circular_moving_block_indices(
    number_of_groups,
    block_length,
    number_of_resamples,
    random_seed,
):
    """
    Create circular moving-block bootstrap indices.

    The output has shape [resamples, number_of_groups]. Each row
    preserves local dependence by sampling contiguous chronological
    blocks of groups rather than individual windows.
    """
    if number_of_groups < 1:
        raise ValueError(
            "number_of_groups must be positive."
        )

    if block_length < 1:
        raise ValueError(
            "block_length must be positive."
        )

    if number_of_resamples < 1:
        raise ValueError(
            "number_of_resamples must be positive."
        )

    rng = np.random.default_rng(random_seed)

    blocks_needed = int(
        np.ceil(number_of_groups / block_length)
    )

    all_indices = np.empty(
        (number_of_resamples, number_of_groups),
        dtype=np.int64,
    )

    within_block_offsets = np.arange(
        block_length,
        dtype=np.int64,
    )

    for resample_index in range(number_of_resamples):
        block_starts = rng.integers(
            low=0,
            high=number_of_groups,
            size=blocks_needed,
        )

        sampled_indices = np.concatenate(
            [
                (block_start + within_block_offsets)
                % number_of_groups
                for block_start in block_starts
            ]
        )[:number_of_groups]

        all_indices[resample_index] = sampled_indices

    return all_indices


def _sum_grouped_statistics(
    grouped_statistics,
    sampled_group_indices,
):
    """
    Sum selected chronological groups into horizon-wise components.
    """
    keys = [
        "valid_count",
        "absolute_error_sum",
        "squared_error_sum",
        "target_absolute_sum",
        "mape_error_sum",
        "mape_count",
    ]

    return {
        key: np.asarray(
            grouped_statistics[key],
            dtype=np.float64,
        )[sampled_group_indices].sum(axis=0)
        for key in keys
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
    Compute percentile bootstrap CIs for MAE, RMSE, MAPE, and WMAPE.

    CIs are computed from raw aggregated sufficient statistics,
    preserving the correct nonlinearity for RMSE and WMAPE.
    """
    groups = _validate_grouped_statistics(
        grouped_statistics
    )

    if not 0.0 < lower_percentile < upper_percentile < 100.0:
        raise ValueError(
            "Bootstrap percentiles must satisfy 0 < lower < upper < 100."
        )

    number_of_groups = len(groups)

    bootstrap_indices = circular_moving_block_indices(
        number_of_groups=number_of_groups,
        block_length=block_length,
        number_of_resamples=number_of_resamples,
        random_seed=random_seed,
    )

    horizon_shape = np.asarray(
        grouped_statistics["valid_count"]
    ).shape[1:]

    metric_names = ["MAE", "RMSE", "MAPE", "WMAPE"]

    bootstrap_values = {
        metric_name: np.full(
            (number_of_resamples,) + horizon_shape,
            np.nan,
            dtype=np.float64,
        )
        for metric_name in metric_names
    }

    for resample_index, sampled_group_indices in enumerate(
        bootstrap_indices
    ):
        sampled_statistics = _sum_grouped_statistics(
            grouped_statistics=grouped_statistics,
            sampled_group_indices=sampled_group_indices,
        )

        sampled_metrics = metrics_from_sufficient_statistics(
            sampled_statistics,
            strict=False,
        )

        for metric_name in metric_names:
            bootstrap_values[metric_name][resample_index] = (
                sampled_metrics[metric_name]
            )

    point_statistics = _sum_grouped_statistics(
        grouped_statistics=grouped_statistics,
        sampled_group_indices=np.arange(number_of_groups),
    )

    point_metrics = metrics_from_sufficient_statistics(
        point_statistics,
        strict=True,
    )

    confidence_intervals = {}

    for metric_name in metric_names:
        metric_bootstrap_values = bootstrap_values[metric_name]

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
        }

    return {
        "groups": groups,
        "bootstrap_indices": bootstrap_indices,
        "block_length": int(block_length),
        "number_of_resamples": int(number_of_resamples),
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
    Paired temporal block-bootstrap CI for metric difference:

        Delta = metric(A) - metric(B)

    The same temporal blocks are resampled for A and B. A CI entirely
    below zero supports lower error for A; entirely above zero supports
    lower error for B.
    """
    groups_a = _validate_grouped_statistics(
        grouped_statistics_a
    )

    groups_b = _validate_grouped_statistics(
        grouped_statistics_b
    )

    if not np.array_equal(groups_a, groups_b):
        raise ValueError(
            "Paired comparison requires identical chronological groups."
        )

    if not 0.0 < lower_percentile < upper_percentile < 100.0:
        raise ValueError(
            "Bootstrap percentiles must satisfy 0 < lower < upper < 100."
        )

    number_of_groups = len(groups_a)

    bootstrap_indices = circular_moving_block_indices(
        number_of_groups=number_of_groups,
        block_length=block_length,
        number_of_resamples=number_of_resamples,
        random_seed=random_seed,
    )

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

    for resample_index, sampled_group_indices in enumerate(
        bootstrap_indices
    ):
        sampled_statistics_a = _sum_grouped_statistics(
            grouped_statistics=grouped_statistics_a,
            sampled_group_indices=sampled_group_indices,
        )

        sampled_statistics_b = _sum_grouped_statistics(
            grouped_statistics=grouped_statistics_b,
            sampled_group_indices=sampled_group_indices,
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
        }

    return {
        "groups": groups_a,
        "bootstrap_indices": bootstrap_indices,
        "block_length": int(block_length),
        "number_of_resamples": int(number_of_resamples),
        "random_seed": int(random_seed),
        "lower_percentile": float(lower_percentile),
        "upper_percentile": float(upper_percentile),
        "confidence_intervals": confidence_intervals,
    }
