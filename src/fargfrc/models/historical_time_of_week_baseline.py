import numpy as np


def fit_historical_time_of_week_profile(
    training_values,
    training_observation_mask,
    training_time_of_week_slots,
    sensor_means,
    number_of_time_slots,
):
    """
    Fit a training-only sensor-by-time-of-week mean profile.

    Parameters
    ----------
    training_values : np.ndarray
        Raw traffic values with shape [timestamps, sensors].
    training_observation_mask : np.ndarray
        Boolean valid-observation mask with the same shape.
    training_time_of_week_slots : np.ndarray
        Integer slot identifier in [0, number_of_time_slots - 1]
        for each training timestamp.
    sensor_means : np.ndarray
        Training-only fallback mean for each sensor.
    number_of_time_slots : int
        Number of weekly five-minute slots (2016 for METR-LA).

    Returns
    -------
    historical_profile : np.ndarray
        Shape [number_of_time_slots, sensors].
    slot_observation_counts : np.ndarray
        Shape [number_of_time_slots, sensors].
    slot_has_direct_observation : np.ndarray
        Shape [number_of_time_slots, sensors].
    """
    training_values = np.asarray(
        training_values,
        dtype=np.float64,
    )

    training_observation_mask = np.asarray(
        training_observation_mask,
        dtype=bool,
    )

    training_time_of_week_slots = np.asarray(
        training_time_of_week_slots,
        dtype=np.int64,
    )

    sensor_means = np.asarray(
        sensor_means,
        dtype=np.float64,
    )

    if training_values.ndim != 2:
        raise ValueError(
            "training_values must have shape [timestamps, sensors]."
        )

    if training_observation_mask.shape != training_values.shape:
        raise ValueError(
            "training_observation_mask must match training_values."
        )

    if training_time_of_week_slots.shape != (
        training_values.shape[0],
    ):
        raise ValueError(
            "One time-of-week slot is required per training timestamp."
        )

    if sensor_means.shape != (training_values.shape[1],):
        raise ValueError(
            "sensor_means must have one value per sensor."
        )

    if number_of_time_slots < 1:
        raise ValueError(
            "number_of_time_slots must be positive."
        )

    if np.any(training_time_of_week_slots < 0) or np.any(
        training_time_of_week_slots >= number_of_time_slots
    ):
        raise ValueError(
            "Training time-of-week slots are out of range."
        )

    if not np.isfinite(
        training_values[training_observation_mask]
    ).all():
        raise ValueError(
            "Valid training observations contain NaN or infinite values."
        )

    slot_sums = np.zeros(
        (number_of_time_slots, training_values.shape[1]),
        dtype=np.float64,
    )

    slot_observation_counts = np.zeros(
        (number_of_time_slots, training_values.shape[1]),
        dtype=np.int64,
    )

    masked_training_values = np.where(
        training_observation_mask,
        training_values,
        0.0,
    )

    np.add.at(
        slot_sums,
        training_time_of_week_slots,
        masked_training_values,
    )

    np.add.at(
        slot_observation_counts,
        training_time_of_week_slots,
        training_observation_mask.astype(np.int64),
    )

    slot_has_direct_observation = (
        slot_observation_counts > 0
    )

    historical_profile = np.divide(
        slot_sums,
        slot_observation_counts,
        out=np.zeros_like(slot_sums),
        where=slot_has_direct_observation,
    )

    historical_profile = np.where(
        slot_has_direct_observation,
        historical_profile,
        sensor_means[None, :],
    )

    if not np.isfinite(historical_profile).all():
        raise RuntimeError(
            "Historical profile contains NaN or infinite values."
        )

    if np.any(historical_profile <= 0):
        raise RuntimeError(
            "Historical profile contains non-positive traffic predictions."
        )

    return (
        historical_profile,
        slot_observation_counts,
        slot_has_direct_observation,
    )


def predict_historical_time_of_week(
    historical_profile,
    slot_observation_counts,
    target_time_of_week_slots,
):
    """
    Generate raw-unit historical time-of-week predictions.

    Parameters
    ----------
    historical_profile : np.ndarray
        Shape [time_slots, sensors].
    slot_observation_counts : np.ndarray
        Shape [time_slots, sensors].
    target_time_of_week_slots : np.ndarray
        Shape [batch, forecast_horizons].

    Returns
    -------
    predictions : np.ndarray
        Shape [batch, forecast_horizons, sensors].
    training_mean_fallback_used : np.ndarray
        Boolean array matching predictions. True means the historical
        slot lacked an observed training value for that sensor.
    """
    historical_profile = np.asarray(
        historical_profile,
        dtype=np.float64,
    )

    slot_observation_counts = np.asarray(
        slot_observation_counts,
        dtype=np.int64,
    )

    target_time_of_week_slots = np.asarray(
        target_time_of_week_slots,
        dtype=np.int64,
    )

    if historical_profile.ndim != 2:
        raise ValueError(
            "historical_profile must have shape [time_slots, sensors]."
        )

    if slot_observation_counts.shape != historical_profile.shape:
        raise ValueError(
            "slot_observation_counts must match historical_profile."
        )

    if target_time_of_week_slots.ndim != 2:
        raise ValueError(
            "target_time_of_week_slots must have shape [batch, horizons]."
        )

    if np.any(target_time_of_week_slots < 0) or np.any(
        target_time_of_week_slots >= historical_profile.shape[0]
    ):
        raise ValueError(
            "Target time-of-week slots are out of range."
        )

    predictions = historical_profile[
        target_time_of_week_slots
    ]

    training_mean_fallback_used = (
        slot_observation_counts[
            target_time_of_week_slots
        ] == 0
    )

    if not np.isfinite(predictions).all():
        raise RuntimeError(
            "Historical baseline predictions contain NaN or infinite values."
        )

    return (
        predictions.astype(np.float64),
        training_mean_fallback_used,
    )
