import numpy as np


def masked_same_time_previous_day(
    input_values,
    input_mask,
    daily_source_values,
    daily_source_mask,
):
    """
    Construct normalized same-time-previous-day predictions.

    Prediction hierarchy for each [sample, horizon, sensor]:
      1. Valid value at the same time on the previous day.
      2. Most recent valid sensor value in the current input history.
      3. Normalized 0.0, corresponding to the training-only sensor mean.

    Parameters
    ----------
    input_values : np.ndarray
        Shape [batch, input_steps, sensors].
    input_mask : np.ndarray
        Shape [batch, input_steps, sensors].
    daily_source_values : np.ndarray
        Shape [batch, output_steps, sensors].
    daily_source_mask : np.ndarray
        Shape [batch, output_steps, sensors].

    Returns
    -------
    prediction_values : np.ndarray
        Shape [batch, output_steps, sensors].
    daily_source_used : np.ndarray
        Boolean array with the same prediction shape.
    input_history_used : np.ndarray
        Boolean array with the same prediction shape.
    training_mean_used : np.ndarray
        Boolean array with the same prediction shape.
    """
    input_values = np.asarray(input_values, dtype=np.float64)
    input_mask = np.asarray(input_mask, dtype=bool)
    daily_source_values = np.asarray(
        daily_source_values,
        dtype=np.float64,
    )
    daily_source_mask = np.asarray(
        daily_source_mask,
        dtype=bool,
    )

    if input_values.ndim != 3:
        raise ValueError(
            "input_values must be [batch, input_steps, sensors]."
        )

    if input_mask.shape != input_values.shape:
        raise ValueError(
            "input_mask must match input_values."
        )

    if daily_source_values.ndim != 3:
        raise ValueError(
            "daily_source_values must be [batch, output_steps, sensors]."
        )

    if daily_source_mask.shape != daily_source_values.shape:
        raise ValueError(
            "daily_source_mask must match daily_source_values."
        )

    if input_values.shape[0] != daily_source_values.shape[0]:
        raise ValueError(
            "Input and daily-source batch dimensions must match."
        )

    if input_values.shape[2] != daily_source_values.shape[2]:
        raise ValueError(
            "Input and daily-source sensor dimensions must match."
        )

    if not np.isfinite(input_values).all():
        raise ValueError(
            "input_values contain NaN or infinite values."
        )

    if not np.isfinite(daily_source_values).all():
        raise ValueError(
            "daily_source_values contain NaN or infinite values."
        )

    reversed_mask = input_mask[:, ::-1, :]
    sensor_has_input_history = reversed_mask.any(axis=1)

    reverse_last_valid_index = reversed_mask.argmax(axis=1)

    input_steps = input_values.shape[1]

    last_valid_index = (
        input_steps - 1 - reverse_last_valid_index
    )

    last_input_values = np.take_along_axis(
        input_values,
        last_valid_index[:, None, :],
        axis=1,
    ).squeeze(axis=1)

    last_input_values = np.where(
        sensor_has_input_history,
        last_input_values,
        0.0,
    )

    input_history_prediction = np.repeat(
        last_input_values[:, None, :],
        repeats=daily_source_values.shape[1],
        axis=1,
    )

    prediction_values = np.where(
        daily_source_mask,
        daily_source_values,
        input_history_prediction,
    )

    daily_source_used = daily_source_mask.copy()

    input_history_used = (
        ~daily_source_mask
        & sensor_has_input_history[:, None, :]
    )

    training_mean_used = (
        ~daily_source_mask
        & ~sensor_has_input_history[:, None, :]
    )

    hierarchy_count = (
        daily_source_used.astype(np.int8)
        + input_history_used.astype(np.int8)
        + training_mean_used.astype(np.int8)
    )

    if not np.all(hierarchy_count == 1):
        raise RuntimeError(
            "Prediction-source hierarchy is not mutually exclusive and exhaustive."
        )

    return (
        prediction_values.astype(np.float64),
        daily_source_used,
        input_history_used,
        training_mean_used,
    )
