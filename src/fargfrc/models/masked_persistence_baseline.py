import numpy as np


def masked_last_observation_persistence(
    input_values,
    input_mask,
    output_steps,
):
    """
    Generate normalized MLOCF predictions.

    Parameters
    ----------
    input_values : np.ndarray
        Shape [batch, input_steps, sensors]. Missing positions must
        contain the fixed normalized placeholder 0.0.
    input_mask : np.ndarray
        Shape [batch, input_steps, sensors]. One denotes an observed
        native reading; zero denotes unavailable input.
    output_steps : int
        Number of future prediction steps.

    Returns
    -------
    prediction_values : np.ndarray
        Shape [batch, output_steps, sensors].
    sensor_has_history : np.ndarray
        Shape [batch, sensors]. True when at least one native
        observation existed in the input history.

    Notes
    -----
    When no history is available for a sensor, the normalized fallback
    is exactly 0.0. After inverse scaling this equals the sensor's
    training-only mean.
    """
    input_values = np.asarray(
        input_values,
        dtype=np.float64,
    )

    input_mask = np.asarray(
        input_mask,
        dtype=bool,
    )

    if input_values.ndim != 3:
        raise ValueError(
            "input_values must have shape [batch, input_steps, sensors]."
        )

    if input_mask.shape != input_values.shape:
        raise ValueError(
            "input_mask must match input_values shape."
        )

    if output_steps < 1:
        raise ValueError(
            "output_steps must be positive."
        )

    if not np.isfinite(input_values).all():
        raise ValueError(
            "input_values contain NaN or infinite values."
        )

    reversed_mask = input_mask[:, ::-1, :]

    sensor_has_history = reversed_mask.any(axis=1)

    reverse_last_valid_index = reversed_mask.argmax(axis=1)

    input_steps = input_values.shape[1]

    last_valid_index = (
        input_steps - 1 - reverse_last_valid_index
    )

    last_valid_values = np.take_along_axis(
        input_values,
        last_valid_index[:, None, :],
        axis=1,
    ).squeeze(axis=1)

    last_valid_values = np.where(
        sensor_has_history,
        last_valid_values,
        0.0,
    )

    prediction_values = np.repeat(
        last_valid_values[:, None, :],
        repeats=int(output_steps),
        axis=1,
    )

    return (
        prediction_values.astype(np.float64),
        sensor_has_history,
    )
