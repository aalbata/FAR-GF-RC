import numpy as np
import torch
from torch.utils.data import Dataset


class CausalMaskedGRUWindowDataset(Dataset):
    """
    Causal direct multi-horizon traffic forecasting dataset.

    Missing normalized values use a 0.0 placeholder, always paired
    with the explicit observation mask.
    """

    def __init__(
        self,
        model_ready_values,
        observation_mask,
        calendar_features,
        elapsed_gap_features,
        anchors,
        input_steps,
        output_steps,
    ):
        self.model_ready_values = np.asarray(
            model_ready_values,
            dtype=np.float32,
        )

        self.observation_mask = np.asarray(
            observation_mask,
            dtype=bool,
        )

        self.calendar_features = np.asarray(
            calendar_features,
            dtype=np.float32,
        )

        self.elapsed_gap_features = np.asarray(
            elapsed_gap_features,
            dtype=np.float32,
        )

        self.anchors = np.asarray(
            anchors,
            dtype=np.int64,
        )

        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)

        if self.model_ready_values.ndim != 2:
            raise ValueError(
                "model_ready_values must be [timestamps, sensors]."
            )

        if self.observation_mask.shape != self.model_ready_values.shape:
            raise ValueError(
                "observation_mask must match model_ready_values."
            )

        if self.elapsed_gap_features.shape != self.model_ready_values.shape:
            raise ValueError(
                "elapsed_gap_features must match model_ready_values."
            )

        if self.calendar_features.ndim != 2:
            raise ValueError(
                "calendar_features must be [timestamps, features]."
            )

        if self.calendar_features.shape[0] != self.model_ready_values.shape[0]:
            raise ValueError(
                "calendar_features timestamps do not align with values."
            )

        if self.anchors.ndim != 1 or self.anchors.size == 0:
            raise ValueError(
                "anchors must be a non-empty one-dimensional array."
            )

        if not np.all(np.diff(self.anchors) > 0):
            raise ValueError(
                "anchors must be strictly increasing."
            )

        if self.anchors.min() < self.input_steps - 1:
            raise ValueError(
                "An anchor lacks adequate input history."
            )

        if self.anchors.max() + self.output_steps >= self.model_ready_values.shape[0]:
            raise ValueError(
                "An anchor lacks adequate future target history."
            )

        if not np.isfinite(self.model_ready_values).all():
            raise ValueError(
                "model_ready_values contain NaN or infinite values."
            )

        if not np.isfinite(self.calendar_features).all():
            raise ValueError(
                "calendar_features contain NaN or infinite values."
            )

        if not np.isfinite(self.elapsed_gap_features).all():
            raise ValueError(
                "elapsed_gap_features contain NaN or infinite values."
            )

    def __len__(self):
        return int(self.anchors.size)

    def __getitem__(self, index):
        anchor = int(self.anchors[index])

        input_start = anchor - self.input_steps + 1
        input_end_exclusive = anchor + 1

        target_start = anchor + 1
        target_end_exclusive = anchor + 1 + self.output_steps

        x_value = self.model_ready_values[
            input_start:input_end_exclusive
        ]

        x_mask = self.observation_mask[
            input_start:input_end_exclusive
        ]

        x_elapsed = self.elapsed_gap_features[
            input_start:input_end_exclusive
        ]

        x_time = self.calendar_features[
            input_start:input_end_exclusive
        ]

        y_value = self.model_ready_values[
            target_start:target_end_exclusive
        ]

        y_mask = self.observation_mask[
            target_start:target_end_exclusive
        ]

        y_time = self.calendar_features[
            target_start:target_end_exclusive
        ]

        return {
            "x_value": torch.from_numpy(
                np.ascontiguousarray(x_value)
            ),
            "x_mask": torch.from_numpy(
                np.ascontiguousarray(
                    x_mask.astype(np.float32)
                )
            ),
            "x_elapsed": torch.from_numpy(
                np.ascontiguousarray(x_elapsed)
            ),
            "x_time": torch.from_numpy(
                np.ascontiguousarray(x_time)
            ),
            "y_value": torch.from_numpy(
                np.ascontiguousarray(y_value)
            ),
            "y_mask": torch.from_numpy(
                np.ascontiguousarray(
                    y_mask.astype(np.bool_)
                )
            ),
            "y_time": torch.from_numpy(
                np.ascontiguousarray(y_time)
            ),
            "anchor": torch.tensor(
                anchor,
                dtype=torch.int64,
            ),
        }
