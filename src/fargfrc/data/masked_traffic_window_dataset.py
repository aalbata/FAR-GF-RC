import numpy as np
import torch
from torch.utils.data import Dataset


class MaskedTrafficWindowDataset(Dataset):
    """
    Lazy window dataset for masked traffic forecasting.

    Returned tensors:
        x_value : [input_steps, sensors]
        x_mask  : [input_steps, sensors]
        x_time  : [input_steps, time_features]
        y_value : [output_steps, sensors]
        y_mask  : [output_steps, sensors]
        y_time  : [output_steps, time_features]
        anchor  : scalar input-history end index
    """

    def __init__(
        self,
        model_values,
        observation_mask,
        time_features,
        anchors,
        input_steps,
        output_steps,
    ):
        self.model_values = np.ascontiguousarray(
            model_values,
            dtype=np.float32,
        )

        self.observation_mask = np.ascontiguousarray(
            observation_mask,
            dtype=np.float32,
        )

        self.time_features = np.ascontiguousarray(
            time_features,
            dtype=np.float32,
        )

        self.anchors = np.asarray(
            anchors,
            dtype=np.int64,
        )

        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)

        if self.model_values.ndim != 2:
            raise ValueError(
                "model_values must have shape [timestamps, sensors]."
            )

        if self.observation_mask.shape != self.model_values.shape:
            raise ValueError(
                "observation_mask shape must match model_values."
            )

        if self.time_features.shape[0] != self.model_values.shape[0]:
            raise ValueError(
                "time_features must align with the timestamp dimension."
            )

        if self.input_steps < 1 or self.output_steps < 1:
            raise ValueError(
                "input_steps and output_steps must both be positive."
            )

        if len(self.anchors) == 0:
            raise ValueError(
                "Dataset anchors cannot be empty."
            )

        if not np.isfinite(self.model_values).all():
            raise ValueError(
                "model_values contain NaN or infinite values."
            )

        if not np.isfinite(self.time_features).all():
            raise ValueError(
                "time_features contain NaN or infinite values."
            )

        self._validate_anchors()

    def _validate_anchors(self):
        minimum_anchor = self.input_steps - 1
        maximum_anchor = (
            self.model_values.shape[0]
            - self.output_steps
            - 1
        )

        if self.anchors.min() < minimum_anchor:
            raise ValueError(
                "At least one anchor does not have enough input history."
            )

        if self.anchors.max() > maximum_anchor:
            raise ValueError(
                "At least one anchor exceeds the available future horizon."
            )

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        anchor = int(self.anchors[index])

        input_start = anchor - self.input_steps + 1
        input_end = anchor + 1

        target_start = anchor + 1
        target_end = anchor + 1 + self.output_steps

        return {
            "x_value": torch.from_numpy(
                self.model_values[input_start:input_end]
            ),
            "x_mask": torch.from_numpy(
                self.observation_mask[input_start:input_end]
            ),
            "x_time": torch.from_numpy(
                self.time_features[input_start:input_end]
            ),
            "y_value": torch.from_numpy(
                self.model_values[target_start:target_end]
            ),
            "y_mask": torch.from_numpy(
                self.observation_mask[target_start:target_end]
            ),
            "y_time": torch.from_numpy(
                self.time_features[target_start:target_end]
            ),
            "anchor": torch.tensor(
                anchor,
                dtype=torch.long,
            ),
        }
