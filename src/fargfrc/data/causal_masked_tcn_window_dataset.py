
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class CausalMaskedTCNWindowDataset(Dataset):
    """
    Causal METR-LA window dataset for a non-graph temporal model.

    Every sample is indexed by target_start:
      history: [target_start-input_steps, target_start)
      target : [target_start, target_start+output_steps)

    Missing traffic values remain zero placeholders after
    normalization and are explicitly accompanied by observation masks.
    """

    def __init__(
        self,
        normalized_values: np.ndarray,
        observation_mask: np.ndarray,
        normalized_elapsed_steps: np.ndarray,
        calendar_features: np.ndarray,
        target_start_indices: np.ndarray,
        input_steps: int,
        output_steps: int,
    ) -> None:
        super().__init__()

        normalized_values = np.asarray(
            normalized_values,
            dtype=np.float32,
        )

        observation_mask = np.asarray(
            observation_mask,
            dtype=np.float32,
        )

        normalized_elapsed_steps = np.asarray(
            normalized_elapsed_steps,
            dtype=np.float32,
        )

        calendar_features = np.asarray(
            calendar_features,
            dtype=np.float32,
        )

        target_start_indices = np.asarray(
            target_start_indices,
            dtype=np.int64,
        )

        if normalized_values.ndim != 2:
            raise ValueError(
                "normalized_values must have shape [time, sensors]."
            )

        if observation_mask.shape != normalized_values.shape:
            raise ValueError(
                "observation_mask shape must match normalized_values."
            )

        if normalized_elapsed_steps.shape != normalized_values.shape:
            raise ValueError(
                "normalized_elapsed_steps shape must match "
                "normalized_values."
            )

        if calendar_features.ndim != 2:
            raise ValueError(
                "calendar_features must have shape [time, features]."
            )

        if calendar_features.shape[0] != normalized_values.shape[0]:
            raise ValueError(
                "calendar_features time dimension must match "
                "normalized_values."
            )

        if target_start_indices.ndim != 1:
            raise ValueError(
                "target_start_indices must be one-dimensional."
            )

        if target_start_indices.size == 0:
            raise ValueError(
                "target_start_indices cannot be empty."
            )

        if input_steps <= 0 or output_steps <= 0:
            raise ValueError(
                "input_steps and output_steps must be positive."
            )

        minimum_anchor = int(target_start_indices.min())
        maximum_anchor = int(target_start_indices.max())

        if minimum_anchor < input_steps:
            raise ValueError(
                "A target start index does not have sufficient "
                "causal history."
            )

        if maximum_anchor + output_steps > normalized_values.shape[0]:
            raise ValueError(
                "A target window extends beyond the supplied data."
            )

        self.normalized_values = normalized_values
        self.observation_mask = observation_mask
        self.normalized_elapsed_steps = normalized_elapsed_steps
        self.calendar_features = calendar_features
        self.target_start_indices = target_start_indices
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)

        self.num_sensors = int(normalized_values.shape[1])
        self.history_feature_dim = int(
            3 + calendar_features.shape[1]
        )
        self.future_calendar_feature_dim = int(
            calendar_features.shape[1]
        )

    def __len__(self) -> int:
        return int(self.target_start_indices.size)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        target_start = int(
            self.target_start_indices[index]
        )

        history_start = target_start - self.input_steps
        target_end = target_start + self.output_steps

        history_values = self.normalized_values[
            history_start:target_start
        ]

        history_mask = self.observation_mask[
            history_start:target_start
        ]

        history_elapsed = self.normalized_elapsed_steps[
            history_start:target_start
        ]

        history_calendar = self.calendar_features[
            history_start:target_start
        ]

        target_values = self.normalized_values[
            target_start:target_end
        ]

        target_mask = self.observation_mask[
            target_start:target_end
        ]

        future_calendar = self.calendar_features[
            target_start:target_end
        ]

        expanded_history_calendar = np.broadcast_to(
            history_calendar[:, None, :],
            (
                self.input_steps,
                self.num_sensors,
                self.future_calendar_feature_dim,
            ),
        )

        history_features = np.concatenate(
            [
                history_values[:, :, None],
                history_mask[:, :, None],
                history_elapsed[:, :, None],
                expanded_history_calendar,
            ],
            axis=-1,
        ).astype(
            np.float32,
            copy=False,
        )

        return {
            "history_features": torch.from_numpy(
                history_features
            ),
            "future_calendar": torch.from_numpy(
                future_calendar.astype(
                    np.float32,
                    copy=False,
                )
            ),
            "target_values": torch.from_numpy(
                target_values.astype(
                    np.float32,
                    copy=False,
                )
            ),
            "target_mask": torch.from_numpy(
                target_mask.astype(
                    np.float32,
                    copy=False,
                )
            ),
            "target_start_index": torch.tensor(
                target_start,
                dtype=torch.int64,
            ),
        }
