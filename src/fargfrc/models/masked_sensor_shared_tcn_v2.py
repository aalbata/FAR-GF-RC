
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as functional


class CausalTemporalConv2d(nn.Module):
    """
    Causal time-only 2D convolution.

    Input layout:
        [batch, channels, sensors, time]

    Kernel layout:
        [out_channels, in_channels, 1, temporal_kernel]

    Kernel height is exactly one; therefore each sensor is processed
    independently and no spatial or cross-sensor mixing can occur.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()

        if kernel_size <= 0:
            raise ValueError(
                "kernel_size must be positive."
            )

        if dilation <= 0:
            raise ValueError(
                "dilation must be positive."
            )

        self.left_padding = int(
            (kernel_size - 1) * dilation
        )

        self.convolution = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, dilation),
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "Expected [batch, channels, sensors, time]."
            )

        padded_x = functional.pad(
            x,
            (
                self.left_padding,
                0,
                0,
                0,
            ),
        )

        return self.convolution(
            padded_x
        )


class CausalResidualTCNBlock2d(nn.Module):
    """
    Two causal time-only Conv2d layers with residual connection.

    No batch normalization is used. Therefore, no batch-dependent
    information flow is introduced across sensors or time positions.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.conv_1 = CausalTemporalConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.conv_2 = CausalTemporalConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(
            p=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        residual = x

        z = self.conv_1(x)
        z = self.activation(z)
        z = self.dropout(z)

        z = self.conv_2(z)
        z = self.activation(z)
        z = self.dropout(z)

        return self.activation(
            z + residual
        )


class MaskedSensorSharedTCNV2(nn.Module):
    """
    MS-TCN v2: parameter-equivalent, graph-free causal TCN backend.

    Each sensor has an independent temporal stream. Temporal weights
    are shared across sensors, while a learned static sensor embedding
    supplies sensor identity to the direct multi-horizon prediction head.

    There is no adjacency matrix, graph convolution, sensor attention,
    adaptive graph, or cross-sensor message-passing operation.
    """

    def __init__(
        self,
        num_sensors: int,
        history_feature_dim: int,
        future_calendar_feature_dim: int,
        channels: int,
        kernel_size: int,
        dilations: Iterable[int],
        dropout: float,
        sensor_embedding_dim: int,
        prediction_head_dim: int,
    ) -> None:
        super().__init__()

        if num_sensors <= 0:
            raise ValueError(
                "num_sensors must be positive."
            )

        if history_feature_dim <= 0:
            raise ValueError(
                "history_feature_dim must be positive."
            )

        if future_calendar_feature_dim <= 0:
            raise ValueError(
                "future_calendar_feature_dim must be positive."
            )

        if channels <= 0:
            raise ValueError(
                "channels must be positive."
            )

        if sensor_embedding_dim <= 0:
            raise ValueError(
                "sensor_embedding_dim must be positive."
            )

        self.num_sensors = int(
            num_sensors
        )

        self.history_feature_dim = int(
            history_feature_dim
        )

        self.future_calendar_feature_dim = int(
            future_calendar_feature_dim
        )

        self.channels = int(
            channels
        )

        self.sensor_embedding_dim = int(
            sensor_embedding_dim
        )

        self.input_projection = nn.Conv2d(
            in_channels=self.history_feature_dim,
            out_channels=self.channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        self.temporal_blocks = nn.ModuleList(
            [
                CausalResidualTCNBlock2d(
                    channels=self.channels,
                    kernel_size=kernel_size,
                    dilation=int(dilation),
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )

        self.sensor_embedding = nn.Embedding(
            num_embeddings=self.num_sensors,
            embedding_dim=self.sensor_embedding_dim,
        )

        prediction_input_dim = (
            self.channels
            + self.sensor_embedding_dim
            + self.future_calendar_feature_dim
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(
                prediction_input_dim,
                prediction_head_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                p=dropout
            ),
            nn.Linear(
                prediction_head_dim,
                1,
            ),
        )

    def encode_history(
        self,
        history_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode independent sensor histories.

        Input:
            [batch, input_steps, sensors, history_features]

        Output:
            [batch, input_steps, sensors, channels]
        """
        if history_features.ndim != 4:
            raise ValueError(
                "history_features must have shape "
                "[batch, time, sensors, features]."
            )

        batch_size, time_steps, sensors, feature_dim = (
            history_features.shape
        )

        if sensors != self.num_sensors:
            raise ValueError(
                "Unexpected sensor count."
            )

        if feature_dim != self.history_feature_dim:
            raise ValueError(
                "Unexpected historical feature dimension."
            )

        # [batch, time, sensors, features]
        # -> [batch, features, sensors, time]
        temporal_tensor = (
            history_features
            .permute(0, 3, 2, 1)
            .contiguous()
        )

        encoded = self.input_projection(
            temporal_tensor
        )

        for temporal_block in self.temporal_blocks:
            encoded = temporal_block(
                encoded
            )

        # [batch, channels, sensors, time]
        # -> [batch, time, sensors, channels]
        return (
            encoded
            .permute(0, 3, 2, 1)
            .contiguous()
        )

    def forward(
        self,
        history_features: torch.Tensor,
        future_calendar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return direct multi-horizon normalized forecasts.

        Input:
            history_features:
                [batch, input_steps, sensors, history_feature_dim]

            future_calendar:
                [batch, output_steps, future_calendar_feature_dim]

        Output:
            [batch, output_steps, sensors]
        """
        if future_calendar.ndim != 3:
            raise ValueError(
                "future_calendar must have shape "
                "[batch, output_steps, calendar_features]."
            )

        encoded_history = self.encode_history(
            history_features
        )

        batch_size = encoded_history.shape[0]
        output_steps = future_calendar.shape[1]

        if future_calendar.shape[0] != batch_size:
            raise ValueError(
                "History and future-calendar batch sizes differ."
            )

        if (
            future_calendar.shape[2]
            != self.future_calendar_feature_dim
        ):
            raise ValueError(
                "Unexpected future-calendar feature dimension."
            )

        final_history_state = encoded_history[
            :,
            -1,
            :,
            :,
        ]

        sensor_indices = torch.arange(
            self.num_sensors,
            device=history_features.device,
            dtype=torch.long,
        )

        sensor_embeddings = self.sensor_embedding(
            sensor_indices
        )

        expanded_history_state = (
            final_history_state
            .unsqueeze(1)
            .expand(
                batch_size,
                output_steps,
                self.num_sensors,
                self.channels,
            )
        )

        expanded_sensor_embeddings = (
            sensor_embeddings
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(
                batch_size,
                output_steps,
                self.num_sensors,
                self.sensor_embedding_dim,
            )
        )

        expanded_future_calendar = (
            future_calendar
            .unsqueeze(2)
            .expand(
                batch_size,
                output_steps,
                self.num_sensors,
                self.future_calendar_feature_dim,
            )
        )

        prediction_features = torch.cat(
            [
                expanded_history_state,
                expanded_sensor_embeddings,
                expanded_future_calendar,
            ],
            dim=-1,
        )

        return self.prediction_head(
            prediction_features
        ).squeeze(
            dim=-1
        )
