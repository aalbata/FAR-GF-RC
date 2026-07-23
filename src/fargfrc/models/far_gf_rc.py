"""
FAR-GF-RC: Failure-Aware Reliability-Calibrated Graph Forecaster
with Robust Failure Curriculum.

The model consumes a seven-channel historical sensor tensor:
[normalized value, observation mask, elapsed gap,
 time-of-day sin/cos, day-of-week sin/cos]

and a four-channel future calendar tensor:
[time-of-day sin/cos, day-of-week sin/cos].

It provides:
- normalized multi-horizon forecasts,
- predictive log-scale estimates for later calibration,
- reconstruction values for artificially masked history,
- reliability weights for every historical sensor state,
- learned adaptive functional graph weights.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class ReliabilityEstimator(nn.Module):
    """
    Estimates the trustworthiness of each sensor-history state.

    Reliability is learned from:
    1. current observation availability,
    2. elapsed time since last available observation,
    3. sensor-specific availability rate within the history window.

    A recency prior ensures that prolonged missing observations
    receive progressively lower reliability.
    """

    def __init__(
        self,
        hidden_dimension: int,
        dropout: float,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                3,
                hidden_dimension,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout,
            ),
            nn.Linear(
                hidden_dimension,
                1,
            ),
        )

    def forward(
        self,
        observation_mask: torch.Tensor,
        elapsed_gap: torch.Tensor,
    ) -> torch.Tensor:
        if observation_mask.shape != elapsed_gap.shape:
            raise ValueError(
                "Observation mask and elapsed-gap tensors "
                "must have identical shapes."
            )

        availability_rate = observation_mask.mean(
            dim=1,
            keepdim=True,
        ).expand_as(
            observation_mask
        )

        reliability_inputs = torch.stack(
            [
                observation_mask,
                elapsed_gap,
                availability_rate,
            ],
            dim=-1,
        )

        learned_reliability = torch.sigmoid(
            self.network(
                reliability_inputs
            ).squeeze(
                dim=-1
            )
        )

        recency_prior = (
            observation_mask
            + (
                1.0
                - observation_mask
            )
            * torch.exp(
                -3.0
                * elapsed_gap
            )
        )

        reliability = (
            learned_reliability
            * recency_prior
        )

        return reliability.clamp(
            min=0.0,
            max=1.0,
        )


class AdaptiveFunctionalGraph(nn.Module):
    """
    Learns a static sensor-to-sensor functional adjacency matrix.
    The graph is row normalized through softmax.
    """

    def __init__(
        self,
        number_of_sensors: int,
        embedding_dimension: int,
    ):
        super().__init__()

        self.source_embedding = nn.Parameter(
            torch.empty(
                number_of_sensors,
                embedding_dimension,
            )
        )

        self.target_embedding = nn.Parameter(
            torch.empty(
                number_of_sensors,
                embedding_dimension,
            )
        )

        nn.init.xavier_uniform_(
            self.source_embedding
        )

        nn.init.xavier_uniform_(
            self.target_embedding
        )

        self.embedding_dimension = (
            embedding_dimension
        )

    def forward(
        self,
    ) -> torch.Tensor:
        affinity_logits = torch.matmul(
            self.source_embedding,
            self.target_embedding.transpose(
                0,
                1,
            ),
        ) / math.sqrt(
            float(
                self.embedding_dimension
            )
        )

        return torch.softmax(
            affinity_logits,
            dim=-1,
        )


class ReliabilityGatedGraphLayer(nn.Module):
    """
    Combines physical and adaptive graph messages.

    Source-node reliability weights both graph supports before
    aggregation. Consequently, information originating from a
    prolonged or unavailable sensor history contributes less to
    the receiving node representation.
    """

    def __init__(
        self,
        latent_dimension: int,
        dropout: float,
    ):
        super().__init__()

        self.update_network = nn.Sequential(
            nn.Linear(
                (
                    3
                    * latent_dimension
                    + 1
                ),
                latent_dimension,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout,
            ),
            nn.Linear(
                latent_dimension,
                latent_dimension,
            ),
        )

        self.normalization = nn.LayerNorm(
            latent_dimension
        )

    @staticmethod
    def _aggregate(
        adjacency: torch.Tensor,
        node_states: torch.Tensor,
        source_reliability: torch.Tensor,
    ) -> torch.Tensor:
        if adjacency.ndim != 2:
            raise ValueError(
                "Adjacency must have shape [sensors, sensors]."
            )

        if node_states.ndim != 3:
            raise ValueError(
                "Node states must have shape "
                "[batch, sensors, latent]."
            )

        if source_reliability.ndim != 2:
            raise ValueError(
                "Source reliability must have shape "
                "[batch, sensors]."
            )

        reliability_weighted_adjacency = (
            adjacency.unsqueeze(
                dim=0
            )
            * source_reliability.unsqueeze(
                dim=1
            )
        )

        reliability_weighted_adjacency = (
            reliability_weighted_adjacency
            / reliability_weighted_adjacency.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(
                1e-6
            )
        )

        return torch.bmm(
            reliability_weighted_adjacency,
            node_states,
        )

    def forward(
        self,
        node_states: torch.Tensor,
        source_reliability: torch.Tensor,
        physical_adjacency: torch.Tensor,
        adaptive_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        physical_messages = self._aggregate(
            adjacency=physical_adjacency,
            node_states=node_states,
            source_reliability=source_reliability,
        )

        adaptive_messages = self._aggregate(
            adjacency=adaptive_adjacency,
            node_states=node_states,
            source_reliability=source_reliability,
        )

        update_inputs = torch.cat(
            [
                node_states,
                physical_messages,
                adaptive_messages,
                source_reliability.unsqueeze(
                    dim=-1
                ),
            ],
            dim=-1,
        )

        updated_states = self.update_network(
            update_inputs
        )

        return self.normalization(
            node_states
            + updated_states
        )


class FARGFRC(nn.Module):
    """
    Failure-Aware Reliability-Calibrated Graph Forecaster.

    Inputs
    ------
    history_features:
        Tensor [batch, input_steps, sensors, 7].

    future_calendar:
        Tensor [batch, output_steps, 4].

    Outputs
    -------
    Dictionary containing:
    forecast_normalized:
        Tensor [batch, output_steps, sensors].

    forecast_log_scale:
        Tensor [batch, output_steps, sensors].

    reconstruction_normalized:
        Tensor [batch, input_steps, sensors].

    reliability:
        Tensor [batch, input_steps, sensors].

    adaptive_adjacency:
        Tensor [sensors, sensors].
    """

    def __init__(
        self,
        number_of_sensors: int,
        history_feature_dimension: int,
        future_calendar_feature_dimension: int,
        input_steps: int,
        output_steps: int,
        physical_adjacency: torch.Tensor,
        latent_dimension: int = 64,
        reliability_hidden_dimension: int = 32,
        temporal_gru_layers: int = 1,
        graph_layers: int = 2,
        sensor_embedding_dimension: int = 16,
        decoder_hidden_dimension: int = 64,
        dropout: float = 0.10,
        forecast_log_scale_minimum: float = -6.0,
        forecast_log_scale_maximum: float = 3.0,
    ):
        super().__init__()

        if history_feature_dimension != 7:
            raise ValueError(
                "FAR-GF-RC requires seven historical features."
            )

        if future_calendar_feature_dimension != 4:
            raise ValueError(
                "FAR-GF-RC requires four future calendar features."
            )

        if physical_adjacency.shape != (
            number_of_sensors,
            number_of_sensors,
        ):
            raise ValueError(
                "Physical adjacency shape does not match "
                "the number of sensors."
            )

        if graph_layers <= 0:
            raise ValueError(
                "At least one graph layer is required."
            )

        self.number_of_sensors = number_of_sensors
        self.history_feature_dimension = (
            history_feature_dimension
        )
        self.future_calendar_feature_dimension = (
            future_calendar_feature_dimension
        )
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.latent_dimension = latent_dimension
        self.forecast_log_scale_minimum = (
            forecast_log_scale_minimum
        )
        self.forecast_log_scale_maximum = (
            forecast_log_scale_maximum
        )

        physical_adjacency = physical_adjacency.detach().float()

        physical_adjacency = physical_adjacency.clamp_min(
            0.0
        )

        physical_adjacency = (
            physical_adjacency
            + torch.eye(
                number_of_sensors,
                dtype=torch.float32,
                device=physical_adjacency.device,
            )
        )

        physical_adjacency = (
            physical_adjacency
            / physical_adjacency.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(
                1e-6
            )
        )

        self.register_buffer(
            "physical_adjacency",
            physical_adjacency,
        )

        self.reliability_estimator = ReliabilityEstimator(
            hidden_dimension=reliability_hidden_dimension,
            dropout=dropout,
        )

        self.history_projection = nn.Sequential(
            nn.Linear(
                history_feature_dimension + 1,
                latent_dimension,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout,
            ),
        )

        self.temporal_encoder = nn.GRU(
            input_size=latent_dimension,
            hidden_size=latent_dimension,
            num_layers=temporal_gru_layers,
            batch_first=True,
            dropout=(
                dropout
                if temporal_gru_layers > 1
                else 0.0
            ),
        )

        self.adaptive_graph = AdaptiveFunctionalGraph(
            number_of_sensors=number_of_sensors,
            embedding_dimension=sensor_embedding_dimension,
        )

        self.graph_layers = nn.ModuleList(
            [
                ReliabilityGatedGraphLayer(
                    latent_dimension=latent_dimension,
                    dropout=dropout,
                )
                for _ in range(
                    graph_layers
                )
            ]
        )

        self.future_calendar_encoder = nn.Sequential(
            nn.Linear(
                future_calendar_feature_dimension,
                latent_dimension,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout,
            ),
        )

        self.horizon_embedding = nn.Embedding(
            output_steps,
            latent_dimension,
        )

        self.forecast_decoder = nn.Sequential(
            nn.Linear(
                latent_dimension,
                decoder_hidden_dimension,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout,
            ),
            nn.Linear(
                decoder_hidden_dimension,
                2,
            ),
        )

        self.reconstruction_head = nn.Sequential(
            nn.Linear(
                2
                * latent_dimension,
                decoder_hidden_dimension,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout,
            ),
            nn.Linear(
                decoder_hidden_dimension,
                1,
            ),
        )

    def forward(
        self,
        history_features: torch.Tensor,
        future_calendar: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if history_features.ndim != 4:
            raise ValueError(
                "history_features must have shape "
                "[batch, input_steps, sensors, features]."
            )

        if future_calendar.ndim != 3:
            raise ValueError(
                "future_calendar must have shape "
                "[batch, output_steps, calendar_features]."
            )

        batch_size, input_steps, sensor_count, feature_count = (
            history_features.shape
        )

        future_batch_size, output_steps, calendar_feature_count = (
            future_calendar.shape
        )

        if batch_size != future_batch_size:
            raise ValueError(
                "History and future-calendar batch sizes differ."
            )

        if input_steps != self.input_steps:
            raise ValueError(
                "Unexpected number of input steps."
            )

        if output_steps != self.output_steps:
            raise ValueError(
                "Unexpected number of output steps."
            )

        if sensor_count != self.number_of_sensors:
            raise ValueError(
                "Unexpected number of sensors."
            )

        if feature_count != self.history_feature_dimension:
            raise ValueError(
                "Unexpected historical feature dimension."
            )

        if (
            calendar_feature_count
            != self.future_calendar_feature_dimension
        ):
            raise ValueError(
                "Unexpected future calendar feature dimension."
            )

        observation_mask = history_features[
            ...,
            1,
        ].float().clamp(
            min=0.0,
            max=1.0,
        )

        elapsed_gap = history_features[
            ...,
            2,
        ].float().clamp(
            min=0.0,
            max=1.0,
        )

        reliability = self.reliability_estimator(
            observation_mask=observation_mask,
            elapsed_gap=elapsed_gap,
        )

        history_with_reliability = torch.cat(
            [
                history_features,
                reliability.unsqueeze(
                    dim=-1
                ),
            ],
            dim=-1,
        )

        projected_history = self.history_projection(
            history_with_reliability
        )

        projected_history = (
            projected_history
            * (
                0.25
                + 0.75
                * reliability.unsqueeze(
                    dim=-1
                )
            )
        )

        temporal_input = projected_history.permute(
            0,
            2,
            1,
            3,
        ).reshape(
            batch_size
            * sensor_count,
            input_steps,
            self.latent_dimension,
        )

        temporal_output, _ = self.temporal_encoder(
            temporal_input
        )

        temporal_states = temporal_output.reshape(
            batch_size,
            sensor_count,
            input_steps,
            self.latent_dimension,
        ).permute(
            0,
            2,
            1,
            3,
        ).contiguous()

        node_states = temporal_states[
            :,
            -1,
            :,
            :,
        ]

        latest_reliability = reliability[
            :,
            -1,
            :,
        ]

        adaptive_adjacency = self.adaptive_graph()

        for graph_layer in self.graph_layers:
            node_states = graph_layer(
                node_states=node_states,
                source_reliability=latest_reliability,
                physical_adjacency=self.physical_adjacency,
                adaptive_adjacency=adaptive_adjacency,
            )

        future_context = self.future_calendar_encoder(
            future_calendar
        )

        horizon_indices = torch.arange(
            self.output_steps,
            device=history_features.device,
        )

        horizon_context = self.horizon_embedding(
            horizon_indices
        )

        decoder_states = (
            node_states.unsqueeze(
                dim=1
            )
            + future_context.unsqueeze(
                dim=2
            )
            + horizon_context.view(
                1,
                self.output_steps,
                1,
                self.latent_dimension,
            )
        )

        forecast_parameters = self.forecast_decoder(
            decoder_states
        )

        forecast_normalized = forecast_parameters[
            ...,
            0,
        ]

        forecast_log_scale = forecast_parameters[
            ...,
            1,
        ].clamp(
            min=self.forecast_log_scale_minimum,
            max=self.forecast_log_scale_maximum,
        )

        graph_context_for_reconstruction = node_states.unsqueeze(
            dim=1
        ).expand(
            -1,
            input_steps,
            -1,
            -1,
        )

        reconstruction_inputs = torch.cat(
            [
                temporal_states,
                graph_context_for_reconstruction,
            ],
            dim=-1,
        )

        reconstruction_normalized = self.reconstruction_head(
            reconstruction_inputs
        ).squeeze(
            dim=-1
        )

        return {
            "forecast_normalized": forecast_normalized,
            "forecast_log_scale": forecast_log_scale,
            "reconstruction_normalized": reconstruction_normalized,
            "reliability": reliability,
            "adaptive_adjacency": adaptive_adjacency,
        }
