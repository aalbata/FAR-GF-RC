import torch
from torch import nn


class MaskedSensorSharedGRU(nn.Module):
    """
    Non-graph direct multi-horizon forecaster.

    A single GRU is shared across all sensors. Each sensor is processed
    independently through time; no graph, adjacency matrix, attention
    across sensors, or cross-sensor message passing is used.
    """

    def __init__(
        self,
        number_of_sensors,
        input_feature_dimension=7,
        time_feature_dimension=4,
        gru_hidden_dimension=64,
        sensor_embedding_dimension=8,
        head_hidden_dimension=64,
        gru_num_layers=1,
        head_dropout=0.10,
    ):
        super().__init__()

        self.number_of_sensors = int(number_of_sensors)
        self.input_feature_dimension = int(
            input_feature_dimension
        )
        self.time_feature_dimension = int(
            time_feature_dimension
        )
        self.gru_hidden_dimension = int(
            gru_hidden_dimension
        )
        self.sensor_embedding_dimension = int(
            sensor_embedding_dimension
        )

        self.encoder = nn.GRU(
            input_size=self.input_feature_dimension,
            hidden_size=self.gru_hidden_dimension,
            num_layers=int(gru_num_layers),
            batch_first=True,
        )

        self.sensor_embedding = nn.Embedding(
            num_embeddings=self.number_of_sensors,
            embedding_dim=self.sensor_embedding_dimension,
        )

        head_input_dimension = (
            self.gru_hidden_dimension
            + self.sensor_embedding_dimension
            + self.time_feature_dimension
        )

        self.forecast_head = nn.Sequential(
            nn.Linear(
                head_input_dimension,
                int(head_hidden_dimension),
            ),
            nn.ReLU(),
            nn.Dropout(float(head_dropout)),
            nn.Linear(
                int(head_hidden_dimension),
                1,
            ),
        )

    def forward(
        self,
        x_value,
        x_mask,
        x_elapsed,
        x_time,
        y_time,
    ):
        if x_value.ndim != 3:
            raise ValueError(
                "x_value must be [batch, input_steps, sensors]."
            )

        if x_mask.shape != x_value.shape:
            raise ValueError(
                "x_mask must match x_value."
            )

        if x_elapsed.shape != x_value.shape:
            raise ValueError(
                "x_elapsed must match x_value."
            )

        if x_time.ndim != 3 or y_time.ndim != 3:
            raise ValueError(
                "x_time and y_time must be three-dimensional."
            )

        batch_size, input_steps, sensor_count = x_value.shape

        if sensor_count != self.number_of_sensors:
            raise ValueError(
                "Input sensor count differs from model configuration."
            )

        if x_time.shape[:2] != (batch_size, input_steps):
            raise ValueError(
                "x_time does not align with x_value."
            )

        if x_time.shape[2] != self.time_feature_dimension:
            raise ValueError(
                "x_time feature dimension is invalid."
            )

        if y_time.shape[0] != batch_size:
            raise ValueError(
                "y_time batch dimension does not match x_value."
            )

        if y_time.shape[2] != self.time_feature_dimension:
            raise ValueError(
                "y_time feature dimension is invalid."
            )

        expanded_x_time = x_time.unsqueeze(2).expand(
            -1,
            -1,
            sensor_count,
            -1,
        )

        input_features = torch.cat(
            [
                x_value.unsqueeze(-1),
                x_mask.unsqueeze(-1).to(x_value.dtype),
                x_elapsed.unsqueeze(-1),
                expanded_x_time,
            ],
            dim=-1,
        )

        if input_features.shape[-1] != self.input_feature_dimension:
            raise RuntimeError(
                "Constructed GRU input feature dimension is invalid."
            )

        per_sensor_sequences = input_features.permute(
            0,
            2,
            1,
            3,
        ).reshape(
            batch_size * sensor_count,
            input_steps,
            self.input_feature_dimension,
        )

        _, hidden_state = self.encoder(
            per_sensor_sequences
        )

        encoded_state = hidden_state[-1].reshape(
            batch_size,
            sensor_count,
            self.gru_hidden_dimension,
        )

        sensor_ids = torch.arange(
            self.number_of_sensors,
            device=x_value.device,
            dtype=torch.long,
        )

        sensor_embedding = self.sensor_embedding(
            sensor_ids
        ).unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

        output_steps = y_time.shape[1]

        expanded_encoded_state = encoded_state.unsqueeze(1).expand(
            -1,
            output_steps,
            -1,
            -1,
        )

        expanded_sensor_embedding = sensor_embedding.unsqueeze(
            1
        ).expand(
            -1,
            output_steps,
            -1,
            -1,
        )

        expanded_y_time = y_time.unsqueeze(2).expand(
            -1,
            -1,
            sensor_count,
            -1,
        )

        head_features = torch.cat(
            [
                expanded_encoded_state,
                expanded_sensor_embedding,
                expanded_y_time,
            ],
            dim=-1,
        )

        normalized_predictions = self.forecast_head(
            head_features
        ).squeeze(-1)

        expected_shape = (
            batch_size,
            output_steps,
            sensor_count,
        )

        if normalized_predictions.shape != expected_shape:
            raise RuntimeError(
                "MS-GRU prediction shape is invalid."
            )

        return normalized_predictions


def masked_normalized_mae_loss(
    predictions,
    targets,
    target_mask,
):
    """
    Compute MAE using valid target locations only.
    """
    if predictions.shape != targets.shape:
        raise ValueError(
            "predictions and targets must have identical shapes."
        )

    if target_mask.shape != targets.shape:
        raise ValueError(
            "target_mask must match targets."
        )

    numerical_mask = target_mask.to(
        dtype=predictions.dtype
    )

    valid_count = numerical_mask.sum()

    if valid_count.item() <= 0:
        raise ValueError(
            "Masked MAE loss requires at least one valid target."
        )

    return (
        torch.abs(predictions - targets)
        .mul(numerical_mask)
        .sum()
        / valid_count
    )
