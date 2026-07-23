
"""
Canonical PEMS-BAY FAR-GF-RC training runner, revision v2.

This module implements the first formal FAR-GF-RC runner for
PEMS-BAY. It is not presented as a recovered METR-LA runner.

Revision v2 corrects the valid first target window at t = 12.
Its history is [0, ..., 11], and its causal elapsed-gap carry
uses the initialized capped pre-series state rather than a
negative Python index.

Training contract:
- PEMS-BAY protocol v2 only.
- Train and selection windows only.
- Calibration and primary-test windows are inaccessible.
- Dynamic deterministic training corruption by epoch, target start,
  model seed, and curriculum scenario.
- Fixed artificial masks for the nine-scenario selection composite.
- Checkpoint criterion: minimum fixed composite raw-speed MAE.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import DataLoader, Dataset


# -----------------------------------------------------------------
# Immutable PEMS-BAY provenance contract
# -----------------------------------------------------------------
EXPECTED_HASHES = {
    "model_source": (
        "6dd6cc74f27cffd70cfef6e4dda89d31e5db16a31d66517a2da1ba2004070d25"
    ),
    "model_config": (
        "4cbfe658252f4fc500d3343d28472649e0d1994ec8d1ec3cd5d55347e705636e"
    ),
    "training_protocol_v2": (
        "7abedbc31e21bacde366d0403728c698566d28ba95957643cdf0773fd2ddfcef"
    ),
    "data_protocol": (
        "5c466b6c481dcf5585d5fd35ceaaeb9015552aadb3a5f34b5d4a7f0384582f25"
    ),
    "native_mask": (
        "ec8917124fe063cf20b962fc5fb349dca750dfb6a8ab0cbcebf0a53728b4ea5d"
    ),
    "normalization": (
        "bd7733009bbe8cdd04d5acfc4f48ba4d0695f6de5a4a52ac6edbbd7a1a796f12"
    ),
    "temporal_split": (
        "d1e887e798a1c45bcc6bff41c757c1d7a00fcebdb07d218074a335555d6d6534"
    ),
    "physical_graph": (
        "0801aa7ccaae9ef6c8f695c8cf7d6f666cf0da67c11185faf3fe123962924de6"
    ),
    "spatial_topology": (
        "2e11ef5f1e83032318c8afa5a68e56728da7f66ce3e99cda33ffa51d410294df"
    ),
    "selection_mask": (
        "8cf8eb7cba68bf1fda53d5bf1c5c8930b2ab76ff2d0800950d447aaf251ac052"
    ),
    "raw_h5": (
        "65d69fb0a2323dba9867179eb7af47c8b814186bc459ff0a4937d21614153c8f"
    ),
}

EXPECTED_INPUT_STEPS = 12
EXPECTED_OUTPUT_STEPS = 12
EXPECTED_HISTORY_FEATURE_DIMENSION = 7
EXPECTED_FUTURE_CALENDAR_DIMENSION = 4
EXPECTED_SENSOR_COUNT = 325
EXPECTED_ELAPSED_GAP_CAP = 288
EXPECTED_MODEL_SEEDS = (
    17,
    29,
    43,
    71,
    101,
)


class PEMSContractError(RuntimeError):
    """Raised when an immutable PEMS-BAY contract is violated."""


@dataclass(frozen=True)
class RunnerPaths:
    """Resolved project artefact paths."""

    project_root: Path
    raw_h5: Path
    native_mask: Path
    normalization: Path
    temporal_split: Path
    physical_graph: Path
    spatial_topology: Path
    selection_mask: Path
    model_source: Path
    model_config: Path
    training_protocol_v2: Path
    data_protocol: Path
    checkpoint_directory: Path
    history_directory: Path


@dataclass
class RunnerAssets:
    """In-memory train and selection assets only."""

    model_config: Dict[str, Any]
    training_protocol: Dict[str, Any]
    data_protocol: Dict[str, Any]
    normalized_values: np.ndarray
    native_mask: np.ndarray
    native_elapsed_steps: np.ndarray
    calendar_features: np.ndarray
    sensor_means: np.ndarray
    sensor_stds: np.ndarray
    train_target_starts: np.ndarray
    selection_target_starts: np.ndarray
    selection_artificial_masks: np.ndarray
    selection_scenario_ids: Tuple[str, ...]
    geographic_neighbour_order: np.ndarray
    physical_graph: np.ndarray
    raw_access_end_exclusive: int
    provenance_hashes: Dict[str, str]


def sha256_file(
    file_path: Path,
) -> str:
    """Return SHA-256 for a file without loading it all at once."""

    digest = hashlib.sha256()

    with open(file_path, "rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_json(
    file_path: Path,
) -> Dict[str, Any]:
    """Load a UTF-8 JSON object."""

    with open(file_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json_write(
    output_path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Write JSON atomically."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    try:
        with open(
            temporary_path,
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
            )

        os.replace(
            temporary_path,
            output_path,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_torch_save(
    output_path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Save a PyTorch payload atomically."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    try:
        torch.save(
            dict(payload),
            temporary_path,
        )

        os.replace(
            temporary_path,
            output_path,
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def deterministic_seed(
    *tokens: Any,
) -> int:
    """
    Derive a stable 63-bit seed without Python's randomized hash().
    """

    joined_tokens = "::".join(
        str(token)
        for token in tokens
    )

    digest = hashlib.sha256(
        joined_tokens.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    ) % (2**63 - 1)


def deterministic_rng(
    *tokens: Any,
) -> np.random.Generator:
    """Create a stable NumPy random generator."""

    return np.random.default_rng(
        deterministic_seed(*tokens)
    )


def set_reproducible_seed(
    model_seed: int,
) -> None:
    """Set all random seeds for a single model run."""

    np.random.seed(model_seed)
    torch.manual_seed(model_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)


def require_deterministic_cuda(
    device: torch.device,
) -> None:
    """Require the deterministic CUDA policy frozen by Cell 17C-P2A."""

    if device.type != "cuda":
        raise PEMSContractError(
            "FAR-GF-RC PEMS-BAY training requires a CUDA device."
        )

    if not torch.cuda.is_available():
        raise PEMSContractError(
            "CUDA is unavailable."
        )

    if os.environ.get(
        "CUBLAS_WORKSPACE_CONFIG"
    ) != ":4096:8":
        raise PEMSContractError(
            "CUBLAS_WORKSPACE_CONFIG must equal ':4096:8'."
        )

    if not torch.are_deterministic_algorithms_enabled():
        raise PEMSContractError(
            "Deterministic PyTorch algorithms are not enabled."
        )

    if not torch.backends.cudnn.deterministic:
        raise PEMSContractError(
            "cuDNN deterministic mode is not enabled."
        )

    if torch.backends.cudnn.benchmark:
        raise PEMSContractError(
            "cuDNN benchmark mode must remain disabled."
        )


def resolve_runner_paths(
    project_root: Path | str,
) -> RunnerPaths:
    """Resolve all frozen PEMS-BAY artefact paths."""

    root = Path(project_root)

    return RunnerPaths(
        project_root=root,
        raw_h5=(
            root
            / "data"
            / "raw"
            / "PEMSBAY"
            / "pems-bay.h5"
        ),
        native_mask=(
            root
            / "data"
            / "processed"
            / "pems_bay_native_observation_mask_v1.npz"
        ),
        normalization=(
            root
            / "data"
            / "processed"
            / "pems_bay_train_normalization_v1.npz"
        ),
        temporal_split=(
            root
            / "data"
            / "processed"
            / "pems_bay_temporal_split_v1.npz"
        ),
        physical_graph=(
            root
            / "data"
            / "processed"
            / "pems_bay_geographic_knn4_self_tuning_gaussian_physical_graph_v1.npz"
        ),
        spatial_topology=(
            root
            / "data"
            / "processed"
            / "pems_bay_geographic_knn4_spatial_failure_topology_v1.npz"
        ),
        selection_mask=(
            root
            / "data"
            / "processed"
            / "controlled_dropout"
            / "pems_bay_v1"
            / "pems_bay_controlled_sensor_dropout_selection_composite_mask_v1.npz"
        ),
        model_source=(
            root
            / "src"
            / "models"
            / "far_gf_rc.py"
        ),
        model_config=(
            root
            / "configs"
            / "pems_bay_far_gf_rc_config_v1.json"
        ),
        training_protocol_v2=(
            root
            / "configs"
            / "pems_bay_far_gf_rc_training_protocol_v2.json"
        ),
        data_protocol=(
            root
            / "configs"
            / "pems_bay_far_gf_rc_data_protocol_v1.json"
        ),
        checkpoint_directory=(
            root
            / "outputs"
            / "checkpoints"
            / "pems_bay_far_gf_rc_protocol_v2"
        ),
        history_directory=(
            root
            / "outputs"
            / "training_history"
            / "pems_bay_far_gf_rc_protocol_v2"
        ),
    )


def verify_frozen_hashes(
    paths: RunnerPaths,
) -> Dict[str, str]:
    """Verify all source assets needed for training and selection."""

    named_paths = {
        "model_source": paths.model_source,
        "model_config": paths.model_config,
        "training_protocol_v2": paths.training_protocol_v2,
        "data_protocol": paths.data_protocol,
        "native_mask": paths.native_mask,
        "normalization": paths.normalization,
        "temporal_split": paths.temporal_split,
        "physical_graph": paths.physical_graph,
        "spatial_topology": paths.spatial_topology,
        "selection_mask": paths.selection_mask,
        "raw_h5": paths.raw_h5,
    }

    missing_paths = [
        f"{name}: {file_path}"
        for name, file_path in named_paths.items()
        if not file_path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Required frozen PEMS-BAY assets are missing:\n"
            + "\n".join(missing_paths)
        )

    observed_hashes = {
        name: sha256_file(file_path)
        for name, file_path in named_paths.items()
    }

    mismatches = [
        (
            name,
            EXPECTED_HASHES[name],
            observed_hashes[name],
        )
        for name in named_paths
        if observed_hashes[name]
        != EXPECTED_HASHES[name]
    ]

    if mismatches:
        message_lines = [
            "Frozen asset hash mismatch."
        ]

        for name, expected_hash, observed_hash in mismatches:
            message_lines.extend(
                [
                    f"Asset: {name}",
                    f"Expected: {expected_hash}",
                    f"Observed: {observed_hash}",
                ]
            )

        raise PEMSContractError(
            "\n".join(message_lines)
        )

    return observed_hashes


def build_calendar_features(
    timestamp_values_ns: np.ndarray,
) -> np.ndarray:
    """
    Create the frozen four calendar channels.

    Timestamps are interpreted as the source dataset's wall-clock
    time series. No timezone conversion is performed.
    """

    timestamps = pd.DatetimeIndex(
        pd.to_datetime(timestamp_values_ns)
    )

    minutes_of_day = (
        timestamps.hour.to_numpy(dtype=np.float32)
        * 60.0
        + timestamps.minute.to_numpy(dtype=np.float32)
    )

    minute_of_week = (
        timestamps.dayofweek.to_numpy(dtype=np.float32)
        * 1440.0
        + minutes_of_day
    )

    daily_phase = (
        2.0
        * np.pi
        * minutes_of_day
        / 1440.0
    )

    weekly_phase = (
        2.0
        * np.pi
        * minute_of_week
        / 10080.0
    )

    return np.stack(
        [
            np.sin(daily_phase),
            np.cos(daily_phase),
            np.sin(weekly_phase),
            np.cos(weekly_phase),
        ],
        axis=1,
    ).astype(
        np.float32,
        copy=False,
    )


def build_native_elapsed_steps(
    native_mask: np.ndarray,
    maximum_gap_steps: int,
) -> np.ndarray:
    """
    Build causal elapsed-gap steps over native availability only.

    For a timestamp with observed data, elapsed gap is zero.
    For a missing timestamp, it increases from the previous state
    and is capped at maximum_gap_steps.
    """

    if native_mask.ndim != 2:
        raise ValueError(
            "native_mask must have shape [timestamps, sensors]."
        )

    elapsed_steps = np.empty(
        native_mask.shape,
        dtype=np.int16,
    )

    running_gap = np.full(
        native_mask.shape[1],
        fill_value=maximum_gap_steps,
        dtype=np.int16,
    )

    for timestamp_index in range(
        native_mask.shape[0]
    ):
        observed_now = native_mask[
            timestamp_index
        ]

        running_gap[observed_now] = 0

        missing_now = ~observed_now

        running_gap[missing_now] = np.minimum(
            running_gap[missing_now] + 1,
            maximum_gap_steps,
        )

        elapsed_steps[
            timestamp_index
        ] = running_gap

    return elapsed_steps


def build_window_elapsed_feature(
    effective_history_mask: np.ndarray,
    initial_gap_steps: np.ndarray,
    maximum_gap_steps: int,
) -> np.ndarray:
    """
    Build causal elapsed gaps for one corrupted history window.

    effective_history_mask:
        [input_steps, sensors]

    initial_gap_steps:
        Native elapsed state at time t - 13, immediately before
        the first history time t - 12.
    """

    if effective_history_mask.ndim != 2:
        raise ValueError(
            "effective_history_mask must be [steps, sensors]."
        )

    if initial_gap_steps.shape != (
        effective_history_mask.shape[1],
    ):
        raise ValueError(
            "initial_gap_steps has incompatible sensor dimension."
        )

    elapsed_features = np.empty(
        effective_history_mask.shape,
        dtype=np.float32,
    )

    running_gap = np.asarray(
        initial_gap_steps,
        dtype=np.int16,
    ).copy()

    for step_index in range(
        effective_history_mask.shape[0]
    ):
        observed_now = effective_history_mask[
            step_index
        ]

        running_gap[observed_now] = 0

        missing_now = ~observed_now

        running_gap[missing_now] = np.minimum(
            running_gap[missing_now] + 1,
            maximum_gap_steps,
        )

        elapsed_features[
            step_index
        ] = (
            running_gap.astype(np.float32)
            / float(maximum_gap_steps)
        )

    return elapsed_features


def parse_scenario_parameters(
    scenario_identifier: str,
) -> Dict[str, int | float | str]:
    """Parse a frozen scenario identifier."""

    if scenario_identifier == "clean":
        return {
            "type": "clean",
        }

    iid_match = re.fullmatch(
        r"iid_random_(\d+)pct",
        scenario_identifier,
    )

    if iid_match:
        return {
            "type": "iid",
            "fraction": (
                int(iid_match.group(1))
                / 100.0
            ),
        }

    temporal_match = re.fullmatch(
        r"temporal_tail_(\d+)pct_sensors_(\d+)steps",
        scenario_identifier,
    )

    if temporal_match:
        return {
            "type": "temporal_tail",
            "sensor_fraction": (
                int(temporal_match.group(1))
                / 100.0
            ),
            "tail_steps": int(
                temporal_match.group(2)
            ),
        }

    spatial_match = re.fullmatch(
        r"spatial_geographic_knn4_cluster_(\d+)_full_history",
        scenario_identifier,
    )

    if spatial_match:
        return {
            "type": "spatial_cluster",
            "cluster_size": int(
                spatial_match.group(1)
            ),
        }

    raise PEMSContractError(
        "Unknown PEMS-BAY failure scenario:\n"
        f"{scenario_identifier}"
    )


def make_training_artificial_mask(
    native_history_mask: np.ndarray,
    scenario_identifier: str,
    geographic_neighbour_order: np.ndarray,
    model_seed: int,
    epoch_index: int,
    target_start: int,
) -> np.ndarray:
    """
    Generate one dynamic, deterministic training corruption mask.

    True means remove a native-observed historical entry.
    """

    if native_history_mask.shape != (
        EXPECTED_INPUT_STEPS,
        EXPECTED_SENSOR_COUNT,
    ):
        raise ValueError(
            "native_history_mask has incompatible shape."
        )

    scenario_parameters = parse_scenario_parameters(
        scenario_identifier
    )

    artificial_mask = np.zeros_like(
        native_history_mask,
        dtype=bool,
    )

    scenario_type = scenario_parameters[
        "type"
    ]

    if scenario_type == "clean":
        return artificial_mask

    random_generator = deterministic_rng(
        "pems_bay_training_corruption",
        model_seed,
        epoch_index,
        target_start,
        scenario_identifier,
    )

    if scenario_type == "iid":
        removal_probability = float(
            scenario_parameters["fraction"]
        )

        random_values = random_generator.random(
            native_history_mask.shape
        )

        return (
            native_history_mask
            & (
                random_values
                < removal_probability
            )
        )

    if scenario_type == "temporal_tail":
        sensor_fraction = float(
            scenario_parameters["sensor_fraction"]
        )

        tail_steps = int(
            scenario_parameters["tail_steps"]
        )

        sensor_count = int(
            np.ceil(
                sensor_fraction
                * EXPECTED_SENSOR_COUNT
            )
        )

        selected_sensors = random_generator.choice(
            EXPECTED_SENSOR_COUNT,
            size=sensor_count,
            replace=False,
        )

        artificial_mask[
            -tail_steps:,
            selected_sensors,
        ] = True

        return (
            artificial_mask
            & native_history_mask
        )

    if scenario_type == "spatial_cluster":
        cluster_size = int(
            scenario_parameters["cluster_size"]
        )

        if cluster_size > EXPECTED_SENSOR_COUNT:
            raise PEMSContractError(
                "Spatial-cluster size exceeds sensor count."
            )

        centre_sensor = int(
            random_generator.integers(
                low=0,
                high=EXPECTED_SENSOR_COUNT,
            )
        )

        selected_sensors = geographic_neighbour_order[
            centre_sensor,
            :cluster_size,
        ]

        if len(
            np.unique(selected_sensors)
        ) != cluster_size:
            raise PEMSContractError(
                "Geographic neighbour order does not provide a "
                "unique spatial cluster."
            )

        artificial_mask[
            :,
            selected_sensors,
        ] = True

        return (
            artificial_mask
            & native_history_mask
        )

    raise PEMSContractError(
        "Unhandled training scenario type:\n"
        f"{scenario_type}"
    )


class FailureAwareWindowDataset(Dataset):
    """
    Direct 12-to-12 PEMS-BAY forecasting dataset.

    It uses target starts directly:
    - history: [t - 12, ..., t - 1]
    - targets: [t, ..., t + 11]
    - elapsed-gap carry: t - 13; for t = 12, the initialized
      capped pre-series state is used
    """

    def __init__(
        self,
        normalized_values: np.ndarray,
        native_mask: np.ndarray,
        native_elapsed_steps: np.ndarray,
        calendar_features: np.ndarray,
        target_starts: np.ndarray,
        input_steps: int,
        output_steps: int,
        elapsed_gap_cap: int,
        split_name: str,
        model_seed: Optional[int] = None,
        geographic_neighbour_order: Optional[np.ndarray] = None,
        fixed_artificial_masks: Optional[np.ndarray] = None,
    ) -> None:
        self.normalized_values = np.asarray(
            normalized_values,
            dtype=np.float32,
        )

        self.native_mask = np.asarray(
            native_mask,
            dtype=bool,
        )

        self.native_elapsed_steps = np.asarray(
            native_elapsed_steps,
            dtype=np.int16,
        )

        self.calendar_features = np.asarray(
            calendar_features,
            dtype=np.float32,
        )

        self.target_starts = np.asarray(
            target_starts,
            dtype=np.int64,
        )

        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        self.elapsed_gap_cap = int(elapsed_gap_cap)
        self.split_name = str(split_name)
        self.model_seed = model_seed

        self.geographic_neighbour_order = (
            None
            if geographic_neighbour_order is None
            else np.asarray(
                geographic_neighbour_order,
                dtype=np.int64,
            )
        )

        self.fixed_artificial_masks = (
            None
            if fixed_artificial_masks is None
            else np.asarray(
                fixed_artificial_masks,
                dtype=bool,
            )
        )

        self.current_epoch = None
        self.current_scenario_names: Tuple[str, ...] = tuple()
        self.current_scenario_probabilities = np.empty(
            shape=(0,),
            dtype=np.float64,
        )

        self._validate()

    def _validate(
        self,
    ) -> None:
        expected_value_shape = self.native_mask.shape

        if self.normalized_values.shape != expected_value_shape:
            raise ValueError(
                "normalized_values and native_mask must share shape."
            )

        if self.native_elapsed_steps.shape != expected_value_shape:
            raise ValueError(
                "native_elapsed_steps and native_mask must share shape."
            )

        if self.calendar_features.shape != (
            expected_value_shape[0],
            EXPECTED_FUTURE_CALENDAR_DIMENSION,
        ):
            raise ValueError(
                "calendar_features must be [timestamps, 4]."
            )

        if self.target_starts.ndim != 1:
            raise ValueError(
                "target_starts must be one-dimensional."
            )

        if self.target_starts.size == 0:
            raise ValueError(
                "target_starts must not be empty."
            )

        if self.target_starts.min() < self.input_steps:
            raise ValueError(
                "A target start lacks the required input history."
            )

        if self.target_starts.max() + self.output_steps > (
            self.normalized_values.shape[0]
        ):
            raise ValueError(
                "A target start exceeds loaded train/selection data."
            )

        if self.split_name == "selection":
            if self.fixed_artificial_masks is None:
                raise ValueError(
                    "Selection dataset requires fixed artificial masks."
                )

            if self.fixed_artificial_masks.shape != (
                self.target_starts.size,
                self.input_steps,
                EXPECTED_SENSOR_COUNT,
            ):
                raise ValueError(
                    "Selection artificial masks have incompatible shape."
                )

        if self.split_name == "train":
            if self.model_seed is None:
                raise ValueError(
                    "Train dataset requires a model seed."
                )

            if self.geographic_neighbour_order is None:
                raise ValueError(
                    "Train dataset requires geographic neighbour order."
                )

            if self.geographic_neighbour_order.shape != (
                EXPECTED_SENSOR_COUNT,
                EXPECTED_SENSOR_COUNT,
            ):
                raise ValueError(
                    "Geographic neighbour order has incompatible shape."
                )

        if not np.isfinite(
            self.normalized_values
        ).all():
            raise ValueError(
                "normalized_values contain non-finite values."
            )

        if not np.isfinite(
            self.calendar_features
        ).all():
            raise ValueError(
                "calendar_features contain non-finite values."
            )

    def set_epoch(
        self,
        epoch_index: int,
        scenario_probabilities: Mapping[str, float],
    ) -> None:
        """Activate deterministic dynamic corruption for one epoch."""

        if self.split_name != "train":
            raise RuntimeError(
                "set_epoch is valid only for the train dataset."
            )

        if epoch_index < 1:
            raise ValueError(
                "epoch_index must start at one."
            )

        scenario_names = tuple(
            scenario_probabilities.keys()
        )

        probabilities = np.asarray(
            [
                scenario_probabilities[
                    scenario_name
                ]
                for scenario_name in scenario_names
            ],
            dtype=np.float64,
        )

        if len(scenario_names) == 0:
            raise ValueError(
                "A curriculum stage must define at least one scenario."
            )

        if np.any(probabilities < 0.0):
            raise ValueError(
                "Scenario probabilities must be non-negative."
            )

        if not np.isclose(
            probabilities.sum(),
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Scenario probabilities must sum to one."
            )

        self.current_epoch = int(epoch_index)
        self.current_scenario_names = scenario_names
        self.current_scenario_probabilities = probabilities

    def __len__(
        self,
    ) -> int:
        return int(
            self.target_starts.size
        )

    def _sample_training_scenario(
        self,
        target_start: int,
    ) -> str:
        if self.current_epoch is None:
            raise RuntimeError(
                "The train dataset has no active curriculum epoch."
            )

        random_generator = deterministic_rng(
            "pems_bay_training_scenario",
            self.model_seed,
            self.current_epoch,
            target_start,
        )

        scenario_index = int(
            random_generator.choice(
                len(self.current_scenario_names),
                p=self.current_scenario_probabilities,
            )
        )

        return self.current_scenario_names[
            scenario_index
        ]

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, torch.Tensor]:
        target_start = int(
            self.target_starts[index]
        )

        history_start = (
            target_start
            - self.input_steps
        )

        history_end = target_start

        target_end = (
            target_start
            + self.output_steps
        )

        carry_index = (
            target_start
            - self.input_steps
            - 1
        )

        native_history_mask = self.native_mask[
            history_start:history_end
        ]

        native_target_mask = self.native_mask[
            target_start:target_end
        ]

        native_history_values = self.normalized_values[
            history_start:history_end
        ]

        target_values = self.normalized_values[
            target_start:target_end
        ]

        if self.split_name == "train":
            scenario_identifier = self._sample_training_scenario(
                target_start
            )

            artificial_mask = make_training_artificial_mask(
                native_history_mask=native_history_mask,
                scenario_identifier=scenario_identifier,
                geographic_neighbour_order=(
                    self.geographic_neighbour_order
                ),
                model_seed=int(self.model_seed),
                epoch_index=int(self.current_epoch),
                target_start=target_start,
            )

        elif self.split_name == "selection":
            scenario_identifier = "frozen_selection_composite"

            artificial_mask = self.fixed_artificial_masks[
                index
            ]

        else:
            raise RuntimeError(
                "Unsupported split name:\n"
                f"{self.split_name}"
            )

        if np.any(
            artificial_mask
            & ~native_history_mask
        ):
            raise PEMSContractError(
                "Artificial corruption attempted to remove a "
                "native-unavailable history cell."
            )

        effective_history_mask = (
            native_history_mask
            & ~artificial_mask
        )

        effective_history_values = np.where(
            effective_history_mask,
            native_history_values,
            0.0,
        ).astype(
            np.float32,
            copy=False,
        )

        if carry_index >= 0:
            initial_gap_steps = self.native_elapsed_steps[
                carry_index
            ]
        else:
            # The frozen split includes target_start = 12.
            # Its history is [0, ..., 11], so no real t - 13
            # timestamp exists. This matches the global causal
            # elapsed-gap recurrence initialized at the cap before
            # the first timestamp.
            initial_gap_steps = np.full(
                shape=(
                    self.native_mask.shape[1],
                ),
                fill_value=self.elapsed_gap_cap,
                dtype=np.int16,
            )

        history_elapsed = build_window_elapsed_feature(
            effective_history_mask=effective_history_mask,
            initial_gap_steps=initial_gap_steps,
            maximum_gap_steps=self.elapsed_gap_cap,
        )

        history_calendar = self.calendar_features[
            history_start:history_end
        ]

        future_calendar = self.calendar_features[
            target_start:target_end
        ]

        return {
            "history_value": torch.from_numpy(
                np.ascontiguousarray(
                    effective_history_values
                )
            ),
            "history_mask": torch.from_numpy(
                np.ascontiguousarray(
                    effective_history_mask.astype(
                        np.float32
                    )
                )
            ),
            "history_elapsed": torch.from_numpy(
                np.ascontiguousarray(
                    history_elapsed
                )
            ),
            "history_calendar": torch.from_numpy(
                np.ascontiguousarray(
                    history_calendar
                )
            ),
            "future_calendar": torch.from_numpy(
                np.ascontiguousarray(
                    future_calendar
                )
            ),
            "target_value": torch.from_numpy(
                np.ascontiguousarray(
                    target_values
                )
            ),
            "target_mask": torch.from_numpy(
                np.ascontiguousarray(
                    native_target_mask.astype(
                        np.float32
                    )
                )
            ),
            "reconstruction_target": torch.from_numpy(
                np.ascontiguousarray(
                    native_history_values
                )
            ),
            "artificial_mask": torch.from_numpy(
                np.ascontiguousarray(
                    artificial_mask.astype(
                        np.float32
                    )
                )
            ),
            "target_start": torch.tensor(
                target_start,
                dtype=torch.int64,
            ),
        }


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute a stable masked mean."""

    denominator = mask.sum()

    if float(
        denominator.detach().cpu()
    ) <= 0.0:
        return values.new_zeros(())

    return (
        values
        * mask
    ).sum() / denominator


def assemble_history_features(
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """
    Assemble [batch, 12, sensors, 7] history channels.

    Channel order:
    0 normalized value
    1 effective observation mask
    2 normalized causal elapsed gap
    3 daily sine
    4 daily cosine
    5 weekly sine
    6 weekly cosine
    """

    history_value = batch[
        "history_value"
    ]

    history_mask = batch[
        "history_mask"
    ]

    history_elapsed = batch[
        "history_elapsed"
    ]

    history_calendar = batch[
        "history_calendar"
    ]

    sensor_count = history_value.shape[2]

    broadcast_calendar = history_calendar.unsqueeze(
        dim=2
    ).expand(
        -1,
        -1,
        sensor_count,
        -1,
    )

    history_features = torch.cat(
        [
            history_value.unsqueeze(dim=-1),
            history_mask.unsqueeze(dim=-1),
            history_elapsed.unsqueeze(dim=-1),
            broadcast_calendar,
        ],
        dim=-1,
    )

    expected_shape = (
        history_value.shape[0],
        EXPECTED_INPUT_STEPS,
        EXPECTED_SENSOR_COUNT,
        EXPECTED_HISTORY_FEATURE_DIMENSION,
    )

    if tuple(
        history_features.shape
    ) != expected_shape:
        raise PEMSContractError(
            "Assembled history-feature shape mismatch.\n"
            f"Expected: {expected_shape}\n"
            f"Observed: {tuple(history_features.shape)}"
        )

    return history_features


def compute_training_losses(
    model_output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    optimization: Mapping[str, Any],
) -> Dict[str, torch.Tensor]:
    """
    Compute the canonical frozen weighted objective.

    Forecast loss:
        masked normalized L1 over native future targets.

    Scale loss:
        masked Gaussian negative log likelihood using the model's
        predicted log-scale field.

    Reconstruction loss:
        normalized L1 only on artificially removed native history.

    Reliability loss:
        binary cross entropy between predicted reliability and the
        effective historical observation state.
    """

    forecast_prediction = model_output[
        "forecast_normalized"
    ]

    forecast_log_scale = model_output[
        "forecast_log_scale"
    ]

    reconstruction_prediction = model_output[
        "reconstruction_normalized"
    ]

    reliability_prediction = model_output[
        "reliability"
    ]

    target_value = batch[
        "target_value"
    ]

    target_mask = batch[
        "target_mask"
    ]

    reconstruction_target = batch[
        "reconstruction_target"
    ]

    artificial_mask = batch[
        "artificial_mask"
    ]

    history_mask = batch[
        "history_mask"
    ]

    forecast_error = (
        forecast_prediction
        - target_value
    )

    forecast_loss = masked_mean(
        values=torch.abs(
            forecast_error
        ),
        mask=target_mask,
    )

    scale_value = torch.exp(
        forecast_log_scale
    ).clamp_min(
        1e-6
    )

    scale_nll = 0.5 * (
        (
            forecast_error
            / scale_value
        ).square()
        + 2.0
        * forecast_log_scale
    )

    scale_nll_loss = masked_mean(
        values=scale_nll,
        mask=target_mask,
    )

    reconstruction_loss = masked_mean(
        values=torch.abs(
            reconstruction_prediction
            - reconstruction_target
        ),
        mask=artificial_mask,
    )

    reliability_prediction = reliability_prediction.clamp(
        min=1e-6,
        max=1.0 - 1e-6,
    )

    reliability_cross_entropy = functional.binary_cross_entropy(
        input=reliability_prediction,
        target=history_mask,
        reduction="none",
    )

    reliability_loss = reliability_cross_entropy.mean()

    total_loss = (
        float(
            optimization[
                "forecast_loss_weight"
            ]
        )
        * forecast_loss
        + float(
            optimization[
                "scale_nll_loss_weight"
            ]
        )
        * scale_nll_loss
        + float(
            optimization[
                "reconstruction_loss_weight"
            ]
        )
        * reconstruction_loss
        + float(
            optimization[
                "reliability_loss_weight"
            ]
        )
        * reliability_loss
    )

    return {
        "total_loss": total_loss,
        "forecast_loss": forecast_loss,
        "scale_nll_loss": scale_nll_loss,
        "reconstruction_loss": reconstruction_loss,
        "reliability_loss": reliability_loss,
    }


def move_batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Move one dataloader batch to the target device."""

    return {
        key: value.to(
            device,
            non_blocking=True,
        )
        for key, value in batch.items()
    }


def evaluate_selection_raw_mae(
    model: nn.Module,
    selection_loader: DataLoader,
    sensor_means: torch.Tensor,
    sensor_stds: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Compute global raw-speed MAE over all valid selection targets.

    This is exactly the frozen checkpoint criterion.
    """

    model.eval()

    absolute_error_sum = 0.0
    valid_target_count = 0

    with torch.no_grad():
        for cpu_batch in selection_loader:
            batch = move_batch_to_device(
                batch=cpu_batch,
                device=device,
            )

            history_features = assemble_history_features(
                batch
            )

            model_output = model(
                history_features,
                batch[
                    "future_calendar"
                ],
            )

            forecast_normalized = model_output[
                "forecast_normalized"
            ]

            prediction_raw = (
                forecast_normalized
                * sensor_stds.view(
                    1,
                    1,
                    -1,
                )
                + sensor_means.view(
                    1,
                    1,
                    -1,
                )
            )

            target_raw = (
                batch[
                    "target_value"
                ]
                * sensor_stds.view(
                    1,
                    1,
                    -1,
                )
                + sensor_means.view(
                    1,
                    1,
                    -1,
                )
            )

            target_mask = batch[
                "target_mask"
            ].bool()

            absolute_error_sum += float(
                torch.abs(
                    prediction_raw
                    - target_raw
                )[target_mask].sum().item()
            )

            valid_target_count += int(
                target_mask.sum().item()
            )

    if valid_target_count <= 0:
        raise PEMSContractError(
            "Selection evaluation has no valid future targets."
        )

    return (
        absolute_error_sum
        / float(valid_target_count)
    )


def curriculum_stage_for_epoch(
    curriculum_stages: Sequence[Mapping[str, Any]],
    epoch_index: int,
) -> Mapping[str, Any]:
    """Return the unique curriculum stage covering one epoch."""

    matching_stages = [
        stage
        for stage in curriculum_stages
        if int(stage["epoch_start"])
        <= epoch_index
        <= int(stage["epoch_end"])
    ]

    if len(matching_stages) != 1:
        raise PEMSContractError(
            "Curriculum stages do not uniquely cover epoch "
            f"{epoch_index}."
        )

    return matching_stages[0]


def load_fargfrc_class(
    model_source_path: Path,
):
    """Import FARGFRC directly from its frozen source path."""

    module_specification = importlib.util.spec_from_file_location(
        "frozen_far_gf_rc_model",
        model_source_path,
    )

    if (
        module_specification is None
        or module_specification.loader is None
    ):
        raise ImportError(
            "Unable to load FAR-GF-RC source module."
        )

    module = importlib.util.module_from_spec(
        module_specification
    )

    module_specification.loader.exec_module(
        module
    )

    if not hasattr(
        module,
        "FARGFRC",
    ):
        raise ImportError(
            "Frozen model source does not expose FARGFRC."
        )

    return module.FARGFRC


def build_model(
    model_class: type,
    model_config: Mapping[str, Any],
    physical_graph: np.ndarray,
    device: torch.device,
) -> nn.Module:
    """Build FAR-GF-RC using the frozen PEMS-BAY configuration."""

    architecture = dict(
        model_config[
            "architecture"
        ]
    )

    elapsed_gap_cap_steps = architecture.pop(
        "elapsed_gap_cap_steps",
        None,
    )

    if elapsed_gap_cap_steps != EXPECTED_ELAPSED_GAP_CAP:
        raise PEMSContractError(
            "Unexpected elapsed-gap cap in model configuration."
        )

    model = model_class(
        number_of_sensors=int(
            model_config[
                "num_sensors"
            ]
        ),
        history_feature_dimension=int(
            model_config[
                "history_feature_dimension"
            ]
        ),
        future_calendar_feature_dimension=int(
            model_config[
                "future_calendar_feature_dimension"
            ]
        ),
        input_steps=int(
            model_config[
                "input_steps"
            ]
        ),
        output_steps=int(
            model_config[
                "output_steps"
            ]
        ),
        physical_adjacency=torch.from_numpy(
            physical_graph.astype(
                np.float32,
                copy=False,
            )
        ),
        **architecture,
    )

    return model.to(device)


def _read_npz_array(
    npz_path: Path,
    key: str,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """Read one typed NPZ array."""

    with np.load(
        npz_path,
        allow_pickle=False,
    ) as bundle:
        if key not in bundle.files:
            raise KeyError(
                f"NPZ key '{key}' is absent in:\n{npz_path}"
            )

        array = np.asarray(
            bundle[key]
        )

    if dtype is not None:
        array = array.astype(
            dtype,
            copy=False,
        )

    return array


def _decode_text_array(
    values: np.ndarray,
) -> Tuple[str, ...]:
    """Decode a fixed-width NumPy text array."""

    decoded_values: List[str] = []

    for value in np.asarray(
        values
    ).reshape(-1):
        if isinstance(value, bytes):
            decoded_values.append(
                value.decode("utf-8")
            )
        else:
            decoded_values.append(
                str(value)
            )

    return tuple(decoded_values)


def load_runner_assets(
    project_root: Path | str,
) -> RunnerAssets:
    """
    Load only the data needed for train and selection.

    Raw values, masks, elapsed states, and calendar features are
    loaded only through the end of the selection target interval.
    Calibration and primary-test values are never loaded.
    """

    paths = resolve_runner_paths(
        project_root
    )

    provenance_hashes = verify_frozen_hashes(
        paths
    )

    model_config = load_json(
        paths.model_config
    )

    training_protocol = load_json(
        paths.training_protocol_v2
    )

    data_protocol = load_json(
        paths.data_protocol
    )

    if model_config.get(
        "dataset"
    ) != "PEMS-BAY":
        raise PEMSContractError(
            "Model configuration is not PEMS-BAY."
        )

    if training_protocol.get(
        "dataset"
    ) != "PEMS-BAY":
        raise PEMSContractError(
            "Training protocol is not PEMS-BAY."
        )

    if training_protocol.get(
        "protocol_version"
    ) != "v2":
        raise PEMSContractError(
            "PEMS-BAY training protocol v2 is required."
        )

    if tuple(
        training_protocol.get(
            "model_seed_values",
            [],
        )
    ) != EXPECTED_MODEL_SEEDS:
        raise PEMSContractError(
            "Unexpected PEMS-BAY model-seed plan."
        )

    optimization = training_protocol[
        "optimization"
    ]

    if int(
        optimization[
            "train_batch_size"
        ]
    ) != 32:
        raise PEMSContractError(
            "Frozen train batch size must equal 32."
        )

    if int(
        optimization[
            "selection_batch_size"
        ]
    ) != 64:
        raise PEMSContractError(
            "Frozen selection batch size must equal 64."
        )

    with np.load(
        paths.temporal_split,
        allow_pickle=False,
    ) as split_bundle:
        train_target_starts = np.asarray(
            split_bundle[
                "train_target_starts"
            ],
            dtype=np.int64,
        )

        selection_target_starts = np.asarray(
            split_bundle[
                "selection_target_starts"
            ],
            dtype=np.int64,
        )

    if train_target_starts.size != 36458:
        raise PEMSContractError(
            "Unexpected number of train target windows."
        )

    if selection_target_starts.size != 2595:
        raise PEMSContractError(
            "Unexpected number of selection target windows."
        )

    selection_end_exclusive = int(
        selection_target_starts.max()
        + EXPECTED_OUTPUT_STEPS
    )

    if selection_end_exclusive != 39087:
        raise PEMSContractError(
            "Unexpected selection data access boundary."
        )

    with h5py.File(
        paths.raw_h5,
        "r",
    ) as h5_file:
        raw_values = np.asarray(
            h5_file[
                "speed/block0_values"
            ][
                :selection_end_exclusive
            ],
            dtype=np.float32,
        )

        timestamp_values_ns = np.asarray(
            h5_file[
                "speed/axis1"
            ][
                :selection_end_exclusive
            ],
            dtype=np.int64,
        )

    if raw_values.shape != (
        selection_end_exclusive,
        EXPECTED_SENSOR_COUNT,
    ):
        raise PEMSContractError(
            "Loaded raw PEMS-BAY train/selection shape is invalid."
        )

    with np.load(
        paths.native_mask,
        allow_pickle=False,
    ) as native_mask_bundle:
        native_mask = np.asarray(
            native_mask_bundle[
                "native_observation_mask"
            ][
                :selection_end_exclusive
            ],
            dtype=bool,
        )

    with np.load(
        paths.normalization,
        allow_pickle=False,
    ) as normalization_bundle:
        sensor_means = np.asarray(
            normalization_bundle[
                "sensor_means"
            ],
            dtype=np.float32,
        )

        sensor_stds = np.asarray(
            normalization_bundle[
                "sensor_stds"
            ],
            dtype=np.float32,
        )

    if sensor_means.shape != (
        EXPECTED_SENSOR_COUNT,
    ):
        raise PEMSContractError(
            "Sensor means have incompatible shape."
        )

    if sensor_stds.shape != (
        EXPECTED_SENSOR_COUNT,
    ):
        raise PEMSContractError(
            "Sensor standard deviations have incompatible shape."
        )

    if np.any(sensor_stds <= 0.0):
        raise PEMSContractError(
            "Sensor standard deviations must be positive."
        )

    normalized_values = np.zeros_like(
        raw_values,
        dtype=np.float32,
    )

    normalized_values[
        native_mask
    ] = (
        raw_values[
            native_mask
        ]
        - np.broadcast_to(
            sensor_means,
            raw_values.shape,
        )[
            native_mask
        ]
    ) / np.broadcast_to(
        sensor_stds,
        raw_values.shape,
    )[
        native_mask
    ]

    calendar_features = build_calendar_features(
        timestamp_values_ns
    )

    native_elapsed_steps = build_native_elapsed_steps(
        native_mask=native_mask,
        maximum_gap_steps=EXPECTED_ELAPSED_GAP_CAP,
    )

    with np.load(
        paths.selection_mask,
        allow_pickle=False,
    ) as selection_bundle:
        selection_artificial_masks = np.asarray(
            selection_bundle[
                "artificial_dropout_mask"
            ],
            dtype=bool,
        )

        frozen_selection_target_starts = np.asarray(
            selection_bundle[
                "selection_target_starts"
            ],
            dtype=np.int64,
        )

        selection_scenario_ids = _decode_text_array(
            selection_bundle[
                "scenario_ids"
            ]
        )

    if not np.array_equal(
        frozen_selection_target_starts,
        selection_target_starts,
    ):
        raise PEMSContractError(
            "Selection target starts differ between split and mask assets."
        )

    if selection_artificial_masks.shape != (
        selection_target_starts.size,
        EXPECTED_INPUT_STEPS,
        EXPECTED_SENSOR_COUNT,
    ):
        raise PEMSContractError(
            "Selection artificial-mask shape is invalid."
        )

    with np.load(
        paths.spatial_topology,
        allow_pickle=False,
    ) as topology_bundle:
        geographic_neighbour_order = np.asarray(
            topology_bundle[
                "geographic_neighbour_order"
            ],
            dtype=np.int64,
        )

    if geographic_neighbour_order.shape != (
        EXPECTED_SENSOR_COUNT,
        EXPECTED_SENSOR_COUNT,
    ):
        raise PEMSContractError(
            "Geographic neighbour order has invalid shape."
        )

    with np.load(
        paths.physical_graph,
        allow_pickle=False,
    ) as graph_bundle:
        physical_graph = np.asarray(
            graph_bundle[
                "physical_graph"
            ],
            dtype=np.float32,
        )

    if physical_graph.shape != (
        EXPECTED_SENSOR_COUNT,
        EXPECTED_SENSOR_COUNT,
    ):
        raise PEMSContractError(
            "Physical graph has invalid shape."
        )

    if not np.isfinite(
        physical_graph
    ).all():
        raise PEMSContractError(
            "Physical graph contains non-finite values."
        )

    if np.any(
        physical_graph < 0.0
    ):
        raise PEMSContractError(
            "Physical graph contains negative values."
        )

    if not np.allclose(
        np.diag(physical_graph),
        0.0,
        rtol=0.0,
        atol=1e-7,
    ):
        raise PEMSContractError(
            "Physical graph must have a zero diagonal."
        )

    protocol_scenarios = tuple(
        training_protocol[
            "selection_protocol"
        ][
            "selection_scenarios"
        ]
    )

    if protocol_scenarios != selection_scenario_ids:
        raise PEMSContractError(
            "Training protocol v2 and frozen selection scenario order differ."
        )

    return RunnerAssets(
        model_config=model_config,
        training_protocol=training_protocol,
        data_protocol=data_protocol,
        normalized_values=normalized_values,
        native_mask=native_mask,
        native_elapsed_steps=native_elapsed_steps,
        calendar_features=calendar_features,
        sensor_means=sensor_means,
        sensor_stds=sensor_stds,
        train_target_starts=train_target_starts,
        selection_target_starts=selection_target_starts,
        selection_artificial_masks=selection_artificial_masks,
        selection_scenario_ids=selection_scenario_ids,
        geographic_neighbour_order=geographic_neighbour_order,
        physical_graph=physical_graph,
        raw_access_end_exclusive=selection_end_exclusive,
        provenance_hashes=provenance_hashes,
    )


def create_dataloaders(
    assets: RunnerAssets,
    model_seed: int,
) -> Tuple[
    FailureAwareWindowDataset,
    DataLoader,
]:
    """
    Create the fixed selection loader and mutable train dataset.

    A train DataLoader is created separately for each epoch because
    its shuffling generator is epoch-specific.
    """

    train_dataset = FailureAwareWindowDataset(
        normalized_values=assets.normalized_values,
        native_mask=assets.native_mask,
        native_elapsed_steps=assets.native_elapsed_steps,
        calendar_features=assets.calendar_features,
        target_starts=assets.train_target_starts,
        input_steps=EXPECTED_INPUT_STEPS,
        output_steps=EXPECTED_OUTPUT_STEPS,
        elapsed_gap_cap=EXPECTED_ELAPSED_GAP_CAP,
        split_name="train",
        model_seed=model_seed,
        geographic_neighbour_order=(
            assets.geographic_neighbour_order
        ),
    )

    selection_dataset = FailureAwareWindowDataset(
        normalized_values=assets.normalized_values,
        native_mask=assets.native_mask,
        native_elapsed_steps=assets.native_elapsed_steps,
        calendar_features=assets.calendar_features,
        target_starts=assets.selection_target_starts,
        input_steps=EXPECTED_INPUT_STEPS,
        output_steps=EXPECTED_OUTPUT_STEPS,
        elapsed_gap_cap=EXPECTED_ELAPSED_GAP_CAP,
        split_name="selection",
        fixed_artificial_masks=(
            assets.selection_artificial_masks
        ),
    )

    selection_batch_size = int(
        assets.training_protocol[
            "optimization"
        ][
            "selection_batch_size"
        ]
    )

    selection_loader = DataLoader(
        selection_dataset,
        batch_size=selection_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataset, selection_loader


def create_train_loader(
    train_dataset: FailureAwareWindowDataset,
    train_batch_size: int,
    model_seed: int,
    epoch_index: int,
) -> DataLoader:
    """Create deterministic shuffled training batches for one epoch."""

    torch_generator = torch.Generator()

    torch_generator.manual_seed(
        deterministic_seed(
            "pems_bay_train_loader",
            model_seed,
            epoch_index,
        )
    )

    return DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        generator=torch_generator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def create_immutable_output_paths(
    paths: RunnerPaths,
    model_seed: int,
) -> Tuple[Path, Path]:
    """Return checkpoint and history paths for a unique seed."""

    checkpoint_path = (
        paths.checkpoint_directory
        / (
            "pems_bay_far_gf_rc"
            "_protocol_v2"
            f"_seed_{model_seed}.pt"
        )
    )

    history_path = (
        paths.history_directory
        / (
            "pems_bay_far_gf_rc"
            "_protocol_v2"
            f"_seed_{model_seed}.json"
        )
    )

    existing_paths = [
        path
        for path in (
            checkpoint_path,
            history_path,
        )
        if path.exists()
    ]

    if existing_paths:
        raise FileExistsError(
            "Immutable output paths already exist for this seed. "
            "Refusing to overwrite:\n"
            + "\n".join(
                str(path)
                for path in existing_paths
            )
        )

    return checkpoint_path, history_path


def train_fargfrc_seed(
    project_root: Path | str,
    model_seed: int,
    device: str | torch.device = "cuda",
) -> Dict[str, Any]:
    """
    Train one PEMS-BAY FAR-GF-RC seed through 30 frozen epochs.

    This function uses:
    - train windows only for fitting;
    - fixed selection composite only for checkpoint selection;
    - no calibration access;
    - no primary-test access or evaluation.
    """

    if int(model_seed) not in EXPECTED_MODEL_SEEDS:
        raise ValueError(
            "model_seed must be one of:\n"
            f"{EXPECTED_MODEL_SEEDS}"
        )

    resolved_device = torch.device(
        device
    )

    require_deterministic_cuda(
        resolved_device
    )

    paths = resolve_runner_paths(
        project_root
    )

    checkpoint_path, history_path = (
        create_immutable_output_paths(
            paths=paths,
            model_seed=int(model_seed),
        )
    )

    set_reproducible_seed(
        int(model_seed)
    )

    assets = load_runner_assets(
        project_root=project_root
    )

    model_class = load_fargfrc_class(
        paths.model_source
    )

    model = build_model(
        model_class=model_class,
        model_config=assets.model_config,
        physical_graph=assets.physical_graph,
        device=resolved_device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            assets.training_protocol[
                "optimization"
            ][
                "learning_rate"
            ]
        ),
        weight_decay=float(
            assets.training_protocol[
                "optimization"
            ][
                "weight_decay"
            ]
        ),
    )

    optimization = assets.training_protocol[
        "optimization"
    ]

    train_batch_size = int(
        optimization[
            "train_batch_size"
        ]
    )

    maximum_epochs = int(
        optimization[
            "max_epochs"
        ]
    )

    gradient_clip_norm = float(
        optimization[
            "gradient_clip_norm"
        ]
    )

    curriculum_stages = assets.training_protocol[
        "curriculum_stages"
    ]

    train_dataset, selection_loader = create_dataloaders(
        assets=assets,
        model_seed=int(model_seed),
    )

    sensor_means = torch.from_numpy(
        assets.sensor_means
    ).to(
        resolved_device,
        dtype=torch.float32,
    )

    sensor_stds = torch.from_numpy(
        assets.sensor_stds
    ).to(
        resolved_device,
        dtype=torch.float32,
    )

    best_selection_mae = float("inf")
    best_epoch_index = None
    best_model_state = None
    epoch_history: List[Dict[str, Any]] = []

    for epoch_index in range(
        1,
        maximum_epochs + 1,
    ):
        stage = curriculum_stage_for_epoch(
            curriculum_stages=curriculum_stages,
            epoch_index=epoch_index,
        )

        scenario_probabilities = stage[
            "scenario_probabilities"
        ]

        train_dataset.set_epoch(
            epoch_index=epoch_index,
            scenario_probabilities=scenario_probabilities,
        )

        train_loader = create_train_loader(
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
            model_seed=int(model_seed),
            epoch_index=epoch_index,
        )

        model.train()

        accumulated_losses = {
            "total_loss": 0.0,
            "forecast_loss": 0.0,
            "scale_nll_loss": 0.0,
            "reconstruction_loss": 0.0,
            "reliability_loss": 0.0,
        }

        batch_count = 0

        for cpu_batch in train_loader:
            batch = move_batch_to_device(
                batch=cpu_batch,
                device=resolved_device,
            )

            history_features = assemble_history_features(
                batch
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            model_output = model(
                history_features,
                batch[
                    "future_calendar"
                ],
            )

            loss_values = compute_training_losses(
                model_output=model_output,
                batch=batch,
                optimization=optimization,
            )

            loss_values[
                "total_loss"
            ].backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )

            optimizer.step()

            for loss_name in accumulated_losses:
                accumulated_losses[
                    loss_name
                ] += float(
                    loss_values[
                        loss_name
                    ].detach().item()
                )

            batch_count += 1

        if batch_count <= 0:
            raise PEMSContractError(
                "Training loader produced no batches."
            )

        mean_losses = {
            loss_name: (
                accumulated_losses[
                    loss_name
                ]
                / float(batch_count)
            )
            for loss_name in accumulated_losses
        }

        selection_raw_mae = evaluate_selection_raw_mae(
            model=model,
            selection_loader=selection_loader,
            sensor_means=sensor_means,
            sensor_stds=sensor_stds,
            device=resolved_device,
        )

        improved = (
            selection_raw_mae
            < best_selection_mae
        )

        if improved:
            best_selection_mae = float(
                selection_raw_mae
            )

            best_epoch_index = int(
                epoch_index
            )

            best_model_state = {
                parameter_name: parameter_value.detach()
                .cpu()
                .clone()
                for parameter_name, parameter_value
                in model.state_dict().items()
            }

        epoch_history.append(
            {
                "epoch": int(epoch_index),
                "curriculum_stage": stage[
                    "name"
                ],
                "scenario_probabilities": copy.deepcopy(
                    scenario_probabilities
                ),
                "train_loss": mean_losses,
                "selection_raw_speed_mae": float(
                    selection_raw_mae
                ),
                "checkpoint_improved": bool(
                    improved
                ),
            }
        )

    if best_model_state is None:
        raise PEMSContractError(
            "No valid best model state was produced."
        )

    checkpoint_payload = {
        "run_identifier": (
            "pems_bay_far_gf_rc"
            "_protocol_v2"
            f"_seed_{model_seed}"
        ),
        "dataset": "PEMS-BAY",
        "model_seed": int(model_seed),
        "checkpoint_criterion": (
            "minimum_fixed_selection_composite"
            "_overall_raw_speed_mae"
        ),
        "best_epoch": int(
            best_epoch_index
        ),
        "best_selection_raw_speed_mae": float(
            best_selection_mae
        ),
        "model_state_dict": best_model_state,
        "model_config": assets.model_config,
        "training_protocol": assets.training_protocol,
        "provenance_hashes": assets.provenance_hashes,
        "runner_source_path": str(
            Path(__file__).resolve()
        ),
        "runner_source_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "data_access_policy": {
            "train_accessed": True,
            "selection_accessed": True,
            "calibration_accessed": False,
            "primary_test_accessed": False,
            "raw_access_end_exclusive": int(
                assets.raw_access_end_exclusive
            ),
        },
    }

    history_payload = {
        "run_identifier": checkpoint_payload[
            "run_identifier"
        ],
        "dataset": "PEMS-BAY",
        "model_seed": int(model_seed),
        "best_epoch": int(
            best_epoch_index
        ),
        "best_selection_raw_speed_mae": float(
            best_selection_mae
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
        "provenance_hashes": assets.provenance_hashes,
        "runner_source_sha256": checkpoint_payload[
            "runner_source_sha256"
        ],
        "data_access_policy": checkpoint_payload[
            "data_access_policy"
        ],
        "epochs": epoch_history,
    }

    atomic_torch_save(
        output_path=checkpoint_path,
        payload=checkpoint_payload,
    )

    atomic_json_write(
        output_path=history_path,
        payload=history_payload,
    )

    return {
        "model_seed": int(model_seed),
        "best_epoch": int(
            best_epoch_index
        ),
        "best_selection_raw_speed_mae": float(
            best_selection_mae
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
        "history_path": str(
            history_path
        ),
        "calibration_accessed": False,
        "primary_test_accessed": False,
    }
