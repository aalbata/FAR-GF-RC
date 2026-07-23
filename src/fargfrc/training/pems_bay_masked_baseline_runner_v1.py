"""
Canonical PEMS-BAY masked-baseline training runner, revision v1.

Supported model families:
- MS-GRU
- MS-TCN-v2

This runner implements a fair five-seed PEMS-BAY baseline comparison
against FAR-GF-RC under the frozen masked-baseline protocol v2.

Semantic-alignment policy:
- The frozen FAR-GF-RC runner v2 is imported dynamically as a semantic
  dependency after source-hash verification.
- Its FailureAwareWindowDataset, deterministic training-mask generator,
  causal elapsed-gap logic, fixed selection-mask handling, calendar
  construction, epoch-specific train shuffling, and deterministic CUDA
  requirements are reused without modification.
- Baseline-specific code is restricted to model construction, forward
  dispatch, forecast-only masked-L1 optimization, and output payloads.

Data-access policy:
- Train and selection windows only.
- Calibration and primary-test data are inaccessible.
- Raw traffic is loaded only through raw index 39087.
- Checkpoints are selected only by fixed selection-composite raw-speed MAE.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


# -----------------------------------------------------------------
# Immutable PEMS-BAY baseline contract
# -----------------------------------------------------------------
EXPECTED_INPUT_STEPS = 12
EXPECTED_OUTPUT_STEPS = 12
EXPECTED_HISTORY_FEATURE_DIMENSION = 7
EXPECTED_FUTURE_CALENDAR_DIMENSION = 4
EXPECTED_SENSOR_COUNT = 325
EXPECTED_ELAPSED_GAP_CAP = 288
EXPECTED_SELECTION_ACCESS_END_EXCLUSIVE = 39087

EXPECTED_MODEL_SEEDS = (
    17,
    29,
    43,
    71,
    101,
)

SUPPORTED_MODEL_IDENTIFIERS = (
    "MS-GRU",
    "MS-TCN-v2",
)

MODEL_FILE_TAGS = {
    "MS-GRU": "ms_gru",
    "MS-TCN-v2": "ms_tcn_v2",
}

EXPECTED_HASHES = {
    "baseline_protocol_v2": (
        "91d5ff35786a76f03bdfceaf7a3059f65c532c8f064baa23ed67e1369f5ed23e"
    ),
    "ms_gru_config": (
        "b5edb17e18e94228e26c88002f2dbf7de0c1e4f98f79abbf85639124c9e3b127"
    ),
    "ms_tcn_v2_config": (
        "52bf7c3277b1820089bd685a3031a179a7308bb3f2538b6617f246eab28758bd"
    ),
    "ms_gru_source": (
        "17ac07b273476fb4cf7def103e5f4e25fbcbaf488b61e0cf5da9a3456999851e"
    ),
    "ms_tcn_v2_source": (
        "1e46c2dffd2153f0babc5b267b538959f89a86bc1c3d2d8cea7f4706e6906c8b"
    ),
    "fargfrc_runner_v2_raw": (
        "f0c4934c2954e4f528c391ec03aba2c55a9956d930646003049076d29451a433"
    ),
    "fargfrc_runner_v2_text": (
        "58fbaf5a82f458ebc8e0fac85278594403306c1f6d6e1835cf55fad400316805"
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

EXPECTED_SELECTION_SCENARIOS = (
    "iid_random_10pct",
    "iid_random_30pct",
    "iid_random_50pct",
    "temporal_tail_25pct_sensors_3steps",
    "temporal_tail_50pct_sensors_6steps",
    "temporal_tail_75pct_sensors_12steps",
    "spatial_geographic_knn4_cluster_8_full_history",
    "spatial_geographic_knn4_cluster_16_full_history",
    "spatial_geographic_knn4_cluster_32_full_history",
)


class PEMSBaselineContractError(RuntimeError):
    """Raised when an immutable PEMS-BAY baseline contract is violated."""


@dataclass(frozen=True)
class BaselineRunnerPaths:
    """Resolved immutable PEMS-BAY baseline artefact paths."""

    project_root: Path
    raw_h5: Path
    native_mask: Path
    normalization: Path
    temporal_split: Path
    spatial_topology: Path
    selection_mask: Path
    baseline_protocol_v2: Path
    data_protocol: Path
    ms_gru_config: Path
    ms_tcn_v2_config: Path
    ms_gru_source: Path
    ms_tcn_v2_source: Path
    fargfrc_runner_v2_source: Path
    checkpoint_directory: Path
    history_directory: Path


@dataclass
class BaselineRunnerAssets:
    """In-memory train and selection assets only."""

    model_identifier: str
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
    raw_access_end_exclusive: int
    provenance_hashes: Dict[str, str]
    semantic_module: Any


# -----------------------------------------------------------------
# Generic immutable I/O helpers
# -----------------------------------------------------------------
def sha256_file(
    file_path: Path,
) -> str:
    digest = hashlib.sha256()

    with open(
        file_path,
        "rb",
    ) as file_handle:
        for block in iter(
            lambda: file_handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                block
            )

    return digest.hexdigest()


def sha256_text_normalized(
    file_path: Path,
) -> str:
    source_text = Path(
        file_path
    ).read_text(
        encoding="utf-8"
    )

    return hashlib.sha256(
        source_text.encode(
            "utf-8"
        )
    ).hexdigest()


def load_json(
    file_path: Path,
) -> Dict[str, Any]:
    return json.loads(
        Path(
            file_path
        ).read_text(
            encoding="utf-8"
        )
    )


def atomic_json_write(
    output_path: Path,
    payload: Mapping[str, Any],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serialized_payload = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=str(output_path.parent),
            prefix=(
                output_path.name
                + "."
            ),
            suffix=".tmp",
        ) as temporary_file:
            temporary_file.write(
                serialized_payload
            )
            temporary_path = Path(
                temporary_file.name
            )

        os.replace(
            temporary_path,
            output_path,
        )

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


def atomic_torch_save(
    output_path: Path,
    payload: Mapping[str, Any],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=str(output_path.parent),
            prefix=(
                output_path.name
                + "."
            ),
            suffix=".tmp",
        ) as temporary_file:
            temporary_path = Path(
                temporary_file.name
            )

        torch.save(
            payload,
            temporary_path,
        )

        os.replace(
            temporary_path,
            output_path,
        )

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


def _read_npz_array(
    npz_path: Path,
    key: str,
    dtype: Any = None,
) -> np.ndarray:
    with np.load(
        npz_path,
        allow_pickle=False,
    ) as bundle:
        if key not in bundle.files:
            raise KeyError(
                f"NPZ key '{key}' is absent in:\n{npz_path}"
            )

        values = np.asarray(
            bundle[key]
        )

    if dtype is not None:
        values = values.astype(
            dtype,
            copy=False,
        )

    return values


def _decode_text_array(
    values: np.ndarray,
) -> Tuple[str, ...]:
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


# -----------------------------------------------------------------
# Path resolution and frozen-hash validation
# -----------------------------------------------------------------
def resolve_runner_paths(
    project_root: Path | str,
) -> BaselineRunnerPaths:
    """Resolve immutable PEMS-BAY baseline paths."""

    root = Path(
        project_root
    )

    return BaselineRunnerPaths(
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
            / (
                "pems_bay_controlled_sensor_dropout_"
                "selection_composite_mask_v1.npz"
            )
        ),
        baseline_protocol_v2=(
            root
            / "configs"
            / "pems_bay_masked_baseline_training_protocol_v2.json"
        ),
        data_protocol=(
            root
            / "configs"
            / "pems_bay_far_gf_rc_data_protocol_v1.json"
        ),
        ms_gru_config=(
            root
            / "configs"
            / "pems_bay_masked_sensor_shared_gru_config_v1.json"
        ),
        ms_tcn_v2_config=(
            root
            / "configs"
            / "pems_bay_masked_sensor_shared_tcn_v2_config_v1.json"
        ),
        ms_gru_source=(
            root
            / "src"
            / "models"
            / "masked_sensor_shared_gru.py"
        ),
        ms_tcn_v2_source=(
            root
            / "src"
            / "models"
            / "masked_sensor_shared_tcn_v2.py"
        ),
        fargfrc_runner_v2_source=(
            root
            / "src"
            / "training"
            / "pems_bay_far_gf_rc_runner_v2.py"
        ),
        checkpoint_directory=(
            root
            / "outputs"
            / "checkpoints"
            / "pems_bay_masked_baselines_protocol_v1"
        ),
        history_directory=(
            root
            / "outputs"
            / "training_history"
            / "pems_bay_masked_baselines_protocol_v1"
        ),
    )


def verify_frozen_hashes(
    paths: BaselineRunnerPaths,
    model_identifier: str,
) -> Dict[str, str]:
    """Verify all immutable assets required by one baseline model."""

    if model_identifier not in SUPPORTED_MODEL_IDENTIFIERS:
        raise ValueError(
            "Unsupported baseline model identifier:\n"
            f"{model_identifier}"
        )

    if model_identifier == "MS-GRU":
        selected_model_source = paths.ms_gru_source
        selected_model_config = paths.ms_gru_config
        selected_source_key = "ms_gru_source"
        selected_config_key = "ms_gru_config"

    else:
        selected_model_source = paths.ms_tcn_v2_source
        selected_model_config = paths.ms_tcn_v2_config
        selected_source_key = "ms_tcn_v2_source"
        selected_config_key = "ms_tcn_v2_config"

    named_paths = {
        "baseline_protocol_v2": paths.baseline_protocol_v2,
        "data_protocol": paths.data_protocol,
        "native_mask": paths.native_mask,
        "normalization": paths.normalization,
        "temporal_split": paths.temporal_split,
        "spatial_topology": paths.spatial_topology,
        "selection_mask": paths.selection_mask,
        "raw_h5": paths.raw_h5,
        "fargfrc_runner_v2_raw": (
            paths.fargfrc_runner_v2_source
        ),
        selected_source_key: selected_model_source,
        selected_config_key: selected_model_config,
    }

    missing_paths = [
        f"{logical_name}: {file_path}"
        for logical_name, file_path
        in named_paths.items()
        if not file_path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Required frozen PEMS-BAY baseline assets are missing:\n"
            + "\n".join(
                missing_paths
            )
        )

    observed_hashes = {
        logical_name: sha256_file(
            file_path
        )
        for logical_name, file_path
        in named_paths.items()
    }

    fargfrc_text_hash = sha256_text_normalized(
        paths.fargfrc_runner_v2_source
    )

    observed_hashes[
        "fargfrc_runner_v2_text"
    ] = fargfrc_text_hash

    expected_keys = list(
        named_paths.keys()
    ) + [
        "fargfrc_runner_v2_text",
    ]

    mismatches = [
        (
            logical_name,
            EXPECTED_HASHES[
                logical_name
            ],
            observed_hashes[
                logical_name
            ],
        )
        for logical_name in expected_keys
        if observed_hashes[
            logical_name
        ]
        != EXPECTED_HASHES[
            logical_name
        ]
    ]

    if mismatches:
        message_lines = [
            "Frozen baseline asset hash mismatch."
        ]

        for logical_name, expected_hash, observed_hash in mismatches:
            message_lines.extend(
                [
                    f"Asset: {logical_name}",
                    f"Expected: {expected_hash}",
                    f"Observed: {observed_hash}",
                ]
            )

        raise PEMSBaselineContractError(
            "\n".join(
                message_lines
            )
        )

    return observed_hashes


# -----------------------------------------------------------------
# Dynamic semantic dependency on frozen FAR-GF-RC runner v2
# -----------------------------------------------------------------
def load_fargfrc_semantic_module(
    fargfrc_runner_v2_source: Path,
) -> Any:
    """
    Import the frozen FAR-GF-RC runner-v2 semantic module.

    The module import alone does not load traffic data, train models,
    or access calibration or primary-test values.
    """

    module_name = (
        "_pems_bay_fargfrc_semantics_v2"
    )

    sys.modules.pop(
        module_name,
        None,
    )

    module_specification = importlib.util.spec_from_file_location(
        module_name,
        fargfrc_runner_v2_source,
    )

    if (
        module_specification is None
        or module_specification.loader is None
    ):
        raise ImportError(
            "Unable to build FAR-GF-RC semantic-module import."
        )

    semantic_module = importlib.util.module_from_spec(
        module_specification
    )

    sys.modules[
        module_name
    ] = semantic_module

    try:
        module_specification.loader.exec_module(
            semantic_module
        )

    except Exception:
        sys.modules.pop(
            module_name,
            None,
        )
        raise

    required_symbols = (
        "FailureAwareWindowDataset",
        "build_calendar_features",
        "build_native_elapsed_steps",
        "assemble_history_features",
        "masked_mean",
        "create_train_loader",
        "curriculum_stage_for_epoch",
        "set_reproducible_seed",
        "require_deterministic_cuda",
    )

    missing_symbols = [
        symbol_name
        for symbol_name in required_symbols
        if not hasattr(
            semantic_module,
            symbol_name,
        )
    ]

    if missing_symbols:
        raise PEMSBaselineContractError(
            "Frozen FAR-GF-RC semantic module lacks required "
            "symbols:\n"
            + "\n".join(
                missing_symbols
            )
        )

    return semantic_module


# -----------------------------------------------------------------
# Model loading and construction
# -----------------------------------------------------------------
def load_baseline_model_class(
    model_source_path: Path,
    model_identifier: str,
) -> type[nn.Module]:
    """Dynamically load one frozen baseline model source."""

    class_name = {
        "MS-GRU": "MaskedSensorSharedGRU",
        "MS-TCN-v2": "MaskedSensorSharedTCNV2",
    }[
        model_identifier
    ]

    module_name = (
        "_pems_bay_baseline_model_"
        + MODEL_FILE_TAGS[
            model_identifier
        ]
    )

    sys.modules.pop(
        module_name,
        None,
    )

    module_specification = importlib.util.spec_from_file_location(
        module_name,
        model_source_path,
    )

    if (
        module_specification is None
        or module_specification.loader is None
    ):
        raise ImportError(
            "Unable to build baseline-model import specification."
        )

    model_module = importlib.util.module_from_spec(
        module_specification
    )

    sys.modules[
        module_name
    ] = model_module

    try:
        module_specification.loader.exec_module(
            model_module
        )

    except Exception:
        sys.modules.pop(
            module_name,
            None,
        )
        raise

    if not hasattr(
        model_module,
        class_name,
    ):
        raise AttributeError(
            "Frozen baseline source lacks expected class:\n"
            f"{class_name}"
        )

    model_class = getattr(
        model_module,
        class_name,
    )

    if not issubclass(
        model_class,
        nn.Module,
    ):
        raise TypeError(
            "Loaded baseline class is not a torch.nn.Module."
        )

    return model_class


def build_baseline_model(
    model_class: type[nn.Module],
    model_identifier: str,
    model_config: Mapping[str, Any],
    device: torch.device,
) -> nn.Module:
    """Construct one frozen baseline architecture."""

    architecture = model_config[
        "architecture"
    ]

    if model_identifier == "MS-GRU":
        model = model_class(
            number_of_sensors=int(
                architecture[
                    "number_of_sensors"
                ]
            ),
            input_feature_dimension=int(
                architecture[
                    "input_feature_dimension"
                ]
            ),
            time_feature_dimension=int(
                architecture[
                    "time_feature_dimension"
                ]
            ),
            gru_hidden_dimension=int(
                architecture[
                    "gru_hidden_dimension"
                ]
            ),
            sensor_embedding_dimension=int(
                architecture[
                    "sensor_embedding_dimension"
                ]
            ),
            head_hidden_dimension=int(
                architecture[
                    "head_hidden_dimension"
                ]
            ),
            gru_num_layers=int(
                architecture[
                    "gru_num_layers"
                ]
            ),
            head_dropout=float(
                architecture[
                    "head_dropout"
                ]
            ),
        )

    elif model_identifier == "MS-TCN-v2":
        model = model_class(
            num_sensors=int(
                architecture[
                    "num_sensors"
                ]
            ),
            history_feature_dim=int(
                architecture[
                    "history_feature_dim"
                ]
            ),
            future_calendar_feature_dim=int(
                architecture[
                    "future_calendar_feature_dim"
                ]
            ),
            channels=int(
                architecture[
                    "channels"
                ]
            ),
            kernel_size=int(
                architecture[
                    "kernel_size"
                ]
            ),
            dilations=tuple(
                int(dilation)
                for dilation in architecture[
                    "dilations"
                ]
            ),
            dropout=float(
                architecture[
                    "dropout"
                ]
            ),
            sensor_embedding_dim=int(
                architecture[
                    "sensor_embedding_dim"
                ]
            ),
            prediction_head_dim=int(
                architecture[
                    "prediction_head_dim"
                ]
            ),
        )

    else:
        raise ValueError(
            "Unsupported model identifier:\n"
            f"{model_identifier}"
        )

    model = model.to(
        device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    if parameter_count <= 0:
        raise PEMSBaselineContractError(
            "Baseline model has no trainable parameters."
        )

    return model


# -----------------------------------------------------------------
# Asset loading: train and selection only
# -----------------------------------------------------------------
def load_runner_assets(
    project_root: Path | str,
    model_identifier: str,
) -> BaselineRunnerAssets:
    """
    Load only train and selection data required by one baseline.

    Calibration and primary-test values are never loaded.
    """

    if model_identifier not in SUPPORTED_MODEL_IDENTIFIERS:
        raise ValueError(
            "Unsupported model identifier:\n"
            f"{model_identifier}"
        )

    paths = resolve_runner_paths(
        project_root
    )

    provenance_hashes = verify_frozen_hashes(
        paths=paths,
        model_identifier=model_identifier,
    )

    semantic_module = load_fargfrc_semantic_module(
        paths.fargfrc_runner_v2_source
    )

    training_protocol = load_json(
        paths.baseline_protocol_v2
    )

    data_protocol = load_json(
        paths.data_protocol
    )

    if model_identifier == "MS-GRU":
        model_config_path = paths.ms_gru_config
        model_source_path = paths.ms_gru_source
        expected_config_hash_key = "ms_gru_config"
        expected_source_hash_key = "ms_gru_source"

    else:
        model_config_path = paths.ms_tcn_v2_config
        model_source_path = paths.ms_tcn_v2_source
        expected_config_hash_key = "ms_tcn_v2_config"
        expected_source_hash_key = "ms_tcn_v2_source"

    model_config = load_json(
        model_config_path
    )

    if model_config.get(
        "dataset"
    ) != "PEMS-BAY":
        raise PEMSBaselineContractError(
            "Baseline model configuration is not PEMS-BAY."
        )

    if model_config.get(
        "model_identifier"
    ) != model_identifier:
        raise PEMSBaselineContractError(
            "Baseline model identifier disagrees with configuration."
        )

    if training_protocol.get(
        "dataset"
    ) != "PEMS-BAY":
        raise PEMSBaselineContractError(
            "Baseline training protocol is not PEMS-BAY."
        )

    if training_protocol.get(
        "protocol_version"
    ) != "v2":
        raise PEMSBaselineContractError(
            "Executable baseline protocol v2 is required."
        )

    if tuple(
        training_protocol.get(
            "model_seed_values",
            [],
        )
    ) != EXPECTED_MODEL_SEEDS:
        raise PEMSBaselineContractError(
            "Unexpected baseline model-seed plan."
        )

    if model_identifier not in tuple(
        training_protocol.get(
            "model_families",
            [],
        )
    ):
        raise PEMSBaselineContractError(
            "Selected baseline model is not frozen in the protocol."
        )

    model_configuration_contract = training_protocol.get(
        "model_configuration_contract",
        {},
    )

    model_contract_key = {
        "MS-GRU": "ms_gru",
        "MS-TCN-v2": "ms_tcn_v2",
    }[
        model_identifier
    ]

    expected_config_hash = model_configuration_contract[
        model_contract_key
    ][
        "sha256"
    ]

    if expected_config_hash != EXPECTED_HASHES[
        expected_config_hash_key
    ]:
        raise PEMSBaselineContractError(
            "Baseline protocol configuration hash is inconsistent."
        )

    if provenance_hashes[
        expected_config_hash_key
    ] != expected_config_hash:
        raise PEMSBaselineContractError(
            "Observed baseline configuration hash is inconsistent."
        )

    expected_source_hash = model_config[
        "model_source"
    ][
        "sha256"
    ]

    if expected_source_hash != EXPECTED_HASHES[
        expected_source_hash_key
    ]:
        raise PEMSBaselineContractError(
            "Baseline configuration model-source hash is inconsistent."
        )

    if provenance_hashes[
        expected_source_hash_key
    ] != expected_source_hash:
        raise PEMSBaselineContractError(
            "Observed baseline model-source hash is inconsistent."
        )

    if int(
        model_config[
            "architecture"
        ].get(
            "number_of_sensors",
            model_config[
                "architecture"
            ].get(
                "num_sensors",
                -1,
            ),
        )
    ) != EXPECTED_SENSOR_COUNT:
        raise PEMSBaselineContractError(
            "Baseline configuration has an invalid sensor count."
        )

    optimization = training_protocol[
        "optimization"
    ]

    if int(
        optimization[
            "train_batch_size"
        ]
    ) != 32:
        raise PEMSBaselineContractError(
            "Frozen train batch size must equal 32."
        )

    if int(
        optimization[
            "selection_batch_size"
        ]
    ) != 64:
        raise PEMSBaselineContractError(
            "Frozen selection batch size must equal 64."
        )

    if int(
        optimization[
            "max_epochs"
        ]
    ) != 30:
        raise PEMSBaselineContractError(
            "Frozen baseline epoch budget must equal 30."
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
        raise PEMSBaselineContractError(
            "Unexpected number of train target windows."
        )

    if selection_target_starts.size != 2595:
        raise PEMSBaselineContractError(
            "Unexpected number of selection target windows."
        )

    selection_end_exclusive = int(
        selection_target_starts.max()
        + EXPECTED_OUTPUT_STEPS
    )

    if selection_end_exclusive != (
        EXPECTED_SELECTION_ACCESS_END_EXCLUSIVE
    ):
        raise PEMSBaselineContractError(
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
        raise PEMSBaselineContractError(
            "Loaded raw PEMS-BAY train/selection shape is invalid."
        )

    native_mask = _read_npz_array(
        npz_path=paths.native_mask,
        key="native_observation_mask",
        dtype=bool,
    )[ 
        :selection_end_exclusive
    ]

    if native_mask.shape != raw_values.shape:
        raise PEMSBaselineContractError(
            "Native observation-mask shape is incompatible."
        )

    sensor_means = _read_npz_array(
        npz_path=paths.normalization,
        key="sensor_means",
        dtype=np.float32,
    )

    sensor_stds = _read_npz_array(
        npz_path=paths.normalization,
        key="sensor_stds",
        dtype=np.float32,
    )

    if sensor_means.shape != (
        EXPECTED_SENSOR_COUNT,
    ):
        raise PEMSBaselineContractError(
            "Sensor means have incompatible shape."
        )

    if sensor_stds.shape != (
        EXPECTED_SENSOR_COUNT,
    ):
        raise PEMSBaselineContractError(
            "Sensor standard deviations have incompatible shape."
        )

    if np.any(
        sensor_stds <= 0.0
    ):
        raise PEMSBaselineContractError(
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

    calendar_features = semantic_module.build_calendar_features(
        timestamp_values_ns
    )

    native_elapsed_steps = semantic_module.build_native_elapsed_steps(
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
        raise PEMSBaselineContractError(
            "Selection target starts differ between split and mask assets."
        )

    if selection_artificial_masks.shape != (
        selection_target_starts.size,
        EXPECTED_INPUT_STEPS,
        EXPECTED_SENSOR_COUNT,
    ):
        raise PEMSBaselineContractError(
            "Selection artificial-mask shape is invalid."
        )

    if selection_scenario_ids != EXPECTED_SELECTION_SCENARIOS:
        raise PEMSBaselineContractError(
            "Frozen selection-scenario order is invalid."
        )

    geographic_neighbour_order = _read_npz_array(
        npz_path=paths.spatial_topology,
        key="geographic_neighbour_order",
        dtype=np.int64,
    )

    if geographic_neighbour_order.shape != (
        EXPECTED_SENSOR_COUNT,
        EXPECTED_SENSOR_COUNT,
    ):
        raise PEMSBaselineContractError(
            "Geographic neighbour order has invalid shape."
        )

    return BaselineRunnerAssets(
        model_identifier=model_identifier,
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
        raw_access_end_exclusive=selection_end_exclusive,
        provenance_hashes=provenance_hashes,
        semantic_module=semantic_module,
    )


# -----------------------------------------------------------------
# Dataset, model-forward, loss, and selection evaluation
# -----------------------------------------------------------------
def create_dataloaders(
    assets: BaselineRunnerAssets,
    model_seed: int,
) -> Tuple[Any, DataLoader]:
    """Create exact FAR-GF-RC-semantic train and selection datasets."""

    semantic_module = assets.semantic_module

    train_dataset = semantic_module.FailureAwareWindowDataset(
        normalized_values=assets.normalized_values,
        native_mask=assets.native_mask,
        native_elapsed_steps=assets.native_elapsed_steps,
        calendar_features=assets.calendar_features,
        target_starts=assets.train_target_starts,
        input_steps=EXPECTED_INPUT_STEPS,
        output_steps=EXPECTED_OUTPUT_STEPS,
        elapsed_gap_cap=EXPECTED_ELAPSED_GAP_CAP,
        split_name="train",
        model_seed=int(model_seed),
        geographic_neighbour_order=(
            assets.geographic_neighbour_order
        ),
    )

    selection_dataset = semantic_module.FailureAwareWindowDataset(
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

    selection_loader = DataLoader(
        selection_dataset,
        batch_size=int(
            assets.training_protocol[
                "optimization"
            ][
                "selection_batch_size"
            ]
        ),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataset, selection_loader


def move_batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Move one batch to the selected CUDA device."""

    return {
        key: value.to(
            device,
            non_blocking=True,
        )
        for key, value in batch.items()
    }


def forward_baseline(
    model: nn.Module,
    model_identifier: str,
    batch: Mapping[str, torch.Tensor],
    semantic_module: Any,
) -> torch.Tensor:
    """Run the architecture-specific baseline forward pass."""

    if model_identifier == "MS-GRU":
        forecast_normalized = model(
            batch[
                "history_value"
            ],
            batch[
                "history_mask"
            ],
            batch[
                "history_elapsed"
            ],
            batch[
                "history_calendar"
            ],
            batch[
                "future_calendar"
            ],
        )

    elif model_identifier == "MS-TCN-v2":
        history_features = semantic_module.assemble_history_features(
            batch
        )

        forecast_normalized = model(
            history_features,
            batch[
                "future_calendar"
            ],
        )

    else:
        raise ValueError(
            "Unsupported model identifier:\n"
            f"{model_identifier}"
        )

    expected_shape = tuple(
        batch[
            "target_value"
        ].shape
    )

    if tuple(
        forecast_normalized.shape
    ) != expected_shape:
        raise PEMSBaselineContractError(
            "Baseline forecast shape mismatch.\n"
            f"Expected: {expected_shape}\n"
            f"Observed: {tuple(forecast_normalized.shape)}"
        )

    if not torch.isfinite(
        forecast_normalized
    ).all():
        raise PEMSBaselineContractError(
            "Baseline forecast contains non-finite values."
        )

    return forecast_normalized


def compute_training_losses(
    forecast_normalized: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    semantic_module: Any,
) -> Dict[str, torch.Tensor]:
    """
    Compute the frozen baseline objective.

    Both MS-GRU and MS-TCN-v2 use forecast-only masked normalized L1
    over native future observed targets.
    """

    forecast_loss = semantic_module.masked_mean(
        values=torch.abs(
            forecast_normalized
            - batch[
                "target_value"
            ]
        ),
        mask=batch[
            "target_mask"
        ],
    )

    if not torch.isfinite(
        forecast_loss
    ):
        raise PEMSBaselineContractError(
            "Baseline forecast loss is non-finite."
        )

    return {
        "total_loss": forecast_loss,
        "forecast_loss": forecast_loss,
    }


def evaluate_selection_raw_mae(
    model: nn.Module,
    model_identifier: str,
    selection_loader: DataLoader,
    sensor_means: torch.Tensor,
    sensor_stds: torch.Tensor,
    device: torch.device,
    semantic_module: Any,
) -> float:
    """Compute global fixed-selection raw-speed MAE."""

    model.eval()

    absolute_error_sum = 0.0
    valid_target_count = 0

    with torch.no_grad():
        for cpu_batch in selection_loader:
            batch = move_batch_to_device(
                batch=cpu_batch,
                device=device,
            )

            forecast_normalized = forward_baseline(
                model=model,
                model_identifier=model_identifier,
                batch=batch,
                semantic_module=semantic_module,
            )

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
                )[
                    target_mask
                ].sum().item()
            )

            valid_target_count += int(
                target_mask.sum().item()
            )

    if valid_target_count <= 0:
        raise PEMSBaselineContractError(
            "Selection evaluation has no valid targets."
        )

    return (
        absolute_error_sum
        / float(
            valid_target_count
        )
    )


# -----------------------------------------------------------------
# Immutable output-path and model-state helpers
# -----------------------------------------------------------------
def create_immutable_output_paths(
    paths: BaselineRunnerPaths,
    model_identifier: str,
    model_seed: int,
) -> Tuple[Path, Path]:
    """Return unique immutable checkpoint and history paths."""

    model_tag = MODEL_FILE_TAGS[
        model_identifier
    ]

    checkpoint_path = (
        paths.checkpoint_directory
        / (
            "pems_bay_"
            + model_tag
            + "_protocol_v2"
            + f"_seed_{model_seed}.pt"
        )
    )

    history_path = (
        paths.history_directory
        / (
            "pems_bay_"
            + model_tag
            + "_protocol_v2"
            + f"_seed_{model_seed}.json"
        )
    )

    existing_paths = [
        output_path
        for output_path in (
            checkpoint_path,
            history_path,
        )
        if output_path.exists()
    ]

    if existing_paths:
        raise FileExistsError(
            "Immutable baseline output paths already exist. "
            "Refusing to overwrite:\n"
            + "\n".join(
                str(output_path)
                for output_path in existing_paths
            )
        )

    return checkpoint_path, history_path


def clone_state_to_cpu(
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """Clone a finite model state onto CPU for deterministic saving."""

    cloned_state: Dict[str, torch.Tensor] = {}

    for tensor_name, tensor_value in model.state_dict().items():
        if torch.is_floating_point(
            tensor_value
        ) and not torch.isfinite(
            tensor_value
        ).all():
            raise PEMSBaselineContractError(
                "Model state contains non-finite values:\n"
                f"{tensor_name}"
            )

        cloned_state[
            tensor_name
        ] = tensor_value.detach().cpu().clone()

    return cloned_state


# -----------------------------------------------------------------
# Full one-seed training entry point
# -----------------------------------------------------------------
def train_baseline_seed(
    project_root: Path | str,
    model_identifier: str,
    model_seed: int,
    device: str | torch.device = "cuda",
) -> Dict[str, Any]:
    """
    Train one immutable PEMS-BAY baseline seed.

    Data policy:
    - train windows only for optimization;
    - fixed selection composite only for checkpoint selection;
    - calibration inaccessible;
    - primary-test inaccessible.
    """

    if model_identifier not in SUPPORTED_MODEL_IDENTIFIERS:
        raise ValueError(
            "model_identifier must be one of:\n"
            f"{SUPPORTED_MODEL_IDENTIFIERS}"
        )

    if int(
        model_seed
    ) not in EXPECTED_MODEL_SEEDS:
        raise ValueError(
            "model_seed must be one of:\n"
            f"{EXPECTED_MODEL_SEEDS}"
        )

    resolved_device = torch.device(
        device
    )

    paths = resolve_runner_paths(
        project_root
    )

    provenance_hashes = verify_frozen_hashes(
        paths=paths,
        model_identifier=model_identifier,
    )

    semantic_module = load_fargfrc_semantic_module(
        paths.fargfrc_runner_v2_source
    )

    semantic_module.require_deterministic_cuda(
        resolved_device
    )

    checkpoint_path, history_path = create_immutable_output_paths(
        paths=paths,
        model_identifier=model_identifier,
        model_seed=int(model_seed),
    )

    semantic_module.set_reproducible_seed(
        int(model_seed)
    )

    assets = load_runner_assets(
        project_root=project_root,
        model_identifier=model_identifier,
    )

    if assets.provenance_hashes != provenance_hashes:
        raise PEMSBaselineContractError(
            "Independent baseline provenance checks disagree."
        )

    if model_identifier == "MS-GRU":
        model_source_path = paths.ms_gru_source

    else:
        model_source_path = paths.ms_tcn_v2_source

    model_class = load_baseline_model_class(
        model_source_path=model_source_path,
        model_identifier=model_identifier,
    )

    model = build_baseline_model(
        model_class=model_class,
        model_identifier=model_identifier,
        model_config=assets.model_config,
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

    best_selection_mae = float(
        "inf"
    )

    best_epoch_index = None
    best_model_state = None
    epoch_history: List[Dict[str, Any]] = []

    for epoch_index in range(
        1,
        maximum_epochs + 1,
    ):
        stage = semantic_module.curriculum_stage_for_epoch(
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

        train_loader = semantic_module.create_train_loader(
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
            model_seed=int(model_seed),
            epoch_index=epoch_index,
        )

        model.train()

        accumulated_total_loss = 0.0
        accumulated_forecast_loss = 0.0
        train_batch_count = 0

        for cpu_batch in train_loader:
            batch = move_batch_to_device(
                batch=cpu_batch,
                device=resolved_device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            forecast_normalized = forward_baseline(
                model=model,
                model_identifier=model_identifier,
                batch=batch,
                semantic_module=assets.semantic_module,
            )

            losses = compute_training_losses(
                forecast_normalized=forecast_normalized,
                batch=batch,
                semantic_module=assets.semantic_module,
            )

            losses[
                "total_loss"
            ].backward()

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )

            if not math.isfinite(
                float(
                    gradient_norm.detach().cpu()
                )
            ):
                raise PEMSBaselineContractError(
                    "Non-finite baseline gradient norm."
                )

            optimizer.step()

            accumulated_total_loss += float(
                losses[
                    "total_loss"
                ].detach().cpu()
            )

            accumulated_forecast_loss += float(
                losses[
                    "forecast_loss"
                ].detach().cpu()
            )

            train_batch_count += 1

        if train_batch_count <= 0:
            raise PEMSBaselineContractError(
                "Training epoch has no batches."
            )

        selection_raw_mae = evaluate_selection_raw_mae(
            model=model,
            model_identifier=model_identifier,
            selection_loader=selection_loader,
            sensor_means=sensor_means,
            sensor_stds=sensor_stds,
            device=resolved_device,
            semantic_module=assets.semantic_module,
        )

        if not math.isfinite(
            selection_raw_mae
        ):
            raise PEMSBaselineContractError(
                "Selection raw-speed MAE is non-finite."
            )

        if selection_raw_mae < best_selection_mae:
            best_selection_mae = float(
                selection_raw_mae
            )

            best_epoch_index = int(
                epoch_index
            )

            best_model_state = clone_state_to_cpu(
                model
            )

        epoch_history.append(
            {
                "epoch": int(
                    epoch_index
                ),
                "curriculum_epoch_start": int(
                    stage[
                        "epoch_start"
                    ]
                ),
                "curriculum_epoch_end": int(
                    stage[
                        "epoch_end"
                    ]
                ),
                "scenario_probabilities": {
                    scenario_name: float(
                        probability
                    )
                    for scenario_name, probability
                    in scenario_probabilities.items()
                },
                "train_loss": {
                    "total_loss": (
                        accumulated_total_loss
                        / float(
                            train_batch_count
                        )
                    ),
                    "forecast_loss": (
                        accumulated_forecast_loss
                        / float(
                            train_batch_count
                        )
                    ),
                },
                "selection_raw_speed_mae": float(
                    selection_raw_mae
                ),
            }
        )

    if best_epoch_index is None or best_model_state is None:
        raise PEMSBaselineContractError(
            "No baseline checkpoint was selected."
        )

    selection_mae_values = [
        float(
            epoch_record[
                "selection_raw_speed_mae"
            ]
        )
        for epoch_record in epoch_history
    ]

    minimum_selection_mae = min(
        selection_mae_values
    )

    earliest_minimum_epoch = (
        selection_mae_values.index(
            minimum_selection_mae
        )
        + 1
    )

    if best_epoch_index != earliest_minimum_epoch:
        raise PEMSBaselineContractError(
            "Checkpoint is not the earliest minimum-selection epoch."
        )

    if not math.isclose(
        best_selection_mae,
        minimum_selection_mae,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PEMSBaselineContractError(
            "Stored best selection MAE disagrees with epoch history."
        )

    baseline_runner_source = Path(
        __file__
    ).resolve()

    baseline_runner_source_sha256 = sha256_file(
        baseline_runner_source
    )

    data_access_policy = {
        "train_accessed": True,
        "selection_accessed": True,
        "calibration_accessed": False,
        "primary_test_accessed": False,
        "raw_access_end_exclusive": int(
            assets.raw_access_end_exclusive
        ),
    }

    checkpoint_payload = {
        "dataset": "PEMS-BAY",
        "model_identifier": model_identifier,
        "model_seed": int(
            model_seed
        ),
        "model_config": assets.model_config,
        "training_protocol": assets.training_protocol,
        "model_config_sha256": (
            provenance_hashes[
                {
                    "MS-GRU": "ms_gru_config",
                    "MS-TCN-v2": "ms_tcn_v2_config",
                }[
                    model_identifier
                ]
            ]
        ),
        "training_protocol_sha256": (
            provenance_hashes[
                "baseline_protocol_v2"
            ]
        ),
        "runner_source_sha256": (
            baseline_runner_source_sha256
        ),
        "fargfrc_semantic_runner_source_sha256": (
            provenance_hashes[
                "fargfrc_runner_v2_raw"
            ]
        ),
        "best_epoch": int(
            best_epoch_index
        ),
        "best_selection_raw_speed_mae": float(
            best_selection_mae
        ),
        "model_state_dict": best_model_state,
        "data_access_policy": data_access_policy,
        "provenance_hashes": provenance_hashes,
    }

    history_payload = {
        "dataset": "PEMS-BAY",
        "model_identifier": model_identifier,
        "model_seed": int(
            model_seed
        ),
        "model_config_sha256": (
            checkpoint_payload[
                "model_config_sha256"
            ]
        ),
        "training_protocol_sha256": (
            checkpoint_payload[
                "training_protocol_sha256"
            ]
        ),
        "runner_source_sha256": (
            baseline_runner_source_sha256
        ),
        "fargfrc_semantic_runner_source_sha256": (
            checkpoint_payload[
                "fargfrc_semantic_runner_source_sha256"
            ]
        ),
        "best_epoch": int(
            best_epoch_index
        ),
        "best_selection_raw_speed_mae": float(
            best_selection_mae
        ),
        "epochs": epoch_history,
        "data_access_policy": data_access_policy,
        "provenance_hashes": provenance_hashes,
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
        "checkpoint_path": str(
            checkpoint_path
        ),
        "history_path": str(
            history_path
        ),
        "best_epoch": int(
            best_epoch_index
        ),
        "best_selection_raw_speed_mae": float(
            best_selection_mae
        ),
        "completed_epoch_count": int(
            len(epoch_history)
        ),
        "runner_source_sha256": (
            baseline_runner_source_sha256
        ),
        "provenance_hashes": provenance_hashes,
        "data_access_policy": data_access_policy,
    }
