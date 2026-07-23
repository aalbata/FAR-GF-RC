"""
PEMS-BAY Final Multi-Model Primary Evaluation Runner v4.

This evaluator is intentionally separated from all training runners.
It evaluates the already-selected frozen checkpoints for FAR-GF-RC,
MS-GRU, and MS-TCN-v2 over the clean condition and nine immutable
controlled sensor-dropout conditions.

Primary-test access is permitted only after all protocol and source
integrity checks pass. No training, tuning, checkpoint selection,
calibration scoring, or normalization update is implemented here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import sys
import json
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone

import h5py
import numpy as np
import torch


# ============================================================
# 1. Immutable experiment identifiers
# ============================================================
EXPECTED_FINAL_PROTOCOL_SHA256 = (
    "b299159b61b80475a00d2e89382734479550d6f5577e47f3b7bacc18999458fc"
)

EXPECTED_BOUNDARY_HISTORY_ADDENDUM_SHA256 = (
    "f89d07942ddce901d4154408836d0e1a2790ef4ddfd00f752fa1e58f12b8d481"
)

EXPECTED_ELAPSED_STATE_ADDENDUM_SHA256 = (
    "31869b4d20d9cae84e88534fbcf3cb946a3e80f65e7eee0675475809e9556990"
)

EXPECTED_DATA_PROTOCOL_SHA256 = (
    "5c466b6c481dcf5585d5fd35ceaaeb9015552aadb3a5f34b5d4a7f0384582f25"
)

EXPECTED_CONTROLLED_DROPOUT_PROTOCOL_SHA256 = (
    "a5f0ba1d785d03aefffe2262d9129aff07a88f8038c23bbe5200e77084a8dbf4"
)

EXPECTED_TEST_MANIFEST_SHA256 = (
    "0f5e4e4e8bc9248a2aeee194c4e750d0c56b096459ad78c5f04ba44ddfee2e5c"
)

EXPECTED_RAW_H5_SHA256 = (
    "65d69fb0a2323dba9867179eb7af47c8b814186bc459ff0a4937d21614153c8f"
)

EXPECTED_NATIVE_MASK_SHA256 = (
    "ec8917124fe063cf20b962fc5fb349dca750dfb6a8ab0cbcebf0a53728b4ea5d"
)

EXPECTED_NORMALIZATION_SHA256 = (
    "bd7733009bbe8cdd04d5acfc4f48ba4d0695f6de5a4a52ac6edbbd7a1a796f12"
)

EXPECTED_TEMPORAL_SPLIT_SHA256 = (
    "d1e887e798a1c45bcc6bff41c757c1d7a00fcebdb07d218074a335555d6d6534"
)

EXPECTED_PHYSICAL_GRAPH_SHA256 = (
    "0801aa7ccaae9ef6c8f695c8cf7d6f666cf0da67c11185faf3fe123962924de6"
)

EXPECTED_SPATIAL_TOPOLOGY_SHA256 = (
    "2e11ef5f1e83032318c8afa5a68e56728da7f66ce3e99cda33ffa51d410294df"
)

EXPECTED_FAR_MODEL_SHA256 = (
    "6dd6cc74f27cffd70cfef6e4dda89d31e5db16a31d66517a2da1ba2004070d25"
)

EXPECTED_MS_GRU_MODEL_SHA256 = (
    "17ac07b273476fb4cf7def103e5f4e25fbcbaf488b61e0cf5da9a3456999851e"
)

EXPECTED_MS_TCN_V2_MODEL_SHA256 = (
    "1e46c2dffd2153f0babc5b267b538959f89a86bc1c3d2d8cea7f4706e6906c8b"
)

EXPECTED_FAR_CONFIG_SHA256 = (
    "4cbfe658252f4fc500d3343d28472649e0d1994ec8d1ec3cd5d55347e705636e"
)

EXPECTED_MS_GRU_CONFIG_SHA256 = (
    "b5edb17e18e94228e26c88002f2dbf7de0c1e4f98f79abbf85639124c9e3b127"
)

EXPECTED_MS_TCN_V2_CONFIG_SHA256 = (
    "52bf7c3277b1820089bd685a3031a179a7308bb3f2538b6617f246eab28758bd"
)

EXPECTED_FAR_RUNNER_RAW_SHA256 = (
    "f0c4934c2954e4f528c391ec03aba2c55a9956d930646003049076d29451a433"
)

EXPECTED_BASELINE_RUNNER_SHA256 = (
    "563f14a9cea1b9a0517a8017702726e2541caecf8db552df5ad4106c2530ec9e"
)

EXPECTED_FAR_TRAINING_PROTOCOL_SHA256 = (
    "7abedbc31e21bacde366d0403728c698566d28ba95957643cdf0773fd2ddfcef"
)

EXPECTED_BASELINE_TRAINING_PROTOCOL_SHA256 = (
    "91d5ff35786a76f03bdfceaf7a3059f65c532c8f064baa23ed67e1369f5ed23e"
)


# ============================================================
# 2. Frozen primary-test evaluation contract
# ============================================================
MODEL_IDENTIFIERS = (
    "FAR-GF-RC",
    "MS-GRU",
    "MS-TCN-v2",
)

BASELINE_IDENTIFIERS = (
    "MS-GRU",
    "MS-TCN-v2",
)

EXPECTED_MODEL_SEEDS = (
    17,
    29,
    43,
    71,
    101,
)

CONDITION_ORDER = (
    "clean_native_history",
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

FAILURE_SCENARIOS = CONDITION_ORDER[1:]

EXPECTED_ARTIFICIAL_REMOVALS = {
    "iid_random_10pct": 4061785,
    "iid_random_30pct": 12186825,
    "iid_random_50pct": 20298255,
    "temporal_tail_25pct_sensors_3steps": 2561433,
    "temporal_tail_50pct_sensors_6steps": 10183264,
    "temporal_tail_75pct_sensors_12steps": 30487291,
    "spatial_geographic_knn4_cluster_8_full_history": 999586,
    "spatial_geographic_knn4_cluster_16_full_history": 1999187,
    "spatial_geographic_knn4_cluster_32_full_history": 3998327,
}

INPUT_STEPS = 12
OUTPUT_STEPS = 12
NUMBER_OF_SENSORS = 325
ELAPSED_GAP_CAP_STEPS = 288

RAW_SPEED_START = 41680
RAW_SPEED_END = 52116

PRIMARY_TEST_START = 41692
PRIMARY_TEST_END = 52116
PRIMARY_TEST_TARGET_START = 41692
PRIMARY_TEST_TARGET_STOP = 52105
PRIMARY_TEST_WINDOW_COUNT = 10413
PRIMARY_TEST_NATIVE_TARGET_COUNT = 40608096

HORIZON_INDEX_BY_LEAD = {
    3: 2,
    6: 5,
    12: 11,
}

SCOPE_ORDER = (
    "overall",
    "horizon_3",
    "horizon_6",
    "horizon_12",
)

METRIC_NAMES = (
    "mae",
    "rmse",
    "mape_percent",
    "wmape_percent",
)

EVALUATION_BATCH_SIZE = 64


# ============================================================
# 3. Project paths
# ============================================================
def resolve_paths(project_root: Path) -> dict:
    project_root = Path(project_root).resolve()

    return {
        "project_root": project_root,
        "final_protocol": (
            project_root
            / "configs"
            / "pems_bay_final_multi_model_primary_evaluation_protocol_v1.json"
        ),
        "boundary_addendum": (
            project_root
            / "configs"
            / "pems_bay_final_primary_test_boundary_history_context_addendum_v1.json"
        ),
        "elapsed_addendum": (
            project_root
            / "configs"
            / "pems_bay_final_primary_test_native_elapsed_state_prehistory_addendum_v1.json"
        ),
        "data_protocol": (
            project_root
            / "configs"
            / "pems_bay_far_gf_rc_data_protocol_v1.json"
        ),
        "controlled_dropout_protocol": (
            project_root
            / "configs"
            / "pems_bay_controlled_sensor_dropout_protocol_v1.json"
        ),
        "far_training_protocol": (
            project_root
            / "configs"
            / "pems_bay_far_gf_rc_training_protocol_v2.json"
        ),
        "baseline_training_protocol": (
            project_root
            / "configs"
            / "pems_bay_masked_baseline_training_protocol_v2.json"
        ),
        "far_config": (
            project_root
            / "configs"
            / "pems_bay_far_gf_rc_config_v1.json"
        ),
        "ms_gru_config": (
            project_root
            / "configs"
            / "pems_bay_masked_sensor_shared_gru_config_v1.json"
        ),
        "ms_tcn_v2_config": (
            project_root
            / "configs"
            / "pems_bay_masked_sensor_shared_tcn_v2_config_v1.json"
        ),
        "raw_h5": (
            project_root
            / "data"
            / "raw"
            / "PEMSBAY"
            / "pems-bay.h5"
        ),
        "native_mask": (
            project_root
            / "data"
            / "processed"
            / "pems_bay_native_observation_mask_v1.npz"
        ),
        "normalization": (
            project_root
            / "data"
            / "processed"
            / "pems_bay_train_normalization_v1.npz"
        ),
        "temporal_split": (
            project_root
            / "data"
            / "processed"
            / "pems_bay_temporal_split_v1.npz"
        ),
        "physical_graph": (
            project_root
            / "data"
            / "processed"
            / "pems_bay_geographic_knn4_self_tuning_gaussian_physical_graph_v1.npz"
        ),
        "spatial_topology": (
            project_root
            / "data"
            / "processed"
            / "pems_bay_geographic_knn4_spatial_failure_topology_v1.npz"
        ),
        "test_manifest": (
            project_root
            / "data"
            / "processed"
            / "controlled_dropout"
            / "pems_bay_v1"
            / "pems_bay_controlled_sensor_dropout_test_primary_manifest_v1.json"
        ),
        "far_model_source": (
            project_root
            / "src"
            / "models"
            / "far_gf_rc.py"
        ),
        "ms_gru_model_source": (
            project_root
            / "src"
            / "models"
            / "masked_sensor_shared_gru.py"
        ),
        "ms_tcn_v2_model_source": (
            project_root
            / "src"
            / "models"
            / "masked_sensor_shared_tcn_v2.py"
        ),
        "far_runner_source": (
            project_root
            / "src"
            / "training"
            / "pems_bay_far_gf_rc_runner_v2.py"
        ),
        "baseline_runner_source": (
            project_root
            / "src"
            / "training"
            / "pems_bay_masked_baseline_runner_v1.py"
        ),
        "results_directory": (
            project_root
            / "outputs"
            / "results"
            / "controlled_dropout"
            / "pems_bay_final_primary_evaluation_v1"
        ),
    }


# ============================================================
# 4. Immutable integrity helpers
# ============================================================
def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()

    with open(file_path, "rb") as file_handle:
        for block in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def verify_file(
    logical_name: str,
    file_path: Path,
    expected_sha256: str,
) -> str:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing frozen asset: {logical_name}\n{file_path}"
        )

    observed_sha256 = sha256_file(file_path)

    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {logical_name}.\n"
            f"Expected: {expected_sha256}\n"
            f"Observed: {observed_sha256}"
        )

    return observed_sha256


def read_json_file(
    file_path: Path,
) -> dict:
    return json.loads(
        file_path.read_text(
            encoding="utf-8"
        )
    )


def require_equal(
    observed_value,
    expected_value,
    logical_name: str,
) -> None:
    if observed_value != expected_value:
        raise RuntimeError(
            f"Frozen contract mismatch for {logical_name}.\n"
            f"Expected: {expected_value}\n"
            f"Observed: {observed_value}"
        )


def import_module_from_source(
    module_name: str,
    source_path: Path,
):
    specification = importlib.util.spec_from_file_location(
        module_name,
        source_path,
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise ImportError(
            f"Unable to load frozen source module:\n{source_path}"
        )

    module = importlib.util.module_from_spec(
        specification
    )
    sys.modules[module_name] = module

    specification.loader.exec_module(module)

    return module


def decode_text_scalar(
    value,
) -> str:
    values = np.asarray(value)

    if values.size != 1:
        raise RuntimeError(
            "Expected a one-element text array."
        )

    scalar_value = values.reshape(
        -1
    )[0]

    if isinstance(
        scalar_value,
        bytes,
    ):
        return scalar_value.decode(
            "utf-8"
        )

    return str(scalar_value)


# ============================================================
# 5. Frozen protocol reconciliation
# ============================================================
def verify_and_load_protocols(
    paths: dict,
) -> tuple[dict, dict]:
    observed_hashes = {}

    observed_hashes["final_protocol"] = verify_file(
        "final primary-test evaluation protocol",
        paths["final_protocol"],
        EXPECTED_FINAL_PROTOCOL_SHA256,
    )

    observed_hashes["boundary_addendum"] = verify_file(
        "boundary-history context addendum",
        paths["boundary_addendum"],
        EXPECTED_BOUNDARY_HISTORY_ADDENDUM_SHA256,
    )

    observed_hashes["elapsed_addendum"] = verify_file(
        "native elapsed-state prehistory addendum",
        paths["elapsed_addendum"],
        EXPECTED_ELAPSED_STATE_ADDENDUM_SHA256,
    )

    observed_hashes["data_protocol"] = verify_file(
        "PEMS-BAY data protocol",
        paths["data_protocol"],
        EXPECTED_DATA_PROTOCOL_SHA256,
    )

    observed_hashes["controlled_dropout_protocol"] = verify_file(
        "controlled-dropout protocol",
        paths["controlled_dropout_protocol"],
        EXPECTED_CONTROLLED_DROPOUT_PROTOCOL_SHA256,
    )

    observed_hashes["test_manifest"] = verify_file(
        "primary-test mask manifest",
        paths["test_manifest"],
        EXPECTED_TEST_MANIFEST_SHA256,
    )

    observed_hashes["far_training_protocol"] = verify_file(
        "FAR-GF-RC training protocol",
        paths["far_training_protocol"],
        EXPECTED_FAR_TRAINING_PROTOCOL_SHA256,
    )

    observed_hashes["baseline_training_protocol"] = verify_file(
        "masked baseline training protocol",
        paths["baseline_training_protocol"],
        EXPECTED_BASELINE_TRAINING_PROTOCOL_SHA256,
    )

    observed_hashes["far_config"] = verify_file(
        "FAR-GF-RC model configuration",
        paths["far_config"],
        EXPECTED_FAR_CONFIG_SHA256,
    )

    observed_hashes["ms_gru_config"] = verify_file(
        "MS-GRU model configuration",
        paths["ms_gru_config"],
        EXPECTED_MS_GRU_CONFIG_SHA256,
    )

    observed_hashes["ms_tcn_v2_config"] = verify_file(
        "MS-TCN-v2 model configuration",
        paths["ms_tcn_v2_config"],
        EXPECTED_MS_TCN_V2_CONFIG_SHA256,
    )

    observed_hashes["raw_h5"] = verify_file(
        "raw PEMS-BAY HDF",
        paths["raw_h5"],
        EXPECTED_RAW_H5_SHA256,
    )

    observed_hashes["native_mask"] = verify_file(
        "native observation-mask archive",
        paths["native_mask"],
        EXPECTED_NATIVE_MASK_SHA256,
    )

    observed_hashes["normalization"] = verify_file(
        "train normalization archive",
        paths["normalization"],
        EXPECTED_NORMALIZATION_SHA256,
    )

    observed_hashes["temporal_split"] = verify_file(
        "temporal split archive",
        paths["temporal_split"],
        EXPECTED_TEMPORAL_SPLIT_SHA256,
    )

    observed_hashes["physical_graph"] = verify_file(
        "physical graph archive",
        paths["physical_graph"],
        EXPECTED_PHYSICAL_GRAPH_SHA256,
    )

    observed_hashes["spatial_topology"] = verify_file(
        "spatial failure-topology archive",
        paths["spatial_topology"],
        EXPECTED_SPATIAL_TOPOLOGY_SHA256,
    )

    observed_hashes["far_model_source"] = verify_file(
        "FAR-GF-RC model source",
        paths["far_model_source"],
        EXPECTED_FAR_MODEL_SHA256,
    )

    observed_hashes["ms_gru_model_source"] = verify_file(
        "MS-GRU model source",
        paths["ms_gru_model_source"],
        EXPECTED_MS_GRU_MODEL_SHA256,
    )

    observed_hashes["ms_tcn_v2_model_source"] = verify_file(
        "MS-TCN-v2 model source",
        paths["ms_tcn_v2_model_source"],
        EXPECTED_MS_TCN_V2_MODEL_SHA256,
    )

    observed_hashes["far_runner_source"] = verify_file(
        "FAR-GF-RC semantic runner",
        paths["far_runner_source"],
        EXPECTED_FAR_RUNNER_RAW_SHA256,
    )

    observed_hashes["baseline_runner_source"] = verify_file(
        "baseline semantic runner",
        paths["baseline_runner_source"],
        EXPECTED_BASELINE_RUNNER_SHA256,
    )

    final_protocol = read_json_file(
        paths["final_protocol"]
    )

    boundary_addendum = read_json_file(
        paths["boundary_addendum"]
    )

    elapsed_addendum = read_json_file(
        paths["elapsed_addendum"]
    )

    condition_order = final_protocol[
        "controlled_dropout_protocol"
    ][
        "condition_order"
    ]

    require_equal(
        condition_order,
        list(CONDITION_ORDER),
        "final condition order",
    )

    require_equal(
        final_protocol["model_seeds"],
        list(EXPECTED_MODEL_SEEDS),
        "final model seeds",
    )

    require_equal(
        final_protocol[
            "controlled_dropout_protocol"
        ][
            "expected_artificial_removals"
        ],
        EXPECTED_ARTIFICIAL_REMOVALS,
        "expected artificial-removal counts",
    )

    require_equal(
        final_protocol[
            "partition_policy"
        ][
            "primary_test_interval"
        ],
        [PRIMARY_TEST_START, PRIMARY_TEST_END],
        "primary-test raw interval",
    )

    require_equal(
        final_protocol[
            "partition_policy"
        ][
            "primary_test_target_start_interval"
        ],
        [PRIMARY_TEST_TARGET_START, PRIMARY_TEST_TARGET_STOP - 1],
        "primary-test target-start interval",
    )

    require_equal(
        int(
            final_protocol[
                "partition_policy"
            ][
                "primary_test_window_count"
            ]
        ),
        PRIMARY_TEST_WINDOW_COUNT,
        "primary-test window count",
    )

    require_equal(
        int(
            final_protocol[
                "partition_policy"
            ][
                "primary_test_native_target_cells"
            ]
        ),
        PRIMARY_TEST_NATIVE_TARGET_COUNT,
        "primary-test native target-cell count",
    )

    require_equal(
        boundary_addendum[
            "boundary_history_context"
        ][
            "required_raw_history_interval"
        ],
        [RAW_SPEED_START, PRIMARY_TEST_START],
        "boundary-history raw interval",
    )

    require_equal(
        elapsed_addendum[
            "authorized_native_mask_access"
        ][
            "access_interval"
        ],
        [0, PRIMARY_TEST_END],
        "authorized native-mask interval",
    )

    require_equal(
        elapsed_addendum[
            "raw_speed_protection_policy"
        ][
            "authorized_raw_speed_access_interval"
        ],
        [RAW_SPEED_START, RAW_SPEED_END],
        "authorized raw-speed interval",
    )

    return {
        "final_protocol": final_protocol,
        "boundary_addendum": boundary_addendum,
        "elapsed_addendum": elapsed_addendum,
        "hashes": observed_hashes,
    }, final_protocol


# ============================================================
# 6. Required data loading and semantic validation
# ============================================================
def load_required_evaluation_assets(
    paths: dict,
    protocol_bundle: dict,
) -> dict:
    with np.load(
        paths["native_mask"],
        allow_pickle=False,
    ) as native_mask_archive:
        if "native_observation_mask" not in native_mask_archive.files:
            raise RuntimeError(
                "native_observation_mask is missing from the "
                "frozen native-mask archive."
            )

        native_observation_mask = np.asarray(
            native_mask_archive[
                "native_observation_mask"
            ],
            dtype=bool,
        )

    if native_observation_mask.shape != (
        PRIMARY_TEST_END,
        NUMBER_OF_SENSORS,
    ):
        raise RuntimeError(
            "Unexpected native-observation-mask shape."
        )

    with np.load(
        paths["normalization"],
        allow_pickle=False,
    ) as normalization_archive:
        sensor_means = np.asarray(
            normalization_archive[
                "sensor_means"
            ],
            dtype=np.float32,
        )

        sensor_stds = np.asarray(
            normalization_archive[
                "sensor_stds"
            ],
            dtype=np.float32,
        )

    if sensor_means.shape != (
        NUMBER_OF_SENSORS,
    ):
        raise RuntimeError(
            "Unexpected sensor_means shape."
        )

    if sensor_stds.shape != (
        NUMBER_OF_SENSORS,
    ):
        raise RuntimeError(
            "Unexpected sensor_stds shape."
        )

    if not np.all(
        np.isfinite(sensor_means)
    ):
        raise RuntimeError(
            "sensor_means contains non-finite values."
        )

    if not np.all(
        np.isfinite(sensor_stds)
    ):
        raise RuntimeError(
            "sensor_stds contains non-finite values."
        )

    if np.any(sensor_stds <= 0.0):
        raise RuntimeError(
            "sensor_stds must be strictly positive."
        )

    with np.load(
        paths["temporal_split"],
        allow_pickle=False,
    ) as split_archive:
        primary_test_target_starts = np.asarray(
            split_archive[
                "primary_test_target_starts"
            ],
            dtype=np.int64,
        )

    expected_target_starts = np.arange(
        PRIMARY_TEST_TARGET_START,
        PRIMARY_TEST_TARGET_STOP,
        dtype=np.int64,
    )

    if not np.array_equal(
        primary_test_target_starts,
        expected_target_starts,
    ):
        raise RuntimeError(
            "The frozen primary_test_target_starts sequence does "
            "not match the chronological final-test contract."
        )

    with np.load(
        paths["physical_graph"],
        allow_pickle=False,
    ) as physical_graph_archive:
        physical_graph = np.asarray(
            physical_graph_archive[
                "physical_graph"
            ],
            dtype=np.float32,
        )

    if physical_graph.shape != (
        NUMBER_OF_SENSORS,
        NUMBER_OF_SENSORS,
    ):
        raise RuntimeError(
            "Unexpected physical_graph shape."
        )

    if not np.all(
        np.isfinite(physical_graph)
    ):
        raise RuntimeError(
            "physical_graph contains non-finite values."
        )

    with h5py.File(
        paths["raw_h5"],
        "r",
    ) as raw_h5:
        raw_speed_values = np.asarray(
            raw_h5[
                "speed/block0_values"
            ][
                RAW_SPEED_START:RAW_SPEED_END
            ],
            dtype=np.float32,
        )

        timestamp_values_ns = np.asarray(
            raw_h5[
                "speed/axis1"
            ][
                RAW_SPEED_START:RAW_SPEED_END
            ],
            dtype=np.int64,
        )

    expected_raw_shape = (
        RAW_SPEED_END - RAW_SPEED_START,
        NUMBER_OF_SENSORS,
    )

    if raw_speed_values.shape != expected_raw_shape:
        raise RuntimeError(
            "Raw-speed slice has an unexpected shape."
        )

    if timestamp_values_ns.shape != (
        RAW_SPEED_END - RAW_SPEED_START,
    ):
        raise RuntimeError(
            "Timestamp slice has an unexpected shape."
        )

    if not np.all(
        np.isfinite(raw_speed_values)
    ):
        raise RuntimeError(
            "The authorized raw-speed slice contains non-finite values."
        )

    if int(
        primary_test_target_starts.shape[0]
    ) != PRIMARY_TEST_WINDOW_COUNT:
        raise RuntimeError(
            "Primary-test target-start count is invalid."
        )

    history_global_indices = (
        primary_test_target_starts[
            :,
            None,
        ]
        - INPUT_STEPS
        + np.arange(
            INPUT_STEPS,
            dtype=np.int64,
        )[
            None,
            :
        ]
    )

    if history_global_indices.min() != RAW_SPEED_START:
        raise RuntimeError(
            "First primary-test history index does not equal 41680."
        )

    if history_global_indices.max() != (
        PRIMARY_TEST_TARGET_STOP - 2
    ):
        raise RuntimeError(
            "Last primary-test history index is invalid."
        )

    native_history_mask_all = native_observation_mask[
        history_global_indices
    ]

    if native_history_mask_all.shape != (
        PRIMARY_TEST_WINDOW_COUNT,
        INPUT_STEPS,
        NUMBER_OF_SENSORS,
    ):
        raise RuntimeError(
            "Native historical-window mask has an unexpected shape."
        )

    return {
        "raw_speed_values": raw_speed_values,
        "timestamp_values_ns": timestamp_values_ns,
        "native_observation_mask": native_observation_mask,
        "sensor_means": sensor_means,
        "sensor_stds": sensor_stds,
        "physical_graph": physical_graph,
        "primary_test_target_starts": primary_test_target_starts,
        "native_history_mask_all": native_history_mask_all,
        "protocol_bundle": protocol_bundle,
    }


# ============================================================
# 7. Exact frozen scenario-mask loading
# ============================================================
def load_scenario_mask_records(
    paths: dict,
    assets: dict,
) -> dict:
    manifest = read_json_file(
        paths["test_manifest"]
    )

    primary_test = manifest[
        "primary_test"
    ]

    require_equal(
        int(
            primary_test[
                "target_window_count"
            ]
        ),
        PRIMARY_TEST_WINDOW_COUNT,
        "manifest primary-test target-window count",
    )

    require_equal(
        int(
            primary_test[
                "native_observed_history_count"
            ]
        ),
        int(assets["native_history_mask_all"].sum(dtype=np.int64)),
        "manifest native observed history count",
    )

    require_equal(
        int(
            primary_test[
                "native_target_count"
            ]
        ),
        PRIMARY_TEST_NATIVE_TARGET_COUNT,
        "manifest native target count",
    )

    records = primary_test[
        "scenario_mask_records"
    ]

    record_order = [
        record["scenario_id"]
        for record in records
    ]

    require_equal(
        record_order,
        list(FAILURE_SCENARIOS),
        "primary-test scenario-mask order",
    )

    resolved_masks = {}

    for record in records:
        scenario_identifier = record[
            "scenario_id"
        ]

        if scenario_identifier not in FAILURE_SCENARIOS:
            raise RuntimeError(
                f"Unexpected scenario in test manifest: "
                f"{scenario_identifier}"
            )

        require_equal(
            record["status"],
            "created",
            f"{scenario_identifier} manifest status",
        )

        require_equal(
            int(
                record[
                    "artificial_removed_history_count"
                ]
            ),
            EXPECTED_ARTIFICIAL_REMOVALS[
                scenario_identifier
            ],
            f"{scenario_identifier} expected removal count",
        )

        mask_path = Path(
            record["path"]
        )

        verify_file(
            f"{scenario_identifier} primary-test mask archive",
            mask_path,
            record["sha256"],
        )

        with np.load(
            mask_path,
            allow_pickle=False,
        ) as mask_archive:
            required_mask_keys = {
                "artificial_dropout_mask",
                "primary_test_target_starts",
                "scenario_id",
                "scenario_specification_json",
                "scenario_specification_sha256",
            }

            missing_keys = sorted(
                required_mask_keys
                - set(mask_archive.files)
            )

            if missing_keys:
                raise RuntimeError(
                    f"{scenario_identifier} mask archive lacks "
                    f"required keys: {missing_keys}"
                )

            artificial_dropout_mask = np.asarray(
                mask_archive[
                    "artificial_dropout_mask"
                ],
                dtype=bool,
            )

            archived_target_starts = np.asarray(
                mask_archive[
                    "primary_test_target_starts"
                ],
                dtype=np.int64,
            )

            archived_scenario_identifier = decode_text_scalar(
                mask_archive[
                    "scenario_id"
                ]
            )

        if artificial_dropout_mask.shape != (
            PRIMARY_TEST_WINDOW_COUNT,
            INPUT_STEPS,
            NUMBER_OF_SENSORS,
        ):
            raise RuntimeError(
                f"{scenario_identifier} artificial-dropout mask "
                "has an unexpected shape."
            )

        if not np.array_equal(
            archived_target_starts,
            assets[
                "primary_test_target_starts"
            ],
        ):
            raise RuntimeError(
                f"{scenario_identifier} target starts differ from "
                "the frozen temporal split."
            )

        require_equal(
            archived_scenario_identifier,
            scenario_identifier,
            f"{scenario_identifier} archived scenario identifier",
        )

        artificial_removed_history_count = int(
            artificial_dropout_mask.sum(
                dtype=np.int64
            )
        )

        require_equal(
            artificial_removed_history_count,
            int(
                record[
                    "artificial_removed_history_count"
                ]
            ),
            f"{scenario_identifier} artificial removed count",
        )

        forbidden_artificial_removals = (
            artificial_dropout_mask
            & ~assets[
                "native_history_mask_all"
            ]
        )

        if np.any(
            forbidden_artificial_removals
        ):
            raise RuntimeError(
                f"{scenario_identifier} removes a historically "
                "native-missing entry, which is prohibited."
            )

        remaining_observed_history_count = int(
            assets[
                "native_history_mask_all"
            ].sum(
                dtype=np.int64
            )
            - artificial_removed_history_count
        )

        require_equal(
            remaining_observed_history_count,
            int(
                record[
                    "remaining_observed_history_count"
                ]
            ),
            f"{scenario_identifier} remaining observed history count",
        )

        resolved_masks[
            scenario_identifier
        ] = artificial_dropout_mask

    return resolved_masks


# ============================================================
# 8. Frozen causal feature construction
# ============================================================
def build_batch_elapsed_feature(
    effective_history_mask: np.ndarray,
    initial_gap_steps: np.ndarray,
) -> np.ndarray:
    """
    Vectorized batch equivalent of the frozen FAR-GF-RC runner's
    build_window_elapsed_feature(...) contract.

    The initialization is the native elapsed state at t-13.
    The recurrence then uses the effective history mask after the
    scenario-specific artificial dropout mask is applied.
    """
    if effective_history_mask.ndim != 3:
        raise ValueError(
            "effective_history_mask must be [batch, steps, sensors]."
        )

    if initial_gap_steps.shape != (
        effective_history_mask.shape[0],
        effective_history_mask.shape[2],
    ):
        raise ValueError(
            "initial_gap_steps has an incompatible shape."
        )

    elapsed_feature = np.empty(
        effective_history_mask.shape,
        dtype=np.float32,
    )

    running_gap = np.asarray(
        initial_gap_steps,
        dtype=np.int16,
    ).copy()

    for step_index in range(
        effective_history_mask.shape[1]
    ):
        observed_now = effective_history_mask[
            :,
            step_index,
            :,
        ]

        running_gap[observed_now] = 0

        missing_now = ~observed_now

        running_gap[missing_now] = np.minimum(
            running_gap[missing_now] + 1,
            ELAPSED_GAP_CAP_STEPS,
        )

        elapsed_feature[
            :,
            step_index,
            :,
        ] = (
            running_gap.astype(
                np.float32
            )
            / float(
                ELAPSED_GAP_CAP_STEPS
            )
        )

    return elapsed_feature


def build_batch_inputs(
    target_starts: np.ndarray,
    artificial_dropout_mask: np.ndarray | None,
    assets: dict,
    calendar_features: np.ndarray,
    native_elapsed_steps: np.ndarray,
) -> dict:
    """
    Assemble the frozen seven-channel history tensor without
    imputation and without accessing any raw speed before 41680.
    """
    local_target_starts = (
        target_starts
        - RAW_SPEED_START
    )

    history_local_indices = (
        local_target_starts[
            :,
            None,
        ]
        - INPUT_STEPS
        + np.arange(
            INPUT_STEPS,
            dtype=np.int64,
        )[
            None,
            :
        ]
    )

    future_local_indices = (
        local_target_starts[
            :,
            None,
        ]
        + np.arange(
            OUTPUT_STEPS,
            dtype=np.int64,
        )[
            None,
            :
        ]
    )

    if history_local_indices.min() < 0:
        raise RuntimeError(
            "History construction attempted to access raw speed "
            "before the authorized boundary-history interval."
        )

    if future_local_indices.max() >= (
        RAW_SPEED_END
        - RAW_SPEED_START
    ):
        raise RuntimeError(
            "Future construction attempted to access beyond the "
            "authorized primary-test raw interval."
        )

    history_raw_speed = assets[
        "raw_speed_values"
    ][
        history_local_indices
    ]

    target_raw_speed = assets[
        "raw_speed_values"
    ][
        future_local_indices
    ]

    history_native_mask = assets[
        "native_observation_mask"
    ][
        (
            target_starts[
                :,
                None,
            ]
            - INPUT_STEPS
            + np.arange(
                INPUT_STEPS,
                dtype=np.int64,
            )[
                None,
                :
            ]
        )
    ]

    target_native_mask = assets[
        "native_observation_mask"
    ][
        (
            target_starts[
                :,
                None,
            ]
            + np.arange(
                OUTPUT_STEPS,
                dtype=np.int64,
            )[
                None,
                :
            ]
        )
    ]

    if artificial_dropout_mask is None:
        effective_history_mask = history_native_mask.copy()

    else:
        if artificial_dropout_mask.shape != (
            target_starts.shape[0],
            INPUT_STEPS,
            NUMBER_OF_SENSORS,
        ):
            raise RuntimeError(
                "Batch artificial-dropout mask has an incompatible "
                "shape."
            )

        effective_history_mask = (
            history_native_mask
            & ~artificial_dropout_mask
        )

    initial_gap_steps = native_elapsed_steps[
        target_starts
        - INPUT_STEPS
        - 1
    ]

    elapsed_history_feature = build_batch_elapsed_feature(
        effective_history_mask=effective_history_mask,
        initial_gap_steps=initial_gap_steps,
    )

    normalized_history_value = (
        history_raw_speed
        - assets[
            "sensor_means"
        ][
            None,
            None,
            :
        ]
    ) / assets[
        "sensor_stds"
    ][
        None,
        None,
        :
    ]

    normalized_history_value = np.where(
        effective_history_mask,
        normalized_history_value,
        0.0,
    ).astype(
        np.float32,
        copy=False,
    )

    history_calendar = calendar_features[
        history_local_indices
    ]

    future_calendar = calendar_features[
        future_local_indices
    ]

    history_features = np.empty(
        (
            target_starts.shape[0],
            INPUT_STEPS,
            NUMBER_OF_SENSORS,
            7,
        ),
        dtype=np.float32,
    )

    history_features[
        :,
        :,
        :,
        0,
    ] = normalized_history_value

    history_features[
        :,
        :,
        :,
        1,
    ] = effective_history_mask.astype(
        np.float32
    )

    history_features[
        :,
        :,
        :,
        2,
    ] = elapsed_history_feature

    for calendar_feature_index in range(4):
        history_features[
            :,
            :,
            :,
            3 + calendar_feature_index,
        ] = history_calendar[
            :,
            :,
            None,
            calendar_feature_index,
        ]

    if not np.all(
        np.isfinite(history_features)
    ):
        raise RuntimeError(
            "History-feature construction produced non-finite values."
        )

    if not np.all(
        np.isfinite(target_raw_speed)
    ):
        raise RuntimeError(
            "Target raw-speed slice contains non-finite values."
        )

    if np.any(
        target_raw_speed[
            target_native_mask
        ]
        <= 0.0
    ):
        raise RuntimeError(
            "Native observed targets must be strictly positive."
        )

    return {
        "history_features": history_features,
        "future_calendar": future_calendar.astype(
            np.float32,
            copy=False,
        ),
        "target_raw_speed": target_raw_speed.astype(
            np.float32,
            copy=False,
        ),
        "target_native_mask": target_native_mask,
    }


# ============================================================
# 9. Metric accumulation
# ============================================================
def empty_metric_accumulator() -> dict:
    return {
        "count": 0,
        "absolute_error_sum": 0.0,
        "squared_error_sum": 0.0,
        "absolute_percentage_error_sum": 0.0,
        "absolute_target_sum": 0.0,
    }


def update_metric_accumulator(
    accumulator: dict,
    prediction_raw: torch.Tensor,
    target_raw: torch.Tensor,
    target_mask: torch.Tensor,
) -> None:
    if prediction_raw.shape != target_raw.shape:
        raise RuntimeError(
            "Prediction and target shapes differ."
        )

    if target_mask.shape != target_raw.shape:
        raise RuntimeError(
            "Target metric mask has an incompatible shape."
        )

    valid_count = int(
        target_mask.sum().item()
    )

    if valid_count <= 0:
        raise RuntimeError(
            "Metric update received no valid target cells."
        )

    valid_prediction = prediction_raw[
        target_mask
    ]

    valid_target = target_raw[
        target_mask
    ]

    if torch.any(
        valid_target <= 0.0
    ):
        raise RuntimeError(
            "MAPE denominator must be strictly positive for every "
            "scored native target."
        )

    absolute_errors = torch.abs(
        valid_prediction
        - valid_target
    )

    accumulator["count"] += valid_count

    accumulator[
        "absolute_error_sum"
    ] += float(
        absolute_errors.sum(
            dtype=torch.float64
        ).item()
    )

    accumulator[
        "squared_error_sum"
    ] += float(
        torch.square(
            absolute_errors
        ).sum(
            dtype=torch.float64
        ).item()
    )

    accumulator[
        "absolute_percentage_error_sum"
    ] += float(
        (
            absolute_errors
            / torch.abs(
                valid_target
            )
        ).sum(
            dtype=torch.float64
        ).item()
    )

    accumulator[
        "absolute_target_sum"
    ] += float(
        torch.abs(
            valid_target
        ).sum(
            dtype=torch.float64
        ).item()
    )


def finalize_metric_accumulator(
    accumulator: dict,
) -> dict:
    count = int(
        accumulator["count"]
    )

    if count <= 0:
        raise RuntimeError(
            "Cannot finalize an empty metric accumulator."
        )

    absolute_target_sum = float(
        accumulator[
            "absolute_target_sum"
        ]
    )

    if absolute_target_sum <= 0.0:
        raise RuntimeError(
            "WMAPE denominator must be positive."
        )

    mae = (
        float(
            accumulator[
                "absolute_error_sum"
            ]
        )
        / count
    )

    rmse = float(
        np.sqrt(
            float(
                accumulator[
                    "squared_error_sum"
                ]
            )
            / count
        )
    )

    mape_percent = (
        100.0
        * float(
            accumulator[
                "absolute_percentage_error_sum"
            ]
        )
        / count
    )

    wmape_percent = (
        100.0
        * float(
            accumulator[
                "absolute_error_sum"
            ]
        )
        / absolute_target_sum
    )

    return {
        "scored_cells": count,
        "mae": mae,
        "rmse": rmse,
        "mape_percent": mape_percent,
        "wmape_percent": wmape_percent,
    }


# ============================================================
# 10. Frozen model construction
# ============================================================
def build_models_from_frozen_sources(
    paths: dict,
    device: torch.device,
):
    far_module = import_module_from_source(
        "frozen_far_gf_rc_model",
        paths["far_model_source"],
    )

    ms_gru_module = import_module_from_source(
        "frozen_masked_sensor_shared_gru",
        paths["ms_gru_model_source"],
    )

    ms_tcn_v2_module = import_module_from_source(
        "frozen_masked_sensor_shared_tcn_v2",
        paths["ms_tcn_v2_model_source"],
    )

    far_model_class = getattr(
        far_module,
        "FARGFRC",
    )

    ms_gru_model_class = getattr(
        ms_gru_module,
        "MaskedSensorSharedGRU",
    )

    ms_tcn_v2_model_class = getattr(
        ms_tcn_v2_module,
        "MaskedSensorSharedTCNV2",
    )

    far_config = read_json_file(
        paths["far_config"]
    )

    ms_gru_config = read_json_file(
        paths["ms_gru_config"]
    )

    ms_tcn_v2_config = read_json_file(
        paths["ms_tcn_v2_config"]
    )

    return {
        "FAR-GF-RC": {
            "class": far_model_class,
            "config": far_config,
        },
        "MS-GRU": {
            "class": ms_gru_model_class,
            "config": ms_gru_config,
        },
        "MS-TCN-v2": {
            "class": ms_tcn_v2_model_class,
            "config": ms_tcn_v2_config,
        },
    }


def instantiate_model(
    model_identifier: str,
    model_registry: dict,
    physical_graph: np.ndarray,
    device: torch.device,
):
    model_class = model_registry[
        model_identifier
    ][
        "class"
    ]

    model_config = model_registry[
        model_identifier
    ][
        "config"
    ]

    architecture = model_config[
        "architecture"
    ]

    if model_identifier == "FAR-GF-RC":
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
                physical_graph
            ),
            latent_dimension=int(
                architecture[
                    "latent_dimension"
                ]
            ),
            reliability_hidden_dimension=int(
                architecture[
                    "reliability_hidden_dimension"
                ]
            ),
            temporal_gru_layers=int(
                architecture[
                    "temporal_gru_layers"
                ]
            ),
            graph_layers=int(
                architecture[
                    "graph_layers"
                ]
            ),
            sensor_embedding_dimension=int(
                architecture[
                    "sensor_embedding_dimension"
                ]
            ),
            decoder_hidden_dimension=int(
                architecture[
                    "decoder_hidden_dimension"
                ]
            ),
            dropout=float(
                architecture[
                    "dropout"
                ]
            ),
            forecast_log_scale_minimum=float(
                architecture[
                    "forecast_log_scale_minimum"
                ]
            ),
            forecast_log_scale_maximum=float(
                architecture[
                    "forecast_log_scale_maximum"
                ]
            ),
        )

    elif model_identifier == "MS-GRU":
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
            dilations=[
                int(value)
                for value in architecture[
                    "dilations"
                ]
            ],
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
        raise RuntimeError(
            f"Unsupported model identifier: {model_identifier}"
        )

    return model.to(
        device
    )


def load_selected_checkpoint(
    model: torch.nn.Module,
    model_identifier: str,
    model_seed: int,
    final_protocol: dict,
) -> dict:
    model_specification = final_protocol[
        "models"
    ][
        model_identifier
    ]

    checkpoint_directory = Path(
        model_specification[
            "checkpoint_directory"
        ]
    )

    checkpoint_filename = model_specification[
        "checkpoint_filename_template"
    ].format(
        seed=int(model_seed)
    )

    checkpoint_path = (
        checkpoint_directory
        / checkpoint_filename
    )

    expected_checkpoint_sha256 = model_specification[
        "expected_checkpoint_hashes"
    ][
        str(model_seed)
    ]

    observed_checkpoint_sha256 = verify_file(
        f"{model_identifier}, Seed {model_seed} checkpoint",
        checkpoint_path,
        expected_checkpoint_sha256,
    )

    checkpoint_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(
        checkpoint_payload,
        dict,
    ):
        raise RuntimeError(
            f"{model_identifier}, Seed {model_seed}: checkpoint "
            "payload is not a mapping."
        )

    if "model_state_dict" not in checkpoint_payload:
        raise RuntimeError(
            f"{model_identifier}, Seed {model_seed}: checkpoint "
            "does not contain model_state_dict."
        )

    if int(
        checkpoint_payload[
            "model_seed"
        ]
    ) != int(model_seed):
        raise RuntimeError(
            f"{model_identifier}, Seed {model_seed}: checkpoint "
            "model_seed metadata is inconsistent."
        )

    if model_identifier == "FAR-GF-RC":
        checkpoint_model_name = checkpoint_payload[
            "model_config"
        ][
            "model_name"
        ]

        require_equal(
            checkpoint_model_name,
            "FAR-GF-RC",
            "FAR-GF-RC checkpoint model name",
        )

    else:
        require_equal(
            checkpoint_payload[
                "model_identifier"
            ],
            model_identifier,
            f"{model_identifier} checkpoint identifier",
        )

    model.load_state_dict(
        checkpoint_payload[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()

    return {
        "checkpoint_path": str(
            checkpoint_path
        ),
        "checkpoint_sha256": observed_checkpoint_sha256,
        "best_epoch": int(
            checkpoint_payload[
                "best_epoch"
            ]
        ),
        "best_selection_raw_speed_mae": float(
            checkpoint_payload[
                "best_selection_raw_speed_mae"
            ]
        ),
    }


# ============================================================
# 11. Model-family forward dispatch
# ============================================================
def forward_model(
    model_identifier: str,
    model: torch.nn.Module,
    history_features: torch.Tensor,
    future_calendar: torch.Tensor,
) -> torch.Tensor:
    if model_identifier == "FAR-GF-RC":
        model_output = model(
            history_features,
            future_calendar,
        )

        if not isinstance(
            model_output,
            dict,
        ):
            raise RuntimeError(
                "FAR-GF-RC must return a dictionary."
            )

        if "forecast_normalized" not in model_output:
            raise RuntimeError(
                "FAR-GF-RC output lacks forecast_normalized."
            )

        normalized_prediction = model_output[
            "forecast_normalized"
        ]

    elif model_identifier == "MS-GRU":
        normalized_prediction = model(
            history_features[
                :,
                :,
                :,
                0,
            ],
            history_features[
                :,
                :,
                :,
                1,
            ],
            history_features[
                :,
                :,
                :,
                2,
            ],
            history_features[
                :,
                :,
                0,
                3:7,
            ],
            future_calendar,
        )

    elif model_identifier == "MS-TCN-v2":
        normalized_prediction = model(
            history_features,
            future_calendar,
        )

    else:
        raise RuntimeError(
            f"Unsupported model identifier: {model_identifier}"
        )

    if normalized_prediction.shape != (
        history_features.shape[0],
        OUTPUT_STEPS,
        NUMBER_OF_SENSORS,
    ):
        raise RuntimeError(
            f"{model_identifier} returned an unexpected prediction shape: "
            f"{tuple(normalized_prediction.shape)}"
        )

    return normalized_prediction


# ============================================================
# 12. Scenario evaluation
# ============================================================
def evaluate_one_model_seed_condition(
    model_identifier: str,
    model: torch.nn.Module,
    condition_identifier: str,
    artificial_dropout_mask: np.ndarray | None,
    assets: dict,
    calendar_features: np.ndarray,
    native_elapsed_steps: np.ndarray,
    device: torch.device,
) -> dict:
    target_starts = assets[
        "primary_test_target_starts"
    ]

    accumulators = {
        "overall": empty_metric_accumulator(),
        "horizon_3": empty_metric_accumulator(),
        "horizon_6": empty_metric_accumulator(),
        "horizon_12": empty_metric_accumulator(),
    }

    means_tensor = torch.from_numpy(
        assets[
            "sensor_means"
        ]
    ).to(
        device=device,
        dtype=torch.float32,
    ).view(
        1,
        1,
        NUMBER_OF_SENSORS,
    )

    stds_tensor = torch.from_numpy(
        assets[
            "sensor_stds"
        ]
    ).to(
        device=device,
        dtype=torch.float32,
    ).view(
        1,
        1,
        NUMBER_OF_SENSORS,
    )

    with torch.no_grad():
        for batch_start in range(
            0,
            PRIMARY_TEST_WINDOW_COUNT,
            EVALUATION_BATCH_SIZE,
        ):
            batch_stop = min(
                batch_start
                + EVALUATION_BATCH_SIZE,
                PRIMARY_TEST_WINDOW_COUNT,
            )

            target_starts_batch = target_starts[
                batch_start:batch_stop
            ]

            if artificial_dropout_mask is None:
                artificial_dropout_mask_batch = None

            else:
                artificial_dropout_mask_batch = (
                    artificial_dropout_mask[
                        batch_start:batch_stop
                    ]
                )

            batch = build_batch_inputs(
                target_starts=target_starts_batch,
                artificial_dropout_mask=artificial_dropout_mask_batch,
                assets=assets,
                calendar_features=calendar_features,
                native_elapsed_steps=native_elapsed_steps,
            )

            history_features_tensor = torch.from_numpy(
                np.ascontiguousarray(
                    batch[
                        "history_features"
                    ]
                )
            ).to(
                device=device,
                dtype=torch.float32,
                non_blocking=False,
            )

            future_calendar_tensor = torch.from_numpy(
                np.ascontiguousarray(
                    batch[
                        "future_calendar"
                    ]
                )
            ).to(
                device=device,
                dtype=torch.float32,
                non_blocking=False,
            )

            target_raw_tensor = torch.from_numpy(
                np.ascontiguousarray(
                    batch[
                        "target_raw_speed"
                    ]
                )
            ).to(
                device=device,
                dtype=torch.float32,
                non_blocking=False,
            )

            target_mask_tensor = torch.from_numpy(
                np.ascontiguousarray(
                    batch[
                        "target_native_mask"
                    ]
                )
            ).to(
                device=device,
                dtype=torch.bool,
                non_blocking=False,
            )

            normalized_prediction = forward_model(
                model_identifier=model_identifier,
                model=model,
                history_features=history_features_tensor,
                future_calendar=future_calendar_tensor,
            )

            raw_prediction = (
                normalized_prediction
                * stds_tensor
                + means_tensor
            )

            update_metric_accumulator(
                accumulator=accumulators[
                    "overall"
                ],
                prediction_raw=raw_prediction,
                target_raw=target_raw_tensor,
                target_mask=target_mask_tensor,
            )

            for lead_step, lead_index in (
                HORIZON_INDEX_BY_LEAD.items()
            ):
                update_metric_accumulator(
                    accumulator=accumulators[
                        f"horizon_{lead_step}"
                    ],
                    prediction_raw=raw_prediction[
                        :,
                        lead_index,
                        :,
                    ],
                    target_raw=target_raw_tensor[
                        :,
                        lead_index,
                        :,
                    ],
                    target_mask=target_mask_tensor[
                        :,
                        lead_index,
                        :,
                    ],
                )

    finalized_metrics = {
        scope_identifier: finalize_metric_accumulator(
            accumulator
        )
        for scope_identifier, accumulator in accumulators.items()
    }

    overall_count = finalized_metrics[
        "overall"
    ][
        "scored_cells"
    ]

    if overall_count != PRIMARY_TEST_NATIVE_TARGET_COUNT:
        raise RuntimeError(
            f"{model_identifier}, {condition_identifier}: scored "
            "overall native target-cell count differs from the "
            "frozen primary-test contract.\n"
            f"Expected: {PRIMARY_TEST_NATIVE_TARGET_COUNT}\n"
            f"Observed: {overall_count}"
        )

    return finalized_metrics


# ============================================================
# 13. Aggregation and reporting
# ============================================================
def aggregate_seed_metrics(
    per_seed_rows: list[dict],
) -> list[dict]:
    grouped_rows = {}

    for row in per_seed_rows:
        group_key = (
            row["model_identifier"],
            row["condition_identifier"],
            row["scope_identifier"],
        )

        grouped_rows.setdefault(
            group_key,
            [],
        ).append(row)

    summary_rows = []

    for (
        model_identifier,
        condition_identifier,
        scope_identifier,
    ), rows in sorted(
        grouped_rows.items()
    ):
        observed_seed_order = sorted(
            int(
                row["model_seed"]
            )
            for row in rows
        )

        if observed_seed_order != list(
            EXPECTED_MODEL_SEEDS
        ):
            raise RuntimeError(
                "Seed aggregation does not contain the exact frozen "
                f"five-seed set for {model_identifier}, "
                f"{condition_identifier}, {scope_identifier}."
            )

        summary_row = {
            "model_identifier": model_identifier,
            "condition_identifier": condition_identifier,
            "scope_identifier": scope_identifier,
            "seed_count": len(rows),
        }

        for metric_name in METRIC_NAMES:
            metric_values = np.asarray(
                [
                    float(
                        row[
                            metric_name
                        ]
                    )
                    for row in rows
                ],
                dtype=np.float64,
            )

            summary_row[
                f"{metric_name}_mean"
            ] = float(
                metric_values.mean()
            )

            summary_row[
                f"{metric_name}_sample_std"
            ] = float(
                metric_values.std(
                    ddof=1
                )
            )

        summary_rows.append(
            summary_row
        )

    return summary_rows


def compute_clean_to_dropout_degradation(
    per_seed_rows: list[dict],
) -> list[dict]:
    lookup = {
        (
            row["model_identifier"],
            int(
                row["model_seed"]
            ),
            row["condition_identifier"],
            row["scope_identifier"],
        ): row
        for row in per_seed_rows
    }

    degradation_rows = []

    for model_identifier in MODEL_IDENTIFIERS:
        for model_seed in EXPECTED_MODEL_SEEDS:
            for scope_identifier in SCOPE_ORDER:
                clean_row = lookup[
                    (
                        model_identifier,
                        model_seed,
                        "clean_native_history",
                        scope_identifier,
                    )
                ]

                clean_mae = float(
                    clean_row["mae"]
                )

                if clean_mae <= 0.0:
                    raise RuntimeError(
                        "Clean MAE must be positive."
                    )

                for failure_scenario in FAILURE_SCENARIOS:
                    scenario_row = lookup[
                        (
                            model_identifier,
                            model_seed,
                            failure_scenario,
                            scope_identifier,
                        )
                    ]

                    scenario_mae = float(
                        scenario_row["mae"]
                    )

                    absolute_degradation = (
                        scenario_mae
                        - clean_mae
                    )

                    relative_degradation_percent = (
                        100.0
                        * absolute_degradation
                        / clean_mae
                    )

                    degradation_rows.append(
                        {
                            "model_identifier": model_identifier,
                            "model_seed": int(
                                model_seed
                            ),
                            "condition_identifier": failure_scenario,
                            "scope_identifier": scope_identifier,
                            "clean_mae": clean_mae,
                            "scenario_mae": scenario_mae,
                            "absolute_mae_degradation": absolute_degradation,
                            "relative_mae_degradation_percent": (
                                relative_degradation_percent
                            ),
                        }
                    )

    return degradation_rows


def aggregate_degradation(
    degradation_rows: list[dict],
) -> list[dict]:
    grouped_rows = {}

    for row in degradation_rows:
        group_key = (
            row["model_identifier"],
            row["condition_identifier"],
            row["scope_identifier"],
        )

        grouped_rows.setdefault(
            group_key,
            [],
        ).append(row)

    summary_rows = []

    for group_key, rows in sorted(
        grouped_rows.items()
    ):
        model_identifier, condition_identifier, scope_identifier = (
            group_key
        )

        absolute_values = np.asarray(
            [
                row[
                    "absolute_mae_degradation"
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        relative_values = np.asarray(
            [
                row[
                    "relative_mae_degradation_percent"
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        summary_rows.append(
            {
                "model_identifier": model_identifier,
                "condition_identifier": condition_identifier,
                "scope_identifier": scope_identifier,
                "seed_count": len(rows),
                "absolute_mae_degradation_mean": float(
                    absolute_values.mean()
                ),
                "absolute_mae_degradation_sample_std": float(
                    absolute_values.std(
                        ddof=1
                    )
                ),
                "relative_mae_degradation_percent_mean": float(
                    relative_values.mean()
                ),
                "relative_mae_degradation_percent_sample_std": float(
                    relative_values.std(
                        ddof=1
                    )
                ),
            }
        )

    return summary_rows


def compute_paired_far_improvement(
    per_seed_rows: list[dict],
) -> list[dict]:
    lookup = {
        (
            row["model_identifier"],
            int(
                row["model_seed"]
            ),
            row["condition_identifier"],
            row["scope_identifier"],
        ): row
        for row in per_seed_rows
    }

    paired_rows = []

    for baseline_identifier in BASELINE_IDENTIFIERS:
        for model_seed in EXPECTED_MODEL_SEEDS:
            for condition_identifier in CONDITION_ORDER:
                for scope_identifier in SCOPE_ORDER:
                    far_mae = float(
                        lookup[
                            (
                                "FAR-GF-RC",
                                model_seed,
                                condition_identifier,
                                scope_identifier,
                            )
                        ][
                            "mae"
                        ]
                    )

                    baseline_mae = float(
                        lookup[
                            (
                                baseline_identifier,
                                model_seed,
                                condition_identifier,
                                scope_identifier,
                            )
                        ][
                            "mae"
                        ]
                    )

                    if baseline_mae <= 0.0:
                        raise RuntimeError(
                            "Baseline MAE must be positive for paired "
                            "relative-improvement computation."
                        )

                    paired_improvement_percent = (
                        100.0
                        * (
                            baseline_mae
                            - far_mae
                        )
                        / baseline_mae
                    )

                    paired_rows.append(
                        {
                            "reference_model": "FAR-GF-RC",
                            "baseline_model": baseline_identifier,
                            "model_seed": int(
                                model_seed
                            ),
                            "condition_identifier": condition_identifier,
                            "scope_identifier": scope_identifier,
                            "far_mae": far_mae,
                            "baseline_mae": baseline_mae,
                            "paired_mae_improvement_percent": (
                                paired_improvement_percent
                            ),
                        }
                    )

    return paired_rows


def aggregate_paired_improvement(
    paired_rows: list[dict],
) -> list[dict]:
    grouped_rows = {}

    for row in paired_rows:
        group_key = (
            row["reference_model"],
            row["baseline_model"],
            row["condition_identifier"],
            row["scope_identifier"],
        )

        grouped_rows.setdefault(
            group_key,
            [],
        ).append(row)

    summary_rows = []

    for group_key, rows in sorted(
        grouped_rows.items()
    ):
        (
            reference_model,
            baseline_model,
            condition_identifier,
            scope_identifier,
        ) = group_key

        values = np.asarray(
            [
                row[
                    "paired_mae_improvement_percent"
                ]
                for row in rows
            ],
            dtype=np.float64,
        )

        summary_rows.append(
            {
                "reference_model": reference_model,
                "baseline_model": baseline_model,
                "condition_identifier": condition_identifier,
                "scope_identifier": scope_identifier,
                "seed_count": len(rows),
                "paired_mae_improvement_percent_mean": float(
                    values.mean()
                ),
                "paired_mae_improvement_percent_sample_std": float(
                    values.std(
                        ddof=1
                    )
                ),
            }
        )

    return summary_rows


def write_csv(
    output_path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        raise RuntimeError(
            f"Cannot write an empty CSV: {output_path.name}"
        )

    field_names = sorted(
        {
            field_name
            for row in rows
            for field_name in row.keys()
        }
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=field_names,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                row
            )


def build_metric_tensor(
    per_seed_rows: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    tensor = np.full(
        (
            len(MODEL_IDENTIFIERS),
            len(EXPECTED_MODEL_SEEDS),
            len(CONDITION_ORDER),
            len(SCOPE_ORDER),
            len(METRIC_NAMES),
        ),
        np.nan,
        dtype=np.float64,
    )

    count_tensor = np.zeros(
        (
            len(MODEL_IDENTIFIERS),
            len(EXPECTED_MODEL_SEEDS),
            len(CONDITION_ORDER),
            len(SCOPE_ORDER),
        ),
        dtype=np.int64,
    )

    model_index = {
        model_identifier: index
        for index, model_identifier in enumerate(
            MODEL_IDENTIFIERS
        )
    }

    seed_index = {
        int(model_seed): index
        for index, model_seed in enumerate(
            EXPECTED_MODEL_SEEDS
        )
    }

    condition_index = {
        condition_identifier: index
        for index, condition_identifier in enumerate(
            CONDITION_ORDER
        )
    }

    scope_index = {
        scope_identifier: index
        for index, scope_identifier in enumerate(
            SCOPE_ORDER
        )
    }

    for row in per_seed_rows:
        row_index = (
            model_index[
                row[
                    "model_identifier"
                ]
            ],
            seed_index[
                int(
                    row[
                        "model_seed"
                    ]
                )
            ],
            condition_index[
                row[
                    "condition_identifier"
                ]
            ],
            scope_index[
                row[
                    "scope_identifier"
                ]
            ],
        )

        for metric_position, metric_name in enumerate(
            METRIC_NAMES
        ):
            tensor[
                row_index
                + (
                    metric_position,
                )
            ] = float(
                row[
                    metric_name
                ]
            )

        count_tensor[
            row_index
        ] = int(
            row[
                "scored_cells"
            ]
        )

    if np.any(
        ~np.isfinite(tensor)
    ):
        raise RuntimeError(
            "Metric tensor contains unfilled entries."
        )

    if np.any(
        count_tensor <= 0
    ):
        raise RuntimeError(
            "Metric count tensor contains an empty entry."
        )

    return tensor, count_tensor


def write_final_results_atomically(
    results_directory: Path,
    provenance: dict,
    per_seed_rows: list[dict],
    seed_summary_rows: list[dict],
    degradation_rows: list[dict],
    degradation_summary_rows: list[dict],
    paired_improvement_rows: list[dict],
    paired_improvement_summary_rows: list[dict],
) -> None:
    if results_directory.exists():
        raise FileExistsError(
            "Final result directory already exists and must not be "
            f"overwritten:\n{results_directory}"
        )

    results_directory.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=(
                results_directory.name
                + "_temporary_"
            ),
            dir=results_directory.parent,
        )
    )

    try:
        write_csv(
            temporary_directory
            / "per_seed_metrics.csv",
            per_seed_rows,
        )

        write_csv(
            temporary_directory
            / "seed_summary_metrics.csv",
            seed_summary_rows,
        )

        write_csv(
            temporary_directory
            / "per_seed_clean_to_dropout_degradation.csv",
            degradation_rows,
        )

        write_csv(
            temporary_directory
            / "clean_to_dropout_degradation_summary.csv",
            degradation_summary_rows,
        )

        write_csv(
            temporary_directory
            / "paired_far_mae_improvement.csv",
            paired_improvement_rows,
        )

        write_csv(
            temporary_directory
            / "paired_far_mae_improvement_summary.csv",
            paired_improvement_summary_rows,
        )

        metric_tensor, metric_count_tensor = (
            build_metric_tensor(
                per_seed_rows
            )
        )

        np.savez_compressed(
            temporary_directory
            / "primary_test_metric_tensor.npz",
            model_identifiers=np.asarray(
                MODEL_IDENTIFIERS
            ),
            model_seeds=np.asarray(
                EXPECTED_MODEL_SEEDS,
                dtype=np.int64,
            ),
            condition_identifiers=np.asarray(
                CONDITION_ORDER
            ),
            scope_identifiers=np.asarray(
                SCOPE_ORDER
            ),
            metric_names=np.asarray(
                METRIC_NAMES
            ),
            metric_tensor=metric_tensor,
            scored_cell_tensor=metric_count_tensor,
        )

        final_json = {
            "evaluation_identifier": (
                "pems_bay_final_multi_model_primary_evaluation_v1"
            ),
            "created_utc": datetime.now(
                timezone.utc
            ).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "provenance": provenance,
            "metric_definitions": {
                "mae": (
                    "Mean absolute error in raw traffic-speed units."
                ),
                "rmse": (
                    "Root mean squared error in raw traffic-speed units."
                ),
                "mape_percent": (
                    "Mean absolute percentage error multiplied by 100."
                ),
                "wmape_percent": (
                    "Sum absolute error divided by sum absolute target "
                    "value, multiplied by 100."
                ),
            },
            "aggregation": {
                "model_seeds": list(
                    EXPECTED_MODEL_SEEDS
                ),
                "mean": "Arithmetic mean across five frozen seeds.",
                "sample_standard_deviation": (
                    "Sample standard deviation across five frozen seeds "
                    "with ddof=1."
                ),
                "paired_far_improvement": (
                    "100 * (baseline_seed_MAE - FAR_seed_MAE) / "
                    "baseline_seed_MAE, aggregated across paired seeds."
                ),
            },
            "per_seed_metrics": per_seed_rows,
            "seed_summary_metrics": seed_summary_rows,
            "per_seed_clean_to_dropout_degradation": degradation_rows,
            "clean_to_dropout_degradation_summary": (
                degradation_summary_rows
            ),
            "paired_far_mae_improvement": paired_improvement_rows,
            "paired_far_mae_improvement_summary": (
                paired_improvement_summary_rows
            ),
        }

        (
            temporary_directory
            / "primary_test_evaluation_results.json"
        ).write_text(
            json.dumps(
                final_json,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_directory.rename(
            results_directory
        )

    except Exception:
        if temporary_directory.exists():
            shutil.rmtree(
                temporary_directory,
                ignore_errors=True,
            )

        raise


# ============================================================
# 14. Main final-evaluation procedure
# ============================================================
def run_final_evaluation(
    project_root: Path,
    device_name: str,
) -> None:
    paths = resolve_paths(
        project_root
    )

    if paths[
        "results_directory"
    ].exists():
        raise FileExistsError(
            "Final primary-test outputs already exist and must not "
            "be overwritten:\n"
            f"{paths['results_directory']}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the frozen full primary-test "
            "evaluation."
        )

    device = torch.device(
        device_name
    )

    if device.type != "cuda":
        raise RuntimeError(
            "The evaluator accepts only a CUDA device."
        )

    protocol_bundle, final_protocol = (
        verify_and_load_protocols(
            paths
        )
    )

    assets = load_required_evaluation_assets(
        paths=paths,
        protocol_bundle=protocol_bundle,
    )

    scenario_masks = load_scenario_mask_records(
        paths=paths,
        assets=assets,
    )

    semantic_runner_module = import_module_from_source(
        "frozen_far_gf_rc_semantic_runner",
        paths["far_runner_source"],
    )

    if not hasattr(
        semantic_runner_module,
        "build_calendar_features",
    ):
        raise RuntimeError(
            "Frozen FAR-GF-RC runner lacks build_calendar_features."
        )

    if not hasattr(
        semantic_runner_module,
        "build_native_elapsed_steps",
    ):
        raise RuntimeError(
            "Frozen FAR-GF-RC runner lacks build_native_elapsed_steps."
        )

    calendar_features = (
        semantic_runner_module.build_calendar_features(
            assets[
                "timestamp_values_ns"
            ]
        )
    )

    if calendar_features.shape != (
        RAW_SPEED_END
        - RAW_SPEED_START,
        4,
    ):
        raise RuntimeError(
            "Frozen calendar-feature helper returned an unexpected shape."
        )

    if calendar_features.dtype != np.float32:
        calendar_features = calendar_features.astype(
            np.float32,
            copy=False,
        )

    native_elapsed_steps = (
        semantic_runner_module.build_native_elapsed_steps(
            assets[
                "native_observation_mask"
            ],
            ELAPSED_GAP_CAP_STEPS,
        )
    )

    if native_elapsed_steps.shape != (
        PRIMARY_TEST_END,
        NUMBER_OF_SENSORS,
    ):
        raise RuntimeError(
            "Frozen native elapsed-state helper returned an "
            "unexpected shape."
        )

    if native_elapsed_steps.dtype != np.int16:
        native_elapsed_steps = native_elapsed_steps.astype(
            np.int16,
            copy=False,
        )

    model_registry = build_models_from_frozen_sources(
        paths=paths,
        device=device,
    )

    per_seed_rows = []
    checkpoint_provenance = {}

    for model_identifier in MODEL_IDENTIFIERS:
        checkpoint_provenance[
            model_identifier
        ] = {}

        for model_seed in EXPECTED_MODEL_SEEDS:
            model = instantiate_model(
                model_identifier=model_identifier,
                model_registry=model_registry,
                physical_graph=assets[
                    "physical_graph"
                ],
                device=device,
            )

            checkpoint_metadata = load_selected_checkpoint(
                model=model,
                model_identifier=model_identifier,
                model_seed=model_seed,
                final_protocol=final_protocol,
            )

            checkpoint_provenance[
                model_identifier
            ][
                str(model_seed)
            ] = checkpoint_metadata

            for condition_identifier in CONDITION_ORDER:
                if condition_identifier == "clean_native_history":
                    scenario_mask = None

                else:
                    scenario_mask = scenario_masks[
                        condition_identifier
                    ]

                scope_metrics = evaluate_one_model_seed_condition(
                    model_identifier=model_identifier,
                    model=model,
                    condition_identifier=condition_identifier,
                    artificial_dropout_mask=scenario_mask,
                    assets=assets,
                    calendar_features=calendar_features,
                    native_elapsed_steps=native_elapsed_steps,
                    device=device,
                )

                for scope_identifier in SCOPE_ORDER:
                    metric_record = scope_metrics[
                        scope_identifier
                    ]

                    per_seed_rows.append(
                        {
                            "model_identifier": model_identifier,
                            "model_seed": int(
                                model_seed
                            ),
                            "condition_identifier": condition_identifier,
                            "scope_identifier": scope_identifier,
                            "scored_cells": int(
                                metric_record[
                                    "scored_cells"
                                ]
                            ),
                            "mae": float(
                                metric_record[
                                    "mae"
                                ]
                            ),
                            "rmse": float(
                                metric_record[
                                    "rmse"
                                ]
                            ),
                            "mape_percent": float(
                                metric_record[
                                    "mape_percent"
                                ]
                            ),
                            "wmape_percent": float(
                                metric_record[
                                    "wmape_percent"
                                ]
                            ),
                        }
                    )

                print(
                    "Completed "
                    f"{model_identifier} | Seed {model_seed} | "
                    f"Condition {condition_identifier}"
                )

            del model

            torch.cuda.empty_cache()

    expected_per_seed_row_count = (
        len(MODEL_IDENTIFIERS)
        * len(EXPECTED_MODEL_SEEDS)
        * len(CONDITION_ORDER)
        * len(SCOPE_ORDER)
    )

    if len(per_seed_rows) != expected_per_seed_row_count:
        raise RuntimeError(
            "Per-seed metric row count is incomplete.\n"
            f"Expected: {expected_per_seed_row_count}\n"
            f"Observed: {len(per_seed_rows)}"
        )

    seed_summary_rows = aggregate_seed_metrics(
        per_seed_rows
    )

    degradation_rows = compute_clean_to_dropout_degradation(
        per_seed_rows
    )

    degradation_summary_rows = aggregate_degradation(
        degradation_rows
    )

    paired_improvement_rows = compute_paired_far_improvement(
        per_seed_rows
    )

    paired_improvement_summary_rows = (
        aggregate_paired_improvement(
            paired_improvement_rows
        )
    )

    provenance = {
        "final_protocol_sha256": (
            EXPECTED_FINAL_PROTOCOL_SHA256
        ),
        "boundary_history_addendum_sha256": (
            EXPECTED_BOUNDARY_HISTORY_ADDENDUM_SHA256
        ),
        "elapsed_state_addendum_sha256": (
            EXPECTED_ELAPSED_STATE_ADDENDUM_SHA256
        ),
        "raw_speed_access_interval": [
            RAW_SPEED_START,
            RAW_SPEED_END,
        ],
        "native_mask_access_interval": [
            0,
            PRIMARY_TEST_END,
        ],
        "calibration_target_evaluation_accessed": False,
        "calibration_targets_scored": False,
        "calibration_metrics_computed": False,
        "checkpoint_selection_performed": False,
        "hyperparameter_tuning_performed": False,
        "normalization_update_performed": False,
        "primary_test_target_start_order": (
            "chronological_non_shuffled"
        ),
        "evaluation_batch_size": EVALUATION_BATCH_SIZE,
        "checkpoint_provenance": checkpoint_provenance,
        "verified_asset_hashes": protocol_bundle[
            "hashes"
        ],
        "controlled_dropout_artificial_removal_counts": (
            EXPECTED_ARTIFICIAL_REMOVALS
        ),
    }

    write_final_results_atomically(
        results_directory=paths[
            "results_directory"
        ],
        provenance=provenance,
        per_seed_rows=per_seed_rows,
        seed_summary_rows=seed_summary_rows,
        degradation_rows=degradation_rows,
        degradation_summary_rows=degradation_summary_rows,
        paired_improvement_rows=paired_improvement_rows,
        paired_improvement_summary_rows=(
            paired_improvement_summary_rows
        ),
    )

    print(
        "FINAL PRIMARY-TEST EVALUATION COMPLETED."
    )

    print(
        "Results directory: "
        f"{paths['results_directory']}"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen PEMS-BAY final multi-model primary-test "
            "evaluation exactly once."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help=(
            "Absolute path to traffic_robust_forecasting."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=(
            "CUDA device identifier, for example cuda or cuda:0."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    run_final_evaluation(
        project_root=arguments.project_root,
        device_name=arguments.device,
    )


if __name__ == "__main__":
    main()
