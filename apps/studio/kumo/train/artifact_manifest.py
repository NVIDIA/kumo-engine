# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hosted adapter for the shared prediction artifact owner."""

from functools import partial

from kumo_core.train import artifact_manifest as _shared
from kumo_core.train._artifact_storage import _ManifestStorage

from io_utils import fsutil
from io_utils.json import save_json_atomic

_STORAGE = _ManifestStorage(fsutil, save_json_atomic)

PREDICTION_ARTIFACT_MANIFEST_VERSION = _shared.PREDICTION_ARTIFACT_MANIFEST_VERSION
PREDICTION_SHARD_MANIFEST_MAX_BYTES = _shared.PREDICTION_SHARD_MANIFEST_MAX_BYTES
PREDICTION_SELECTION_MANIFEST_MAX_BYTES = _shared.PREDICTION_SELECTION_MANIFEST_MAX_BYTES
PREDICTION_MANIFEST_MAX_ROOTS = _shared.PREDICTION_MANIFEST_MAX_ROOTS
PREDICTION_CONFIG_MAX_BYTES = _shared.PREDICTION_CONFIG_MAX_BYTES
PREDICTION_PRODUCER_ABI = _shared.PREDICTION_PRODUCER_ABI
prediction_shard_producer_abi = _shared.prediction_shard_producer_abi
PredictionProducerMismatchError = _shared.PredictionProducerMismatchError
hash_prediction_config = partial(_shared.hash_prediction_config,
                                 _storage=_STORAGE)
require_owned_path = _shared.require_owned_path
PredictionArtifactInventory = _shared.PredictionArtifactInventory
PredictionShardManifest = _shared.PredictionShardManifest
PredictionManifestRef = _shared.PredictionManifestRef
SelectedPredictionArtifact = _shared.SelectedPredictionArtifact
PredictionSelectionManifest = _shared.PredictionSelectionManifest
PredictionSelectionRef = _shared.PredictionSelectionRef
publish_shard_manifest = partial(_shared.publish_shard_manifest,
                                 _storage=_STORAGE)
load_shard_manifest = partial(_shared.load_shard_manifest, _storage=_STORAGE)
publish_prediction_selection = partial(_shared.publish_prediction_selection,
                                       _storage=_STORAGE)
load_prediction_selection = partial(_shared.load_prediction_selection,
                                    _storage=_STORAGE)
load_current_prediction_selection = partial(
    _shared.load_current_prediction_selection, _storage=_STORAGE)
load_current_prediction_selection_metadata = partial(
    _shared.load_current_prediction_selection_metadata, _storage=_STORAGE)
current_prediction_selection_ref = partial(
    _shared.current_prediction_selection_ref, _storage=_STORAGE)
inventory_generation = _shared.inventory_generation
