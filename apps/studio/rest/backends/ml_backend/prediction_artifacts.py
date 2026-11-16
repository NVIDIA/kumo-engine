# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve selected batch-prediction artifacts without globbing attempts."""

import os.path as osp
from functools import partial

from kumo_core.train import artifact_manifest as _shared
from kumo_core.types import PredictionArtifactType

from async_utils.sync import blocking_func
from io_utils import fsutil
from kumo.home import HomeDirManager
from kumo.train.artifact_manifest import (
    _STORAGE,
    current_prediction_selection_ref,
    load_current_prediction_selection_metadata,
)

ResolvedPredictionArtifact = _shared.ResolvedPredictionArtifact
selected_prediction_artifact_files = partial(
    _shared.selected_prediction_artifact_files, _storage=_STORAGE)
resolve_prediction_selection_artifact = partial(
    _shared.resolve_prediction_selection_artifact, _storage=_STORAGE)


@blocking_func()
def resolve_prediction_artifact(
    job_home: HomeDirManager,
    artifact_type: PredictionArtifactType,
    *,
    validate_objects: bool = True,
    include_selection_ref: bool = False,
) -> ResolvedPredictionArtifact | None:
    """Resolve a schema-v2 selected artifact; legacy roots are unsupported."""
    output_root = job_home.prediction_output_dir
    pointer_path = osp.join(output_root, '_selected_artifacts.json')
    if fsutil.exists(pointer_path):
        selection = load_current_prediction_selection_metadata(output_root)
        selected = next((artifact for artifact in selection.artifacts
                         if artifact.artifact_type is artifact_type), None)
        if selected is None:
            return None
        return ResolvedPredictionArtifact(
            root=selected.root,
            files=(selected_prediction_artifact_files(selected)
                   if validate_objects else None),
            roots=selected.roots,
            manifest_backed=True,
            selection_ref=(current_prediction_selection_ref(output_root)
                           if include_selection_ref else None),
        )

    return None
