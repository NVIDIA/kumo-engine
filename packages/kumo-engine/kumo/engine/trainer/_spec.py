# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Versioned local training metadata stored in every completed bundle.

The spec records the query, model plan, snapshot identity, and effective table
generation settings needed to reconstruct a fitted ``Trainer``. The bundle
manifest integrity-covers the spec. Graph storage and source data remain
separate and must be supplied through a compatible prediction snapshot.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from kumoapi.model_plan import ModelPlan, TrainingTableGenerationPlan

#: Bundle metadata filename and schema version.
SPEC_FILE = 'training-spec.json'
SPEC_VERSION = 1

#: What `kumo_core`'s `VitalArtifactPublicationPolicy` names its own
#: manifest. `write_training_spec` amends this file's `artifacts` list so
#: `training-spec.json` is covered the same way every other bundle file is.
MANIFEST_FILE = 'publication-manifest.json'
SPEC_ROLE = 'training_spec'


@dataclass(frozen=True)
class TrainingSpec:
    r"""Metadata needed to restore a fitted local ``Trainer``.

    ``snapshot_id`` records the fit-time graph and data versions for
    provenance. The graph itself is not embedded in the model bundle;
    prediction receives its graph through an explicit snapshot.
    """

    spec_version: int
    predictive_query: str
    model_plan: Dict[str, Any]
    snapshot_id: str
    training_table_generation: Dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    """Replace one metadata file without exposing partial JSON."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,
                                         prefix=f'.{path.name}.',
                                         delete=False) as out:
            temporary = Path(out.name)
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
        temporary = None
        if hasattr(os, 'O_DIRECTORY'):
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_training_spec(
    bundle_dir: Path,
    *,
    predictive_query: str,
    model_plan: ModelPlan,
    snapshot_id: str,
    training_table_generation: TrainingTableGenerationPlan,
) -> None:
    r"""Write an integrity-covered training spec into a completed bundle."""
    from kumo_core.io.json import to_json_dict

    spec = TrainingSpec(
        spec_version=SPEC_VERSION,
        predictive_query=predictive_query,
        model_plan=to_json_dict(model_plan),
        snapshot_id=snapshot_id,
        training_table_generation=to_json_dict(training_table_generation),
    )
    spec_path = bundle_dir / SPEC_FILE
    _atomic_write(
        spec_path,
        (json.dumps(asdict(spec), indent=2, sort_keys=True) +
         '\n').encode('utf-8'),
    )

    manifest_path = bundle_dir / MANIFEST_FILE
    with open(manifest_path, encoding='utf-8') as handle:
        manifest = json.load(handle)
    manifest['artifacts'] = [
        entry for entry in manifest.get('artifacts') or []
        if entry.get('relative_path') != SPEC_FILE
    ] + [{
        'role': SPEC_ROLE,
        'relative_path': SPEC_FILE,
        'size_bytes': spec_path.stat().st_size,
        'sha256': _sha256(spec_path),
    }]
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode('utf-8'),
    )


def read_training_spec(bundle_dir: Path) -> TrainingSpec:
    r"""Reads back what `write_training_spec` wrote.

    Raises:
        ValueError: If `bundle_dir` carries no `training-spec.json` (not
            published by this SDK) or `publication-manifest.json` (not a
            `kumo_core` bundle at all), or the spec version is one this SDK
            cannot read.
    """
    spec_path = bundle_dir / SPEC_FILE
    if not spec_path.is_file():
        raise ValueError(
            f"No {SPEC_FILE} in {bundle_dir}: it was not written by "
            "Trainer.fit(), so what it was trained from "
            "is not recorded and it cannot be loaded.")
    with open(spec_path, encoding='utf-8') as handle:
        data = json.load(handle)
    if data.get('spec_version') != SPEC_VERSION:
        raise ValueError(
            f"{bundle_dir} carries training-spec version "
            f"{data.get('spec_version')!r}, which this SDK reads only "
            f"version {SPEC_VERSION}.")
    return TrainingSpec(
        spec_version=data['spec_version'],
        predictive_query=data['predictive_query'],
        model_plan=data['model_plan'],
        snapshot_id=data['snapshot_id'],
        training_table_generation=data['training_table_generation'],
    )


def read_bundle_artifact_paths(bundle_dir: Path) -> Dict[str, str]:
    r"""Rebuild the verified ``role -> path`` artifact mapping."""
    from kumo_core.artifacts import verify_model_bundle

    verify_model_bundle(bundle_dir)
    manifest_path = bundle_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise ValueError(
            f"No {MANIFEST_FILE} in {bundle_dir}: it is not a Kumo model "
            "bundle.")
    with open(manifest_path, encoding='utf-8') as handle:
        manifest = json.load(handle)
    artifacts = manifest.get('artifacts') or []
    spec_entries = [
        entry for entry in artifacts if entry.get('role') == SPEC_ROLE
    ]
    if (len(spec_entries) != 1
            or spec_entries[0].get('relative_path') != SPEC_FILE):
        raise ValueError(
            f"The model bundle at {bundle_dir} does not declare exactly one "
            f"integrity-covered {SPEC_FILE} artifact. Fit and export the "
            "model again.")
    return {
        entry['role']: str(bundle_dir / entry['relative_path'])
        for entry in artifacts
        if entry['role'] != SPEC_ROLE
    }
