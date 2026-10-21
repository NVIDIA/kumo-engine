# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Descriptive fit record stored as ``provenance.json`` in a model bundle.

It is integrity-covered by the publication manifest but excluded from the
bundle id, so its volatile fields (times, environment, scheduler) never change
which model a bundle is.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from kumo.engine._manifest import new_manifest, read_manifest, write_manifest
from kumo.engine.pquery.training_table import TrainingTable
from kumo.engine.trainer._spec import MANIFEST_FILE, PROVENANCE_ROLE


def write_provenance(
    bundle_dir: Path,
    *,
    train_table: TrainingTable,
    snapshot_id: str,
    warm_start_bundle_id: Optional[str],
    started_at: datetime,
    published_model: Optional[str],
) -> None:
    from kumo_core.artifacts import PROVENANCE_FILE, add_bundle_file

    config = _read_json(bundle_dir / 'model' / 'config.json') or {}
    finished_at = datetime.now(timezone.utc)
    tt_id = train_table.tt_id
    parents = {
        'training_table': tt_id,
        'snapshot': snapshot_id,
        'warm_start': warm_start_bundle_id,
    }
    write_manifest(
        bundle_dir / PROVENANCE_FILE,
        new_manifest(
            'model_bundle', None, parents,
            fit={
                'started_at': started_at.isoformat(timespec='seconds'),
                'finished_at': finished_at.isoformat(timespec='seconds'),
                'seconds': round((finished_at - started_at).total_seconds(),
                                 3),
            },
            training_table=_training_table(tt_id, train_table.manifest),
            selected_trial=config.get('version'),
            published_model=published_model,
        ))
    add_bundle_file(bundle_dir, PROVENANCE_ROLE, PROVENANCE_FILE)


def _training_table(tt_id: str,
                    manifest: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    manifest = manifest or {}
    parents = manifest.get('parents') or {}
    record = {
        'id': tt_id,
        'source': manifest.get('source'),
        'snapshot': parents.get('snapshot'),
    }
    if record['source'] == 'custom':
        source_table = parents.get('source_table') or {}
        record.update({
            key: source_table.get(key)
            for key in ('connector', 'location', 'version')
        })
    return record


def read_provenance(bundle_dir: Path) -> Optional[Dict[str, Any]]:
    manifest = _read_json(bundle_dir / MANIFEST_FILE) or {}
    for entry in manifest.get('artifacts') or []:
        if entry.get('role') == PROVENANCE_ROLE:
            return read_manifest(bundle_dir / entry['relative_path'])
    return None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
