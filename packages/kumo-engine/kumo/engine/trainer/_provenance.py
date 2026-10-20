# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Descriptive fit record stored as ``provenance.json`` in a model bundle.

It is integrity-covered by the publication manifest but excluded from the
bundle id, so its volatile fields (times, environment, scheduler) never change
which model a bundle is.
"""

import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from kumo.engine._manifest import (
    _flatten,
    new_manifest,
    read_manifest,
    write_manifest,
)
from kumo.engine.pquery.training_table import TrainingTable
from kumo.engine.trainer._spec import (
    MANIFEST_FILE,
    PROVENANCE_ROLE,
    SPEC_VERSION,
)


def write_provenance(
    bundle_dir: Path,
    *,
    train_table: TrainingTable,
    snapshot_id: str,
    warm_start_bundle_id: Optional[str],
    started_at: datetime,
    home_dir: Path,
) -> None:
    from kumo_core.artifacts import PROVENANCE_FILE, add_bundle_file

    config = _read_json(bundle_dir / 'model' / 'config.json') or {}
    finished_at = datetime.now(timezone.utc)
    parents = {
        'training_table': train_table.tt_id,
        'training_table_content_sha256':
        (train_table.manifest or {}).get('content_sha256'),
        'snapshot': snapshot_id,
        'warm_start': warm_start_bundle_id,
    }
    write_manifest(
        bundle_dir / PROVENANCE_FILE,
        new_manifest(
            'model_bundle', None, parents,
            bundle_format_version=SPEC_VERSION,
            seed=config.get('random_state_seed'),
            fit={
                'started_at': started_at.isoformat(timespec='seconds'),
                'finished_at': finished_at.isoformat(timespec='seconds'),
                'seconds': round((finished_at - started_at).total_seconds(),
                                 3),
            },
            selected_trial=config.get('version'),
            trials=_trials(home_dir, Path(config['root_dir'])),
        ))
    add_bundle_file(bundle_dir, PROVENANCE_ROLE, PROVENANCE_FILE)


def read_provenance(bundle_dir: Path) -> Optional[Dict[str, Any]]:
    manifest = _read_json(bundle_dir / MANIFEST_FILE) or {}
    for entry in manifest.get('artifacts') or []:
        if entry.get('role') == PROVENANCE_ROLE:
            return read_manifest(bundle_dir / entry['relative_path'])
    return None


def _trials(home_dir: Path, automl_dir: Path) -> List[Dict[str, Any]]:
    status = {
        entry['trial']: entry
        for entry in (_read_json(home_dir / 'status.json') or {}).get(
            'trials') or []
    }
    versions = {
        int(match.group(1))
        for path in automl_dir.iterdir()
        if (match := re.fullmatch(r'version_(\d+)', path.name))
    }
    trials, plans = [], {}
    for trial in sorted(status.keys() | versions):
        log_dir = automl_dir / f'version_{trial}'
        config = _read_json(log_dir / 'config.json') or {}
        progress = _read_json(log_dir / 'train_progress.json') or {}
        best_epoch = (config.get('refit_cfg') or {}).get('best_val_epoch', -1)
        succeeded = status.get(trial, {}).get('succeed')
        trials.append({
            'trial': trial,
            'status': (None if succeeded is None else
                       'succeeded' if succeeded else 'failed'),
            'seconds': progress.get('elapsed_time'),
            'best_epoch': best_epoch if best_epoch >= 0 else None,
            'metrics': _val_metrics(log_dir / 'val_result.csv'),
        })
        plans[trial] = {
            key: json.dumps(value, sort_keys=True)
            for key, value in _flatten(
                _read_json(log_dir / 'model_plan.json') or {}).items()
        }
    varying = sorted(
        key for key in {key
                        for plan in plans.values() for key in plan}
        if len({plan.get(key)
                for plan in plans.values()}) > 1)
    for entry in trials:
        plan = plans[entry['trial']]
        entry['params'] = {
            key: json.loads(plan[key]) if key in plan else None
            for key in varying
        }
    return trials


def _val_metrics(path: Path) -> Dict[str, float]:
    try:
        with open(path, newline='', encoding='utf-8') as handle:
            row = next(csv.DictReader(handle), None) or {}
    except OSError:
        return {}
    metrics = {}
    for key, value in row.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if key.startswith('val_') and math.isfinite(number):
            metrics[key] = number
    return metrics


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
