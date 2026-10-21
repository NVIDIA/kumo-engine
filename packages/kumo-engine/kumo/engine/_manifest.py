# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared JSON manifest format for engine artifacts."""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from kumo_core.io.filesystem import atomic_write_bytes

SCHEMA_VERSION = 1


def new_manifest(kind: str, id: str, parents: Mapping[str, Any],
                 **fields: Any) -> Dict[str, Any]:
    from kumo.engine._provenance import environment, scheduler_context

    return {
        'kind': kind,
        'schema_version': SCHEMA_VERSION,
        'id': id,
        'parents': dict(parents),
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'environment': environment(),
        'scheduler': scheduler_context(),
        **fields,
    }


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    payload = json.dumps(manifest, indent=2, sort_keys=True) + '\n'
    atomic_write_bytes(path, payload.encode('utf-8'))


def read_manifest(path: Path) -> Optional[Dict[str, Any]]:
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        warnings.warn(f"Ignoring unreadable manifest {path}: {error}",
                      UserWarning, stacklevel=3)
        return None
    version = (manifest.get('schema_version')
               if isinstance(manifest, dict) else None)
    if type(version) is not int or not 1 <= version <= SCHEMA_VERSION:
        warnings.warn(
            f"Ignoring {path}: manifest schema_version {version!r} is not "
            f"supported by this kumo-engine (up to {SCHEMA_VERSION}).",
            UserWarning,
            stacklevel=3,
        )
        return None
    return manifest
