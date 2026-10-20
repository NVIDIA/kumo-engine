# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared JSON manifest format for engine artifacts."""

import json
import os
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

SCHEMA_VERSION = 1


def atomic_write(path: Path, payload: bytes) -> None:
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
    atomic_write(path, (json.dumps(manifest, indent=2, sort_keys=True) +
                        '\n').encode('utf-8'))


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
    if not isinstance(version, int) or version > SCHEMA_VERSION:
        warnings.warn(
            f"Ignoring {path}: manifest schema_version {version!r} is not "
            f"supported by this kumo-engine (up to {SCHEMA_VERSION}).",
            UserWarning,
            stacklevel=3,
        )
        return None
    return manifest


def _flatten(value: Mapping[str, Any], prefix: str = '') -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, item in value.items():
        name = f'{prefix}{key}'
        if isinstance(item, Mapping) and item:
            flat.update(_flatten(item, f'{name}.'))
        else:
            flat[name] = item
    return flat


def diff_manifests(a: Mapping[str, Any],
                   b: Mapping[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    """Return ``{dotted_key: (a_value, b_value)}`` for every differing key."""
    left, right = _flatten(a), _flatten(b)
    return {
        key: (left.get(key), right.get(key))
        for key in sorted(left.keys() | right.keys())
        if key not in left or key not in right or left[key] != right[key]
    }
