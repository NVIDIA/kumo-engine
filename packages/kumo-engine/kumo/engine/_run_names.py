# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Run names, the same on every tracker."""

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from kumo.engine._text import entity

if TYPE_CHECKING:
    from kumo.engine._tracking import RunRecord

_KIND_WORDS = {'prediction': 'predict'}


def model_label(registered: Optional[str], query: Optional[str],
                task_type: Optional[str]) -> str:
    r"""The registered name's last part, else <entity>-<task type>."""
    if registered:
        return registered.rsplit('.', 1)[-1]
    return '-'.join(filter(None, (entity(query), task_type))) or 'model'


def run_name(kind: str, core: str, suffix: Optional[str] = None) -> str:
    r"""<kind>-<core><suffix>, the kind not repeated."""
    word = _KIND_WORDS.get(kind, kind)
    named = re.match(rf'{word}([-_]|$)', core, re.IGNORECASE)
    return (core if named else f'{word}-{core}') + (suffix or '')


def versioned(model: str, version: str) -> str:
    r"""<model>-v<version>, or <model>-<version> for a named version."""
    return f'{model}-v{version}' if version.isdigit() else f'{model}-{version}'


def scored_model(model: str, version: Optional[str], bundle_id: str) -> str:
    r"""<model>-v<version>, else <model>-<bundle id prefix>."""
    return versioned(model, version) if version else f'{model}-{bundle_id[:8]}'


def fit_name(model: str, started: datetime) -> str:
    started = started.astimezone(timezone.utc)
    return run_name('fit', f'{model}-{started:%Y%m%d-%H%M%S}')


def trial_name(trial: int, attempt: int, fit: str) -> str:
    return f'trial-{trial}-{fit}' + (f'-attempt-{attempt}' if attempt > 1
                                     else '')


def record_name(record: 'RunRecord', named: Optional[str] = None) -> str:
    r"""The run name of ``record``; ``named`` replaces <model>-v<version>."""
    tags = record.tags
    kind = tags.get('kumo.kind')
    model = record.model_label or 'model'
    if kind == 'fit':
        return run_name('fit', named) if named else fit_name(
            model, record.started or datetime.now(timezone.utc))
    suffix = record.name_suffix or (
        f"-{tags['kumo.as_of']}" if tags.get('kumo.as_of') else '')
    word = (kind if kind != 'monitoring' else
            'eval' if tags.get('kumo.monitor') == 'evaluate' else 'drift')
    core = named or scored_model(model, tags.get('kumo.model_version'),
                                 tags.get('kumo.bundle_id', ''))
    return run_name(word, core, suffix)
