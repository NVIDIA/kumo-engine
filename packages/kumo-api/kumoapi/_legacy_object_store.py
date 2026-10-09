# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: MIT

from typing import Any, Callable, Optional


def _enum_value(value: Any) -> Any:
    return getattr(value, 'value', value)


def migrate_connector_config(config: Any,
                             provider: Optional[str] = None) -> Any:
    """Return a copy of a flat GCS/ADLS config using the target union."""
    if not isinstance(config, dict):
        return config
    migrated = dict(config)
    provider = provider or _enum_value(migrated.get('type'))
    fields = {
        'gcs': ('bucket', 'prefix'),
        'adls': ('storage_account', 'filesystem', 'prefix', 'endpoint_suffix'),
    }.get(provider)
    if ('target' in migrated or fields is None
            or not any(name in migrated for name in fields)):
        return migrated
    migrated['target'] = {
        'mode': 'rooted',
        **{
            name: migrated.pop(name)
            for name in fields if name in migrated
        },
    }
    return migrated


def migrate_source_table(source_table: Any,
                         provider: Optional[str] = None) -> Any:
    """Return a copy of a legacy GCS/ADLS table using a relative source."""
    if not isinstance(source_table, dict):
        return source_table
    migrated = dict(source_table)
    provider = provider or _enum_value(migrated.get('data_source_type'))
    if (str(provider).upper() in {'GCS', 'ADLS'} and 'source' not in migrated
            and 'source_table_name' in migrated):
        migrated['source'] = {
            'mode': 'relative',
            'path': migrated.pop('source_table_name'),
        }
    return migrated


def migrate_batch_prediction(config: Any) -> Any:
    """Tag the former S3-only custom prediction source contract."""
    if not isinstance(config, dict):
        return config
    migrated = dict(config)
    source = migrated.get('pred_table_source')
    if isinstance(source, dict) and 'storage_type' not in source:
        migrated['pred_table_source'] = {**source, 'storage_type': 'S3'}
    return migrated


def migrate_validator_values(values: Any, migration: Callable[[Any],
                                                              Any]) -> Any:
    """Apply one pure migration to Pydantic v1/v2 dataclass inputs."""
    kwargs = getattr(values, 'kwargs', None)
    if kwargs is not None:
        migrated = migration(dict(kwargs))
        kwargs.clear()
        kwargs.update(migrated)
        return values
    return migration(values)
