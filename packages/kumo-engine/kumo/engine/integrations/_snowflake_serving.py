# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Scoring inside a Snowflake container with a registered version."""

import json
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import pandas as pd

from kumo.engine.connector.base import Connector
from kumo.engine.graph import GraphSnapshot
from kumo.engine.integrations._serving import parsed_query, snapshot_location

_lock = threading.Lock()
_predictors: Dict[str, Any] = {}
_snapshots: Dict[str, GraphSnapshot] = {}


def score_registered(bundle: str, serving: str, rows: pd.DataFrame,
                     params: Mapping[str, Any]) -> pd.DataFrame:
    r"""Scores ``rows`` against ``params``' or ``serving``'s snapshot."""
    from kumo.engine._predictor import load_predictor

    record = json.loads(Path(serving).read_text())
    columns = record['columns']
    if rows.empty:
        return pd.DataFrame(columns=columns)
    uri = params.get('snapshot') or record.get('snapshot')
    if not uri:
        raise ValueError(
            "No snapshot to score against: fit with a stage artifact root "
            "or pass InputSpec(params={'snapshot': '@DB.SCHEMA.STAGE/...'})")
    predictor = _cached(_predictors, bundle,
                        lambda: _staged(bundle, record, load_predictor))
    snapshot = _cached(_snapshots, uri, lambda: load_snapshot(uri, record))
    trainer = getattr(predictor, 'trainer', None)
    if trainer is not None:
        rows = rows.assign(
            ENTITY=_entities(rows['ENTITY'], trainer, snapshot))
    frame = predictor.predict(
        rows, snapshot, {k: v
                         for k, v in params.items() if k != 'snapshot'})
    frame = frame.reindex(columns=columns)
    for column in ('ENTITY', 'CLASS'):
        if column in frame:
            frame[column] = frame[column].astype(str)
    return frame


def _staged(bundle: str, target: Mapping[str, Any],
            load: Callable[[str], Any]) -> Any:
    r"""``load(bundle)``, reading its staged assets as the container."""
    from kumo.engine._deps import asset_locations, stored_roots
    from kumo.engine.trainer._provenance import read_provenance

    roots = stored_roots(read_provenance(Path(bundle)),
                         lambda location: location.startswith('@'))
    if not roots:
        return load(bundle)
    connector = _container_connector(target)
    with asset_locations([(connector, root) for root in roots]):
        return load(bundle)


def load_snapshot(uri: str, target: Mapping[str, Any]) -> GraphSnapshot:
    r"""Loads an exported snapshot, a stage's as the container."""
    if uri.startswith('@'):
        return GraphSnapshot.load_from_connector(_container_connector(target),
                                                 path=uri)
    connector, path = snapshot_location(uri)
    return GraphSnapshot.load_from_connector(connector, path=path)


def _container_connector(target: Mapping[str, Any]) -> Connector:
    from kumo.engine.connector.snowflake import SnowflakeConnector

    return SnowflakeConnector.from_spcs_env(
        warehouse=target.get('warehouse'), database=target.get('database'),
        schema_name=target.get('schema'))


def _cached(cache: Dict[str, Any], key: str, load: Callable[[], Any]) -> Any:
    with _lock:
        if key not in cache:
            cache[key] = load()
        return cache[key]


def _entities(values: pd.Series, trainer: Any,
              snapshot: GraphSnapshot) -> pd.Series:
    entity = snapshot._native_graph.table(
        parsed_query(trainer, snapshot).entity_table)
    dtype = str(entity.column(entity.primary_key).dtype)
    if dtype.startswith('int') or dtype.startswith('uint'):
        return pd.to_numeric(values).astype('int64')
    if dtype.startswith('float'):
        return pd.to_numeric(values).astype('float64')
    return values.astype(str)
