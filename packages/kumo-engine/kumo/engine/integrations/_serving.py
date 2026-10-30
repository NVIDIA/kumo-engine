# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Scoring rows with a registered bundle, for MLflow and Snowflake."""

import json
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from kumo.engine.connector.base import Connector
from kumo.engine.graph import GraphSnapshot

logger = logging.getLogger(__name__)
_ANCHOR = 'ANCHOR_TIMESTAMP'
_lock = threading.Lock()
_predictors: Dict[str, Any] = {}
_snapshots: Dict[str, GraphSnapshot] = {}


def predict(trainer: Any, snapshot: GraphSnapshot, rows: pd.DataFrame,
            params: Mapping[str, Any]) -> pd.DataFrame:
    from kumo.engine._mlops import resolve
    from kumo.engine._scratch import scoped_scratch_dir
    from kumo.engine.connector.file import FileConnector

    time = params.get('prediction_time')
    time = None if time is None else pd.Timestamp(time)
    options = dict(
        binary_classification_threshold=params.get(
            'binary_classification_threshold'),
        num_classes_to_return=params.get('num_classes_to_return'),
    )
    _require_inputs(trainer, snapshot, rows, time)
    trainer.scratch_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=trainer.scratch_dir) as tmp, \
            scoped_scratch_dir(Path(tmp) / 'scratch'):
        if len(rows) == 0:
            return trainer.predict(
                snapshot, prediction_time=None
                if time is None else time.to_pydatetime(), **options).frame
        rows = rows.rename(columns={'ENTITY': 'ENTITY1'})
        if time is not None and 'TIMESTAMP' not in rows:
            rows = rows.assign(TIMESTAMP=time)
        (Path(tmp) / 'rows').mkdir()
        rows.to_parquet(Path(tmp) / 'rows' / 'rows.parquet', index=False)
        entities = FileConnector(root_dir=str(Path(tmp) / 'rows'))[
            'rows.parquet']
        timed = 'TIMESTAMP' in rows and _temporal(trainer, snapshot)
        scored = trainer._predict(
            trainer._prediction_table(snapshot, entities, None),
            snapshot=snapshot, extra_fields=[_ANCHOR] if timed else [],
            settings=resolve(), **options).frame
        return _in_input_order(rows, scored)


def _in_input_order(rows: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    r"""``scored`` in the order of ``rows``, which scoring does not keep;
    rows scoring skipped stay out, and repeated rows pair up in order."""
    keys = ['ENTITY']
    left = pd.DataFrame({'ENTITY': rows['ENTITY1'].to_numpy()})
    right = scored[['ENTITY']].copy()
    if _ANCHOR in scored:
        keys.append(_ANCHOR)
        left[_ANCHOR] = _seconds(rows['TIMESTAMP'])
        right[_ANCHOR] = _seconds(scored[_ANCHOR])
    try:
        left = left.astype({'ENTITY': right['ENTITY'].dtype})
        left['_n'] = left.groupby(keys, sort=False, dropna=False).cumcount()
        right['_n'] = right.groupby(keys, sort=False, dropna=False).cumcount()
        right['_row'] = range(len(right))
        order = left.merge(right, on=[*keys, '_n'])['_row']
    except (TypeError, ValueError):
        order = None
    scored = scored.drop(columns=_ANCHOR, errors='ignore')
    if order is None or len(order) != len(scored):
        logger.warning('Could not match the scores to the input rows; they '
                       'are returned in scoring order: join them on ENTITY')
        return scored
    return scored.iloc[order.to_numpy()].reset_index(drop=True)


def _parsed(trainer: Any, snapshot: GraphSnapshot) -> Any:
    from kumo.engine.pquery import PredictiveQuery

    return PredictiveQuery._from_snapshot(snapshot,
                                          trainer._training_query)._parsed


def _temporal(trainer: Any, snapshot: GraphSnapshot) -> bool:
    from kumo_core.pquery import query_is_temporal

    return query_is_temporal(_parsed(trainer, snapshot),
                             snapshot._native_graph)


def _seconds(values: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(values)
    if stamps.dt.tz is not None:
        stamps = stamps.dt.tz_convert(None)
    return stamps.astype('datetime64[s]')


def _require_inputs(trainer: Any, snapshot: GraphSnapshot, rows: pd.DataFrame,
                    time: Optional[pd.Timestamp]) -> None:
    if len(rows) == 0 or time is not None or 'TIMESTAMP' in rows:
        return
    if _temporal(trainer, snapshot):
        raise ValueError("temporal query: pass params={'prediction_time': "
                         "...} or a TIMESTAMP column")


def snapshot_location(uri: str) -> Tuple[Connector, str]:
    from kumo.engine.connector import connector_for_root

    parent, slash, name = uri.rstrip('/').rpartition('/')
    return connector_for_root(parent or slash or '.'), name


def task_type(bundle: Path) -> Any:
    from kumo_core.types import TaskType

    schema = json.loads((bundle / 'prediction-schema.json').read_text())
    return TaskType(schema['task_type'])


def output_columns(bundle: Path) -> List[str]:
    from kumo_core.train.formatting import prediction_output_columns

    return ['ENTITY', *prediction_output_columns(task_type(bundle))]


def score_registered(bundle: str, serving: str, rows: pd.DataFrame,
                     params: Mapping[str, Any]) -> pd.DataFrame:
    r"""Scores ``rows`` against ``params``' or ``serving``'s snapshot."""
    from kumo.engine._predictor import load_predictor

    record = json.loads(Path(serving).read_text())
    columns = record.get('columns') or output_columns(Path(bundle))
    if rows.empty:
        return pd.DataFrame(columns=columns)
    uri = params.get('snapshot') or record.get('snapshot')
    if not uri:
        raise ValueError(
            "No snapshot to score against: fit with a stage artifact root, "
            "which stores the training snapshot there, or pass "
            "InputSpec(params={'snapshot': '@DB.SCHEMA.STAGE/...'})")
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
    from kumo.engine._deps import Dependencies, asset_locations
    from kumo.engine.trainer._provenance import read_provenance

    roots = sorted({
        record.location.rsplit('/', 2)[0]
        for record in Dependencies.of(read_provenance(Path(bundle))).records
        if (record.location or '').startswith('@')
    })
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
        _parsed(trainer, snapshot).entity_table)
    dtype = str(entity.column(entity.primary_key).dtype)
    if dtype.startswith('int') or dtype.startswith('uint'):
        return pd.to_numeric(values).astype('int64')
    if dtype.startswith('float'):
        return pd.to_numeric(values).astype('float64')
    return values.astype(str)
