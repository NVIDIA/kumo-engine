# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Scoring rows against a registered bundle, shared by the MLflow pyfunc
and the Snowflake Model Registry model."""

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
_trainers: Dict[str, Any] = {}
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
    _require_inputs(trainer, snapshot, rows, params, time)
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


def _temporal(trainer: Any, snapshot: GraphSnapshot) -> bool:
    from kumo_core.pquery import query_is_temporal

    from kumo.engine.pquery import PredictiveQuery

    return query_is_temporal(
        PredictiveQuery._from_snapshot(snapshot,
                                       trainer._training_query)._parsed,
        snapshot._native_graph)


def _seconds(values: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(values)
    if stamps.dt.tz is not None:
        stamps = stamps.dt.tz_convert(None)
    return stamps.astype('datetime64[s]')


def _require_inputs(trainer: Any, snapshot: GraphSnapshot, rows: pd.DataFrame,
                    params: Mapping[str, Any],
                    time: Optional[pd.Timestamp]) -> None:
    from kumo_core.pquery import query_is_temporal
    from kumo_core.types import TaskType

    from kumo.engine.pquery import PredictiveQuery

    untimed = len(rows) > 0 and time is None and 'TIMESTAMP' not in rows
    unthresholded = (
        params.get('binary_classification_threshold') is None and
        trainer.result.scoring_defaults.binary_classification_threshold is None)
    if not (untimed or unthresholded):
        return
    parsed = PredictiveQuery._from_snapshot(snapshot,
                                            trainer._training_query)._parsed
    if untimed and query_is_temporal(parsed, snapshot._native_graph):
        raise ValueError("temporal query: pass params={'prediction_time': "
                         "...} or a TIMESTAMP column")
    if unthresholded and (TaskType(parsed.task_type)
                          is TaskType.BINARY_CLASSIFICATION):
        raise ValueError("binary classification: pass "
                         "params={'binary_classification_threshold': ...}")


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
    r"""Scores ``rows`` with the bundle at ``bundle`` against the snapshot
    in ``params`` or the ``serving`` record. The bundle and each snapshot
    load once per process."""
    from kumo.engine.trainer.trainer import Trainer

    record = json.loads(Path(serving).read_text())
    columns = output_columns(Path(bundle))
    if rows.empty:
        return pd.DataFrame(columns=columns)
    uri = params.get('snapshot') or record.get('snapshot')
    if not uri:
        raise ValueError(
            "No snapshot to score against: register with ModelOutputConfig("
            "..., serving_snapshot='@DB.SCHEMA.STAGE/...') or pass "
            "InputSpec(params={'snapshot': ...})")
    trainer = _cached(_trainers, bundle, lambda: Trainer.load(bundle))
    snapshot = _cached(_snapshots, uri, lambda: load_snapshot(uri, record))
    rows = rows.assign(ENTITY=_entities(rows['ENTITY'], trainer, snapshot))
    frame = predict(trainer, snapshot, rows,
                    {k: v for k, v in params.items() if k != 'snapshot'})
    frame = frame.reindex(columns=columns)
    for column in ('ENTITY', 'CLASS'):
        if column in frame:
            frame[column] = frame[column].astype(str)
    return frame


def load_snapshot(uri: str, target: Mapping[str, Any]) -> GraphSnapshot:
    r"""Loads an exported snapshot; an ``@`` stage path is read with the
    Snowpark Container Services identity, in ``target``'s warehouse."""
    if uri.startswith('@'):
        from kumo.engine.connector.snowflake import SnowflakeConnector

        connector: Connector = SnowflakeConnector.from_spcs_env(
            warehouse=target.get('warehouse'),
            database=target.get('database'),
            schema_name=target.get('schema'))
        return GraphSnapshot.load_from_connector(connector, path=uri)
    connector, path = snapshot_location(uri)
    return GraphSnapshot.load_from_connector(connector, path=path)


def _cached(cache: Dict[str, Any], key: str, load: Callable[[], Any]) -> Any:
    with _lock:
        if key not in cache:
            cache[key] = load()
        return cache[key]


def _entities(values: pd.Series, trainer: Any,
              snapshot: GraphSnapshot) -> pd.Series:
    from kumo.engine.pquery import PredictiveQuery

    parsed = PredictiveQuery._from_snapshot(snapshot,
                                            trainer._training_query)._parsed
    entity = snapshot._native_graph.table(parsed.entity_table)
    dtype = str(entity.column(entity.primary_key).dtype)
    if dtype.startswith('int') or dtype.startswith('uint'):
        return pd.to_numeric(values).astype('int64')
    if dtype.startswith('float'):
        return pd.to_numeric(values).astype('float64')
    return values.astype(str)
