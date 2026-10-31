# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Scoring rows with a registered bundle, for MLflow and Snowflake."""

import logging
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import pandas as pd

from kumo.engine.connector.base import Connector
from kumo.engine.graph import GraphSnapshot

logger = logging.getLogger(__name__)
_ANCHOR = 'ANCHOR_TIMESTAMP'


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
    r"""``scored`` in the order of ``rows``."""
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


def parsed_query(trainer: Any, snapshot: GraphSnapshot) -> Any:
    from kumo.engine.pquery import PredictiveQuery

    return PredictiveQuery._from_snapshot(snapshot,
                                          trainer._training_query)._parsed


def _temporal(trainer: Any, snapshot: GraphSnapshot) -> bool:
    from kumo_core.pquery import query_is_temporal

    return query_is_temporal(parsed_query(trainer, snapshot),
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
