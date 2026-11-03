# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""The predictor a model bundle's manifest names."""

import logging
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Tuple, Union

import pandas as pd

from kumo.engine._deps import imported

TRAINER_PREDICTOR = 'kumo.engine._predictor:TrainerPredictor'
_ANCHOR = 'ANCHOR_TIMESTAMP'
logger = logging.getLogger(__name__)


class Predictor(Protocol):
    r"""Scores ``rows`` (a frame or `PredictionTable`) on ``snapshot``."""
    def predict(self, rows: Any, snapshot: Any,
                params: Mapping[str, Any]) -> Union[pd.DataFrame,
                                                    Tuple[str, ...]]:
        ...


def load_predictor(bundle_dir: Union[str, Path]) -> Predictor:
    r"""The predictor the bundle at ``bundle_dir`` names, loaded."""
    from kumo.engine.trainer._provenance import read_provenance

    manifest = read_provenance(Path(bundle_dir))
    entry = next((record.name
                  for record in (manifest and manifest.dependencies) or ()
                  if record.kind == 'predictor'), TRAINER_PREDICTOR)
    try:
        predictor = imported(entry)
    except (ImportError, AttributeError, ValueError) as error:
        raise ValueError(
            f"The model bundle at {bundle_dir} names predictor {entry!r}, "
            f"which cannot be imported here: {error}") from error
    return predictor.load(bundle_dir)


class TrainerPredictor:
    r"""Scores with the `Trainer` the bundle restores."""
    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    @classmethod
    def load(cls, bundle_dir: Union[str, Path]) -> 'TrainerPredictor':
        from kumo.engine.trainer.trainer import Trainer

        return cls(Trainer.load(bundle_dir))

    def predict(self, rows: Any, snapshot: Any,
                params: Mapping[str, Any]) -> Union[pd.DataFrame,
                                                    Tuple[str, ...]]:
        from kumo.engine.pquery.prediction_table import PredictionTable

        if isinstance(rows, PredictionTable):
            return self.trainer.predict(snapshot, prediction_table=rows,
                                        **params).files
        return predict_frame(self.trainer, snapshot, rows, params)


def predict_frame(trainer: Any, snapshot: Any, rows: pd.DataFrame,
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


def _temporal(trainer: Any, snapshot: Any) -> bool:
    from kumo_core.pquery import query_is_temporal

    from kumo.engine.integrations._serving import parsed_query

    return query_is_temporal(parsed_query(trainer, snapshot),
                             snapshot._native_graph)


def _seconds(values: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(values)
    if stamps.dt.tz is not None:
        stamps = stamps.dt.tz_convert(None)
    return stamps.astype('datetime64[s]')


def _require_inputs(trainer: Any, snapshot: Any, rows: pd.DataFrame,
                    time: Optional[pd.Timestamp]) -> None:
    if len(rows) == 0 or time is not None or 'TIMESTAMP' in rows:
        return
    if _temporal(trainer, snapshot):
        raise ValueError("temporal query: pass params={'prediction_time': "
                         "...} or a TIMESTAMP column")
